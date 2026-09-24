"""电价、燃料库存、送出线路和提名的事务用例。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Iterable, Mapping

from .clock import SystemClock, parse_utc, utc_text
from .errors import Conflict, Forbidden, InvalidState, NotFound, SupplyError, ValidationFailed
from .models import (
    IndexQuote,
    Facility,
    InventoryLot,
    NominationRequest,
    Route,
    ScenarioSet,
    SupplyScenario,
    date_text,
    positive_integer,
)
from .planning import (
    ALGORITHM_VERSION,
    AllocationRequest,
    PricePoint,
    ZERO,
    allocate_capacity,
    canonical_json,
    decimal_text,
    delivered_after_loss,
    digest,
    effective_capacity,
    latest_streak,
    moving_average,
    quantize_money,
    quantize_volume,
    scenario_projection,
    weighted_inventory_cost,
)
from .storage import initialize, transaction


ROLE_PERMISSIONS = {
    "planner": {
        "quote.write",
        "catalog.write",
        "scenario.write",
        "scenario.run",
        "scenario_set.write",
        "scenario_set.run",
        "scenario_set.read",
    },
    "dispatcher": {"nomination.write", "allocation.run", "transfer.write", "inventory.write"},
    "risk": {
        "outage.write",
        "scenario.approve",
        "scenario_set.approve",
        "scenario_set.read",
        "report.read",
    },
    "auditor": {"scenario_set.read", "report.read", "audit.read"},
}


class SupplyService:
    def __init__(self, connection: sqlite3.Connection, clock=None) -> None:
        self.connection = connection
        self.clock = clock or SystemClock()
        initialize(connection)

    def _now(self) -> str:
        return utc_text(self.clock.now())

    def _user(self, user_id: str) -> sqlite3.Row:
        row = self.connection.execute(
            "SELECT * FROM supply_users WHERE user_id=?", (user_id,)
        ).fetchone()
        if row is None:
            raise NotFound("用户不存在")
        if not row["active"]:
            raise Forbidden("用户已停用")
        return row

    def _require(self, user_id: str, permission: str) -> sqlite3.Row:
        user = self._user(user_id)
        if permission not in ROLE_PERMISSIONS[user["role"]]:
            raise Forbidden(f"角色 {user['role']} 无权执行 {permission}")
        return user

    def _audit(
        self,
        entity_type: str,
        entity_id: str,
        event_type: str,
        actor_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        previous = self.connection.execute(
            "SELECT event_hash FROM supply_audit_events ORDER BY event_id DESC LIMIT 1"
        ).fetchone()
        previous_hash = "0" * 64 if previous is None else previous["event_hash"]
        body = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "event_type": event_type,
            "actor_id": actor_id,
            "payload": payload,
            "created_at": self._now(),
            "previous_hash": previous_hash,
        }
        event_hash = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
        self.connection.execute(
            "INSERT INTO supply_audit_events(entity_type,entity_id,event_type,actor_id,payload_json,"
            "previous_hash,event_hash,created_at) VALUES(?,?,?,?,?,?,?,?)",
            (
                entity_type,
                entity_id,
                event_type,
                actor_id,
                canonical_json(payload),
                previous_hash,
                event_hash,
                body["created_at"],
            ),
        )

    def create_user(self, user_id: str, display_name: str, role: str) -> dict[str, Any]:
        if role not in ROLE_PERMISSIONS:
            raise ValidationFailed("未知角色")
        if not user_id.strip() or not display_name.strip():
            raise ValidationFailed("用户编号和名称不能为空")
        try:
            with transaction(self.connection, immediate=True):
                self.connection.execute(
                    "INSERT INTO supply_users(user_id,display_name,role,created_at) VALUES(?,?,?,?)",
                    (user_id.strip(), display_name.strip(), role, self._now()),
                )
        except sqlite3.IntegrityError as exc:
            raise Conflict("用户已经存在") from exc
        return {"user_id": user_id.strip(), "role": role}

    def record_quote(self, actor_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
        self._require(actor_id, "quote.write")
        quote = IndexQuote.from_dict(raw)
        previous = self.connection.execute(
            "SELECT quote_id,source_revision FROM market_index_quotes WHERE market_index=? AND trade_date=? "
            "ORDER BY quote_id DESC LIMIT 1",
            (quote.market_index, quote.trade_date),
        ).fetchone()
        if previous is not None and previous["source_revision"] == quote.source_revision:
            raise Conflict("同一来源修订已登记")
        try:
            with transaction(self.connection, immediate=True):
                cursor = self.connection.execute(
                    "INSERT INTO market_index_quotes(market_index,trade_date,close_cny,source_revision,observed_at,"
                    "supersedes_quote_id,recorded_by,recorded_at) VALUES(?,?,?,?,?,?,?,?)",
                    (
                        quote.market_index,
                        quote.trade_date,
                        decimal_text(quote.close_cny),
                        quote.source_revision,
                        quote.observed_at,
                        None if previous is None else previous["quote_id"],
                        actor_id,
                        self._now(),
                    ),
                )
                quote_id = int(cursor.lastrowid)
                self._audit(
                    "quote",
                    str(quote_id),
                    "quote.recorded",
                    actor_id,
                    {"market_index": quote.market_index, "trade_date": quote.trade_date},
                )
        except sqlite3.IntegrityError as exc:
            raise Conflict("电价版本冲突") from exc
        return {"quote_id": quote_id, "market_index": quote.market_index, "trade_date": quote.trade_date}

    def price_summary(self, market_index: str, sessions: int = 20) -> dict[str, Any]:
        rows = self.connection.execute(
            "SELECT q.trade_date,q.close_cny FROM market_index_quotes q "
            "JOIN (SELECT trade_date,max(quote_id) quote_id FROM market_index_quotes "
            "WHERE market_index=? GROUP BY trade_date) latest ON latest.quote_id=q.quote_id "
            "ORDER BY q.trade_date DESC LIMIT ?",
            (market_index.upper(), sessions),
        ).fetchall()
        points = [PricePoint(row["trade_date"], Decimal(row["close_cny"])) for row in rows]
        if not points:
            raise NotFound("没有基准电价")
        streak = latest_streak(points)
        average = moving_average(points, min(5, len(points)))
        latest = max(points, key=lambda item: item.trade_date)
        return {
            "market_index": market_index.upper(),
            "latest": {"trade_date": latest.trade_date, "close_cny": decimal_text(latest.close)},
            "latest_streak": None if streak is None else streak.as_dict(),
            "moving_average": None if average is None else decimal_text(average),
            "observations": len(points),
        }

    def create_facility(self, actor_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
        self._require(actor_id, "catalog.write")
        facility = Facility.from_dict(raw)
        try:
            with transaction(self.connection, immediate=True):
                self.connection.execute(
                    "INSERT INTO facilities(facility_id,name,kind,timezone,capacity_mwh,created_at) "
                    "VALUES(?,?,?,?,?,?)",
                    (
                        facility.facility_id,
                        facility.name,
                        facility.kind,
                        facility.timezone,
                        decimal_text(facility.capacity_mwh),
                        self._now(),
                    ),
                )
                self._audit("facility", facility.facility_id, "facility.created", actor_id, raw)
        except sqlite3.IntegrityError as exc:
            raise Conflict("设施编号已经存在") from exc
        return dict(raw)

    def create_route(self, actor_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
        self._require(actor_id, "catalog.write")
        route = Route.from_dict(raw)
        try:
            with transaction(self.connection, immediate=True):
                self.connection.execute(
                    "INSERT INTO routes(route_id,origin_id,destination_id,product,daily_capacity,"
                    "loss_basis_points,transit_hours,created_at) VALUES(?,?,?,?,?,?,?,?)",
                    (
                        route.route_id,
                        route.origin_id,
                        route.destination_id,
                        route.product,
                        decimal_text(route.daily_capacity),
                        route.loss_basis_points,
                        route.transit_hours,
                        self._now(),
                    ),
                )
                self._audit("route", route.route_id, "route.created", actor_id, raw)
        except sqlite3.IntegrityError as exc:
            raise Conflict("送出线路编号冲突或设施不存在") from exc
        return self.route(route.route_id)

    def route(self, route_id: str) -> dict[str, Any]:
        row = self.connection.execute("SELECT * FROM routes WHERE route_id=?", (route_id,)).fetchone()
        if row is None:
            raise NotFound("送出线路不存在")
        return dict(row)

    def announce_outage(
        self,
        actor_id: str,
        route_id: str,
        starts_at: str,
        ends_at: str | None,
        capacity_percent: object,
        reason: str,
    ) -> dict[str, Any]:
        self._require(actor_id, "outage.write")
        self.route(route_id)
        try:
            start = parse_utc(starts_at, "starts_at")
            end = None if ends_at is None else parse_utc(ends_at, "ends_at")
        except ValueError as exc:
            raise ValidationFailed(str(exc)) from exc
        if end is not None and end <= start:
            raise ValidationFailed("ends_at 必须晚于 starts_at")
        percentage = Decimal(str(capacity_percent))
        if percentage < 0 or percentage > 100:
            raise ValidationFailed("capacity_percent 必须在 0 到 100 之间")
        with transaction(self.connection, immediate=True):
            cursor = self.connection.execute(
                "INSERT INTO route_outages(route_id,starts_at,ends_at,capacity_percent,reason,created_by,created_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (route_id, utc_text(start), None if end is None else utc_text(end), decimal_text(percentage), reason, actor_id, self._now()),
            )
            outage_id = int(cursor.lastrowid)
            self._audit("route", route_id, "outage.announced", actor_id, {"outage_id": outage_id})
        return {"outage_id": outage_id, "route_id": route_id, "state": "announced"}

    def add_inventory_lot(self, actor_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
        self._require(actor_id, "inventory.write")
        lot = InventoryLot.from_dict(raw)
        try:
            with transaction(self.connection, immediate=True):
                self.connection.execute(
                    "INSERT INTO inventory_lots(lot_id,facility_id,product,grade,quantity_mwh,available_mwh,"
                    "unit_cost_cny,received_at,created_by,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        lot.lot_id,
                        lot.facility_id,
                        lot.product,
                        lot.grade,
                        decimal_text(lot.quantity_mwh),
                        decimal_text(lot.quantity_mwh),
                        decimal_text(lot.unit_cost_cny),
                        lot.received_at,
                        actor_id,
                        self._now(),
                    ),
                )
                self._audit("inventory_lot", lot.lot_id, "inventory.received", actor_id, raw)
        except sqlite3.IntegrityError as exc:
            raise Conflict("燃料批次冲突或设施不存在") from exc
        return self.inventory_lot(lot.lot_id)

    def inventory_lot(self, lot_id: str) -> dict[str, Any]:
        row = self.connection.execute("SELECT * FROM inventory_lots WHERE lot_id=?", (lot_id,)).fetchone()
        if row is None:
            raise NotFound("燃料批次不存在")
        return dict(row)

    def inventory_summary(self, facility_id: str, product: str) -> dict[str, Any]:
        rows = self.connection.execute(
            "SELECT * FROM inventory_lots WHERE facility_id=? AND product=? ORDER BY received_at,lot_id",
            (facility_id, product),
        ).fetchall()
        return {"facility_id": facility_id, "product": product, **weighted_inventory_cost(rows)}

    def submit_nomination(self, actor_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
        self._require(actor_id, "nomination.write")
        nomination = NominationRequest.from_dict(raw)
        request_digest = digest(raw)
        stored = self.connection.execute(
            "SELECT request_sha256,response_json FROM supply_idempotency WHERE scope='nomination' AND idempotency_key=?",
            (nomination.idempotency_key,),
        ).fetchone()
        if stored is not None:
            if stored["request_sha256"] != request_digest:
                raise Conflict("幂等键对应不同提名内容")
            return json.loads(stored["response_json"])
        route = self.route(nomination.route_id)
        if route["state"] != "active":
            raise InvalidState("送出线路当前不可提名")
        response = {
            "nomination_id": nomination.nomination_id,
            "route_id": nomination.route_id,
            "state": "submitted",
            "revision": 1,
        }
        try:
            with transaction(self.connection, immediate=True):
                self.connection.execute(
                    "INSERT INTO nominations(nomination_id,route_id,shipper_id,service_date,requested_mwh,"
                    "priority,idempotency_key,submitted_by,submitted_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        nomination.nomination_id,
                        nomination.route_id,
                        nomination.shipper_id,
                        nomination.service_date,
                        decimal_text(nomination.requested_mwh),
                        nomination.priority,
                        nomination.idempotency_key,
                        actor_id,
                        self._now(),
                    ),
                )
                self.connection.execute(
                    "INSERT INTO supply_idempotency(scope,idempotency_key,request_sha256,response_json,created_at) "
                    "VALUES('nomination',?,?,?,?)",
                    (nomination.idempotency_key, request_digest, canonical_json(response), self._now()),
                )
                self._audit("nomination", nomination.nomination_id, "nomination.submitted", actor_id, raw)
        except sqlite3.IntegrityError as exc:
            raise Conflict("提名编号或幂等键冲突") from exc
        return response

    def _capacity_for_date(self, route: sqlite3.Row, service_date: str) -> Decimal:
        start = service_date + "T00:00:00Z"
        end = service_date + "T23:59:59Z"
        rows = self.connection.execute(
            "SELECT capacity_percent FROM route_outages WHERE route_id=? AND state IN ('announced','active') "
            "AND starts_at<=? AND (ends_at IS NULL OR ends_at>=?) ORDER BY outage_id",
            (route["route_id"], end, start),
        ).fetchall()
        percentages = [Decimal(row["capacity_percent"]) for row in rows]
        return effective_capacity(Decimal(route["daily_capacity"]), percentages)

    def allocate(self, actor_id: str, route_id: str, service_date: str) -> dict[str, Any]:
        self._require(actor_id, "allocation.run")
        route = self.connection.execute("SELECT * FROM routes WHERE route_id=?", (route_id,)).fetchone()
        if route is None:
            raise NotFound("送出线路不存在")
        nominations = self.connection.execute(
            "SELECT * FROM nominations WHERE route_id=? AND service_date=? AND state='submitted' "
            "ORDER BY priority,submitted_at,nomination_id",
            (route_id, service_date),
        ).fetchall()
        if not nominations:
            raise InvalidState("没有待分配提名")
        requests = [
            AllocationRequest(
                row["nomination_id"],
                Decimal(row["requested_mwh"]),
                int(row["priority"]),
                row["submitted_at"],
            )
            for row in nominations
        ]
        available = self._capacity_for_date(route, service_date)
        input_value = [dict(row) for row in nominations]
        input_sha256 = digest({"route": dict(route), "nominations": input_value, "capacity": str(available)})
        result_rows = allocate_capacity(available, requests)
        result = {
            "route_id": route_id,
            "service_date": service_date,
            "available_capacity": decimal_text(available),
            "allocations": result_rows,
        }
        with transaction(self.connection, immediate=True):
            cursor = self.connection.execute(
                "INSERT INTO allocation_runs(route_id,service_date,input_sha256,available_capacity,result_json,"
                "created_by,created_at) VALUES(?,?,?,?,?,?,?)",
                (route_id, service_date, input_sha256, decimal_text(available), canonical_json(result), actor_id, self._now()),
            )
            for item in result_rows:
                state = "allocated" if Decimal(item["allocated_mwh"]) > 0 else "cancelled"
                self.connection.execute(
                    "UPDATE nominations SET allocated_mwh=?,state=?,revision=revision+1 "
                    "WHERE nomination_id=? AND state='submitted'",
                    (item["allocated_mwh"], state, item["nomination_id"]),
                )
            allocation_id = int(cursor.lastrowid)
            self._audit("route", route_id, "allocation.completed", actor_id, {"allocation_id": allocation_id})
        return {"allocation_id": allocation_id, **result}

    def dispatch_transfer(
        self,
        actor_id: str,
        transfer_id: str,
        nomination_id: str,
        lot_id: str,
        expected_revision: int,
    ) -> dict[str, Any]:
        self._require(actor_id, "transfer.write")
        nomination = self.connection.execute(
            "SELECT n.*,r.loss_basis_points,r.transit_hours,r.origin_id FROM nominations n "
            "JOIN routes r ON r.route_id=n.route_id WHERE n.nomination_id=?",
            (nomination_id,),
        ).fetchone()
        if nomination is None:
            raise NotFound("提名不存在")
        if nomination["state"] != "allocated" or nomination["revision"] != expected_revision:
            raise InvalidState("提名不是当前可送电版本")
        lot = self.connection.execute("SELECT * FROM inventory_lots WHERE lot_id=?", (lot_id,)).fetchone()
        if lot is None:
            raise NotFound("燃料批次不存在")
        allocated = Decimal(nomination["allocated_mwh"])
        available = Decimal(lot["available_mwh"])
        if lot["facility_id"] != nomination["origin_id"] or lot["product"] != self.route(nomination["route_id"])["product"]:
            raise Conflict("燃料批次与送出线路起点或电源类型不匹配")
        if available < allocated:
            raise Conflict("燃料库存不足以完成分配")
        expected_delivery = delivered_after_loss(allocated, int(nomination["loss_basis_points"]))
        departed_at = self._now()
        with transaction(self.connection, immediate=True):
            self.connection.execute(
                "UPDATE inventory_lots SET available_mwh=?,revision=revision+1 WHERE lot_id=? AND revision=?",
                (decimal_text(quantize_volume(available - allocated)), lot_id, lot["revision"]),
            )
            self.connection.execute(
                "UPDATE nominations SET state='in_transit',revision=revision+1 WHERE nomination_id=? AND revision=?",
                (nomination_id, expected_revision),
            )
            self.connection.execute(
                "INSERT INTO transfers(transfer_id,nomination_id,inventory_lot_id,loaded_mwh,"
                "expected_delivered_mwh,departed_at,created_by,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (
                    transfer_id,
                    nomination_id,
                    lot_id,
                    decimal_text(allocated),
                    decimal_text(expected_delivery),
                    departed_at,
                    actor_id,
                    departed_at,
                ),
            )
            self._audit("transfer", transfer_id, "transfer.dispatched", actor_id, {"nomination_id": nomination_id})
        return {
            "transfer_id": transfer_id,
            "state": "in_transit",
            "loaded_mwh": decimal_text(allocated),
            "expected_delivered_mwh": decimal_text(expected_delivery),
            "expected_arrival": utc_text(parse_utc(departed_at) + timedelta(hours=int(nomination["transit_hours"]))),
        }

    def create_scenario(self, actor_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
        self._require(actor_id, "scenario.write")
        scenario = SupplyScenario.from_dict(raw)
        definition = canonical_json(raw)
        content_sha256 = hashlib.sha256(definition.encode("utf-8")).hexdigest()
        try:
            with transaction(self.connection, immediate=True):
                self.connection.execute(
                    "INSERT INTO supply_scenarios(scenario_id,name,definition_json,content_sha256,created_by,created_at) "
                    "VALUES(?,?,?,?,?,?)",
                    (scenario.scenario_id, scenario.name, definition, content_sha256, actor_id, self._now()),
                )
                self._audit("scenario", scenario.scenario_id, "scenario.created", actor_id, {"sha256": content_sha256})
        except sqlite3.IntegrityError as exc:
            raise Conflict("情景编号或内容已经存在") from exc
        return {"scenario_id": scenario.scenario_id, "state": "draft", "sha256": content_sha256}

    def approve_scenario(self, actor_id: str, scenario_id: str, expected_revision: int) -> dict[str, Any]:
        self._require(actor_id, "scenario.approve")
        with transaction(self.connection, immediate=True):
            cursor = self.connection.execute(
                "UPDATE supply_scenarios SET state='approved',revision=revision+1 "
                "WHERE scenario_id=? AND state='draft' AND revision=?",
                (scenario_id, expected_revision),
            )
            if cursor.rowcount != 1:
                raise InvalidState("情景不是当前草稿版本")
            self._audit("scenario", scenario_id, "scenario.approved", actor_id, {})
        return {"scenario_id": scenario_id, "state": "approved", "revision": expected_revision + 1}

    def run_scenario(self, actor_id: str, scenario_id: str, as_of_date: str) -> dict[str, Any]:
        self._require(actor_id, "scenario.run")
        row = self.connection.execute(
            "SELECT * FROM supply_scenarios WHERE scenario_id=?", (scenario_id,)
        ).fetchone()
        if row is None:
            raise NotFound("情景不存在")
        if row["state"] != "approved":
            raise InvalidState("只有已批准情景可以运行")
        outcome = self._execute_scenario(row, as_of_date)
        run_id, _, replayed = self._store_scenario_run(row, as_of_date, outcome, actor_id, audit_event=True)
        return {"run_id": run_id, **outcome["result"], "replayed": replayed}

    def _day_outages(self, as_of_date: str) -> dict[str, list[sqlite3.Row]]:
        rows = self.connection.execute(
            "SELECT * FROM route_outages WHERE state IN ('announced','active') "
            "AND starts_at<=? AND (ends_at IS NULL OR ends_at>=?) ORDER BY outage_id",
            (as_of_date + "T23:59:59Z", as_of_date + "T00:00:00Z"),
        ).fetchall()
        grouped: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            grouped.setdefault(row["route_id"], []).append(row)
        return grouped

    def _execute_scenario(self, scenario_row: sqlite3.Row, as_of_date: str) -> dict[str, Any]:
        """组装输入快照并执行投影。任何输入缺口都抛 SupplyError 由调用方记录为失败。"""
        scenario = SupplyScenario.from_dict(json.loads(scenario_row["definition_json"]))
        price_row = self.connection.execute(
            "SELECT * FROM market_index_quotes WHERE trade_date<=? ORDER BY trade_date DESC,quote_id DESC LIMIT 1",
            (as_of_date,),
        ).fetchone()
        if price_row is None:
            raise InvalidState("截止日期没有可用电价")
        routes = self.connection.execute(
            "SELECT * FROM routes WHERE state='active' ORDER BY route_id"
        ).fetchall()
        outages = self._day_outages(as_of_date)
        route_limits: list[dict[str, Any]] = []
        for route in routes:
            active_outages = outages.get(route["route_id"], [])
            percentages = [Decimal(item["capacity_percent"]) for item in active_outages]
            effective = effective_capacity(Decimal(route["daily_capacity"]), percentages)
            route_limits.append({
                "route_id": route["route_id"],
                "nominal_capacity": decimal_text(Decimal(route["daily_capacity"])),
                "effective_capacity": decimal_text(effective),
                "outages": [
                    {
                        "outage_id": item["outage_id"],
                        "capacity_percent": item["capacity_percent"],
                        "reason": item["reason"],
                    }
                    for item in active_outages
                ],
            })
        lots = self.connection.execute(
            "SELECT lot_id,facility_id,product,grade,quantity_mwh,available_mwh,unit_cost_cny,"
            "received_at,revision FROM inventory_lots ORDER BY lot_id"
        ).fetchall()
        totals: dict[tuple[str, str], Decimal] = {}
        for lot in lots:
            key = (lot["facility_id"], lot["product"])
            totals[key] = totals.get(key, ZERO) + Decimal(lot["available_mwh"])
        inventory = [
            {"facility_id": facility_id, "product": product, "available_mwh": decimal_text(quantize_volume(quantity))}
            for (facility_id, product), quantity in sorted(totals.items())
        ]
        inventory_snapshot = {
            "as_of_date": as_of_date,
            "lots": [dict(lot) for lot in lots],
            "totals": inventory,
        }
        price_snapshot = {
            "quote_id": price_row["quote_id"],
            "market_index": price_row["market_index"],
            "trade_date": price_row["trade_date"],
            "close_cny": price_row["close_cny"],
            "source_revision": price_row["source_revision"],
        }
        # 摘要只包含稳定业务字段，排除 created_at/revision 等随库变化的元数据，
        # 保证在不同数据库中重建相同输入时得到相同的 input_sha256。
        route_inputs = [
            {
                "route_id": item["route_id"],
                "origin_id": item["origin_id"],
                "destination_id": item["destination_id"],
                "product": item["product"],
                "daily_capacity": item["daily_capacity"],
                "loss_basis_points": item["loss_basis_points"],
                "transit_hours": item["transit_hours"],
            }
            for item in routes
        ]
        demand_changes = {key: decimal_text(value) for key, value in sorted(scenario.demand_changes.items())}
        input_value = {
            "algorithm_version": ALGORITHM_VERSION,
            "scenario_sha256": scenario_row["content_sha256"],
            "as_of_date": as_of_date,
            "price": price_row["close_cny"],
            "routes": route_inputs,
            "route_limits": route_limits,
            "inventory": inventory,
        }
        result = scenario_projection(
            current_price=Decimal(price_row["close_cny"]),
            market_index_drop_percent=scenario.market_index_drop_percent,
            routes=routes,
            inventory=inventory,
            route_capacity_changes=scenario.route_capacity_changes,
            demand_changes=scenario.demand_changes,
        )
        return {
            "result": result,
            "input_sha256": digest(input_value),
            "price_snapshot": price_snapshot,
            "demand_changes": demand_changes,
            "route_limits": route_limits,
            "inventory_snapshot": inventory_snapshot,
        }

    def _store_scenario_run(
        self,
        scenario_row: sqlite3.Row,
        as_of_date: str,
        outcome: Mapping[str, Any],
        actor_id: str,
        *,
        audit_event: bool,
    ) -> tuple[int, dict[str, Any], bool]:
        existing = self.connection.execute(
            "SELECT run_id,result_json FROM scenario_runs WHERE scenario_id=? AND as_of_date=? "
            "AND input_sha256=? AND (algorithm_version=? OR algorithm_version IS NULL)",
            (scenario_row["scenario_id"], as_of_date, outcome["input_sha256"], ALGORITHM_VERSION),
        ).fetchone()
        if existing is not None:
            return int(existing["run_id"]), json.loads(existing["result_json"]), True
        with transaction(self.connection, immediate=True):
            cursor = self.connection.execute(
                "INSERT INTO scenario_runs(scenario_id,as_of_date,input_sha256,algorithm_version,"
                "result_json,created_by,created_at) VALUES(?,?,?,?,?,?,?)",
                (
                    scenario_row["scenario_id"],
                    as_of_date,
                    outcome["input_sha256"],
                    ALGORITHM_VERSION,
                    canonical_json(outcome["result"]),
                    actor_id,
                    self._now(),
                ),
            )
            run_id = int(cursor.lastrowid)
            if audit_event:
                self._audit("scenario", scenario_row["scenario_id"], "scenario.executed", actor_id, {"run_id": run_id})
        return run_id, outcome["result"], False

    def _scenario_set_row(self, set_id: str) -> sqlite3.Row:
        row = self.connection.execute("SELECT * FROM scenario_sets WHERE set_id=?", (set_id,)).fetchone()
        if row is None:
            raise NotFound("情景集合不存在")
        return row

    def _validate_set_members(self, scenario_set: ScenarioSet) -> dict[str, sqlite3.Row]:
        placeholders = ",".join("?" for _ in scenario_set.scenario_ids)
        rows = self.connection.execute(
            f"SELECT * FROM supply_scenarios WHERE scenario_id IN ({placeholders})",
            scenario_set.scenario_ids,
        ).fetchall()
        found = {row["scenario_id"]: row for row in rows}
        missing = [scenario_id for scenario_id in scenario_set.scenario_ids if scenario_id not in found]
        if missing:
            raise ValidationFailed(f"情景不存在: {', '.join(missing)}")
        return found

    def _insert_set_members(
        self,
        set_id: str,
        scenario_set: ScenarioSet,
        members: Mapping[str, sqlite3.Row],
    ) -> None:
        for position, scenario_id in enumerate(scenario_set.scenario_ids, start=1):
            scenario = members[scenario_id]
            self.connection.execute(
                "INSERT INTO scenario_set_members(set_id,scenario_id,position,scenario_sha256,scenario_revision) "
                "VALUES(?,?,?,?,?)",
                (set_id, scenario_id, position, scenario["content_sha256"], scenario["revision"]),
            )

    def create_scenario_set(self, actor_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
        self._require(actor_id, "scenario_set.write")
        scenario_set = ScenarioSet.from_dict(raw)
        members = self._validate_set_members(scenario_set)
        definition = canonical_json(raw)
        content_sha256 = hashlib.sha256(definition.encode("utf-8")).hexdigest()
        try:
            with transaction(self.connection, immediate=True):
                self.connection.execute(
                    "INSERT INTO scenario_sets(set_id,name,definition_json,content_sha256,created_by,created_at) "
                    "VALUES(?,?,?,?,?,?)",
                    (scenario_set.set_id, scenario_set.name, definition, content_sha256, actor_id, self._now()),
                )
                self._insert_set_members(scenario_set.set_id, scenario_set, members)
                self._audit("scenario_set", scenario_set.set_id, "scenario_set.created", actor_id, {"sha256": content_sha256})
        except sqlite3.IntegrityError as exc:
            raise Conflict("情景集合编号或内容已经存在") from exc
        return {"set_id": scenario_set.set_id, "state": "draft", "revision": 1, "sha256": content_sha256}

    def update_scenario_set(self, actor_id: str, set_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
        self._require(actor_id, "scenario_set.write")
        row = self._scenario_set_row(set_id)
        if row["state"] != "draft":
            raise InvalidState("只有草稿集合可以编辑")
        expected_revision = positive_integer(raw.get("expected_revision"), "expected_revision")
        if str(raw.get("set_id", set_id)) != set_id:
            raise ValidationFailed("set_id 与路径不一致")
        scenario_set = ScenarioSet.from_dict(raw)
        members = self._validate_set_members(scenario_set)
        definition = canonical_json(raw)
        content_sha256 = hashlib.sha256(definition.encode("utf-8")).hexdigest()
        try:
            with transaction(self.connection, immediate=True):
                cursor = self.connection.execute(
                    "UPDATE scenario_sets SET name=?,definition_json=?,content_sha256=?,revision=revision+1 "
                    "WHERE set_id=? AND state='draft' AND revision=?",
                    (scenario_set.name, definition, content_sha256, set_id, expected_revision),
                )
                if cursor.rowcount != 1:
                    raise InvalidState("情景集合不是当前草稿版本")
                self.connection.execute("DELETE FROM scenario_set_members WHERE set_id=?", (set_id,))
                self._insert_set_members(set_id, scenario_set, members)
                self._audit(
                    "scenario_set", set_id, "scenario_set.updated", actor_id,
                    {"revision": expected_revision + 1, "sha256": content_sha256},
                )
        except sqlite3.IntegrityError as exc:
            raise Conflict("情景集合内容与已有集合重复") from exc
        return {"set_id": set_id, "state": "draft", "revision": expected_revision + 1, "sha256": content_sha256}

    def approve_scenario_set(self, actor_id: str, set_id: str, expected_revision: int) -> dict[str, Any]:
        self._require(actor_id, "scenario_set.approve")
        self._scenario_set_row(set_id)
        with transaction(self.connection, immediate=True):
            drafts = self.connection.execute(
                "SELECT s.scenario_id FROM scenario_set_members m "
                "JOIN supply_scenarios s ON s.scenario_id=m.scenario_id "
                "WHERE m.set_id=? AND s.state<>'approved'",
                (set_id,),
            ).fetchall()
            if drafts:
                raise InvalidState("集合包含未批准情景: " + ", ".join(row["scenario_id"] for row in drafts))
            cursor = self.connection.execute(
                "UPDATE scenario_sets SET state='frozen',revision=revision+1,approved_by=?,approved_at=? "
                "WHERE set_id=? AND state='draft' AND revision=?",
                (actor_id, self._now(), set_id, expected_revision),
            )
            if cursor.rowcount != 1:
                raise InvalidState("情景集合不是当前草稿版本")
            self.connection.execute(
                "UPDATE scenario_set_members SET scenario_sha256=(SELECT content_sha256 FROM supply_scenarios "
                "WHERE supply_scenarios.scenario_id=scenario_set_members.scenario_id), "
                "scenario_revision=(SELECT revision FROM supply_scenarios "
                "WHERE supply_scenarios.scenario_id=scenario_set_members.scenario_id) WHERE set_id=?",
                (set_id,),
            )
            self._audit("scenario_set", set_id, "scenario_set.frozen", actor_id, {"revision": expected_revision + 1})
        return {"set_id": set_id, "state": "frozen", "revision": expected_revision + 1}

    def scenario_set(self, actor_id: str, set_id: str) -> dict[str, Any]:
        self._require(actor_id, "scenario_set.read")
        row = self._scenario_set_row(set_id)
        members = self.connection.execute(
            "SELECT m.scenario_id,m.position,m.scenario_sha256,m.scenario_revision,s.state AS scenario_state "
            "FROM scenario_set_members m JOIN supply_scenarios s ON s.scenario_id=m.scenario_id "
            "WHERE m.set_id=? ORDER BY m.position",
            (set_id,),
        ).fetchall()
        return {
            "set_id": set_id,
            "name": row["name"],
            "state": row["state"],
            "revision": row["revision"],
            "sha256": row["content_sha256"],
            "definition": json.loads(row["definition_json"]),
            "approved_by": row["approved_by"],
            "approved_at": row["approved_at"],
            "members": [dict(item) for item in members],
            "created_at": row["created_at"],
        }

    def run_scenario_set(
        self,
        actor_id: str,
        set_id: str,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, Any]:
        self._require(actor_id, "scenario_set.run")
        row = self._scenario_set_row(set_id)
        if row["state"] != "frozen":
            raise InvalidState("只有审批后冻结的情景集合可以执行")
        definition = json.loads(row["definition_json"])
        date_from = date_text(date_from or definition["date_from"], "date_from")
        date_to = date_text(date_to or definition["date_to"], "date_to")
        if date_from < definition["date_from"] or date_to > definition["date_to"] or date_to < date_from:
            raise ValidationFailed("执行日期区间必须是冻结区间的子集")
        existing = self.connection.execute(
            "SELECT set_run_id,summary_json FROM scenario_set_runs WHERE set_id=? AND content_sha256=? "
            "AND date_from=? AND date_to=? AND algorithm_version=?",
            (set_id, row["content_sha256"], date_from, date_to, ALGORITHM_VERSION),
        ).fetchone()
        if existing is not None:
            return {"set_run_id": existing["set_run_id"], **json.loads(existing["summary_json"]), "replayed": True}

        members = self.connection.execute(
            "SELECT m.scenario_id,m.position,m.scenario_sha256,m.scenario_revision "
            "FROM scenario_set_members m WHERE m.set_id=? ORDER BY m.position",
            (set_id,),
        ).fetchall()
        start = date.fromisoformat(date_from)
        dates = [
            (start + timedelta(days=offset)).isoformat()
            for offset in range((date.fromisoformat(date_to) - start).days + 1)
        ]
        plan = [(member, day) for day in dates for member in members]
        if len(plan) > 5000:
            raise ValidationFailed("批量回放单次最多执行 5000 个情景日")
        with transaction(self.connection, immediate=True):
            cursor = self.connection.execute(
                "INSERT INTO scenario_set_runs(set_id,content_sha256,date_from,date_to,algorithm_version,"
                "state,total_items,summary_json,created_by,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    set_id, row["content_sha256"], date_from, date_to, ALGORITHM_VERSION,
                    "running", len(plan), canonical_json({}), actor_id, self._now(),
                ),
            )
            set_run_id = int(cursor.lastrowid)

        items: list[dict[str, Any]] = []
        succeeded = 0
        failed = 0
        for position, (member, as_of_date) in enumerate(plan, start=1):
            scenario_row = self.connection.execute(
                "SELECT * FROM supply_scenarios WHERE scenario_id=?", (member["scenario_id"],)
            ).fetchone()
            if scenario_row is None:
                raise NotFound("情景不存在")
            item: dict[str, Any] = {
                "position": position,
                "scenario_id": member["scenario_id"],
                "as_of_date": as_of_date,
                "scenario_sha256": member["scenario_sha256"],
                "scenario_revision": member["scenario_revision"],
            }
            try:
                outcome = self._execute_scenario(scenario_row, as_of_date)
                run_id, _, _ = self._store_scenario_run(scenario_row, as_of_date, outcome, actor_id, audit_event=False)
                item.update({
                    "state": "succeeded",
                    "scenario_run_id": run_id,
                    "result": outcome["result"],
                    "input_sha256": outcome["input_sha256"],
                    "algorithm_version": ALGORITHM_VERSION,
                    "price_snapshot": outcome["price_snapshot"],
                    "demand_changes": outcome["demand_changes"],
                    "route_limits": outcome["route_limits"],
                    "inventory_snapshot": outcome["inventory_snapshot"],
                })
                succeeded += 1
            except SupplyError as exc:
                item.update({"state": "failed", "failure_code": exc.code, "failure_reason": str(exc)})
                failed += 1
            self._store_set_run_item(set_run_id, item)
            items.append(item)

        summary = self._build_set_summary(row, date_from, date_to, items, succeeded, failed)
        final_state = "completed" if failed == 0 else "completed_with_failures"
        with transaction(self.connection, immediate=True):
            self.connection.execute(
                "UPDATE scenario_set_runs SET state=?,succeeded_items=?,failed_items=?,summary_json=? "
                "WHERE set_run_id=?",
                (final_state, succeeded, failed, canonical_json(summary), set_run_id),
            )
            self._audit(
                "scenario_set", set_id, "scenario_set.executed", actor_id,
                {"set_run_id": set_run_id, "succeeded": succeeded, "failed": failed},
            )
        return {"set_run_id": set_run_id, **summary, "replayed": False}

    def _store_set_run_item(self, set_run_id: int, item: Mapping[str, Any]) -> None:
        succeeded = item["state"] == "succeeded"
        with transaction(self.connection, immediate=True):
            self.connection.execute(
                "INSERT INTO scenario_set_run_items(set_run_id,position,scenario_id,as_of_date,state,"
                "scenario_run_id,result_json,input_sha256,algorithm_version,scenario_sha256,scenario_revision,"
                "price_snapshot_json,demand_changes_json,route_limits_json,inventory_snapshot_json,"
                "failure_code,failure_reason,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    set_run_id,
                    item["position"],
                    item["scenario_id"],
                    item["as_of_date"],
                    item["state"],
                    item.get("scenario_run_id"),
                    canonical_json(item["result"]) if succeeded else None,
                    item.get("input_sha256"),
                    item.get("algorithm_version"),
                    item["scenario_sha256"],
                    item["scenario_revision"],
                    canonical_json(item["price_snapshot"]) if succeeded else None,
                    canonical_json(item["demand_changes"]) if succeeded else None,
                    canonical_json(item["route_limits"]) if succeeded else None,
                    canonical_json(item["inventory_snapshot"]) if succeeded else None,
                    None if succeeded else item["failure_code"],
                    None if succeeded else item["failure_reason"],
                    self._now(),
                ),
            )

    def _build_set_summary(
        self,
        set_row: sqlite3.Row,
        date_from: str,
        date_to: str,
        items: list[Mapping[str, Any]],
        succeeded: int,
        failed: int,
    ) -> dict[str, Any]:
        sum_capacity = ZERO
        sum_inventory = ZERO
        price_total = ZERO
        fingerprints = []
        for item in items:
            if item["state"] == "succeeded":
                result = item["result"]
                sum_capacity += Decimal(result["total_projected_capacity"])
                sum_inventory += Decimal(result["demand_adjusted_inventory"])
                price_total += Decimal(result["projected_market_index_cny"])
                fingerprint = {
                    "scenario_id": item["scenario_id"],
                    "as_of_date": item["as_of_date"],
                    "state": "succeeded",
                    "scenario_sha256": item["scenario_sha256"],
                    "input_sha256": item["input_sha256"],
                    "algorithm_version": item["algorithm_version"],
                    "result_sha256": digest(result),
                }
            else:
                fingerprint = {
                    "scenario_id": item["scenario_id"],
                    "as_of_date": item["as_of_date"],
                    "state": "failed",
                    "scenario_sha256": item["scenario_sha256"],
                    "failure_code": item["failure_code"],
                    "failure_reason": item["failure_reason"],
                }
            fingerprints.append(fingerprint)
        aggregate = {
            "sum_projected_capacity_mwh": decimal_text(quantize_volume(sum_capacity)),
            "sum_demand_adjusted_inventory_mwh": decimal_text(quantize_volume(sum_inventory)),
            "average_projected_index_cny": (
                decimal_text(quantize_money(price_total / Decimal(succeeded))) if succeeded else None
            ),
        }
        return {
            "set_id": set_row["set_id"],
            "content_sha256": set_row["content_sha256"],
            "date_from": date_from,
            "date_to": date_to,
            "algorithm_version": ALGORITHM_VERSION,
            "total_items": len(items),
            "succeeded_items": succeeded,
            "failed_items": failed,
            "aggregate": aggregate,
            "items_digest": digest(fingerprints),
        }

    def _verify_chain(self) -> dict[str, Any]:
        rows = self.connection.execute("SELECT * FROM supply_audit_events ORDER BY event_id").fetchall()
        previous_hash = "0" * 64
        valid = True
        for row in rows:
            body = {
                "entity_type": row["entity_type"],
                "entity_id": row["entity_id"],
                "event_type": row["event_type"],
                "actor_id": row["actor_id"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
                "previous_hash": row["previous_hash"],
            }
            calculated = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
            if row["previous_hash"] != previous_hash or row["event_hash"] != calculated:
                valid = False
                break
            previous_hash = row["event_hash"]
        return {"valid": valid, "events": len(rows), "head_hash": previous_hash}

    def scenario_set_run_report(self, actor_id: str, set_run_id: int) -> dict[str, Any]:
        self._require(actor_id, "scenario_set.read")
        run = self.connection.execute("SELECT * FROM scenario_set_runs WHERE set_run_id=?", (set_run_id,)).fetchone()
        if run is None:
            raise NotFound("批量回放不存在")
        item_rows = self.connection.execute(
            "SELECT * FROM scenario_set_run_items WHERE set_run_id=? ORDER BY position", (set_run_id,)
        ).fetchall()
        items = []
        for row in item_rows:
            item = {
                "position": row["position"],
                "scenario_id": row["scenario_id"],
                "as_of_date": row["as_of_date"],
                "state": row["state"],
                "scenario_sha256": row["scenario_sha256"],
                "scenario_revision": row["scenario_revision"],
            }
            if row["state"] == "succeeded":
                item.update({
                    "scenario_run_id": row["scenario_run_id"],
                    "result": json.loads(row["result_json"]),
                    "input_sha256": row["input_sha256"],
                    "algorithm_version": row["algorithm_version"],
                    "snapshots": {
                        "price_version": json.loads(row["price_snapshot_json"]),
                        "demand_changes": json.loads(row["demand_changes_json"]),
                        "route_limits": json.loads(row["route_limits_json"]),
                        "inventory": json.loads(row["inventory_snapshot_json"]),
                    },
                })
            else:
                item.update({"failure": {"code": row["failure_code"], "reason": row["failure_reason"]}})
            items.append(item)
        set_id = run["set_id"]
        member_ids = [
            row["scenario_id"]
            for row in self.connection.execute(
                "SELECT scenario_id FROM scenario_set_members WHERE set_id=? ORDER BY position", (set_id,)
            ).fetchall()
        ]
        entity_ids = [set_id, *member_ids]
        placeholders = ",".join("?" for _ in entity_ids)
        event_rows = self.connection.execute(
            "SELECT event_id,entity_type,entity_id,event_type,actor_id,payload_json,previous_hash,event_hash,created_at "
            f"FROM supply_audit_events WHERE (entity_type='scenario_set' AND entity_id=?) "
            f"OR (entity_type='scenario' AND entity_id IN ({placeholders})) ORDER BY event_id",
            [set_id, *entity_ids],
        ).fetchall()
        audit_trail = [
            {
                "event_id": row["event_id"],
                "entity_type": row["entity_type"],
                "entity_id": row["entity_id"],
                "event_type": row["event_type"],
                "actor_id": row["actor_id"],
                "payload": json.loads(row["payload_json"]),
                "previous_hash": row["previous_hash"],
                "event_hash": row["event_hash"],
                "created_at": row["created_at"],
            }
            for row in event_rows
        ]
        return {
            "set_run_id": set_run_id,
            "set_id": set_id,
            "state": run["state"],
            "summary": json.loads(run["summary_json"]),
            "items": items,
            "audit_trail": audit_trail,
            "chain": self._verify_chain(),
            "created_by": run["created_by"],
            "created_at": run["created_at"],
        }

    def audit_chain(self, actor_id: str) -> dict[str, Any]:
        self._require(actor_id, "audit.read")
        return self._verify_chain()
