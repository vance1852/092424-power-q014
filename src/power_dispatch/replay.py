"""情景集合批量回放：输入快照、确定性投影与摘要。

每次回放都在执行时刻为每个“情景 × 日期”冻结一套完整输入快照
（价格版本、需求变更、能力限制、库存快照、情景假设），配合算法
版本计算结果。只要快照和算法版本不变，同一集合的重放必然得到
相同摘要。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Mapping, Sequence

from .models import SupplyScenario
from .planning import (
    HUNDRED,
    ZERO,
    canonical_json,
    decimal_text,
    digest,
    quantize_money,
    quantize_volume,
    scenario_projection,
)

# 投影算法版本。快照结构、收入口径或量化规则发生不兼容变化时必须提升，
# 历史结果仍然保留，但新回放会在新版本下重新计算。
ALGORITHM_VERSION = "dispatch-projection-1.0.0"


def enumerate_dates(start_date: str, end_date: str) -> list[str]:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if end < start:
        raise ValueError("结束日期不能早于开始日期")
    span = (end - start).days
    if span > 366:
        raise ValueError("日期区间不能超过 366 天")
    return [(start + timedelta(days=offset)).isoformat() for offset in range(span + 1)]


def latest_price(connection: sqlite3.Connection, service_date: str) -> sqlite3.Row | None:
    return connection.execute(
        "SELECT quote_id,market_index,trade_date,close_cny,source_revision,recorded_at "
        "FROM market_index_quotes WHERE trade_date<=? "
        "ORDER BY trade_date DESC,quote_id DESC LIMIT 1",
        (service_date,),
    ).fetchone()


def route_restrictions(
    connection: sqlite3.Connection, service_date: str, route_capacity_changes: Mapping[str, Decimal]
) -> list[dict[str, Any]]:
    """构造送出线路能力限制快照（含当日停运降容与情景调整）。"""
    routes = connection.execute(
        "SELECT * FROM routes WHERE state='active' ORDER BY route_id"
    ).fetchall()
    day_start = service_date + "T00:00:00Z"
    day_end = service_date + "T23:59:59Z"
    rows: list[dict[str, Any]] = []
    for route in routes:
        outages = connection.execute(
            "SELECT outage_id,capacity_percent,reason,state FROM route_outages "
            "WHERE route_id=? AND state IN ('announced','active') "
            "AND starts_at<=? AND (ends_at IS NULL OR ends_at>=?) ORDER BY outage_id",
            (route["route_id"], day_end, day_start),
        ).fetchall()
        nominal = Decimal(route["daily_capacity"])
        effective = nominal
        for outage in outages:
            percentage = Decimal(outage["capacity_percent"])
            effective *= max(ZERO, min(HUNDRED, percentage)) / HUNDRED
        scenario_change = route_capacity_changes.get(route["route_id"], ZERO)
        effective = max(ZERO, effective * (Decimal(1) + scenario_change / HUNDRED))
        rows.append({
            "route_id": route["route_id"],
            "product": route["product"],
            "nominal_daily_capacity": decimal_text(quantize_volume(nominal)),
            "scenario_change_percent": decimal_text(scenario_change),
            "effective_capacity": decimal_text(quantize_volume(effective)),
            "outage_restrictions": [
                {
                    "outage_id": int(item["outage_id"]),
                    "capacity_percent": str(item["capacity_percent"]),
                    "reason": item["reason"],
                    "state": item["state"],
                }
                for item in outages
            ],
        })
    return rows


def inventory_snapshot(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    lots = connection.execute(
        "SELECT lot_id,facility_id,product,grade,quantity_mwh,available_mwh,unit_cost_cny,"
        "received_at,revision FROM inventory_lots ORDER BY lot_id"
    ).fetchall()
    return [dict(lot) for lot in lots]


def build_snapshot(
    *,
    connection: sqlite3.Connection,
    scenario_row: sqlite3.Row,
    service_date: str,
) -> dict[str, Any]:
    price = latest_price(connection, service_date)
    definition = json.loads(scenario_row["definition_json"])
    scenario = SupplyScenario.from_dict(definition)
    snapshot = {
        "scenario_id": scenario_row["scenario_id"],
        "scenario_sha256": scenario_row["content_sha256"],
        "service_date": service_date,
        "algorithm_version": ALGORITHM_VERSION,
        "price_version": None
        if price is None
        else {
            "quote_id": int(price["quote_id"]),
            "market_index": price["market_index"],
            "trade_date": price["trade_date"],
            "close_cny": price["close_cny"],
            "source_revision": price["source_revision"],
            "recorded_at": price["recorded_at"],
        },
        "scenario_assumptions": {
            "market_index_drop_percent": decimal_text(scenario.market_index_drop_percent),
            "demand_changes": {
                key: decimal_text(value) for key, value in sorted(scenario.demand_changes.items())
            },
        },
        "capacity_restrictions": route_restrictions(
            connection, service_date, scenario.route_capacity_changes
        ),
        "inventory_snapshot": inventory_snapshot(connection),
    }
    snapshot["snapshot_sha256"] = digest(
        {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    )
    return snapshot


def _aggregate_inventory(lots: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    aggregated: dict[tuple[str, str], Decimal] = {}
    for lot in lots:
        key = (str(lot["facility_id"]), str(lot["product"]))
        aggregated[key] = aggregated.get(key, ZERO) + Decimal(str(lot["available_mwh"]))
    return [
        {
            "facility_id": facility,
            "product": product,
            "available_mwh": decimal_text(quantize_volume(available)),
        }
        for (facility, product), available in sorted(aggregated.items())
    ]


def project_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """对冻结快照执行确定性投影，产出含燃料余额与收入口径的结果。"""
    price_version = snapshot["price_version"]
    if price_version is None:
        raise ValueError("截止日期没有可用电价版本")
    assumptions = snapshot["scenario_assumptions"]
    drop_percent = Decimal(str(assumptions["market_index_drop_percent"]))
    demand_changes = {
        key: Decimal(str(value)) for key, value in sorted(assumptions["demand_changes"].items())
    }
    restrictions = snapshot["capacity_restrictions"]
    # 能力限制已折叠停运降容与情景调整，投影时不再重复施加路线变化。
    projection_routes = [
        {"route_id": item["route_id"], "daily_capacity": item["effective_capacity"]}
        for item in restrictions
    ]
    inventory_rows = _aggregate_inventory(snapshot["inventory_snapshot"])
    projection = scenario_projection(
        current_price=Decimal(price_version["close_cny"]),
        market_index_drop_percent=drop_percent,
        routes=projection_routes,
        inventory=inventory_rows,
        route_capacity_changes={},
        demand_changes=demand_changes,
    )
    projected_price = Decimal(projection["projected_market_index_cny"])
    total_capacity = Decimal(projection["total_projected_capacity"])
    fuel_balance_mwh = Decimal(projection["demand_adjusted_inventory"])
    # 收入口径：以情景价格对受能力限制后的可送出电量估值；
    # 燃料余额按需求调整后库存乘以库存加权单位成本折算货币价值。
    total_value = ZERO
    total_quantity = ZERO
    for lot in snapshot["inventory_snapshot"]:
        available = Decimal(str(lot["available_mwh"]))
        total_quantity += available
        total_value += available * Decimal(str(lot["unit_cost_cny"]))
    weighted_cost = ZERO if total_quantity == ZERO else total_value / total_quantity
    return {
        "projected_price_cny": decimal_text(projected_price),
        "total_effective_capacity_mwh": decimal_text(total_capacity),
        "demand_adjusted_inventory_mwh": decimal_text(fuel_balance_mwh),
        "estimated_revenue_cny": decimal_text(quantize_money(total_capacity * projected_price)),
        "fuel_balance_value_cny": decimal_text(quantize_money(fuel_balance_mwh * weighted_cost)),
        "routes": projection["routes"],
        "inventory": projection["inventory"],
    }


def summarize_items(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """对一批回放结果计算确定性集合级摘要。"""
    total_revenue = ZERO
    total_fuel_value = ZERO
    succeeded = 0
    failed = 0
    by_scenario: dict[str, dict[str, Any]] = {}
    for item in sorted(items, key=lambda row: (row["scenario_id"], row["service_date"])):
        if item["status"] != "succeeded":
            failed += 1
            by_scenario.setdefault(
                item["scenario_id"],
                {"runs": 0, "failures": 0, "revenue_cny": "0.00", "fuel_balance_cny": "0.00"},
            )["failures"] += 1
            continue
        succeeded += 1
        result = json.loads(item["result_json"])
        revenue = Decimal(result["estimated_revenue_cny"])
        fuel_value = Decimal(result["fuel_balance_value_cny"])
        total_revenue += revenue
        total_fuel_value += fuel_value
        bucket = by_scenario.setdefault(
            item["scenario_id"],
            {"runs": 0, "failures": 0, "revenue_cny": "0.00", "fuel_balance_cny": "0.00"},
        )
        bucket["runs"] += 1
        bucket["revenue_cny"] = decimal_text(Decimal(bucket["revenue_cny"]) + revenue)
        bucket["fuel_balance_cny"] = decimal_text(Decimal(bucket["fuel_balance_cny"]) + fuel_value)
    return {
        "results_count": succeeded,
        "failure_count": failed,
        "total_estimated_revenue_cny": decimal_text(quantize_money(total_revenue)),
        "total_fuel_balance_value_cny": decimal_text(quantize_money(total_fuel_value)),
        "by_scenario": {key: by_scenario[key] for key in sorted(by_scenario)},
    }


def items_digest(items: Sequence[Mapping[str, Any]]) -> str:
    canonical_items = [
        {
            "scenario_id": row["scenario_id"],
            "service_date": row["service_date"],
            "status": row["status"],
            "input_sha256": row["input_sha256"],
            "result_json": None if row["result_json"] is None else json.loads(row["result_json"]),
            "failure_code": row["failure_code"],
            "failure_message": row["failure_message"],
        }
        for row in sorted(items, key=lambda entry: (entry["scenario_id"], entry["service_date"]))
    ]
    return digest(canonical_items)
