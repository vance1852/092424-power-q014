from __future__ import annotations

import json
import sqlite3
import unittest
from datetime import datetime, timezone
from decimal import Decimal

from power_dispatch.api import JsonApplication
from power_dispatch.clock import FrozenClock
from power_dispatch.errors import Forbidden, InvalidState
from power_dispatch.planning import ALGORITHM_VERSION
from power_dispatch.service import SupplyService


def scenario_payload(scenario_id: str, *, drop: str, route_change: str, demand: str) -> dict[str, object]:
    return {
        "scenario_id": scenario_id,
        "name": f"情景 {scenario_id}",
        "market_index_drop_percent": drop,
        "route_capacity_changes": {"pipe-a-b": route_change},
        "demand_changes": {"field-a:crude": demand},
    }


class ScenarioSetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.connection = sqlite3.connect(":memory:", isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.clock = FrozenClock(datetime(2026, 9, 24, 8, 0, tzinfo=timezone.utc))
        self.service = SupplyService(self.connection, self.clock)
        for user_id, role in (("plan", "planner"), ("dispatch", "dispatcher"), ("risk", "risk"), ("audit", "auditor")):
            self.service.create_user(user_id, user_id, role)
        self.service.create_facility("plan", {"facility_id": "field-a", "name": "北部电厂", "kind": "storage", "timezone": "Asia/Shanghai", "capacity_mwh": "500000"})
        self.service.create_facility("plan", {"facility_id": "terminal-b", "name": "沿海终端", "kind": "terminal", "timezone": "Asia/Shanghai", "capacity_mwh": "800000"})
        self.service.create_route("plan", {"route_id": "pipe-a-b", "origin_id": "field-a", "destination_id": "terminal-b", "product": "crude", "daily_capacity": "100000", "loss_basis_points": 25, "transit_hours": 36})
        for day in (21, 22, 23):
            self.service.record_quote("plan", {"market_index": "PEAK_VALLEY", "trade_date": f"2026-09-{day}", "close_cny": str(100 - day), "source_revision": f"rev-{day}", "observed_at": f"2026-09-{day}T21:00:00Z"})
        self.service.add_inventory_lot("dispatch", {"lot_id": "lot-1", "facility_id": "field-a", "product": "crude", "grade": "PEAK_VALLEY", "quantity_mwh": "120000", "unit_cost_cny": "91.5", "received_at": "2026-09-20T06:00:00Z"})
        self.service.create_scenario("plan", scenario_payload("s-restart", drop="9", route_change="20", demand="-5"))
        self.service.create_scenario("plan", scenario_payload("s-surge", drop="-4", route_change="-15", demand="8"))
        self.service.approve_scenario("risk", "s-restart", 1)
        self.service.approve_scenario("risk", "s-surge", 1)
        self.set_payload = {
            "set_id": "pack-1",
            "name": "三套峰谷假设对比",
            "scenario_ids": ["s-restart", "s-surge"],
            "date_from": "2026-09-21",
            "date_to": "2026-09-23",
        }

    def tearDown(self) -> None:
        self.connection.close()

    def freeze_set(self, payload: dict[str, object] | None = None) -> dict[str, object]:
        payload = payload or self.set_payload
        self.service.create_scenario_set("plan", payload)
        return self.service.approve_scenario_set("risk", payload["set_id"], 1)

    def test_draft_editing_uses_optimistic_revision_and_blocks_after_freeze(self) -> None:
        created = self.service.create_scenario_set("plan", self.set_payload)
        self.assertEqual(created["revision"], 1)
        updated_payload = dict(self.set_payload, name="改名后的集合", scenario_ids=["s-restart"], expected_revision=1)
        updated = self.service.update_scenario_set("plan", "pack-1", updated_payload)
        self.assertEqual(updated["revision"], 2)
        with self.assertRaises(InvalidState):
            self.service.update_scenario_set("plan", "pack-1", dict(updated_payload, expected_revision=1))
        frozen = self.service.approve_scenario_set("risk", "pack-1", 2)
        self.assertEqual(frozen["state"], "frozen")
        with self.assertRaises(InvalidState):
            self.service.update_scenario_set("plan", "pack-1", dict(updated_payload, expected_revision=2))
        detail = self.service.scenario_set("plan", "pack-1")
        self.assertEqual([m["scenario_id"] for m in detail["members"]], ["s-restart"])
        self.assertEqual(detail["approved_by"], "risk")

    def test_approval_freezes_member_versions(self) -> None:
        self.freeze_set()
        members = self.connection.execute(
            "SELECT scenario_id,scenario_sha256,scenario_revision FROM scenario_set_members ORDER BY position"
        ).fetchall()
        scenarios = {
            row["scenario_id"]: row
            for row in self.connection.execute("SELECT scenario_id,content_sha256,revision FROM supply_scenarios").fetchall()
        }
        for member in members:
            self.assertEqual(member["scenario_sha256"], scenarios[member["scenario_id"]]["content_sha256"])
            self.assertEqual(member["scenario_revision"], scenarios[member["scenario_id"]]["revision"])

    def test_set_with_unapproved_scenario_cannot_freeze_or_run(self) -> None:
        self.service.create_scenario("plan", scenario_payload("s-draft", drop="0", route_change="0", demand="0"))
        payload = dict(self.set_payload, set_id="pack-draft", scenario_ids=["s-restart", "s-draft"])
        self.service.create_scenario_set("plan", payload)
        with self.assertRaises(InvalidState):
            self.service.approve_scenario_set("risk", "pack-draft", 1)
        with self.assertRaises(InvalidState):
            self.service.run_scenario_set("plan", "pack-draft")
        # 草稿集合绝不落账：没有执行审计事件
        count = self.connection.execute(
            "SELECT count(*) FROM supply_audit_events WHERE event_type='scenario_set.executed'"
        ).fetchone()[0]
        self.assertEqual(count, 0)

    def test_batch_runs_date_range_and_replays_with_identical_summary(self) -> None:
        self.freeze_set()
        first = self.service.run_scenario_set("plan", "pack-1")
        self.assertFalse(first["replayed"])
        self.assertEqual(first["total_items"], 6)
        self.assertEqual(first["succeeded_items"], 6)
        self.assertEqual(first["failed_items"], 0)
        self.clock.advance(hours=2)
        second = self.service.run_scenario_set("plan", "pack-1")
        self.assertTrue(second["replayed"])
        self.assertEqual(second["set_run_id"], first["set_run_id"])
        self.assertEqual(second["items_digest"], first["items_digest"])
        self.assertEqual(second["aggregate"], first["aggregate"])

    def test_replay_summary_is_independent_of_clock_and_database(self) -> None:
        self.freeze_set()
        first = self.service.run_scenario_set("plan", "pack-1")

        other_connection = sqlite3.connect(":memory:", isolation_level=None)
        other_connection.row_factory = sqlite3.Row
        other_clock = FrozenClock(datetime(2030, 1, 1, 0, 0, tzinfo=timezone.utc))
        other = SupplyService(other_connection, other_clock)
        for user_id, role in (("plan", "planner"), ("dispatch", "dispatcher"), ("risk", "risk"), ("audit", "auditor")):
            other.create_user(user_id, user_id, role)
        other.create_facility("plan", {"facility_id": "field-a", "name": "北部电厂", "kind": "storage", "timezone": "Asia/Shanghai", "capacity_mwh": "500000"})
        other.create_facility("plan", {"facility_id": "terminal-b", "name": "沿海终端", "kind": "terminal", "timezone": "Asia/Shanghai", "capacity_mwh": "800000"})
        other.create_route("plan", {"route_id": "pipe-a-b", "origin_id": "field-a", "destination_id": "terminal-b", "product": "crude", "daily_capacity": "100000", "loss_basis_points": 25, "transit_hours": 36})
        for day in (21, 22, 23):
            other.record_quote("plan", {"market_index": "PEAK_VALLEY", "trade_date": f"2026-09-{day}", "close_cny": str(100 - day), "source_revision": f"rev-{day}", "observed_at": f"2026-09-{day}T21:00:00Z"})
        other.add_inventory_lot("dispatch", {"lot_id": "lot-1", "facility_id": "field-a", "product": "crude", "grade": "PEAK_VALLEY", "quantity_mwh": "120000", "unit_cost_cny": "91.5", "received_at": "2026-09-20T06:00:00Z"})
        other.create_scenario("plan", scenario_payload("s-restart", drop="9", route_change="20", demand="-5"))
        other.create_scenario("plan", scenario_payload("s-surge", drop="-4", route_change="-15", demand="8"))
        other.approve_scenario("risk", "s-restart", 1)
        other.approve_scenario("risk", "s-surge", 1)
        other.create_scenario_set("plan", self.set_payload)
        other.approve_scenario_set("risk", "pack-1", 1)
        rebuilt = other.run_scenario_set("plan", "pack-1")
        self.assertEqual(rebuilt["items_digest"], first["items_digest"])
        self.assertEqual(rebuilt["aggregate"], first["aggregate"])
        other_connection.close()

    def test_partial_failures_keep_successes_and_reasons(self) -> None:
        payload = dict(self.set_payload, date_from="2026-09-19", date_to="2026-09-23")
        self.freeze_set(payload)
        batch = self.service.run_scenario_set("plan", "pack-1")
        self.assertEqual(batch["total_items"], 10)
        self.assertEqual(batch["succeeded_items"], 6)
        self.assertEqual(batch["failed_items"], 4)
        failed_rows = self.connection.execute(
            "SELECT scenario_id,as_of_date,failure_code,failure_reason,result_json "
            "FROM scenario_set_run_items WHERE state='failed' ORDER BY position"
        ).fetchall()
        self.assertEqual({row["as_of_date"] for row in failed_rows}, {"2026-09-19", "2026-09-20"})
        self.assertTrue(all(row["failure_code"] == "invalid_state" for row in failed_rows))
        self.assertTrue(all(row["failure_reason"] for row in failed_rows))
        self.assertTrue(all(row["result_json"] is None for row in failed_rows))
        state = self.connection.execute("SELECT state FROM scenario_set_runs").fetchone()[0]
        self.assertEqual(state, "completed_with_failures")

    def test_each_result_keeps_full_input_snapshot_and_algorithm_version(self) -> None:
        self.service.announce_outage("risk", "pipe-a-b", "2026-09-22T00:00:00Z", "2026-09-22T23:59:59Z", "50", "检修")
        self.freeze_set()
        batch = self.service.run_scenario_set("plan", "pack-1")
        report = self.service.scenario_set_run_report("audit", batch["set_run_id"])
        for item in report["items"]:
            self.assertEqual(item["algorithm_version"], ALGORITHM_VERSION)
            self.assertRegex(item["input_sha256"], r"^[0-9a-f]{64}$")
            snapshots = item["snapshots"]
            self.assertEqual(snapshots["price_version"]["market_index"], "PEAK_VALLEY")
            self.assertIn("quote_id", snapshots["price_version"])
            self.assertIn("source_revision", snapshots["price_version"])
            self.assertEqual(snapshots["demand_changes"], {"field-a:crude": {"s-restart": "-5", "s-surge": "8"}[item["scenario_id"]]})
            self.assertEqual(snapshots["route_limits"][0]["route_id"], "pipe-a-b")
            self.assertEqual(snapshots["inventory"]["totals"][0]["available_mwh"], "120000.000")
            self.assertEqual(snapshots["inventory"]["lots"][0]["lot_id"], "lot-1")
        surge_22 = next(
            item for item in report["items"]
            if item["scenario_id"] == "s-surge" and item["as_of_date"] == "2026-09-22"
        )
        limit = surge_22["snapshots"]["route_limits"][0]
        self.assertEqual(limit["effective_capacity"], "50000.000")
        self.assertEqual(limit["outages"][0]["reason"], "检修")
        # v1.0.0 投影把情景能力变化作用于线路额定能力；停运降额单独保存在能力限制快照中
        self.assertEqual(surge_22["result"]["total_projected_capacity"], "85000.000")
        stored_algo = self.connection.execute("SELECT DISTINCT algorithm_version FROM scenario_runs").fetchall()
        self.assertEqual([row[0] for row in stored_algo], [ALGORITHM_VERSION])

    def test_snapshot_is_frozen_even_when_inventory_changes_later(self) -> None:
        self.freeze_set()
        batch = self.service.run_scenario_set("plan", "pack-1")
        # 回放后真实库存发生变化（新批次到货），报告中的库存快照必须保持执行时状态
        self.service.add_inventory_lot("dispatch", {"lot_id": "lot-2", "facility_id": "field-a", "product": "crude", "grade": "PEAK_VALLEY", "quantity_mwh": "30000", "unit_cost_cny": "93", "received_at": "2026-09-24T06:00:00Z"})
        report = self.service.scenario_set_run_report("audit", batch["set_run_id"])
        for item in report["items"]:
            self.assertEqual([lot["lot_id"] for lot in item["snapshots"]["inventory"]["lots"]], ["lot-1"])
            self.assertEqual(item["snapshots"]["inventory"]["totals"][0]["available_mwh"], "120000.000")

    def test_execution_date_range_must_be_within_frozen_range(self) -> None:
        self.freeze_set()
        with self.assertRaises(Exception):
            self.service.run_scenario_set("plan", "pack-1", "2026-09-20", "2026-09-23")
        sub = self.service.run_scenario_set("plan", "pack-1", "2026-09-22", "2026-09-23")
        self.assertEqual(sub["total_items"], 4)
        self.assertFalse(sub["replayed"])
        again = self.service.run_scenario_set("plan", "pack-1", "2026-09-22", "2026-09-23")
        self.assertTrue(again["replayed"])
        self.assertEqual(again["set_run_id"], sub["set_run_id"])

    def test_report_requires_read_permission_and_returns_filtered_audit_chain(self) -> None:
        self.freeze_set()
        batch = self.service.run_scenario_set("plan", "pack-1")
        # 无关事件不应出现在集合审计链中
        self.service.record_quote("plan", {"market_index": "PEAK_VALLEY", "trade_date": "2026-09-24", "close_cny": "95", "source_revision": "rev-24", "observed_at": "2026-09-24T21:00:00Z"})
        with self.assertRaises(Forbidden):
            self.service.scenario_set_run_report("dispatch", batch["set_run_id"])
        report = self.service.scenario_set_run_report("risk", batch["set_run_id"])
        self.assertTrue(report["chain"]["valid"])
        event_types = [event["event_type"] for event in report["audit_trail"]]
        self.assertEqual(
            event_types,
            [
                "scenario.created",
                "scenario.created",
                "scenario.approved",
                "scenario.approved",
                "scenario_set.created",
                "scenario_set.frozen",
                "scenario_set.executed",
            ],
        )
        for event in report["audit_trail"]:
            self.assertIn("event_hash", event)
            self.assertIn("previous_hash", event)
        self.assertEqual(report["summary"]["items_digest"], batch["items_digest"])
        self.assertEqual(self.service.scenario_set_run_report("plan", batch["set_run_id"])["state"], "completed")

    def test_role_separation_for_set_lifecycle(self) -> None:
        with self.assertRaises(Forbidden):
            self.service.create_scenario_set("risk", self.set_payload)
        self.service.create_scenario_set("plan", self.set_payload)
        with self.assertRaises(Forbidden):
            self.service.approve_scenario_set("plan", "pack-1", 1)
        with self.assertRaises(Forbidden):
            self.service.approve_scenario_set("audit", "pack-1", 1)
        self.service.approve_scenario_set("risk", "pack-1", 1)
        with self.assertRaises(Forbidden):
            self.service.run_scenario_set("audit", "pack-1")

    def test_api_exposes_scenario_set_endpoints(self) -> None:
        app = JsonApplication(self.service)
        created = app.handle("POST", "/scenario-sets", {"X-Actor-Id": "plan"}, json.dumps(self.set_payload).encode())
        self.assertEqual(created.status, 201)
        forbidden = app.handle(
            "POST", "/scenario-sets/pack-1/approve", {"X-Actor-Id": "plan"}, b'{"expected_revision":1}'
        )
        self.assertEqual(forbidden.status, 403)
        approved = app.handle(
            "POST", "/scenario-sets/pack-1/approve", {"X-Actor-Id": "risk"}, b'{"expected_revision":1}'
        )
        self.assertEqual(approved.status, 200)
        edited = app.handle(
            "PUT", "/scenario-sets/pack-1", {"X-Actor-Id": "plan"},
            json.dumps(dict(self.set_payload, name="冻结后不可改")).encode(),
        )
        self.assertEqual(edited.status, 409)
        executed = app.handle("POST", "/scenario-sets/pack-1/runs", {"X-Actor-Id": "plan"}, b"{}")
        self.assertEqual(executed.status, 200)
        set_run_id = executed.body["set_run_id"]
        report = app.handle("GET", f"/scenario-set-runs/{set_run_id}", {"X-Actor-Id": "audit"})
        self.assertEqual(report.status, 200)
        self.assertEqual(len(report.body["items"]), 6)
        self.assertTrue(report.body["chain"]["valid"])
        no_actor = app.handle("GET", f"/scenario-set-runs/{set_run_id}")
        self.assertEqual(no_actor.status, 422)


if __name__ == "__main__":
    unittest.main()
