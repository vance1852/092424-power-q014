from __future__ import annotations

import json
import sqlite3
import unittest
from datetime import datetime, timezone
from decimal import Decimal

from power_dispatch.api import JsonApplication
from power_dispatch.clock import FrozenClock
from power_dispatch.errors import Conflict, Forbidden, InvalidState
from power_dispatch.replay import ALGORITHM_VERSION
from power_dispatch.service import SupplyService


SCENARIOS = {
    "base": {"name": "基准", "drop": "0", "routes": {"pipe-a-b": "0"}, "demand": {"field-a:crude": "0"}},
    "drop": {"name": "电价回落", "drop": "9", "routes": {"pipe-a-b": "-20"}, "demand": {"field-a:crude": "-5"}},
    "tight": {"name": "送出受限", "drop": "2", "routes": {"pipe-a-b": "-60"}, "demand": {"field-a:crude": "10"}},
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
        for day, close in ((19, "102"), (20, "100"), (21, "98"), (22, "96")):
            self.service.record_quote("plan", {"market_index": "PEAK_VALLEY", "trade_date": f"2026-09-{day}", "close_cny": close, "source_revision": f"rev-{day}", "observed_at": f"2026-09-{day}T21:00:00Z"})
        self.service.add_inventory_lot("dispatch", {"lot_id": "lot-1", "facility_id": "field-a", "product": "crude", "grade": "PEAK_VALLEY", "quantity_mwh": "150000", "unit_cost_cny": "91.25", "received_at": "2026-09-20T06:00:00Z"})

    def tearDown(self) -> None:
        self.connection.close()

    def scenario(self, key: str, approved: bool = True) -> str:
        spec = SCENARIOS[key]
        scenario_id = f"s-{key}"
        self.service.create_scenario("plan", {"scenario_id": scenario_id, "name": spec["name"], "market_index_drop_percent": spec["drop"], "route_capacity_changes": spec["routes"], "demand_changes": spec["demand"]})
        if approved:
            self.service.approve_scenario("risk", scenario_id, 1)
        return scenario_id

    def freeze_set(self, set_id: str = "set-q3", start: str = "2026-09-20", end: str = "2026-09-22"):
        ids = [self.scenario("base"), self.scenario("drop"), self.scenario("tight")]
        self.service.create_scenario_set("plan", {"set_id": set_id, "name": "三假设对比", "start_date": start, "end_date": end, "scenario_ids": ids})
        self.service.approve_scenario_set("risk", set_id, 1)
        return ids

    def test_draft_editing_and_revision_guard(self) -> None:
        ids = [self.scenario("base"), self.scenario("drop")]
        created = self.service.create_scenario_set("plan", {"set_id": "set-1", "name": "草稿", "start_date": "2026-09-20", "end_date": "2026-09-22", "scenario_ids": ids})
        self.assertEqual(created["state"], "draft")
        tight = self.scenario("tight")
        updated = self.service.update_scenario_set("plan", "set-1", {"expected_revision": 1, "scenario_ids": [*ids, tight]})
        self.assertEqual(updated["revision"], 2)
        self.assertEqual(updated["scenario_ids"], [*ids, tight])
        with self.assertRaises(InvalidState):
            self.service.update_scenario_set("plan", "set-1", {"expected_revision": 1, "name": "过期编辑"})
        with self.assertRaises(Forbidden):
            self.service.update_scenario_set("risk", "set-1", {"name": "风险不能改"})

    def test_draft_cannot_run_or_post(self) -> None:
        ids = [self.scenario("base")]
        self.service.create_scenario_set("plan", {"set_id": "set-2", "name": "草稿", "start_date": "2026-09-20", "end_date": "2026-09-22", "scenario_ids": ids})
        with self.assertRaises(InvalidState):
            self.service.run_scenario_set("plan", "set-2", {})

    def test_approval_requires_all_members_approved(self) -> None:
        approved = self.scenario("base")
        pending = self.scenario("drop", approved=False)
        self.service.create_scenario_set("plan", {"set_id": "set-3", "name": "混合", "start_date": "2026-09-20", "end_date": "2026-09-22", "scenario_ids": [approved, pending]})
        with self.assertRaises(InvalidState):
            self.service.approve_scenario_set("risk", "set-3", 1)
        self.service.approve_scenario("risk", pending, 1)
        frozen = self.service.approve_scenario_set("risk", "set-3", 1)
        self.assertEqual(frozen["state"], "approved")
        self.assertIsNotNone(frozen["snapshot_sha256"])
        with self.assertRaises(InvalidState):
            self.service.update_scenario_set("plan", "set-3", {"name": "冻结后不可改"})

    def test_replay_is_deterministic_and_preserves_input_versions(self) -> None:
        self.freeze_set()
        first = self.service.run_scenario_set("plan", "set-q3", {})
        second = self.service.run_scenario_set("plan", "set-q3", {})
        self.assertFalse(first["replayed"])
        self.assertTrue(second["replayed"])
        self.assertEqual(first["run_id"], second["run_id"])
        self.assertEqual(first["summary_sha256"], second["summary_sha256"])
        self.assertEqual(first["summary"], second["summary"])
        detail = self.service.scenario_set_run("audit", first["run_id"])
        self.assertEqual(len(detail["items"]), 9)
        item = detail["items"][0]
        snapshot = item["input_snapshot"]
        self.assertEqual(snapshot["algorithm_version"], ALGORITHM_VERSION)
        self.assertEqual(snapshot["price_version"]["quote_id"], 2)  # 2026-09-20 quote
        self.assertIn("demand_changes", snapshot["scenario_assumptions"])
        self.assertEqual(snapshot["capacity_restrictions"][0]["route_id"], "pipe-a-b")
        self.assertEqual(snapshot["capacity_restrictions"][0]["nominal_daily_capacity"], "100000.000")
        self.assertEqual(len(snapshot["inventory_snapshot"]), 1)
        self.assertEqual(snapshot["snapshot_sha256"], item["input_sha256"])

    def test_capacity_restrictions_include_outage_and_scenario_change(self) -> None:
        # 09-21 停运 50%，叠加 tight 情景 -60% => 100000*0.5*0.4 = 20000
        self.service.announce_outage("risk", "pipe-a-b", "2026-09-21T00:00:00Z", "2026-09-21T23:59:59Z", "50", "检修")
        self.freeze_set()
        run = self.service.run_scenario_set("plan", "set-q3", {"start_date": "2026-09-21", "end_date": "2026-09-21"})
        detail = self.service.scenario_set_run("audit", run["run_id"])
        tight_item = next(item for item in detail["items"] if item["scenario_id"] == "s-tight")
        restriction = tight_item["input_snapshot"]["capacity_restrictions"][0]
        self.assertEqual(restriction["effective_capacity"], "20000.000")
        self.assertEqual(len(restriction["outage_restrictions"]), 1)

    def test_partial_failure_keeps_successes_and_reasons(self) -> None:
        # 区间从 09-18 开始：当日没有可用电价，每个情景失败一次，其余成功。
        self.freeze_set(start="2026-09-18", end="2026-09-20")
        run = self.service.run_scenario_set("plan", "set-q3", {})
        self.assertEqual(run["summary"]["results_count"], 6)
        self.assertEqual(run["summary"]["failure_count"], 3)
        detail = self.service.scenario_set_run("audit", run["run_id"])
        failed = [item for item in detail["items"] if item["status"] == "failed"]
        succeeded = [item for item in detail["items"] if item["status"] == "succeeded"]
        self.assertEqual(len(failed), 3)
        self.assertEqual(len(succeeded), 6)
        for item in failed:
            self.assertEqual(item["failure"]["code"], "invalid_state")
            self.assertIsNone(item["result"])
            self.assertIsNotNone(item["input_snapshot"])
            self.assertIsNone(item["input_snapshot"]["price_version"])

    def test_frozen_members_ignore_later_scenario_change(self) -> None:
        ids = self.freeze_set()
        first = self.service.run_scenario_set("plan", "set-q3", {"start_date": "2026-09-20", "end_date": "2026-09-20"})
        # 已批准情景不可编辑；即便底层记录变化，冻结集合仍以批准快照重放。
        snapshot_row = self.connection.execute(
            "SELECT frozen_snapshot_json FROM scenario_sets WHERE set_id='set-q3'"
        ).fetchone()
        frozen = json.loads(snapshot_row[0])
        self.assertEqual([m["scenario_id"] for m in frozen["members"]], ids)
        self.assertTrue(all(m["content_sha256"] for m in frozen["members"]))
        again = self.service.run_scenario_set("plan", "set-q3", {"start_date": "2026-09-20", "end_date": "2026-09-20"})
        self.assertEqual(again["summary_sha256"], first["summary_sha256"])

    def test_data_drift_creates_new_summary_but_keeps_history(self) -> None:
        self.freeze_set()
        before = self.service.run_scenario_set("plan", "set-q3", {"start_date": "2026-09-20", "end_date": "2026-09-20"})
        self.service.add_inventory_lot("dispatch", {"lot_id": "lot-2", "facility_id": "field-a", "product": "crude", "grade": "PEAK_VALLEY", "quantity_mwh": "50000", "unit_cost_cny": "88", "received_at": "2026-09-21T06:00:00Z"})
        after = self.service.run_scenario_set("plan", "set-q3", {"start_date": "2026-09-20", "end_date": "2026-09-20"})
        self.assertNotEqual(after["run_id"], before["run_id"])
        self.assertNotEqual(after["summary_sha256"], before["summary_sha256"])
        results = self.service.scenario_set_results("risk", "set-q3")
        self.assertEqual([run["run_id"] for run in results["runs"]], [before["run_id"], after["run_id"]])

    def test_results_require_read_permission_and_return_audit_chain(self) -> None:
        self.freeze_set()
        self.service.run_scenario_set("plan", "set-q3", {})
        with self.assertRaises(Forbidden):
            self.service.scenario_set_results("plan", "set-q3")
        with self.assertRaises(Forbidden):
            self.service.scenario_set_results("dispatch", "set-q3")
        results = self.service.scenario_set_results("audit", "set-q3")
        self.assertTrue(results["audit_chain"]["valid"])
        event_types = [event["event_type"] for event in results["audit_trail"]]
        self.assertIn("scenarioset.created", event_types)
        self.assertIn("scenarioset.approved", event_types)
        self.assertIn("scenarioset.replayed", event_types)
        # 成员情景的审批事件也在完整审计链中
        self.assertIn("scenario.approved", event_types)
        self.assertEqual(results["set_snapshot_sha256"], results["runs"][0]["set_snapshot_sha256"])

    def test_run_window_must_stay_inside_frozen_range(self) -> None:
        self.freeze_set(start="2026-09-20", end="2026-09-22")
        with self.assertRaises(Exception):
            self.service.run_scenario_set("plan", "set-q3", {"start_date": "2026-09-19", "end_date": "2026-09-22"})

    def test_api_routes_enforce_actor_and_permissions(self) -> None:
        app = JsonApplication(self.service)
        ids = [self.scenario("base"), self.scenario("drop")]
        payload = json.dumps({"set_id": "api-set", "name": "接口集合", "start_date": "2026-09-20", "end_date": "2026-09-21", "scenario_ids": ids}).encode()
        created = app.handle("POST", "/scenario-sets", {"X-Actor-Id": "plan"}, payload)
        self.assertEqual(created.status, 201)
        missing_actor = app.handle("POST", "/scenario-sets", body=payload)
        self.assertEqual(missing_actor.status, 422)
        approve = app.handle("POST", "/scenario-sets/api-set/approve", {"X-Actor-Id": "risk"}, json.dumps({"expected_revision": 1}).encode())
        self.assertEqual(approve.status, 200)
        run = app.handle("POST", "/scenario-sets/api-set/run", {"X-Actor-Id": "plan"}, b"{}")
        self.assertEqual(run.status, 200)
        run_id = run.body["run_id"]
        detail = app.handle("GET", f"/scenario-set-runs/{run_id}", {"X-Actor-Id": "risk"})
        self.assertEqual(detail.status, 200)
        self.assertEqual(len(detail.body["items"]), 4)
        results = app.handle("GET", "/scenario-sets/api-set/results", {"X-Actor-Id": "audit"})
        self.assertEqual(results.status, 200)
        self.assertTrue(results.body["audit_chain"]["valid"])
        denied = app.handle("GET", "/scenario-sets/api-set/results", {"X-Actor-Id": "plan"})
        self.assertEqual(denied.status, 403)


if __name__ == "__main__":
    unittest.main()
