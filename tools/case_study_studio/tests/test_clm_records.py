from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.case_study_studio.clm_records import CLMRecords, TURN_FILES
from tools.case_study_studio.control import APIError


class CLMRecordsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "clm-task2-test"
        self.turn = self.root / "turns/0001"
        self.turn.mkdir(parents=True)
        self.write("summary.json", {"status": "submitted", "model_calls": 1, "accepted_edits": 1,
                                    "task_id": "drb2:2", "report": "/private/path/report.md"})
        self.write("manifest.json", {"settings": {"model_name": "Qwen", "api_key": "secret"},
                                     "bcg_status": "private file names"})
        self.write("turns/0001/input.json", {"messages": [{"role": "user", "content": "task"}]})
        self.write("turns/0001/output.json", {"choices": [{"message": {"reasoning": "original"}}],
                                            "usage": {"total_tokens": 42}})
        self.write("turns/0001/step.json", {"context_edit_applied": True, "file_changed": True,
                                          "command": "write context"})
        (self.turn / "context.after.txt").write_bytes(b"VERIFIED\r\nNEXT: research\r\n")
        self.store = CLMRecords([self.root])

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_inventory_and_exact_text(self):
        detail = self.store.detail(self.root.name)
        self.assertTrue(detail["turns"][0]["edited"])
        self.assertEqual(detail["turns"][0]["usage"]["total_tokens"], 42)
        record = self.store.turn(self.root.name, "1")
        self.assertEqual(set(record["files"]), set(TURN_FILES))
        self.assertEqual(record["files"]["context.after.txt"], "VERIFIED\r\nNEXT: research\r\n")
        self.assertIsNone(record["files"]["context.before.txt"])
        self.assertIn("original", record["files"]["output.json"])

    def test_only_selected_metadata_exposed(self):
        raw = json.dumps(self.store.runs())
        for secret in ["secret", "api_key", "/private", "private file names"]:
            self.assertNotIn(secret, raw)

    def test_no_path_input_or_unknown_runs(self):
        for run in ["..", "/etc", "unpublished", "../clm-task2-test"]:
            with self.assertRaises(APIError):
                self.store.detail(run)
        for number in ["0", "../1", "1.0", "-1", "10000"]:
            with self.assertRaises(APIError):
                self.store.turn(self.root.name, number)
        with self.assertRaises(APIError):
            self.store.turn(self.root.name, "999")

    def test_symlink_cannot_escape_run(self):
        external = Path(self.temp.name) / "secret.txt"
        external.write_text("private")
        (self.turn / "context.before.txt").symlink_to(external)
        with self.assertRaises(APIError):
            self.store.turn(self.root.name, "1")

    def test_report_path_is_fixed_not_taken_from_summary(self):
        data = self.store.report(self.root.name)
        self.assertIsNone(data["report"])
        (self.root / "workspace").mkdir()
        (self.root / "workspace/report.md").write_text("# Final report")
        (self.root / "task.txt").write_text("Original task")
        self.assertEqual(self.store.report(self.root.name), {"task": "Original task", "report": "# Final report"})

    def test_missing_and_duplicate_roots_rejected(self):
        with self.assertRaises(ValueError):
            CLMRecords([self.root / "missing"])
        with self.assertRaises(ValueError):
            CLMRecords([self.root, self.root])


if __name__ == "__main__":
    unittest.main()
