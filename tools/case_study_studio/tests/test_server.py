"""Read-only checks for the standalone server and its static asset boundary."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("studio_server", ROOT / "serve.py")
assert spec is not None and spec.loader is not None
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class ServerTests(unittest.TestCase):
    def test_only_known_assets_are_served(self) -> None:
        for path in ("/../.env", "/%2e%2e/.env", "/README.md", "/tests/", "/serve.py", "/.git/config", "/apply-copy.mjs", "/build_demo.py", "/build_review_pipeline.py"):
            handler = object.__new__(server.Handler)
            handler.path = path
            with patch.object(handler, "send_error") as error:
                self.assertIsNone(handler.send_head())
                error.assert_called_once_with(404, "Not found")

    def test_allowed_files_exist(self) -> None:
        for name in server.ASSETS:
            self.assertTrue((ROOT / name).is_file())

    def test_mjs_has_javascript_mime_type(self) -> None:
        self.assertIn("javascript", server.Handler.extensions_map[".mjs"])

    def test_method_page_is_linked_and_its_assets_are_public_only(self) -> None:
        self.assertIn('href="./review-pipeline.html"', (ROOT / "runs.html").read_text())
        for name in ("review-pipeline.html", "review-pipeline.css", "review-pipeline.mjs", "review-pipeline-data.mjs", "review-pipeline-contract.json"):
            self.assertIn(name, server.ASSETS)
        self.assertNotIn("build_review_pipeline.py", server.ASSETS)

    def test_shared_run_list_has_no_local_draft_entry(self) -> None:
        html = (ROOT / "runs.html").read_text()
        self.assertIn('id="run-list"', html)
        self.assertNotIn('class="local-link"', html)
        self.assertNotIn('href="./index.html"', html)
        self.assertNotIn('打开原有本地草稿 / 导入旧标注', html)

    def test_run_header_has_english_method_style_label(self) -> None:
        html = (ROOT / "runs.html").read_text()
        self.assertIn('<strong>BCG</strong><span class="runs-brand-label">RUNS &amp; ANNOTATIONS</span>', html)
        self.assertNotIn('<small>运行与标注</small>', html)

    def test_no_third_party_assets(self) -> None:
        for name in ("index.html", "styles.css", "app.mjs", "model.mjs", "browser_crypto.mjs", "view-state.mjs", "bundles.mjs",
                     "copy-editor.html", "copy-editor.css", "copy-editor.mjs",
                     "copy-runtime.mjs", "copy-model.mjs", "markdown.mjs"):
            contents = (ROOT / name).read_text()
            self.assertNotIn("https://", contents)
            self.assertNotIn("http://", contents)

    def test_run_list_has_no_footer_explanation(self) -> None:
        source = (ROOT / "runs.mjs").read_text()
        self.assertNotIn('class="run-footnote"', source)
        self.assertNotIn('产出报告不等于答案正确', source)
        self.assertNotIn('每位标注者独立填写', source)

    def test_run_forms_do_not_request_admin_code(self) -> None:
        html = (ROOT / "runs.html").read_text()
        self.assertNotIn('name="admin"', html)
        self.assertNotIn('启动码', html)
        self.assertIn('确认启动测试', html)
        self.assertIn('确认停止', html)
        self.assertIn('id="identity-form"', html)


if __name__ == "__main__":
    unittest.main()
