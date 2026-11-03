"""Optional Playwright QA against a local Studio; no model/service calls."""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> None:
    from playwright.sync_api import expect, sync_playwright

    origin = os.environ.get("STUDIO_URL", "http://127.0.0.1:8877")
    output = Path(os.environ.get("BCG_RETRIEVAL_QA_OUT", "/tmp/bcg-retrieval-browser-qa"))
    output.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    network: list[str] = []
    with TemporaryDirectory(prefix="retrieval-fixture-") as folder, sync_playwright() as p:
        root = Path(folder)
        graph = {"problem_id": "qa", "n_turns_ingested": 5, "nodes": [
            {"id": 1, "belief": "A fact <script>unsafe()</script>", "confidence": 0.8},
            {"id": 2, "belief": "A necessary bridge", "confidence": 0.0},
            {"id": 3, "belief": "Another piece of evidence", "confidence": 0.6},
        ], "relations": [
            {"id": 10, "from_id": 1, "to_id": 2, "type": "depends_on"},
            {"id": 11, "from_id": 3, "to_id": 2, "type": "supplements"},
        ]}
        audit = {"call": 1, "problem_id": "qa", "graph_n_turns_ingested": 5,
                 "selected_node_ids": [1, 2], "selected_relation_ids": [10], "graph_chars": 200,
                 "graph_chars_budget": 6600, "selector_result": {"strategy": "query_pcst",
                 "objective": 0.472, "parameters": {"edge_cost": 0.2, "confidence_floor": 0.2},
                 "node_scores": {"1": {"confidence": .8, "relevance": .8, "prize": .672},
                                 "2": {"confidence": 0, "relevance": 0, "prize": 0},
                                 "3": {"confidence": .6, "relevance": .2, "prize": .136}}}}
        files = {
            "case.json": {"task_id": "qa", "benchmark": "test", "question": "Retrieval inspection QA"},
            "requests.jsonl": [audit, {**audit, "call": 2, "selected_node_ids": [2, 3], "selected_relation_ids": [11]}],
            "graph_snapshots.jsonl": [{"call": 1, "problem_id": "qa", "snapshot_scope": "acknowledged_update", "snapshot": graph}],
            "trace.jsonl": [{"t": "llm", "turn": 1, "content": "First response"},
                            {"t": "llm", "turn": 2, "content": "Second response"}],
        }
        for name, value in files.items():
            raw = "\n".join(json.dumps(row) for row in value) if isinstance(value, list) else json.dumps(value)
            (root / name).write_text(raw)
        browser = p.chromium.launch(headless=True, executable_path=os.environ.get("CHROME_BIN") or None,
                                    args=["--no-sandbox"])
        context = browser.new_context(viewport={"width": 1600, "height": 1200})
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: network.append(request.url))
        page.goto(origin)
        page.locator("#file-input").set_input_files([root / name for name in files])
        page.locator('[data-tab="retrieval"]').click()
        expect(page.locator(".retrieval-node")).to_have_count(3)
        expect(page.locator(".retrieval-node.is-selected")).to_have_count(2)
        expect(page.locator(".retrieval-edge.is-selected")).to_have_count(1)
        expect(page.locator(".retrieval-detail")).to_contain_text("A fact <script>unsafe()</script>")
        page.locator('.retrieval-node[data-retrieval-node="2"]').click()
        expect(page.locator(".retrieval-detail")).to_have_attribute("data-detail-node", "2")
        expect(page.locator(".retrieval-node-metrics dd")).to_have_text(["0", "0", "0"])
        page.locator('[data-retrieval-zoom="in"]').click()
        assert page.locator(".retrieval-svg").get_attribute("viewBox") != "0 0 1000 620"
        page.locator('[data-retrieval-scope="selected"]').click()
        expect(page.locator(".retrieval-node")).to_have_count(2)
        expect(page.locator(".retrieval-edge")).to_have_count(1)
        page.locator("#retrieval-labels").check()
        expect(page.locator(".retrieval-snippet")).to_have_count(2)
        page.locator('[data-retrieval-step="1"]').click()
        expect(page.locator('.retrieval-node[data-retrieval-node="3"]')).to_have_count(1)
        expect(page.locator(".retrieval-output")).to_contain_text("Second response")
        page.locator("#retrieval-filter").select_option("unselected")
        expect(page.locator(".retrieval-table tbody tr")).to_have_count(1)
        page.locator("#retrieval-search").fill("not present")
        expect(page.locator(".retrieval-table tbody tr")).to_have_count(0)
        page.locator("#retrieval-search").fill("")
        page.locator("#retrieval-filter").select_option("all")
        page.locator('[data-tab="context"]').click()
        expect(page.locator(".annotation-panel")).to_be_visible()
        page.locator('[data-tab="retrieval"]').click()
        expect(page.locator(".annotation-panel")).to_be_hidden()
        page.set_viewport_size({"width": 430, "height": 932})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
        page.screenshot(path=str(output / "retrieval-mobile.png"))
        page.set_viewport_size({"width": 1600, "height": 1200})
        page.screenshot(path=str(output / "retrieval-synthetic.png"))

        real = os.environ.get("BCG_RETRIEVAL_CASE")
        checked = 0
        if real:
            bundle = json.loads(gzip.decompress(Path(real).read_bytes()))
            entry = bundle["cases"][0]
            audit_file = next(f for f in entry["files"] if f["path"].endswith("bcg/requests.jsonl"))
            snapshots_file = next(f for f in entry["files"] if f["path"].endswith("bcg/graph_snapshots.jsonl"))
            audits = [row for line in audit_file["text"].splitlines() if (row := json.loads(line)).get("call") is not None]
            snapshots = [json.loads(line) for line in snapshots_file["text"].splitlines()]
            page.locator("#file-input").set_input_files(real)
            expect(page.locator(".case-heading h1")).not_to_have_text("Retrieval inspection QA", timeout=60_000)
            page.locator('[data-tab="retrieval"]').click()
            page.locator('[data-retrieval-scope="all"]').click()
            expect(page.locator("#retrieval-round option")).to_have_count(len(audits))
            for index, event in enumerate(audits):
                page.locator("#retrieval-round").select_option(index=index)
                expected = [s for s in snapshots if s["problem_id"] == event["problem_id"] and s["call"] <= event["call"]]
                if "graph_n_turns_ingested" in event:
                    expected = [s for s in expected if s["snapshot"]["n_turns_ingested"] == event["graph_n_turns_ingested"]]
                snapshot = expected[-1]["snapshot"]
                expect(page.locator(".retrieval-node")).to_have_count(len(snapshot["nodes"]))
                ids = page.locator(".retrieval-node.is-selected").evaluate_all("els => els.map(el=>Number(el.dataset.retrievalNode))")
                edges = page.locator(".retrieval-edge.is-selected").evaluate_all("els => els.map(el=>Number(el.dataset.retrievalEdge))")
                assert sorted(ids) == sorted(event["selected_node_ids"]), event["call"]
                assert sorted(edges) == sorted(event["selected_relation_ids"]), event["call"]
                assert page.locator(".retrieval-output").count() == 1, event["call"]
                checked += 1
            page.locator(".retrieval-graph-layout").scroll_into_view_if_needed()
            page.screenshot(path=str(output / "retrieval-real-full.png"))
            page.locator('[data-retrieval-scope="selected"]').click()
            page.locator("#retrieval-labels").check()
            expect(page.locator(".retrieval-node")).to_have_count(len(audits[-1]["selected_node_ids"]))
            node_id = 322 if 322 in audits[-1]["selected_node_ids"] else audits[-1]["selected_node_ids"][0]
            page.locator(f'.retrieval-node[data-retrieval-node="{node_id}"]').click()
            page.locator(".retrieval-graph-layout").scroll_into_view_if_needed()
            page.screenshot(path=str(output / "retrieval-real-selected.png"))
        assert not errors, errors
        assert all(url.startswith(origin) or url.startswith("blob:") or url.startswith("data:") for url in network), network
        browser.close()
        print(json.dumps({"synthetic_interactions": "passed", "real_calls_checked": checked,
                          "page_errors": errors, "screenshots": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
