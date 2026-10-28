"""Offline QA server. Queue scheduling is disabled: never calls a model."""
import argparse
import sys
from contextlib import nullcontext
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.case_study_studio.control import ROOT
from tools.case_study_studio.serve import Handler, StudioHTTPServer
from tools.case_study_studio.tests.mock_ai_review import mock_reviews
from tools.case_study_studio.tests.test_control import build_fixture

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--port", type=int, default=8877)
parser.add_argument("--mock-ai", action="store_true", help="Offline simulated review completion for browser QA")
args = parser.parse_args()
control = build_fixture(args.root)
with StudioHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(ROOT / "tools/case_study_studio"))) as server:
    server.control = control
    print(f"Offline QA server ready: {server.server_port}", flush=True)
    with mock_reviews(control) if args.mock_ai else nullcontext():
        server.serve_forever()
