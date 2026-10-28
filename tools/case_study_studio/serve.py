#!/usr/bin/env python3
"""Serve the local workbench, or the shared run/annotation studio with --shared."""

from __future__ import annotations

import argparse
import ipaddress
import json
import socket
import sys
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, ClassVar
from urllib.parse import unquote, urlsplit

if TYPE_CHECKING:
    from tools.case_study_studio.control import Control

ASSETS = {"index.html", "styles.css", "app.mjs", "model.mjs", "browser_crypto.mjs", "view-state.mjs", "bundles.mjs",
          "demo.mjs", "demo-case.json", "icon.svg", "copy-editor.html",
          "copy-editor.css", "copy-editor.mjs", "copy-model.mjs", "copy-runtime.mjs",
          "copy-runtime.css", "copy-overrides.json", "markdown.mjs", "vendor/marked.mjs",
          "vendor/marked.LICENSE", "runs.html", "runs.css", "runs.mjs", "shared.mjs",
          "review-pipeline.html", "review-pipeline.css", "review-pipeline.mjs",
          "review-pipeline-data.mjs", "review-pipeline-contract.json",
          "evidence-loader.mjs", "evidence-worker.mjs", "evidence-cache.mjs"}


class Handler(SimpleHTTPRequestHandler):
    extensions_map: ClassVar[dict[str, str]] = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".mjs": "text/javascript; charset=utf-8",
        ".svg": "image/svg+xml",
    }

    def send_head(self) -> BinaryIO | None:
        path = unquote(urlsplit(self.path).path)
        if path == "/" and getattr(self.server, "control", None):
            self.path = "/runs.html"
            path = "/runs.html"
        name = "index.html" if path == "/" else path.removeprefix("/")
        if name not in ASSETS:
            self.send_error(404, "Not found")
            return None
        return super().send_head()

    def _json(self, value: object, status: int = 200) -> None:
        raw = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _api(self) -> Control:
        from tools.case_study_studio.control import APIError
        control = getattr(self.server, "control", None)
        if not control:
            raise APIError(404, "共享服务未启用")
        host = urlsplit("http://" + self.headers.get("Host", "")).hostname or ""
        try:
            trusted = ipaddress.ip_address(host).is_private
        except ValueError:
            trusted = host in {"localhost", socket.gethostname(), socket.getfqdn()}
        if not trusted:
            raise APIError(403, "请使用本机或内网 IP 访问")
        origin = self.headers.get("Origin")
        if origin and (urlsplit(origin).netloc != self.headers.get("Host") or urlsplit(origin).scheme not in {"http", "https"}):
            raise APIError(403, "不接受跨站请求")
        return control

    def _body(self) -> dict:
        from tools.case_study_studio.control import APIError
        if self.headers.get_content_type() != "application/json":
            raise APIError(415, "请使用 JSON 请求")
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise APIError(400, "请求长度无效") from None
        if not 0 < size <= 1200 * 1024:
            raise APIError(413, "请求不能超过 1.2 MiB")
        self.connection.settimeout(15)
        try:
            value = json.loads(self.rfile.read(size))
        except (ValueError, TimeoutError):
            raise APIError(400, "JSON 请求不完整") from None
        if not isinstance(value, dict):
            raise APIError(400, "请求必须是 JSON object")
        return value

    def _user(self, control: Control) -> dict:
        return control.authenticate(self.headers.get("Authorization", "").removeprefix("Bearer "))

    def _dispatch(self) -> None:
        from tools.case_study_studio.control import APIError
        control = self._api()
        parts = unquote(urlsplit(self.path).path).strip("/").split("/")
        method = self.command
        if parts == ["api", "settings"] and method == "GET":
            return self._json(control.settings())
        if parts == ["api", "session"]:
            if method == "POST":
                # Limit creation/recovery attempts per network client.
                key = (self.client_address[0], int(time.time() // 60))
                with control.lock:
                    counts = getattr(self.server, "identity_requests", {})
                    counts = {k: v for k, v in counts.items() if k[1] >= key[1]}
                    counts[key] = counts.get(key, 0) + 1
                    self.server.identity_requests = counts
                    if counts[key] > 15:
                        raise APIError(429, "尝试过于频繁，请一分钟后再试")
                body = self._body()
                if body.get("token"):
                    return self._json({"user": control.authenticate(body["token"]), "token": body["token"]})
                return self._json(control.register(body.get("name")), 201)
            if method == "GET":
                return self._json({"user": self._user(control)})
        user = self._user(control)
        if parts == ["api", "runs"]:
            if method == "GET":
                return self._json({"runs": control.list_runs(user)})
            if method == "POST":
                return self._json(control.create_run(self._body(), user), 201)
        if len(parts) >= 3 and parts[:2] == ["api", "runs"] and parts[2].isdigit():
            run_id = int(parts[2])
            if len(parts) == 3 and method == "GET":
                return self._json(control.detail(run_id, user))
            if parts[3:] == ["cancel"] and method == "POST":
                self._body()
                return self._json(control.cancel_run(run_id, user))
            if len(parts) == 7 and parts[3] == "tasks":
                benchmark, task_id, action = parts[4:]
                if action in {"evidence", "evidence-preview"} and method == "GET":
                    preview = action == "evidence-preview"
                    path, digest = (control.evidence_preview if preview else control.evidence)(run_id, benchmark, task_id)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/gzip")
                    self.send_header("Content-Length", str(path.stat().st_size))
                    self.send_header("X-Evidence-SHA256", digest)
                    if preview:
                        self.send_header("X-Evidence-Projection", "review-preview-v1")
                    self.end_headers()
                    with path.open("rb") as handle:
                        self.copyfile(handle, self.wfile)
                    return
                if action == "annotation" and method == "GET":
                    return self._json(control.review(run_id, benchmark, task_id, user))
                if action == "annotation" and method == "PUT":
                    return self._json(control.save_review(run_id, benchmark, task_id, user, self._body()))
                if action == "ai-review" and method == "GET":
                    return self._json(control.ai_reviews.get(run_id, benchmark, task_id))
                if action == "ai-review" and method == "POST":
                    body = self._body()
                    if set(body) - {"retry"} or type(body.get("retry", False)) is not bool:
                        raise APIError(400, "AI Review 请求格式不正确")
                    return self._json(control.ai_reviews.enqueue(
                        run_id, benchmark, task_id, user, retry=body.get("retry", False)), 202)
        raise APIError(404, "找不到这个接口")

    def _handle_api(self) -> None:
        from tools.case_study_studio.control import APIError
        try:
            self._dispatch()
        except APIError as exc:
            self._json({"error": exc.message}, exc.status)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            self._json({"error": "服务处理失败，原始结果和已保存标注未改动"}, 500)

    def do_GET(self) -> None:
        if urlsplit(self.path).path.startswith("/api/"):
            self._handle_api()
        else:
            super().do_GET()

    def do_POST(self) -> None:
        self._handle_api()

    def do_PUT(self) -> None:
        self._handle_api()

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        # The copy editor embeds only this same-origin static app.
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.send_header("Content-Security-Policy", "frame-ancestors 'self'")
        super().end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8876)
    parser.add_argument("--shared", action="store_true", help="Shared runs, progress and server-side annotations")
    parser.add_argument("--state-dir", type=Path, help="Private SQLite/evidence directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    handler = partial(Handler, directory=str(root))
    with ThreadingHTTPServer((args.host, args.port), handler) as server:
        sys.path.insert(0, str(root.parents[1]))
        if args.shared:
            from tools.case_study_studio.control import Control
            server.control = Control(root.parents[1], args.state_dir, schedule=True)
        print(f"BCG Case Study Studio: http://{args.host}:{server.server_port}", flush=True)
        if args.shared:
            print("Shared mode. Signed-in reviewers can start/stop tests without an administrator code.", flush=True)
        else:
            print("Local mode. Files and annotations remain in the visitor's browser.", flush=True)
        if args.host not in {"127.0.0.1", "localhost", "::1"}:
            print("Plain HTTP: use only on a trusted LAN. Do not expose this service to the Internet.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            if getattr(server, "control", None):
                server.control.closed.set()


if __name__ == "__main__":
    main()
