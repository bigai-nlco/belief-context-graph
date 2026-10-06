"""Loopback-only HTTP/SSE gateway. It never executes Codex tools itself."""

from __future__ import annotations

import hmac
import json
import os
import re
import threading
import time
import uuid
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

from .config import ProxyConfig
from .context import ContextEngine, RequestError, SessionStore
from .graph import HttpGraphBackend


class AuditLog:
    """Bounded metadata-only log: no prompts, keys, arguments or reasoning."""

    def __init__(self, directory: str | None):
        self.path = Path(directory) / "requests.jsonl" if directory else None
        self.lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)

    def write(self, metadata: dict) -> None:
        if self.path is None:
            return
        line = (
            json.dumps({"timestamp": time.time(), **metadata}, ensure_ascii=False)
            + "\n"
        )
        with self.lock:
            # At most ~20 MiB per gateway audit directory.
            if self.path.exists() and self.path.stat().st_size >= 10 * 1024 * 1024:
                self.path.replace(self.path.with_suffix(".jsonl.1"))
            fd = os.open(self.path, os.O_APPEND | os.O_WRONLY | os.O_CREAT, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as handle:
                handle.write(line)


class StreamUsage:
    """Observe bounded SSE frames without reserializing the forwarded stream."""

    def __init__(self, max_frame_bytes: int):
        self.pending = b""
        self.max_frame_bytes = max_frame_bytes
        self.disabled = False
        self.usage: dict = {}
        self.terminal: str | None = None

    def feed(self, chunk: bytes) -> None:
        if self.disabled:
            return
        self.pending += chunk
        while match := re.search(rb"\r?\n\r?\n", self.pending):
            frame, self.pending = (
                self.pending[: match.start()],
                self.pending[match.end() :],
            )
            lines = frame.splitlines()
            data = b"\n".join(
                line[5:].lstrip(b" ") for line in lines if line.startswith(b"data:")
            )
            try:
                event = json.loads(data)
            except (ValueError, UnicodeDecodeError):
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") in {
                "response.completed",
                "response.incomplete",
                "response.failed",
            }:
                self.terminal = event["type"]
                response = event.get("response")
                if isinstance(response, dict) and isinstance(
                    response.get("usage"), dict
                ):
                    self.usage = response["usage"]
        if len(self.pending) > self.max_frame_bytes:
            self.pending = b""
            self.disabled = True


class ProxyServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        config: ProxyConfig,
        *,
        upstream_client: httpx.Client | None = None,
        graph_client: httpx.Client | None = None,
    ):
        self.config = config
        self.gateway_token = config.secret(config.gateway_token_env)
        self.upstream_key = config.secret(config.upstream_api_key_env)
        graph_key = (
            config.secret(config.graph_api_key_env)
            if config.graph_api_key_env
            else None
        )
        self.upstream = upstream_client or httpx.Client(
            follow_redirects=False, trust_env=False
        )
        self.graph_http = graph_client or httpx.Client(
            follow_redirects=False, trust_env=False
        )
        self.sessions = SessionStore(config.max_sessions)
        self.audit = AuditLog(config.audit_dir)
        self.engine = ContextEngine(
            config,
            HttpGraphBackend(
                self.graph_http,
                config.graph_base_url,
                config.graph_timeout,
                graph_key,
            ),
        )
        super().__init__((config.host, config.port), ProxyHandler)

    def server_close(self) -> None:
        super().server_close()
        self.upstream.close()
        self.graph_http.close()


class ProxyHandler(BaseHTTPRequestHandler):
    server: ProxyServer
    protocol_version = "HTTP/1.1"
    server_version = "BCG-Codex-Prototype/0.1"

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(30)

    def log_message(self, format: str, *args: object) -> None:
        # BaseHTTPRequestHandler logs paths/raw request details to stderr.
        return

    def _json(self, status: int, value: dict) -> None:
        payload = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        self.close_connection = True

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(
                200,
                {"status": "ok", "prototype": True, "mode": self.server.config.mode},
            )
        else:
            self._json(
                404,
                {
                    "error": {
                        "message": "Only HTTP POST /v1/responses is supported (no WebSocket transport)"
                    }
                },
            )

    def _request_body(self) -> dict:
        if self.headers.get("Transfer-Encoding"):
            raise RequestError("Chunked request bodies are unsupported", 411)
        if self.headers.get("Content-Encoding", "identity") != "identity":
            raise RequestError(
                "Disable request compression for this Codex provider", 415
            )
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise RequestError("Invalid Content-Length") from exc
        if length <= 0 or length > self.server.config.max_request_bytes:
            raise RequestError("Invalid or oversized request body", 413)
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise RequestError("Incomplete request body")
        try:
            value = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            raise RequestError("Request body is not valid JSON") from exc
        if not isinstance(value, dict):
            raise RequestError("Request body must be a JSON object")
        return value

    def do_POST(self) -> None:
        sent_headers = False
        locked_session = None
        started = time.monotonic()
        metadata: dict = {"request_id": uuid.uuid4().hex}
        try:
            token = self.headers.get("Authorization", "")
            if not hmac.compare_digest(token, f"Bearer {self.server.gateway_token}"):
                raise RequestError("Invalid gateway bearer token", 401)
            if self.path not in {"/v1/responses", "/responses"}:
                raise RequestError(
                    "This prototype does not support remote compaction, Chat Completions, or other API routes",
                    501,
                )
            session = self.server.sessions.get(self.headers.get("X-BCG-Session-ID", ""))
            if not session.lock.acquire(blocking=False):
                raise RequestError(
                    "Concurrent requests for one session are unsupported; use a distinct session ID per Codex thread",
                    409,
                )
            locked_session = session
            body = self._request_body()
            graph_started = time.monotonic()
            outgoing, prepared = self.server.engine.prepare(body, session)
            metadata.update(prepared)
            metadata["context_prepare_seconds"] = time.monotonic() - graph_started
            with self.server.upstream.stream(
                "POST",
                self.server.config.upstream_base_url + "/responses",
                json=outgoing,
                headers={
                    "Authorization": f"Bearer {self.server.upstream_key}",
                    "Content-Type": "application/json",
                    "Accept-Encoding": "identity",
                    "Accept": "text/event-stream"
                    if outgoing.get("stream")
                    else "application/json",
                },
                timeout=self.server.config.upstream_timeout,
            ) as response:
                metadata["upstream_status"] = response.status_code
                content_type = response.headers.get("content-type", "application/json")
                self.send_response(response.status_code)
                self.send_header("Content-Type", content_type)
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-BCG-Evicted-Steps", str(metadata["evicted_steps"]))
                self.send_header("X-BCG-Fallback", metadata.get("fallback") or "none")
                if response.headers.get("x-request-id"):
                    self.send_header("X-Request-ID", response.headers["x-request-id"])
                self.end_headers()
                sent_headers = True
                observer = StreamUsage(self.server.config.max_request_bytes)
                plain = bytearray()
                oversized = False
                for chunk in response.iter_bytes():
                    if not chunk:
                        continue
                    # HTTP framing changes, Responses event bytes do not.
                    self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                    self.wfile.flush()
                    if "text/event-stream" in content_type:
                        observer.feed(chunk)
                    elif not oversized:
                        if (
                            len(plain) + len(chunk)
                            <= self.server.config.max_request_bytes
                        ):
                            plain.extend(chunk)
                        else:
                            oversized = True
                            plain.clear()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
                if "text/event-stream" in content_type:
                    metadata["agent_usage"] = observer.usage
                    metadata["stream_terminal"] = observer.terminal
                elif not oversized:
                    try:
                        result = json.loads(plain)
                        if isinstance(result, dict):
                            metadata["agent_usage"] = result.get("usage", {})
                    except (ValueError, UnicodeDecodeError):
                        pass
        except RequestError as exc:
            metadata["error"] = str(exc)
            self._json(
                exc.status,
                {"error": {"message": str(exc), "type": "bcg_gateway_error"}},
            )
        except (httpx.HTTPError, OSError):
            metadata["error"] = "upstream_or_client_connection_failed"
            if not sent_headers:
                with suppress(OSError):
                    self._json(
                        502,
                        {
                            "error": {
                                "message": "Upstream connection failed",
                                "type": "bcg_gateway_error",
                            }
                        },
                    )
            self.close_connection = True
        finally:
            if locked_session:
                locked_session.lock.release()
            metadata["elapsed_seconds"] = time.monotonic() - started
            # Disk/log failures must not turn an already completed model
            # response into another request (and another API charge).
            with suppress(OSError):
                self.server.audit.write(metadata)
