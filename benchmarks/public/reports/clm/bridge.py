"""Host-only strict sandbox, existing report tools and wire I/O recorder."""
from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
import shutil
import tempfile
import time
from pathlib import Path


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temporary.replace(path)


def save_bytes(path: Path, value: bytes) -> None:
    """Publish a complete wire body so live readers cannot see partial JSON."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


class Bridge:
    def __init__(self, policy, llm: dict, out: Path, trial: Path):
        self.policy, self.llm, self.out, self.trial = policy, llm, out, trial
        self.token = secrets.token_urlsafe(32)
        self.call = self.attempt = 0
        self.servers = []
        self.connections: set[asyncio.Task] = set()
        self.sandbox = None
        self.rpc_root = None
        self.started = False
        self.closing = False
        self.trace = trial / "agent/trajectories/react_agent.jsonl"

    def event(self, event: dict) -> None:
        self.trace.parent.mkdir(parents=True, exist_ok=True)
        with self.trace.open("a") as handle:
            handle.write(json.dumps({**event, "ts": time.time()}, ensure_ascii=False) + "\n")

    async def __aenter__(self):
        from plugins.tools._sandbox import BwrapSandbox

        allow = {"web_search", "web_fetch", "read_source", "find_in_source"}
        self.tools = {t.name: t for t in self.policy.bind([]) if t.name in allow}
        if set(self.tools) != allow:
            raise RuntimeError("CLM requires the existing four report research tools")
        self.schemas = [t.to_openai_schema() for t in self.tools.values()]
        self.rpc_root = Path(tempfile.mkdtemp(prefix="clm-report-rpc-"))
        public, cli = self.rpc_root / "public", self.rpc_root / "cli"
        public.mkdir(mode=0o700)
        cli.mkdir()
        shutil.copy2(Path(__file__).with_name("research.py"), cli / "research.py")
        self.control = self.rpc_root / "control.sock"
        try:
            self.sandbox = BwrapSandbox(
                workspace=self.policy.workspace, strict_isolation=True, mem_limit_mb=1024,
                binds=((str(public), "/research-rpc", True), (str(cli), "/tools", True),
                       (str(self.policy.sources.root), "/sources", True)))
            self.servers.append(await asyncio.start_unix_server(
                lambda r, w: self.rpc(r, w, self.tool_request),
                path=str(public / "tools.sock"), limit=16 * 1024 * 1024))
            self.servers.append(await asyncio.start_unix_server(
                lambda r, w: self.rpc(r, w, self.control_request),
                path=str(self.control), limit=16 * 1024 * 1024))
            http = await asyncio.start_server(self.proxy, "127.0.0.1", 0, limit=128 * 1024)
            self.servers.append(http)
            self.api_base = f"http://127.0.0.1:{http.sockets[0].getsockname()[1]}/v1"
            return self
        except BaseException:
            await self.__aexit__(None, None, None)
            raise

    async def __aexit__(self, *_):
        self.closing = True
        for server in self.servers:
            server.close()
        for task in tuple(self.connections):
            task.cancel()
        # Python 3.12 Server.wait_closed waits for clients too. Cancel RPCs and
        # kill in-flight sandbox commands FIRST, or stopping a long Bash call
        # would wait for the command's original timeout instead of cancelling it.
        if self.sandbox:
            await asyncio.to_thread(self.sandbox.kill)
        if self.connections:
            await asyncio.gather(*tuple(self.connections), return_exceptions=True)
        for server in self.servers:
            await server.wait_closed()
        if self.rpc_root:
            shutil.rmtree(self.rpc_root, ignore_errors=True)

    def status(self):
        return {"model_call": self.call, "search_remaining": self.policy.search_remaining,
                "fetch_remaining": self.policy.fetch_remaining, "tool_events": len(self.policy.trace)}

    async def tool_request(self, request):
        name = request.get("name")
        if name == "help":
            return self.schemas
        if name not in self.tools or not isinstance(request.get("args", {}), dict):
            raise ValueError("Only the four documented research tools are available")
        result = await self.tools[name].ainvoke(request.get("args", {}))
        save(self.out / "tool_status.json", self.status())
        return result

    async def control_request(self, request):
        op = request.get("op")
        if op == "info":
            return {"question": self.policy.instruction(), "task_id": self.policy.task.task_id,
                    "benchmark": self.policy.task.benchmark, "tools": self.schemas, **self.status()}
        if op == "begin_call":
            self.call, self.attempt = int(request["call"]), 0
            return self.status()
        if op == "exec":
            result = await asyncio.to_thread(self.sandbox.commands.run, request["command"],
                                            timeout=min(180, int(request.get("timeout", 120))))
            return {"stdout": result.stdout, "stderr": result.stderr, "return_code": result.exit_code}
        if op in {"upload", "download"}:
            if request["path"] != "/tmp/.live_ctx/LIVE_CTX_MAIN.txt":
                raise ValueError("Only the context mirror can be transferred")
            root = Path(self.sandbox._tmpdir).resolve()
            path = (root / ".live_ctx/LIVE_CTX_MAIN.txt").resolve()
            if not path.is_relative_to(root):
                raise ValueError("Context mirror escaped the sandbox tmp")
            turn = self.out / "turns" / f"{self.call:04d}"
            turn.mkdir(parents=True, exist_ok=True)
            if op == "upload":
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(request["text"], encoding="utf-8")
                (turn / "context.before.txt").write_text(request["text"], encoding="utf-8")
                return True
            if path.stat().st_size > 16 * 1024 * 1024:
                raise ValueError("Context mirror exceeds logging limit")
            text = path.read_text(encoding="utf-8")
            (turn / "context.after.txt").write_text(text, encoding="utf-8")
            return text
        if op == "status":
            return self.status()
        if op == "step":
            self.event({"t": "tool", "turn": self.call, "name": "bash",
                        "args": {"command": request["command"]}, "result": request["tool_message"],
                        "context_edit_applied": request["context_edit_applied"]})
            return True
        raise ValueError("Unknown private control operation")

    async def rpc(self, reader, writer, handler):
        if self.closing:
            writer.close()
            return
        task = asyncio.current_task()
        self.connections.add(task)
        try:
            try:
                request = json.loads(await asyncio.wait_for(reader.readline(), 180))
                reply = {"ok": True, "result": await handler(request)}
            except Exception as exc:
                reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            writer.write(json.dumps(reply, ensure_ascii=False, default=str).encode() + b"\n")
            with contextlib.suppress(ConnectionError):
                await writer.drain()
        finally:
            writer.close()
            self.connections.discard(task)

    def payload(self, incoming: dict) -> dict:
        if incoming.get("model") != self.llm["model"] or incoming.get("stream"):
            raise ValueError("Unexpected CLM model/stream settings")
        # The harness supplies messages and native bash. The selected Agent
        # profile supplies generation settings, not another context policy.
        payload = dict(incoming)
        for name in ("temperature", "top_p"):
            payload.pop(name, None)
            if self.llm.get(name) is not None:
                payload[name] = self.llm[name]
        for name in ("reasoning_effort", "chat_template_kwargs", "top_k", "min_p",
                     "presence_penalty", "frequency_penalty", "repetition_penalty"):
            if name in (self.llm.get("extra_body") or {}):
                payload[name] = self.llm["extra_body"][name]
        payload.pop("max_tokens", None)
        payload.pop("max_completion_tokens", None)
        key = "max_completion_tokens" if self.llm["model"].lower().startswith(("gpt-5", "gpt-6", "o1", "o3", "o4")) else "max_tokens"
        payload[key] = int(self.llm.get("max_tokens", 8192))
        return payload

    async def proxy(self, reader, writer):
        import httpx

        if self.closing:
            writer.close()
            return
        task = asyncio.current_task()
        self.connections.add(task)
        turn, code, started = None, 500, time.time()
        call = self.call
        try:
            try:
                headers = (await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 30)).decode("latin1")
                first, *rest = headers.split("\r\n")
                method, path, _ = first.split()
                fields = {k.strip().lower(): v.strip() for k, v in (line.split(":", 1) for line in rest if ":" in line)}
                if not secrets.compare_digest(fields.get("authorization", ""), "Bearer " + self.token):
                    raise ValueError("Invalid recorder credentials")
                if method != "POST" or path != "/v1/chat/completions":
                    raise ValueError("Only non-streaming Chat Completions is supported")
                length = int(fields.get("content-length", "0"))
                if not 0 < length <= 16 * 1024 * 1024:
                    raise ValueError("Invalid request size")
                payload = self.payload(json.loads(await reader.readexactly(length)))
                body = json.dumps(payload, ensure_ascii=False).encode()
                self.attempt += 1
                turn = self.out / "turns" / f"{call:04d}"
                turn.mkdir(parents=True, exist_ok=True)
                stem = f"attempt-{self.attempt:02d}"
                save_bytes(turn / f"{stem}.request.json", body)
                save_bytes(turn / "input.json", body)
                if not self.started:
                    self.started = True
                    self.event({"t": "start", "task_id": self.policy.task.task_id, "role_id": "clm",
                                "model_name": self.llm["model"], "system_prompt": payload["messages"][0]["content"],
                                "user_message": self.policy.task.question, "tool_names": ["bash"]})
                url = self.llm["base_url"].rstrip("/")
                if not url.endswith("/chat/completions"):
                    url += "/chat/completions"
                # Credentials are never put in request/response recording files.
                upstream_headers = {"Content-Type": "application/json"}
                if self.llm.get("api_key"):
                    upstream_headers["Authorization"] = "Bearer " + self.llm["api_key"]
                async with httpx.AsyncClient(timeout=610, trust_env=False) as client:
                    response = await client.post(url, content=body, headers=upstream_headers)
                code, result = response.status_code, response.content
                save_bytes(turn / f"{stem}.response.json", result)
                save_bytes(turn / "output.json", result)
                save(turn / f"{stem}.timing.json", {"seconds": time.time() - started, "http_status": code})
                if code == 200:
                    data = response.json()
                    choice = (data.get("choices") or [{}])[0]
                    message = choice.get("message") or {}
                    calls = []
                    for call in message.get("tool_calls") or []:
                        function = call.get("function") or {}
                        args = function.get("arguments", "{}")
                        with contextlib.suppress(ValueError):
                            args = json.loads(args) if isinstance(args, str) else args
                        calls.append({"id": call.get("id"), "name": function.get("name"), "args": args})
                    self.event({"t": "llm", "turn": call, "content": message.get("content"),
                                "thinking": message.get("reasoning_content") or message.get("reasoning") or "",
                                "tool_calls": calls, "usage": data.get("usage") or {},
                                "finish_reason": choice.get("finish_reason")})
            except Exception as exc:
                # Never echo HTTP exceptions containing private request headers.
                result = json.dumps({"error": {"message": type(exc).__name__, "type": "clm_adapter_error"}}).encode()
                if turn:
                    save(turn / "proxy_error.json", {"error": type(exc).__name__})
            writer.write(f"HTTP/1.1 {code} Response\r\nContent-Type: application/json\r\nContent-Length: {len(result)}\r\nConnection: close\r\n\r\n".encode() + result)
            with contextlib.suppress(ConnectionError):
                await writer.drain()
        finally:
            writer.close()
            self.connections.discard(task)
