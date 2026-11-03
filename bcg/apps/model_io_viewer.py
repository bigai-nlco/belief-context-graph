"""Self-contained HTML viewer for exact model request/response traces."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

MODEL_IO_SCHEMA = "bcg.model_io.v1"


class ModelIoViewerError(ValueError):
    """Raised when a model-I/O trace cannot be located or parsed."""


def _trace_identity(root: Path, trace: Path, key: str) -> dict[str, str]:
    """Return stable display metadata for one benchmark model-I/O trace."""

    mode = trace.parent.parent.name if trace.parent.name == "model-io" else ""
    benchmark = trace.parent.parent.parent.name if mode else ""
    prefix = f"{benchmark}-{mode}-" if benchmark and mode else ""
    task_id = trace.stem.removeprefix(prefix) if prefix else trace.stem
    label = task_id
    qualifiers = [value for value in (mode, benchmark) if value]
    if qualifiers:
        label = f"{task_id} · {' / '.join(qualifiers)}"
    return {
        "key": key,
        "task_id": task_id,
        "mode": mode,
        "benchmark": benchmark,
        "label": label,
        "source": str(trace),
        "relative_source": str(trace.relative_to(root)),
    }


def discover_model_io_traces(source: Path) -> list[dict[str, str]]:
    """Discover task traces below a benchmark result directory."""

    root = source.expanduser().resolve()
    if not root.is_dir():
        raise ModelIoViewerError(f"Viewer directory does not exist: {root}")
    traces = sorted(
        path.resolve()
        for path in root.rglob("*.jsonl")
        if path.parent.name == "model-io"
    )
    if not traces:
        raise ModelIoViewerError(f"No model-I/O traces found below {root}.")
    entries = [
        _trace_identity(root, trace, str(index)) for index, trace in enumerate(traces)
    ]
    return sorted(
        entries,
        key=lambda item: (
            item["task_id"].casefold(),
            item["mode"].casefold(),
            item["benchmark"].casefold(),
        ),
    )


def load_model_io_trace(path: Path) -> list[dict[str, Any]]:
    """Group append-only request/response records into ordered model calls."""

    calls: dict[int, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ModelIoViewerError(f"Cannot read model-I/O trace: {path}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ModelIoViewerError(
                f"Invalid JSON on line {line_number} of {path}."
            ) from exc
        if not isinstance(record, dict) or record.get("schema") != MODEL_IO_SCHEMA:
            raise ModelIoViewerError(
                f"{path} is not a {MODEL_IO_SCHEMA} trace (line {line_number})."
            )
        try:
            call_id = int(record["call_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelIoViewerError(
                f"Missing or invalid call_id on line {line_number} of {path}."
            ) from exc
        call = calls.setdefault(call_id, {"call_id": call_id})
        record_type = record.get("type")
        if record_type == "request":
            call["request"] = {
                "timestamp": record.get("timestamp"),
                "model": record.get("model"),
                "payload": record.get("payload"),
            }
        elif record_type == "response":
            call["response"] = {
                "timestamp": record.get("timestamp"),
                "message": record.get("message"),
            }
        elif record_type == "error":
            call["error"] = {
                "timestamp": record.get("timestamp"),
                "message": record.get("error"),
            }

    if not calls:
        raise ModelIoViewerError(f"No model calls found in {path}.")
    return [calls[call_id] for call_id in sorted(calls)]


def resolve_model_io_trace(source: Path, task: str | None = None) -> Path:
    """Resolve a trace from a trace file, task result, or benchmark directory."""

    source = source.expanduser().resolve()
    if not source.exists():
        raise ModelIoViewerError(f"Input does not exist: {source}")
    if source.is_dir():
        candidates = sorted(
            path
            for path in source.rglob("*.jsonl")
            if path.parent.name == "model-io"
            and (not task or task.casefold() in str(path).casefold())
        )
        if not candidates:
            qualifier = f" matching {task!r}" if task else ""
            raise ModelIoViewerError(
                f"No model-I/O traces{qualifier} found below {source}."
            )
        if len(candidates) > 1:
            raise ModelIoViewerError(
                f"Found {len(candidates)} traces below {source}; use --task to select one."
            )
        return candidates[0]
    if source.suffix.casefold() == ".jsonl":
        return source
    if source.suffix.casefold() != ".json":
        raise ModelIoViewerError(
            "Input must be a model-I/O .jsonl trace, a task-result .json, "
            "or a benchmark result directory."
        )
    try:
        result = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelIoViewerError(f"Cannot parse task result: {source}") from exc
    trace_value = result.get("model_io_trace") if isinstance(result, dict) else None
    if not isinstance(trace_value, str) or not trace_value.strip():
        raise ModelIoViewerError(
            f"{source} does not reference a model_io_trace. "
            "This artifact predates exact model-I/O tracing."
        )
    trace = Path(trace_value).expanduser()
    if not trace.is_absolute():
        trace = source.parent / trace
    if not trace.is_file():
        raise ModelIoViewerError(f"Referenced model-I/O trace does not exist: {trace}")
    return trace.resolve()


def render_model_io_viewer(
    source: Path,
    *,
    output: Path | None = None,
    task: str | None = None,
) -> Path:
    """Render one trace as a portable, dependency-free HTML viewer."""

    trace = resolve_model_io_trace(source, task=task)
    calls = load_model_io_trace(trace)
    destination = (
        output.expanduser().resolve()
        if output is not None
        else trace.with_name(f"{trace.stem}-viewer.html")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        {"source": str(trace), "calls": calls},
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    destination.write_text(_html(data), encoding="utf-8")
    return destination


def create_model_io_viewer_server(
    source: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    task: str | None = None,
) -> tuple[ThreadingHTTPServer, str]:
    """Create a lazy directory Viewer server without starting its event loop."""

    root = source.expanduser().resolve()
    entries = discover_model_io_traces(root)
    trace_by_key = {entry["key"]: Path(entry["source"]) for entry in entries}
    initial = entries[0]
    if task:
        folded = task.casefold()
        initial = next(
            (
                entry
                for entry in entries
                if folded in entry["task_id"].casefold()
                or folded in entry["label"].casefold()
            ),
            initial,
        )
    bootstrap = json.dumps(
        {
            "source": str(root),
            "calls": [],
            "tasks": entries,
            "initial_task_key": initial["key"],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    page = _html(bootstrap).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._send(200, "text/html; charset=utf-8", page)
                return
            if parsed.path == "/api/task":
                key = parse_qs(parsed.query).get("key", [""])[0]
                trace = trace_by_key.get(key)
                entry = next((item for item in entries if item["key"] == key), None)
                if trace is None or entry is None:
                    self._json(404, {"error": "unknown task trace"})
                    return
                try:
                    calls = load_model_io_trace(trace)
                except ModelIoViewerError as exc:
                    self._json(422, {"error": str(exc)})
                    return
                self._json(200, {**entry, "calls": calls})
                return
            if parsed.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            self._json(404, {"error": "not found"})

        def _json(self, status: int, value: object) -> None:
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self._send(status, "application/json; charset=utf-8", body)

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except OSError as exc:
        raise ModelIoViewerError(
            f"Cannot start Viewer server on {host}:{port}: {exc}"
        ) from exc
    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    return server, f"http://{browser_host}:{server.server_port}/"


def _html(data: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BCG Model I/O Viewer</title>
<style>
:root{{--bg:#f4f6f8;--panel:#fff;--line:#dce2e8;--text:#17202a;--muted:#66717e;--accent:#d65a1f;--blue:#58789d;--code:#f7f8fa}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
header{{height:70px;padding:14px 22px;background:#111820;color:#fff;display:flex;align-items:center;gap:18px;border-bottom:3px solid var(--accent)}}
header h1{{font-size:18px;margin:0;white-space:nowrap}} header .source{{color:#b9c3ce;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
.task-picker{{margin-left:auto;display:none;align-items:center;gap:8px;min-width:320px}} .task-picker label{{color:#aeb9c4;font-size:12px;white-space:nowrap}} .task-picker select{{width:100%;min-width:0;padding:8px 30px 8px 10px;border:1px solid #435261;border-radius:7px;background:#202c38;color:#f2f5f7;font:13px ui-monospace,SFMono-Regular,Menlo,monospace;outline:none}} .task-picker select:focus{{border-color:#7f96ad}}
.shell{{height:calc(100vh - 70px);display:grid;grid-template-columns:260px minmax(0,1fr)}}
aside{{background:#1b2530;color:#e6ebf0;overflow:auto;border-right:1px solid #0e141b;padding:12px}}
.summary{{padding:6px 8px 12px;color:#9eabb8;font-size:12px}} .call{{width:100%;border:1px solid transparent;border-radius:8px;background:transparent;color:inherit;text-align:left;padding:10px;margin:0 0 6px;cursor:pointer}}
.call:hover{{background:#263442}} .call.active{{background:#304256;border-color:#61758a}} .call strong{{display:block;font-size:13px}} .call span{{display:block;color:#aeb9c4;font-size:11px;margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
main{{min-width:0;overflow:auto;padding:18px}} .toolbar{{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;gap:12px}} .toolbar h2{{font-size:17px;margin:0}} .badges{{display:flex;gap:7px;flex-wrap:wrap;justify-content:flex-end}}
.badge{{padding:4px 8px;border-radius:999px;background:#e8edf2;color:#465360;font-size:11px}} .badge.stop{{background:#fff0e8;color:#a54317}}
.columns{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:14px;align-items:start}} .panel{{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden;box-shadow:0 1px 2px #18212a10}}
.panel-title{{padding:11px 14px;font-weight:700;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:10px}} .panel-title.input{{border-left:4px solid var(--blue)}} .panel-title.output{{border-left:4px solid var(--accent)}}
.panel-body{{padding:12px;max-height:calc(100vh - 170px);overflow:auto}} .message{{border:1px solid var(--line);border-radius:8px;margin-bottom:10px;overflow:hidden}} .role{{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;padding:6px 9px;background:#eef2f5;color:#52606d}}
.role.system{{background:#e9eef6;color:#38577b}} .role.assistant{{background:#fff0e8;color:#a54317}} .role.tool,.role.toolresult{{background:#edf5ed;color:#32623d}} .content{{padding:9px 10px}}
.block{{margin:0 0 8px}} .block:last-child{{margin-bottom:0}} .block-label{{font-size:10px;font-weight:700;text-transform:uppercase;color:var(--muted);margin-bottom:3px}} pre{{margin:0;white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--code);border-radius:6px;padding:8px}}
details{{margin-top:10px;border-top:1px solid var(--line);padding-top:9px}} summary{{cursor:pointer;color:var(--muted);font-size:12px}} .empty{{padding:30px;text-align:center;color:var(--muted)}} .error{{background:#fff0f0;color:#9f2525;padding:10px;border-radius:7px}}
@media(max-width:900px){{header{{flex-wrap:wrap;height:auto}}.task-picker{{order:3;width:100%;min-width:0;margin-left:0}}.shell{{height:calc(100vh - 116px);grid-template-columns:190px minmax(0,1fr)}}.columns{{grid-template-columns:1fr}}.panel-body{{max-height:none}}}} @media(max-width:600px){{.shell{{height:auto;display:block}}aside{{max-height:220px}}main{{padding:10px}}}}
</style>
</head>
<body>
<header><h1>BCG Model I/O Viewer</h1><div class="source" id="source"></div><div class="task-picker" id="task-picker"><label for="task-select">Task ID</label><select id="task-select"></select></div></header>
<div class="shell"><aside><div class="summary" id="summary"></div><div id="calls"></div></aside><main><div id="detail"></div></main></div>
<script>
const BOOT={data}; let DATA={{source:BOOT.source,calls:BOOT.calls||[]}}; let selected=0;
const esc=(v)=>String(v??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const pretty=(v)=>JSON.stringify(v,null,2);
function modelName(call){{const m=call.request?.model||{{}};return m.id||m.model||'unknown model'}}
function outputMessage(call){{return call.response?.message||null}}
function usageText(call){{const u=outputMessage(call)?.usage;if(!u)return 'no usage';const total=u.totalTokens??u.total_tokens??u.total;return total!=null?`${{Number(total).toLocaleString()}} tokens`:'usage recorded'}}
function contentBlocks(content){{if(typeof content==='string')return `<div class="block"><pre>${{esc(content)}}</pre></div>`;if(!Array.isArray(content))return `<pre>${{esc(pretty(content))}}</pre>`;return content.map(block=>{{if(!block||typeof block!=='object')return `<pre>${{esc(String(block))}}</pre>`;const type=block.type||'content';let value=block.text??block.thinking??block.reasoning??block.content;if(type.toLowerCase().includes('tool'))value=pretty({{name:block.name,arguments:block.arguments??block.input,id:block.id}});if(value===undefined)value=pretty(block);return `<div class="block"><div class="block-label">${{esc(type)}}</div><pre>${{esc(typeof value==='string'?value:pretty(value))}}</pre></div>`}}).join('')}}
function messageCard(message){{if(!message||typeof message!=='object')return `<div class="message"><div class="content"><pre>${{esc(pretty(message))}}</pre></div></div>`;const role=message.role||message.type||'message';const extras={{...message}};delete extras.role;delete extras.content;return `<div class="message"><div class="role ${{esc(String(role).toLowerCase())}}">${{esc(role)}}</div><div class="content">${{contentBlocks(message.content)}}${{Object.keys(extras).length?`<details><summary>Message metadata</summary><pre>${{esc(pretty(extras))}}</pre></details>`:''}}</div></div>`}}
function inputView(payload){{if(!payload||typeof payload!=='object')return `<div class="empty">Request payload was not recorded.</div>`;let html='';if(payload.instructions)html+=messageCard({{role:'instructions',content:payload.instructions}});const messages=Array.isArray(payload.messages)?payload.messages:(Array.isArray(payload.input)?payload.input:[]);html+=messages.map(messageCard).join('');if(!html)html=`<pre>${{esc(pretty(payload))}}</pre>`;const tools=payload.tools;if(Array.isArray(tools))html+=`<details><summary>${{tools.length}} tools sent to the model</summary><pre>${{esc(pretty(tools))}}</pre></details>`;html+=`<details><summary>Exact provider payload</summary><pre>${{esc(pretty(payload))}}</pre></details>`;return html}}
function outputView(call){{const message=outputMessage(call);let html=message?messageCard(message):'<div class="empty">No finalized model response.</div>';if(call.error)html+=`<div class="error">${{esc(call.error.message)}}</div>`;if(message)html+=`<details><summary>Exact assistant response</summary><pre>${{esc(pretty(message))}}</pre></details>`;return html}}
function renderList(){{document.getElementById('source').textContent=DATA.source;document.getElementById('summary').textContent=`${{DATA.calls.length}} model calls · exact provider payloads`;document.getElementById('calls').innerHTML=DATA.calls.map((call,index)=>`<button class="call ${{index===selected?'active':''}}" onclick="selectCall(${{index}})"><strong>Call ${{call.call_id}}</strong><span>${{esc(modelName(call))}}</span><span>${{esc(usageText(call))}}</span></button>`).join('')}}
function renderDetail(){{const call=DATA.calls[selected];if(!call){{document.getElementById('detail').innerHTML='<div class="empty">No model calls.</div>';return}}const message=outputMessage(call);const stop=message?.stopReason??message?.stop_reason??(call.error?'error':'pending');document.getElementById('detail').innerHTML=`<div class="toolbar"><h2>Model call ${{call.call_id}}</h2><div class="badges"><span class="badge">${{esc(modelName(call))}}</span><span class="badge">${{esc(usageText(call))}}</span><span class="badge stop">${{esc(stop)}}</span></div></div><div class="columns"><section class="panel"><div class="panel-title input"><span>Model input</span><span>${{esc(call.request?.timestamp||'')}}</span></div><div class="panel-body">${{inputView(call.request?.payload)}}</div></section><section class="panel"><div class="panel-title output"><span>Model output</span><span>${{esc(call.response?.timestamp||call.error?.timestamp||'')}}</span></div><div class="panel-body">${{outputView(call)}}</div></section></div>`}}
function selectCall(index){{selected=index;renderList();renderDetail()}}
async function loadTask(key){{document.getElementById('detail').innerHTML='<div class="empty">Loading task trace…</div>';try{{const response=await fetch(`/api/task?key=${{encodeURIComponent(key)}}`,{{cache:'no-store'}});const value=await response.json();if(!response.ok)throw new Error(value.error||`HTTP ${{response.status}}`);DATA=value;selected=0;renderList();renderDetail()}}catch(error){{document.getElementById('detail').innerHTML=`<div class="error">${{esc(error)}}</div>`}}}}
function initializeTasks(){{const picker=document.getElementById('task-picker');const select=document.getElementById('task-select');if(!Array.isArray(BOOT.tasks)||!BOOT.tasks.length){{renderList();renderDetail();return}}picker.style.display='flex';select.innerHTML=BOOT.tasks.map(task=>`<option value="${{esc(task.key)}}">${{esc(task.label)}}</option>`).join('');select.value=BOOT.initial_task_key||BOOT.tasks[0].key;select.addEventListener('change',()=>loadTask(select.value));loadTask(select.value)}} initializeTasks();
</script>
</body></html>"""


__all__ = [
    "MODEL_IO_SCHEMA",
    "ModelIoViewerError",
    "create_model_io_viewer_server",
    "discover_model_io_traces",
    "load_model_io_trace",
    "render_model_io_viewer",
    "resolve_model_io_trace",
]
