"""FastAPI 应用：HTTP 接口层。

安全默认：
- 只绑 127.0.0.1
- 校验 Host 与 Origin（防 DNS rebinding 与跨站写）
- 写请求需要 CSRF token
- 请求体大小上限
- 错误响应不含堆栈、不含本机路径、不含密钥
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from agentloom.core.budget import Budget
from agentloom.core.errors import AgentLoomError, GuardrailBlocked
from agentloom.core.loop import Agent
from agentloom.guardrails.filters import check_output, redact_pii, scan_for_injection
from agentloom.models.client import Message, OpenAICompatClient, ScriptedModel
from agentloom.observability.metrics import Metrics, TraceStore
from agentloom.paths import SAMPLES_DIR, WEB_DIR
from agentloom.rag.retriever import KnowledgeBase
from agentloom.tools.business import (
    ApprovalQueue,
    TaskStore,
    ToolContext,
    build_registry,
    finalize_answer,
)

MAX_BODY_BYTES = 512 * 1024

logger = logging.getLogger("agentloom.api")


# ----------------------------------------------------------------------
# 请求模型
# ----------------------------------------------------------------------

class StrictModel(BaseModel):
    """extra='forbid'：未声明字段一律 422。请求入口也要防参数走私。"""

    model_config = ConfigDict(extra="forbid")


class AskRequest(StrictModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[dict[str, Any]] = Field(default_factory=list, max_length=40)
    max_steps: int | None = Field(default=None, ge=1, le=24)
    max_tool_calls: int | None = Field(default=None, ge=1, le=32)
    timeout_seconds: float | None = Field(default=None, gt=0, le=300)


class ApprovalRequest(StrictModel):
    approved: bool
    note: str = Field(default="", max_length=300)


class DocumentRequest(StrictModel):
    filename: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=1_000_000)


# ----------------------------------------------------------------------
# 应用状态
# ----------------------------------------------------------------------

class AppState:
    def __init__(self) -> None:
        self.knowledge = KnowledgeBase()
        self.approvals = ApprovalQueue()
        self.tasks = TaskStore()
        self.metrics = Metrics()
        self.traces = TraceStore(os.environ.get("AGENTLOOM_TRACE_DB", ":memory:"))
        self.ctx = ToolContext(
            knowledge=self.knowledge,
            approvals=self.approvals,
            tasks=self.tasks,
            metrics=self.metrics,
        )
        self.registry = build_registry(self.ctx)
        self.csrf_token = uuid.uuid4().hex
        self.started_at = time.time()
        self._seed_samples()

    def _seed_samples(self) -> None:
        if not os.path.isdir(SAMPLES_DIR):
            return
        for name in sorted(os.listdir(SAMPLES_DIR)):
            if not name.endswith((".md", ".txt")):
                continue
            path = os.path.join(SAMPLES_DIR, name)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    self.knowledge.add_document(name, fh.read())
            except OSError:
                continue

    def build_model(self):
        """按环境变量决定用真实模型还是脚本模型。"""
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or ""
        backend = os.environ.get("AGENTLOOM_MODEL_BACKEND", "").strip().lower()

        if backend == "scripted" or not api_key:
            return ScriptedModel(self.registry)

        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return OpenAICompatClient(api_key=api_key, base_url=base_url, model=model)

    def build_agent(self, req: AskRequest) -> Agent:
        budget = Budget(
            max_steps=req.max_steps or 12,
            max_tool_calls=req.max_tool_calls or 12,
            timeout_seconds=req.timeout_seconds or 60.0,
        )
        return Agent(
            model=self.build_model(),
            registry=self.registry,
            budget=budget,
            approval_fn=lambda tool, args: (False, "HTTP 接口不自动批准写操作，请用 /api/approvals"),
        )


STATE: AppState | None = None


def get_state() -> AppState:
    global STATE
    if STATE is None:
        STATE = AppState()
    return STATE


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    get_state()
    yield


app = FastAPI(title="AgentLoom", version="1.0.0", lifespan=lifespan)


# ----------------------------------------------------------------------
# 中间件：Host / Origin / 体积 / CSRF
# ----------------------------------------------------------------------

_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
_ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


@app.middleware("http")
async def guard(request: Request, call_next):
    state = get_state()

    host = (request.headers.get("host") or "").split(":")[0]
    if host and host not in _ALLOWED_HOSTS:
        return _error("forbidden_host", f"host {host!r} is not allowed", 403)

    if request.method not in _SAFE_METHODS:
        origin = request.headers.get("origin")
        if origin:
            from urllib.parse import urlparse

            parsed = urlparse(origin)
            if parsed.hostname not in _ALLOWED_HOSTS:
                return _error("forbidden_origin", "cross-origin write rejected", 403)
        token = request.headers.get("x-csrf-token")
        if token != state.csrf_token:
            return _error("csrf_failed", "missing or invalid CSRF token", 403)

    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_BODY_BYTES:
        return _error("payload_too_large", f"body exceeds {MAX_BODY_BYTES} bytes", 413)

    try:
        return await call_next(request)
    except AgentLoomError as exc:
        return _error(exc.code, exc.message, exc.http_status, exc.detail)
    except Exception:
        # 对外不泄漏堆栈，但**必须**在服务端日志里留下完整堆栈，
        # 否则线上排查只能看到一句 "internal server error"。
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return _error("internal_error", "internal server error", 500)


def _error(code: str, message: str, status: int, detail: Any = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "detail": detail or {}}},
    )


# ----------------------------------------------------------------------
# 端点
# ----------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, Any]:
    state = get_state()
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or ""
    backend = os.environ.get("AGENTLOOM_MODEL_BACKEND", "").strip().lower()
    return {
        "status": "ok",
        "model_backend": "scripted" if (backend == "scripted" or not api_key) else "openai-compat",
        "uptime_seconds": round(time.time() - state.started_at, 1),
        "documents": state.knowledge.doc_count,
        "chunks": state.knowledge.chunk_count,
        "tools": len(state.registry),
    }


@app.get("/api/state")
async def api_state() -> dict[str, Any]:
    state = get_state()
    return {
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "dangerous": t.dangerous,
                "params": [
                    {"name": p.name, "type": p.type, "required": p.required}
                    for p in t.params
                ],
            }
            for t in state.registry
        ],
        "documents": state.knowledge.documents(),
        "pending_approvals": state.approvals.list_pending(),
        "tasks": state.tasks.list(),
        "csrf_token": state.csrf_token,
        "model_backend": "scripted"
        if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY"))
        else "openai-compat",
    }


@app.get("/api/tools")
async def api_tools() -> dict[str, Any]:
    state = get_state()
    return {"tools": state.registry.schemas(), "count": len(state.registry)}


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    state = get_state()
    return {"process": state.metrics.snapshot(), "traces": state.traces.stats()}


@app.post("/api/ask")
async def ask(req: AskRequest, request: Request) -> JSONResponse:
    state = get_state()

    hits = scan_for_injection(req.question)
    if hits:
        state.metrics.record_injection(len(hits))

    agent = state.build_agent(req)
    history = [_to_message(h) for h in req.history if isinstance(h, dict)]
    result = await agent.run(req.question, history=history)

    # 引用校验：用**本次运行自己**收集到的引用对象。
    # 之前读的是 ctx.citation_state（跨请求共享），导致无答案的问题
    # 会继承上一个请求的引用，凭空多出几条来源。
    citations = [c for c in result.citations if hasattr(c, "index")]
    if citations:
        answer, check = finalize_answer(result.answer, citations)
        result.answer = answer
        result.citations = [c.to_dict() for c in citations]
        result.stop_detail = {**result.stop_detail, "citation_check": check}
    else:
        result.citations = []

    out = check_output(result.answer)
    state.metrics.record_run(result)
    state.traces.save(req.question, result)

    payload = result.to_dict()
    payload["output_check"] = out.to_dict()
    payload["injection_hits"] = [h.to_dict() for h in hits]
    return JSONResponse(payload)


@app.post("/api/ask/stream")
async def ask_stream(req: AskRequest) -> StreamingResponse:
    state = get_state()
    agent = state.build_agent(req)
    history = [_to_message(h) for h in req.history if isinstance(h, dict)]

    async def gen() -> AsyncIterator[bytes]:
        async for event in agent.run_stream(req.question, history=history):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8")

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/documents")
async def add_document(req: DocumentRequest) -> dict[str, Any]:
    state = get_state()
    if "/" in req.filename or "\\" in req.filename or req.filename.startswith("."):
        return _error("invalid_filename", "filename must not contain a path", 422)
    chunks = state.knowledge.add_document(req.filename, req.content)
    return {"filename": req.filename, "chunks": chunks, "total_chunks": state.knowledge.chunk_count}


@app.get("/api/documents")
async def list_documents() -> dict[str, Any]:
    state = get_state()
    return {"documents": state.knowledge.documents(), "chunks": state.knowledge.chunk_count}


@app.post("/api/search")
async def search(req: dict[str, Any]) -> dict[str, Any]:
    """直接暴露检索，便于评测与调试（不经模型）。"""
    state = get_state()
    query = str(req.get("query") or "").strip()
    top_k = int(req.get("top_k") or 5)
    if not query:
        return _error("invalid_query", "query is required", 422)
    top_k = max(1, min(top_k, 12))
    citations, _ = state.knowledge.retrieve_with_citations(query, top_k=top_k)
    return {"query": query, "citations": [c.to_dict() for c in citations]}


@app.get("/api/approvals")
async def list_approvals() -> dict[str, Any]:
    state = get_state()
    return {"pending": state.approvals.list_pending(), "tasks": state.tasks.list()}


@app.post("/api/approvals/{approval_id}")
async def decide_approval(approval_id: str, req: ApprovalRequest) -> JSONResponse:
    state = get_state()
    try:
        item = state.approvals.decide(approval_id, req.approved, req.note)
    except KeyError as exc:
        return _error("not_found", str(exc), 404)
    except ValueError as exc:
        return _error("already_consumed", str(exc), 409)

    created = None
    if item.status == "approved":
        created = state.tasks.add(
            title=str(item.args.get("title") or ""),
            priority=str(item.args.get("priority") or "normal"),
            note=req.note,
        )
    return JSONResponse({"approval": item.to_dict(), "task": created})


@app.get("/api/runs")
async def list_runs(limit: int = 20) -> dict[str, Any]:
    state = get_state()
    limit = max(1, min(limit, 100))
    return {"runs": state.traces.recent(limit), "stats": state.traces.stats()}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> JSONResponse:
    state = get_state()
    run = state.traces.get(run_id)
    if run is None:
        return _error("not_found", f"run {run_id!r} not found", 404)
    return JSONResponse(run)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    path = os.path.join(WEB_DIR, "index.html")
    if not os.path.exists(path):
        return HTMLResponse("<h1>AgentLoom</h1><p>web/index.html not found</p>")
    with open(path, "r", encoding="utf-8") as fh:
        return HTMLResponse(fh.read())


_STATIC_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".svg": "image/svg+xml",
}


@app.get("/static/{filename}")
async def static_file(filename: str):
    """提供前端静态资源。

    只允许 `web/` 下的扁平文件名：拒绝任何路径分隔符与上跳，
    避免 `/static/../../etc/passwd` 这类穿越。
    """
    if "/" in filename or "\\" in filename or filename.startswith("."):
        return _error("invalid_filename", "invalid static path", 422)

    path = os.path.join(WEB_DIR, filename)
    real = os.path.realpath(path)
    if not real.startswith(os.path.realpath(WEB_DIR) + os.sep):
        return _error("invalid_filename", "path escapes web directory", 422)
    if not os.path.isfile(real):
        return _error("not_found", f"{filename} not found", 404)

    ext = os.path.splitext(filename)[1].lower()
    with open(real, "rb") as fh:
        content = fh.read()
    from fastapi.responses import Response

    return Response(
        content=content,
        media_type=_STATIC_TYPES.get(ext, "application/octet-stream"),
    )


def _to_message(raw: dict[str, Any]) -> Message:
    return Message(
        role=raw.get("role", "user"),
        content=str(raw.get("content") or ""),
    )


def create_app() -> FastAPI:
    return app
