"""Agent 主循环：模型决策 → 工具执行 → 结果回灌 → 继续，直到收敛或触预算。

设计原则：
1. **模型输出不可信**：每次工具调用都过白名单 + schema 校验，失败转拒绝观察结果。
2. **失败不炸循环**：工具异常、模型报错都变成观察结果或结构化停止原因。
3. **可观测**：每一步都记录结构化轨迹（kind / name / status / duration_ms / detail）。
4. **可中断**：支持取消信号与写操作人工确认（human-in-the-loop）。
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Literal, Sequence

from agentloom.core.budget import Budget, BudgetGuard
from agentloom.core.errors import GuardrailBlocked, ModelError
from agentloom.models.client import Message, ModelClient
from agentloom.tools.registry import ToolRegistry, ToolResult

StepKind = Literal["model", "tool", "guardrail", "approval"]
StepStatus = Literal["ok", "rejected", "error", "pending"]


@dataclass
class Step:
    """轨迹中的一步。"""

    index: int
    kind: StepKind
    name: str
    status: StepStatus
    duration_ms: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind,
            "name": self.name,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 2),
            "detail": self.detail,
        }


@dataclass
class AgentResult:
    """一次 Agent 运行的完整结果。"""

    answer: str
    steps: list[Step]
    stopped_reason: str
    stop_detail: dict[str, Any] = field(default_factory=dict)
    tool_calls: int = 0
    model_turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_seconds: float = 0.0
    estimated_cost_usd: float = 0.0
    backend: str = "unknown"
    run_id: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    pending_approvals: list[dict[str, Any]] = field(default_factory=list)

    @property
    def truncated(self) -> bool:
        """被预算截断（而非失败）。调用方据此决定是否重试或告知用户。"""
        return self.stopped_reason not in ("completed", "model_error", "guardrail")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "answer": self.answer,
            "steps": [s.to_dict() for s in self.steps],
            "tool_calls": self.tool_calls,
            "model_turns": self.model_turns,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "stopped_reason": self.stopped_reason,
            "stop_detail": self.stop_detail,
            "truncated": self.truncated,
            "model_backend": self.backend,
            # citations 里可能是 Citation 对象，也可能是已序列化的 dict，两种都要能出 JSON
            "citations": [
                c.to_dict() if hasattr(c, "to_dict") else c for c in self.citations
            ],
            "pending_approvals": self.pending_approvals,
        }


# 写操作确认回调：返回 (approved, note)
ApprovalFn = Callable[[str, dict[str, Any]], tuple[bool, str]]
CancelFn = Callable[[], bool]


class Agent:
    """受约束的工具调用 Agent。"""

    def __init__(
        self,
        model: ModelClient,
        registry: ToolRegistry,
        *,
        system_prompt: str = "",
        budget: Budget | None = None,
        approval_fn: ApprovalFn | None = None,
        cancel_fn: CancelFn | None = None,
        max_history_chars: int = 24_000,
    ) -> None:
        self.model = model
        self.registry = registry
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.budget = budget or Budget()
        self.approval_fn = approval_fn
        self.cancel_fn = cancel_fn
        self.max_history_chars = max_history_chars

    def _default_system_prompt(self) -> str:
        return (
            "你是一个受约束的工具调用助手。\n"
            "可用工具：\n"
            f"{self.registry.describe()}\n\n"
            "规则：\n"
            "1. 只能调用上面列出的工具，不要臆造工具名或参数。\n"
            "2. 需要事实依据时先调用工具，不要凭记忆编造。\n"
            "3. 工具返回 ok=false 时，读懂错误原因并换一种做法，不要重复同样的错误调用。\n"
            "4. 拿到足够信息后直接作答，不要无意义地反复调用工具。\n"
            "5. 如果工具没有提供依据，明确说「没有找到依据」，不要编造。"
        )

    # ------------------------------------------------------------------
    async def run(self, question: str, *, history: Sequence[Message] | None = None) -> AgentResult:
        """跑完整循环，返回答案 + 轨迹。"""
        run_id = uuid.uuid4().hex[:12]
        guard = BudgetGuard(self.budget)
        steps: list[Step] = []
        citations: list[dict[str, Any]] = []
        pending: list[dict[str, Any]] = []

        messages: list[Message] = [Message(role="system", content=self.system_prompt)]
        for m in (history or []):
            messages.append(m)
        messages.append(Message(role="user", content=question))

        answer = ""
        backend = getattr(self.model, "name", "unknown")

        while True:
            reason = guard.check()
            if reason is not None:
                break

            if self.cancel_fn is not None and self.cancel_fn():
                guard.stop("cancelled")
                break

            # ---------- 1) 模型决策 ----------
            # 历史必须在**调用模型之前**截断。放在循环末尾截断会让第一次调用
            # 带着未截断的完整历史进去，长会话直接把上下文撑爆。
            messages = _truncate(messages, self.max_history_chars)
            guard.count_step()
            t0 = time.perf_counter()
            try:
                resp = await self.model.complete(messages, self.registry.schemas())
            except ModelError as exc:
                steps.append(
                    Step(
                        index=len(steps),
                        kind="model",
                        name="model",
                        status="error",
                        duration_ms=(time.perf_counter() - t0) * 1000,
                        detail={"error": exc.message, "code": exc.code},
                    )
                )
                guard.stop("model_error", error=exc.message, code=exc.code)
                break

            guard.count_model_turn(resp.prompt_tokens, resp.completion_tokens)
            backend = resp.backend or backend
            model_step = Step(
                index=len(steps),
                kind="model",
                name="model",
                status="ok",
                duration_ms=(time.perf_counter() - t0) * 1000,
                detail={
                    "tool_calls": [
                        (tc.get("function") or {}).get("name") for tc in resp.tool_calls
                    ],
                    "content_preview": (resp.content or "")[:160],
                    "finish_reason": resp.finish_reason,
                    "prompt_tokens": resp.prompt_tokens,
                    "completion_tokens": resp.completion_tokens,
                },
            )
            steps.append(model_step)

            # ---------- 2) 无工具调用 -> 收敛 ----------
            if not resp.has_tool_calls:
                answer = resp.content or ""
                guard.stop("completed")
                break

            # 记录 assistant 的 tool_calls，保持协议完整
            messages.append(
                Message(role="assistant", content=resp.content or "", tool_calls=resp.tool_calls)
            )

            # ---------- 3) 逐个执行工具 ----------
            for tc in resp.tool_calls:
                fn = tc.get("function") or {}
                tool_name = fn.get("name") or ""
                call_id = tc.get("id") or f"call_{len(steps)}"

                reason = guard.check()
                if reason is not None:
                    break

                # 解析参数：模型给的 arguments 是字符串，可能是非法 JSON
                raw_args, parse_error = _parse_arguments(fn.get("arguments"))
                if parse_error is not None:
                    obs = {"ok": False, "error": parse_error}
                    steps.append(
                        Step(
                            index=len(steps),
                            kind="tool",
                            name=tool_name or "<unnamed>",
                            status="rejected",
                            detail={"error": parse_error, "rejected": True},
                        )
                    )
                    messages.append(
                        Message(
                            role="tool",
                            content=json.dumps(obs, ensure_ascii=False),
                            tool_call_id=call_id,
                            name=tool_name,
                        )
                    )
                    continue

                guard.count_tool_call()
                t1 = time.perf_counter()

                # 写操作：先过人工确认闸门
                if self.registry.has(tool_name) and self.registry.get(tool_name).dangerous:
                    approved, note = self._ask_approval(tool_name, raw_args)
                    if not approved:
                        obs = {"ok": False, "error": "operation not approved by human", "note": note}
                        steps.append(
                            Step(
                                index=len(steps),
                                kind="approval",
                                name=tool_name,
                                status="rejected",
                                duration_ms=(time.perf_counter() - t1) * 1000,
                                detail={"note": note},
                            )
                        )
                        messages.append(
                            Message(
                                role="tool",
                                content=json.dumps(obs, ensure_ascii=False),
                                tool_call_id=call_id,
                                name=tool_name,
                            )
                        )
                        continue

                result: ToolResult = self.registry.invoke(tool_name, raw_args)
                duration = (time.perf_counter() - t1) * 1000

                status: StepStatus = "ok" if result.ok else "rejected"
                if result.meta.get("exception"):
                    status = "error"

                steps.append(
                    Step(
                        index=len(steps),
                        kind="tool",
                        name=tool_name,
                        status=status,
                        duration_ms=duration,
                        detail={
                            "arguments": _redact(raw_args),
                            "observation": result.to_observation(),
                            # meta 里可能有非 JSON 对象（如 Citation），
                            # 轨迹要能出 JSON，所以这里只挑可序列化的键。
                            **{
                                k: v
                                for k, v in (result.meta or {}).items()
                                if k != "citation_objects"
                            },
                        },
                    )
                )

                # 收集引用（RAG 工具会返回 citations）。
                # 用 meta.citation_objects 拿原始的 Citation 对象（带 index 可校验），
                # 而不是 data.citations 里的 dict 副本。
                if result.ok and isinstance(result.data, dict):
                    objs = result.meta.get("citation_objects")
                    if objs:
                        citations.extend(objs)
                    for c in result.data.get("citations") or []:
                        if not objs:
                            citations.append(c)
                    if result.data.get("pending_approval"):
                        pending.append(result.data["pending_approval"])

                messages.append(
                    Message(
                        role="tool",
                        content=json.dumps(result.to_observation(), ensure_ascii=False),
                        tool_call_id=call_id,
                        name=tool_name,
                    )
                )

            messages = _truncate(messages, self.max_history_chars)

        snap = guard.snapshot()
        return AgentResult(
            answer=answer,
            steps=steps,
            stopped_reason=snap["stopped_reason"],
            stop_detail=snap["stop_detail"],
            tool_calls=snap["tool_calls"],
            model_turns=snap["model_turns"],
            prompt_tokens=snap["prompt_tokens"],
            completion_tokens=snap["completion_tokens"],
            elapsed_seconds=snap["elapsed_seconds"],
            estimated_cost_usd=snap["estimated_cost_usd"],
            backend=backend,
            run_id=run_id,
            citations=citations,
            pending_approvals=pending,
        )

    # ------------------------------------------------------------------
    async def run_stream(self, question: str, *, history: Sequence[Message] | None = None) -> AsyncIterator[dict[str, Any]]:
        """以事件流形式跑循环。

        事件序列：start → step(model/tool，可多次) → token(可多次) → done
        出错时发 error 帧。调用方据此实时看到 Agent 在做什么。
        """
        run_id = uuid.uuid4().hex[:12]
        guard = BudgetGuard(self.budget)
        steps: list[Step] = []
        citations: list[dict[str, Any]] = []

        messages: list[Message] = [Message(role="system", content=self.system_prompt)]
        for m in (history or []):
            messages.append(m)
        messages.append(Message(role="user", content=question))

        yield {
            "event": "start",
            "run_id": run_id,
            "budget": {
                "max_steps": self.budget.max_steps,
                "max_tool_calls": self.budget.max_tool_calls,
                "timeout_seconds": self.budget.timeout_seconds,
            },
        }

        answer = ""
        stopped_reason = "completed"
        stop_detail: dict[str, Any] = {}

        try:
            while True:
                reason = guard.check()
                if reason is not None:
                    stopped_reason = reason
                    stop_detail = guard.state.stop_detail
                    break

                if self.cancel_fn is not None and self.cancel_fn():
                    guard.stop("cancelled")
                    stopped_reason = "cancelled"
                    break

                messages = _truncate(messages, self.max_history_chars)
                guard.count_step()
                t0 = time.perf_counter()
                resp = await self.model.complete(messages, self.registry.schemas())
                guard.count_model_turn(resp.prompt_tokens, resp.completion_tokens)

                step = Step(
                    index=len(steps),
                    kind="model",
                    name="model",
                    status="ok",
                    duration_ms=(time.perf_counter() - t0) * 1000,
                    detail={
                        "tool_calls": [
                            (tc.get("function") or {}).get("name") for tc in resp.tool_calls
                        ],
                        "finish_reason": resp.finish_reason,
                    },
                )
                steps.append(step)
                yield {"event": "step", **step.to_dict()}

                if not resp.has_tool_calls:
                    answer = resp.content or ""
                    # 逐 token 推送（真实客户端可覆盖 stream；这里保证事件形态一致）
                    async for piece in self.model.stream(messages, self.registry.schemas()):
                        if piece:
                            yield {"event": "token", "text": piece}
                    guard.stop("completed")
                    stopped_reason = "completed"
                    break

                messages.append(
                    Message(role="assistant", content=resp.content or "", tool_calls=resp.tool_calls)
                )

                for tc in resp.tool_calls:
                    fn = tc.get("function") or {}
                    tool_name = fn.get("name") or ""
                    call_id = tc.get("id") or f"call_{len(steps)}"
                    raw_args, parse_error = _parse_arguments(fn.get("arguments"))
                    if parse_error is not None:
                        obs = {"ok": False, "error": parse_error}
                        s = Step(
                            index=len(steps),
                            kind="tool",
                            name=tool_name or "<unnamed>",
                            status="rejected",
                            detail={"error": parse_error, "rejected": True},
                        )
                        steps.append(s)
                        yield {"event": "step", **s.to_dict()}
                        messages.append(
                            Message(
                                role="tool",
                                content=json.dumps(obs, ensure_ascii=False),
                                tool_call_id=call_id,
                                name=tool_name,
                            )
                        )
                        continue

                    guard.count_tool_call()
                    t1 = time.perf_counter()

                    if self.registry.has(tool_name) and self.registry.get(tool_name).dangerous:
                        approved, note = self._ask_approval(tool_name, raw_args)
                        if not approved:
                            obs = {"ok": False, "error": "operation not approved by human"}
                            s = Step(
                                index=len(steps),
                                kind="approval",
                                name=tool_name,
                                status="rejected",
                                duration_ms=(time.perf_counter() - t1) * 1000,
                                detail={"note": note},
                            )
                            steps.append(s)
                            yield {"event": "step", **s.to_dict()}
                            messages.append(
                                Message(
                                    role="tool",
                                    content=json.dumps(obs, ensure_ascii=False),
                                    tool_call_id=call_id,
                                    name=tool_name,
                                )
                            )
                            continue

                    result = self.registry.invoke(tool_name, raw_args)
                    status: StepStatus = "ok" if result.ok else "rejected"
                    if result.meta.get("exception"):
                        status = "error"
                    s = Step(
                        index=len(steps),
                        kind="tool",
                        name=tool_name,
                        status=status,
                        duration_ms=(time.perf_counter() - t1) * 1000,
                        detail={
                            "arguments": _redact(raw_args),
                            "observation": result.to_observation(),
                            **{
                                k: v
                                for k, v in (result.meta or {}).items()
                                if k != "citation_objects"
                            },
                        },
                    )
                    steps.append(s)
                    yield {"event": "step", **s.to_dict()}

                    if result.ok and isinstance(result.data, dict):
                        objs = result.meta.get("citation_objects")
                        if objs:
                            citations.extend(objs)
                        else:
                            for c in result.data.get("citations") or []:
                                citations.append(c)

                    messages.append(
                        Message(
                            role="tool",
                            content=json.dumps(result.to_observation(), ensure_ascii=False),
                            tool_call_id=call_id,
                            name=tool_name,
                        )
                    )

                messages = _truncate(messages, self.max_history_chars)

        except ModelError as exc:
            yield {"event": "error", "code": exc.code, "message": exc.message}
            stopped_reason = "model_error"
            stop_detail = {"error": exc.message}

        snap = guard.snapshot()
        yield {
            "event": "done",
            "run_id": run_id,
            "answer": answer,
            "steps": [s.to_dict() for s in steps],
            "tool_calls": snap["tool_calls"],
            "model_turns": snap["model_turns"],
            "elapsed_seconds": snap["elapsed_seconds"],
            "stopped_reason": stopped_reason,
            "stop_detail": stop_detail or snap["stop_detail"],
            "citations": citations,
            "truncated": stopped_reason not in ("completed", "model_error", "guardrail"),
        }

    # ------------------------------------------------------------------
    def _ask_approval(self, tool_name: str, args: dict[str, Any]) -> tuple[bool, str]:
        if self.approval_fn is None:
            # 没有配置审批人时，写操作一律拒绝（安全默认）
            return False, "no approver configured; write operations are denied by default"
        return self.approval_fn(tool_name, args)


# ----------------------------------------------------------------------
# 辅助
# ----------------------------------------------------------------------

def _parse_arguments(raw: Any) -> tuple[dict[str, Any], str | None]:
    """解析模型给的 arguments。模型可能给字符串、dict、None 或非法 JSON。"""
    if raw is None or raw == "":
        return {}, None
    if isinstance(raw, dict):
        return raw, None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            return {}, f"arguments is not valid JSON: {exc.msg}"
        if not isinstance(parsed, dict):
            return {}, f"arguments must decode to a JSON object, got {type(parsed).__name__}"
        return parsed, None
    return {}, f"arguments must be a JSON object or string, got {type(raw).__name__}"


_SENSITIVE_KEYS = {"password", "token", "secret", "api_key", "authorization"}


def _redact(args: dict[str, Any]) -> dict[str, Any]:
    return {
        k: ("***" if k.lower() in _SENSITIVE_KEYS else v)
        for k, v in args.items()
    }


def _truncate(messages: list[Message], max_chars: int) -> list[Message]:
    """按字符预算截断历史。

    两个关键点：
    1. `assistant(tool_calls)` 与它后续的 `tool` 消息是**原子单元**，必须同进同出。
       只留 tool 不留 assistant，模型会看到一条无主的工具结果，协议直接不合法。
    2. 最后一条消息（当前用户问题）**永远保留**，哪怕它自己就超预算——
       把它丢了等于这次请求没有输入。超长时截断它的内容而不是丢弃。
    """
    if not messages:
        return messages

    total = sum(len(m.content or "") for m in messages)
    if total <= max_chars:
        return messages

    head = messages[0]
    rest = messages[1:]
    if not rest:
        return messages

    # 把 rest 切成原子单元
    units: list[list[Message]] = []
    for m in rest:
        if m.role == "tool" and units and units[-1] and units[-1][-1].role == "assistant":
            units[-1].append(m)
        elif m.role == "tool" and units and units[-1] and units[-1][-1].role == "tool":
            units[-1].append(m)
        else:
            units.append([m])

    # 最后一条永远保留
    tail_units: list[list[Message]] = [units[-1]]
    budget = max_chars - len(head.content or "")
    budget -= sum(len(m.content or "") for m in units[-1])

    for unit in reversed(units[:-1]):
        size = sum(len(m.content or "") for m in unit)
        if budget - size < 0:
            break
        budget -= size
        tail_units.append(unit)

    tail_units.reverse()
    out: list[Message] = [head]
    for unit in tail_units:
        out.extend(unit)

    # 兜底：当前问题本身超长时，截断它的内容而不是丢弃它
    if len(out) == 2 and out[1] is messages[-1] and len(out[1].content or "") > max_chars:
        out[1] = Message(
            role=out[1].role,
            content=(out[1].content or "")[:max_chars],
            tool_calls=out[1].tool_calls,
            tool_call_id=out[1].tool_call_id,
            name=out[1].name,
        )
    return out
