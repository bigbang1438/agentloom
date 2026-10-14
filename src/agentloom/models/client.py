"""模型层：窄协议 + 可插拔后端 + 确定性桩模型。

设计要点：
- `ModelClient` 是一个**窄协议**（只有 complete / stream 两个方法），
  真实 HTTP 客户端、脚本模型、录制回放模型都实现它。
  循环层只依赖协议，所以循环的测试完全不需要网络和 API Key。
- 真实客户端负责：指数退避重试、可重试状态码分类、超时映射、
  token 计量、错误信息不泄漏密钥。
"""

from __future__ import annotations

import abc
import asyncio
import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Sequence

from agentloom.core.errors import ModelError, ModelTimeout, RateLimited

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def to_wire(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name
        return msg

    @classmethod
    def from_wire(cls, raw: dict[str, Any]) -> "Message":
        return cls(
            role=raw.get("role", "user"),
            content=raw.get("content") or "",
            tool_calls=list(raw.get("tool_calls") or []),
            tool_call_id=raw.get("tool_call_id"),
            name=raw.get("name"),
        )


@dataclass
class ModelResponse:
    """模型一次回复的归一化结果。"""

    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    backend: str = "unknown"

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ModelClient(abc.ABC):
    """模型窄协议。"""

    name: str = "model"

    @abc.abstractmethod
    async def complete(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float = 0.0,
    ) -> ModelResponse:
        ...

    async def stream(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """默认实现：退回非流式，一次性吐出内容。真实客户端会覆盖它。"""
        resp = await self.complete(messages, tools, temperature=temperature)
        if resp.content:
            yield resp.content


# --------------------------------------------------------------------------
# 确定性脚本模型：让项目零配置可跑、测试可复现
# --------------------------------------------------------------------------

class ScriptedModel(ModelClient):
    """确定性桩模型。

    它**不是智能**，靠规则匹配产出工具调用。价值在于让「模型决策 → 工具执行 →
    结果回灌 → 作答」这条链路在没有任何 API Key 的情况下可跑、可测、可演示。
    """

    name = "scripted"

    def __init__(self, registry: Any | None = None) -> None:
        self.registry = registry
        self._script: list[dict[str, Any]] | None = None

    def load_script(self, script: list[dict[str, Any]]) -> None:
        """载入固定脚本（用于测试：精确控制模型每一步说什么）。"""
        self._script = list(script)

    async def complete(
        self,
        messages: Sequence[Message],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float = 0.0,
    ) -> ModelResponse:
        if self._script is not None:
            return self._next_from_script(messages)

        tool_names = _tool_names(tools or [])
        question = _last_user_text(messages)
        tool_msgs = [m for m in messages if m.role == "tool"]

        # 已有工具结果 -> 基于结果作答
        if tool_msgs:
            return ModelResponse(
                content=self._compose_answer(tool_msgs),
                prompt_tokens=_approx_tokens(messages),
                completion_tokens=32,
                finish_reason="stop",
                backend=self.name,
            )

        # 首轮：按关键词决定调哪个工具
        call = self._pick_tool(question, tool_names)
        if call is None:
            return ModelResponse(
                content=(
                    "我没有找到合适的工具来回答这个问题。"
                    "可以尝试询问订单状态、库存情况或知识库里的内容。"
                ),
                prompt_tokens=_approx_tokens(messages),
                completion_tokens=28,
                finish_reason="stop",
                backend=self.name,
            )
        return ModelResponse(
            content="",
            tool_calls=[call],
            prompt_tokens=_approx_tokens(messages),
            completion_tokens=24,
            finish_reason="tool_calls",
            backend=self.name,
        )

    # ---- 内部 ----
    def _next_from_script(self, messages: Sequence[Message]) -> ModelResponse:
        assert self._script is not None
        idx = len([m for m in messages if m.role in ("assistant",)])
        step = self._script[min(idx, len(self._script) - 1)]
        return ModelResponse(
            content=step.get("content", ""),
            tool_calls=list(step.get("tool_calls") or []),
            prompt_tokens=step.get("prompt_tokens", _approx_tokens(messages)),
            completion_tokens=step.get("completion_tokens", 16),
            finish_reason="tool_calls" if step.get("tool_calls") else "stop",
            backend=self.name,
        )

    def _pick_tool(self, question: str, tool_names: list[str]) -> dict[str, Any] | None:
        """按用户**意图**挑工具，而不是按文本里出现了什么。

        顺序很重要：显式的写操作意图必须排在前面。用户说「记录一下待办：
        检查 SKU-1002 库存」时，句子里确实出现了 SKU，但意图是**创建任务**，
        不能因为顺带提到商品编号就去查库存。
        """
        q = question.lower()
        candidates: list[tuple[str, dict[str, Any]]] = []

        # 1) 显式写操作意图优先级最高
        write_markers = ("待办", "任务", "提醒", "记录一下", "记一下", "帮我记", "创建", "新建")
        if any(k in q for k in write_markers) and "create_task" in tool_names:
            title = question.strip()
            # 去掉「帮我记录一下待办：」这类前缀，留下真正的任务内容
            for prefix in ("帮我记录一下待办：", "帮我记录一下待办:", "记录一下待办：",
                           "记录一下待办:", "帮我记一下：", "记一下：", "待办：", "待办:"):
                if title.startswith(prefix):
                    title = title[len(prefix):].strip()
                    break
            candidates.append(("create_task", {"title": title[:120] or question[:120],
                                               "priority": "normal"}))

        # 2) 统计类意图
        if any(k in q for k in ("统计", "指标", "报表", "多少单", "汇总", "取消率")) and "sales_stats" in tool_names:
            candidates.append(("sales_stats", {"period": "today"}))

        # 3) 具体实体查询
        order_no = _extract_order_no(question)
        if order_no and "lookup_order" in tool_names:
            candidates.append(("lookup_order", {"order_no": order_no}))

        sku = _extract_sku(question)
        if sku and "check_stock" in tool_names:
            candidates.append(("check_stock", {"sku": sku}))

        # 4) 知识类问题：只要是在问「为什么 / 怎么 / 是什么」这类需要依据的问题，
        #    就先去知识库找证据。宁可多检索一次，也不要凭记忆编造。
        knowledge_markers = (
            "知识", "文档", "资料", "检索", "根据", "说明", "依据", "来源",
            "怎么", "如何", "什么是", "为什么", "为何", "是什么", "哪些", "介绍",
            "原理", "机制", "区别", "作用",
        )
        if any(k in q for k in knowledge_markers) and "knowledge_search" in tool_names:
            candidates.append(("knowledge_search", {"query": question.strip()[:200]}))

        if not candidates:
            return None
        name, args = candidates[0]
        return {
            "id": f"call_{abs(hash((name, json.dumps(args, sort_keys=True)))) % 10**8:08d}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        }

    def _compose_answer(self, tool_msgs: list[Message]) -> str:
        parts: list[str] = []
        for m in tool_msgs:
            try:
                obs = json.loads(m.content)
            except (json.JSONDecodeError, TypeError):
                parts.append(str(m.content)[:200])
                continue
            if not isinstance(obs, dict):
                parts.append(str(obs)[:200])
                continue
            if obs.get("ok") is False:
                parts.append(f"[{m.name}] 调用被拒绝：{obs.get('error')}")
            else:
                result = obs.get("result")
                parts.append(f"[{m.name}] {json.dumps(result, ensure_ascii=False)[:400]}")
        return "根据工具返回结果：" + "；".join(parts) if parts else "没有可用的工具结果。"


# --------------------------------------------------------------------------
# 录制 / 回放模型：把真实模型交互录下来，回放给测试与评测用
# --------------------------------------------------------------------------

class ReplayModel(ModelClient):
    """按顺序回放预先录制好的响应。用于把线上问题固化成回归测试。"""

    name = "replay"

    def __init__(self, responses: Sequence[dict[str, Any]]) -> None:
        self._responses = [ModelResponse(**r) for r in responses]
        self._i = 0
        self.calls: list[list[Message]] = []

    async def complete(self, messages, tools=None, *, temperature: float = 0.0) -> ModelResponse:
        self.calls.append(list(messages))
        if self._i >= len(self._responses):
            raise ModelError("replay exhausted: no more recorded responses")
        resp = self._responses[self._i]
        self._i += 1
        return resp


# --------------------------------------------------------------------------
# 真实 HTTP 客户端：OpenAI / DeepSeek 兼容协议
# --------------------------------------------------------------------------

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class OpenAICompatClient(ModelClient):
    """兼容 OpenAI chat-completions 协议（OpenAI / DeepSeek / 各类中转）。"""

    name = "openai-compat"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        *,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        backoff_base: float = 0.5,
        transport: Any | None = None,
    ) -> None:
        if not api_key:
            raise ModelError("api_key is required for the real model backend")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._transport = transport  # 测试注入 httpx.MockTransport
        self._sleep = asyncio.sleep

    async def complete(self, messages, tools=None, *, temperature: float = 0.0) -> ModelResponse:
        import httpx

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_wire() for m in messages],
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout_seconds, transport=self._transport
                ) as client:
                    resp = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
            except httpx.TimeoutException as exc:
                last_exc = ModelTimeout(f"model request timed out after {self.timeout_seconds}s")
                if attempt >= self.max_retries:
                    raise last_exc from exc
                await self._backoff(attempt)
                continue
            except httpx.HTTPError as exc:
                last_exc = ModelError(f"model transport error: {type(exc).__name__}")
                if attempt >= self.max_retries:
                    raise last_exc from exc
                await self._backoff(attempt)
                continue

            if resp.status_code in _RETRYABLE_STATUS and attempt < self.max_retries:
                await self._backoff(attempt, retry_after=resp.headers.get("retry-after"))
                continue

            if resp.status_code == 429:
                raise RateLimited("model rate limited", status=429)
            if resp.status_code >= 400:
                raise ModelError(
                    f"model returned HTTP {resp.status_code}",
                    status=resp.status_code,
                    body=_scrub(resp.text)[:300],
                )

            return self._parse(resp.json())

        raise last_exc or ModelError("model request failed")

    async def stream(self, messages, tools=None, *, temperature: float = 0.0) -> AsyncIterator[str]:
        import httpx

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_wire() for m in messages],
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        async with httpx.AsyncClient(
            timeout=self.timeout_seconds, transport=self._transport
        ) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    raise ModelError(
                        f"model stream returned HTTP {resp.status_code}",
                        status=resp.status_code,
                        body=_scrub(body)[:300],
                    )
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        yield piece

    # ---- 内部 ----
    async def _backoff(self, attempt: int, retry_after: str | None = None) -> None:
        if retry_after:
            try:
                await self._sleep(min(float(retry_after), 10.0))
                return
            except ValueError:
                pass
        delay = self.backoff_base * (2**attempt) + random.uniform(0, 0.1)
        await self._sleep(min(delay, 8.0))

    def _parse(self, body: dict[str, Any]) -> ModelResponse:
        choices = body.get("choices") or []
        if not choices:
            raise ModelError("model response contained no choices")
        choice = choices[0]
        msg = choice.get("message") or {}
        usage = body.get("usage") or {}
        return ModelResponse(
            content=msg.get("content") or "",
            tool_calls=list(msg.get("tool_calls") or []),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            finish_reason=choice.get("finish_reason") or "stop",
            backend=self.name,
        )


# --------------------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------------------

_SECRET_RE = re.compile(r"(sk-[A-Za-z0-9_\-]{4})[A-Za-z0-9_\-]+")
_BEARER_RE = re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+", re.IGNORECASE)


def _scrub(text: str) -> str:
    """从错误信息里抹掉密钥，避免泄漏到日志和响应体。"""
    text = _SECRET_RE.sub(r"\1***", text)
    text = _BEARER_RE.sub(r"\1***", text)
    return text


def _tool_names(tools: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for t in tools:
        fn = (t or {}).get("function") or {}
        if fn.get("name"):
            out.append(fn["name"])
    return out


def _last_user_text(messages: Sequence[Message]) -> str:
    for m in reversed(list(messages)):
        if m.role == "user":
            return m.content or ""
    return ""


def _approx_tokens(messages: Sequence[Message]) -> int:
    """粗略估算 token：中文按 1 字≈1 token，英文按 4 字符≈1 token。"""
    total = 0
    for m in messages:
        text = m.content or ""
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        total += cjk + max(0, (len(text) - cjk)) // 4 + 4
    return total


def _extract_order_no(text: str) -> str | None:
    # 订单号格式是「1 到 4 位大写字母 + 3 到 10 位数字」，例如 A1024、XY123456。
    # 字母数上限不能从 2 起，否则 A1024 这种单字母前缀会被漏掉。
    m = re.search(r"\b([A-Z]{1,4}\d{3,10})\b", text)
    if m:
        return m.group(1)
    m = re.search(r"订单\s*号?\s*[:：]?\s*([A-Za-z0-9\-]{4,20})", text)
    return m.group(1) if m else None


def _extract_sku(text: str) -> str | None:
    m = re.search(r"\b(SKU[-_]?\d{2,10})\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"商品\s*[:：]?\s*([A-Za-z0-9\-]{3,20})", text)
    return m.group(1) if m else None
