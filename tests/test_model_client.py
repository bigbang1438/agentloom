"""模型客户端测试：重试退避、状态码分类、超时、错误信息脱敏、协议解析。

全部跑在 httpx.MockTransport 上——断言的是**协议契约**，不碰网络、不依赖真实模型。
这不是模型质量测试。
"""

from __future__ import annotations

import json

import httpx
import pytest

from agentloom.core.errors import ModelError, ModelTimeout, RateLimited
from agentloom.models.client import (
    Message,
    OpenAICompatClient,
    ReplayModel,
    ScriptedModel,
    _approx_tokens,
    _extract_order_no,
    _extract_sku,
    _scrub,
)


def make_client(handler, **kwargs) -> OpenAICompatClient:
    return OpenAICompatClient(
        api_key="sk-test-key-1234567890",
        base_url="https://api.example.com/v1",
        model="test-model",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def ok_body(content: str = "hello", tool_calls=None) -> dict:
    msg: dict = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "choices": [{"message": msg, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7},
    }


# ---------------------------------------------------------------- 正常路径

@pytest.mark.asyncio
async def test_successful_completion_is_parsed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=ok_body("你好"))

    resp = await make_client(handler).complete([Message(role="user", content="hi")])
    assert resp.content == "你好"
    assert resp.prompt_tokens == 12
    assert resp.completion_tokens == 7
    assert resp.total_tokens == 19
    assert resp.backend == "openai-compat"


@pytest.mark.asyncio
async def test_tool_calls_are_parsed() -> None:
    calls = [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "lookup_order", "arguments": '{"order_no":"A1024"}'},
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=ok_body("", tool_calls=calls))

    resp = await make_client(handler).complete([Message(role="user", content="q")])
    assert resp.has_tool_calls is True
    assert resp.tool_calls[0]["function"]["name"] == "lookup_order"


@pytest.mark.asyncio
async def test_request_carries_auth_and_tools() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=ok_body())

    tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]
    await make_client(handler).complete([Message(role="user", content="q")], tools)

    assert captured["auth"].startswith("Bearer sk-")
    assert captured["body"]["tools"] == tools
    assert captured["body"]["tool_choice"] == "auto"
    assert captured["body"]["model"] == "test-model"


@pytest.mark.asyncio
async def test_tools_omitted_when_none_given() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=ok_body())

    await make_client(handler).complete([Message(role="user", content="q")])
    assert "tools" not in captured["body"]


# ---------------------------------------------------------------- 重试

@pytest.mark.asyncio
async def test_retries_on_503_then_succeeds() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, json=ok_body("recovered"))

    client = make_client(handler, backoff_base=0.001)
    resp = await client.complete([Message(role="user", content="q")])
    assert resp.content == "recovered"
    assert attempts["n"] == 3


@pytest.mark.asyncio
async def test_gives_up_after_max_retries() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(500, text="boom")

    client = make_client(handler, backoff_base=0.001, max_retries=2)
    with pytest.raises(ModelError):
        await client.complete([Message(role="user", content="q")])
    assert attempts["n"] == 3  # 首次 + 2 次重试


@pytest.mark.asyncio
async def test_non_retryable_400_fails_immediately() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(400, text="bad request")

    client = make_client(handler, backoff_base=0.001)
    with pytest.raises(ModelError) as exc:
        await client.complete([Message(role="user", content="q")])
    assert attempts["n"] == 1, "400 不该重试"
    assert exc.value.detail["status"] == 400


@pytest.mark.asyncio
async def test_429_raises_rate_limited() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="slow down")

    client = make_client(handler, backoff_base=0.001, max_retries=0)
    with pytest.raises(RateLimited):
        await client.complete([Message(role="user", content="q")])


@pytest.mark.asyncio
async def test_timeout_is_mapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    client = make_client(handler, backoff_base=0.001, max_retries=0)
    with pytest.raises(ModelTimeout):
        await client.complete([Message(role="user", content="q")])


@pytest.mark.asyncio
async def test_network_error_is_mapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    client = make_client(handler, backoff_base=0.001, max_retries=0)
    with pytest.raises(ModelError) as exc:
        await client.complete([Message(role="user", content="q")])
    assert "transport error" in exc.value.message


# ---------------------------------------------------------------- 脱敏

@pytest.mark.asyncio
async def test_error_body_does_not_leak_api_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            text='{"error":"invalid key sk-live-abcdef1234567890"}',
        )

    client = make_client(handler, max_retries=0)
    with pytest.raises(ModelError) as exc:
        await client.complete([Message(role="user", content="q")])
    body = str(exc.value.detail.get("body", ""))
    assert "sk-live-abcdef1234567890" not in body
    assert "***" in body


def test_scrub_removes_bearer_tokens() -> None:
    assert "abcdef123456" not in _scrub("Authorization: Bearer abcdef123456")


def test_scrub_leaves_normal_text() -> None:
    assert _scrub("model returned HTTP 500") == "model returned HTTP 500"


# ---------------------------------------------------------------- 边界

@pytest.mark.asyncio
async def test_empty_choices_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    with pytest.raises(ModelError):
        await make_client(handler).complete([Message(role="user", content="q")])


@pytest.mark.asyncio
async def test_missing_usage_defaults_to_zero() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    resp = await make_client(handler).complete([Message(role="user", content="q")])
    assert resp.prompt_tokens == 0
    assert resp.completion_tokens == 0


def test_client_requires_api_key() -> None:
    with pytest.raises(ModelError):
        OpenAICompatClient(api_key="", base_url="https://x/v1")


@pytest.mark.asyncio
async def test_stream_yields_content_pieces() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        chunks = [
            'data: {"choices":[{"delta":{"content":"你"}}]}',
            'data: {"choices":[{"delta":{"content":"好"}}]}',
            "data: [DONE]",
        ]
        return httpx.Response(200, text="\n\n".join(chunks))

    client = make_client(handler)
    pieces = [p async for p in client.stream([Message(role="user", content="q")])]
    assert "".join(pieces) == "你好"


# ---------------------------------------------------------------- 桩与回放

@pytest.mark.asyncio
async def test_scripted_model_is_deterministic() -> None:
    m = ScriptedModel()
    m.load_script([{"content": "固定回答"}])
    a = await m.complete([Message(role="user", content="q")])
    b = await m.complete([Message(role="user", content="q")])
    assert a.content == b.content == "固定回答"


@pytest.mark.asyncio
async def test_replay_model_replays_in_order() -> None:
    m = ReplayModel(
        [
            {"content": "first", "backend": "replay"},
            {"content": "second", "backend": "replay"},
        ]
    )
    assert (await m.complete([Message(role="user", content="q")])).content == "first"
    assert (await m.complete([Message(role="user", content="q")])).content == "second"


@pytest.mark.asyncio
async def test_replay_exhaustion_raises() -> None:
    m = ReplayModel([{"content": "only", "backend": "replay"}])
    await m.complete([Message(role="user", content="q")])
    with pytest.raises(ModelError):
        await m.complete([Message(role="user", content="q")])


# ---------------------------------------------------------------- 工具函数

def test_approx_tokens_counts_cjk_heavier() -> None:
    cjk = _approx_tokens([Message(role="user", content="中文内容测试")])
    en = _approx_tokens([Message(role="user", content="abcdefgh")])
    assert cjk > en


@pytest.mark.parametrize(
    "text,expected",
    [
        ("订单 A1024 到哪了", "A1024"),
        ("查一下 B2048", "B2048"),
        ("订单号：XY123456", "XY123456"),
    ],
)
def test_order_no_extraction(text: str, expected: str) -> None:
    assert _extract_order_no(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("SKU-1001 还有货吗", "SKU-1001"),
        ("查 sku_2001", "SKU_2001"),
    ],
)
def test_sku_extraction(text: str, expected: str) -> None:
    assert _extract_sku(text) == expected


def test_message_wire_roundtrip() -> None:
    m = Message(role="tool", content="obs", tool_call_id="c1", name="t")
    wire = m.to_wire()
    assert wire["tool_call_id"] == "c1"
    back = Message.from_wire(wire)
    assert back.role == "tool"
    assert back.name == "t"
