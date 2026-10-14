"""Agent 循环测试：正常链路、轨迹形状、五维预算、拒绝回灌、人工确认、取消。

全部跑在确定性脚本模型上，不需要网络和 API Key。
"""

from __future__ import annotations

import json

import pytest

from agentloom.core.budget import Budget
from agentloom.core.errors import ModelError
from agentloom.core.loop import Agent
from agentloom.models.client import Message, ModelClient, ModelResponse, ScriptedModel
from agentloom.tools.registry import ParamSpec, ToolRegistry


# ----------------------------------------------------------------------
# 测试替身
# ----------------------------------------------------------------------

class AlwaysToolModel(ModelClient):
    """永远要求调用工具的模型，用来把预算打穿。"""

    name = "always-tool"

    def __init__(self, tool_name: str = "noop", args: dict | None = None) -> None:
        self.tool_name = tool_name
        self.args = args or {}
        self.calls = 0

    async def complete(self, messages, tools=None, *, temperature: float = 0.0) -> ModelResponse:
        self.calls += 1
        return ModelResponse(
            content="",
            tool_calls=[
                {
                    "id": f"call_{self.calls}",
                    "type": "function",
                    "function": {
                        "name": self.tool_name,
                        "arguments": json.dumps(self.args),
                    },
                }
            ],
            prompt_tokens=10,
            completion_tokens=5,
            finish_reason="tool_calls",
            backend=self.name,
        )


class ErrorModel(ModelClient):
    name = "error"

    async def complete(self, messages, tools=None, *, temperature: float = 0.0) -> ModelResponse:
        raise ModelError("upstream is down")


def make_registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool("noop", "什么都不做", [])
    def noop():
        return {"ok": 1}

    @reg.tool("echo", "回显", [ParamSpec("text", "string")])
    def echo(text: str):
        return {"echo": text}

    @reg.tool(
        "write_thing",
        "写操作",
        [ParamSpec("name", "string")],
        dangerous=True,
        side_effect=True,
    )
    def write_thing(name: str):
        return {"written": name}

    return reg


# ----------------------------------------------------------------------
# 正常链路
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_happy_path_produces_answer_and_trace() -> None:
    model = ScriptedModel()
    model.load_script(
        [
            {
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": '{"text":"hi"}'},
                    }
                ]
            },
            {"content": "答案是 hi"},
        ]
    )
    agent = Agent(model=model, registry=make_registry())
    result = await agent.run("随便问点什么")

    assert result.answer == "答案是 hi"
    assert result.stopped_reason == "completed"
    assert result.truncated is False
    assert result.tool_calls == 1
    assert result.model_turns == 2

    kinds = [(s.kind, s.name, s.status) for s in result.steps]
    assert kinds == [
        ("model", "model", "ok"),
        ("tool", "echo", "ok"),
        ("model", "model", "ok"),
    ]


@pytest.mark.asyncio
async def test_trace_steps_carry_duration_and_detail() -> None:
    model = ScriptedModel()
    model.load_script(
        [
            {
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": '{"text":"x"}'},
                    }
                ]
            },
            {"content": "done"},
        ]
    )
    result = await Agent(model=model, registry=make_registry()).run("q")

    tool_step = next(s for s in result.steps if s.kind == "tool")
    assert tool_step.duration_ms >= 0
    assert tool_step.detail["observation"]["ok"] is True
    assert tool_step.detail["arguments"] == {"text": "x"}

    d = result.to_dict()
    assert d["stopped_reason"] == "completed"
    assert isinstance(d["steps"], list)
    assert d["run_id"]


@pytest.mark.asyncio
async def test_no_tool_call_converges_immediately() -> None:
    model = ScriptedModel()
    model.load_script([{"content": "直接回答"}])
    result = await Agent(model=model, registry=make_registry()).run("q")
    assert result.answer == "直接回答"
    assert result.tool_calls == 0
    assert len(result.steps) == 1


# ----------------------------------------------------------------------
# 预算
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_max_steps_truncates_without_500() -> None:
    agent = Agent(
        model=AlwaysToolModel("noop"),
        registry=make_registry(),
        budget=Budget(max_steps=4, max_model_turns=99, max_tool_calls=99),
    )
    result = await agent.run("q")
    assert result.stopped_reason == "max_steps"
    assert result.truncated is True
    assert result.stop_detail["limit"] == 4


@pytest.mark.asyncio
async def test_max_tool_calls_truncates() -> None:
    agent = Agent(
        model=AlwaysToolModel("noop"),
        registry=make_registry(),
        budget=Budget(max_steps=99, max_model_turns=99, max_tool_calls=3),
    )
    result = await agent.run("q")
    assert result.stopped_reason == "max_tool_calls"
    assert result.tool_calls == 3


@pytest.mark.asyncio
async def test_max_model_turns_truncates() -> None:
    agent = Agent(
        model=AlwaysToolModel("noop"),
        registry=make_registry(),
        budget=Budget(max_steps=99, max_model_turns=2, max_tool_calls=99),
    )
    result = await agent.run("q")
    assert result.stopped_reason == "max_model_turns"
    assert result.model_turns == 2


@pytest.mark.asyncio
async def test_token_budget_truncates() -> None:
    agent = Agent(
        model=AlwaysToolModel("noop"),
        registry=make_registry(),
        budget=Budget(
            max_steps=99, max_model_turns=99, max_tool_calls=99, max_tokens=25
        ),
    )
    result = await agent.run("q")
    assert result.stopped_reason == "max_tokens"
    assert result.prompt_tokens + result.completion_tokens >= 25


@pytest.mark.asyncio
async def test_cost_budget_truncates() -> None:
    agent = Agent(
        model=AlwaysToolModel("noop"),
        registry=make_registry(),
        budget=Budget(
            max_steps=99,
            max_model_turns=99,
            max_tool_calls=99,
            max_cost_usd=0.00001,
            price_per_1k_input=1.0,
            price_per_1k_output=1.0,
        ),
    )
    result = await agent.run("q")
    assert result.stopped_reason == "max_cost"
    assert result.estimated_cost_usd > 0


@pytest.mark.asyncio
async def test_timeout_budget_truncates() -> None:
    agent = Agent(
        model=AlwaysToolModel("noop"),
        registry=make_registry(),
        budget=Budget(max_steps=99, max_model_turns=99, max_tool_calls=99, timeout_seconds=0.0),
    )
    result = await agent.run("q")
    assert result.stopped_reason == "timeout"


# ----------------------------------------------------------------------
# 拒绝回灌
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rejected_tool_call_is_fed_back_and_loop_continues() -> None:
    """参数走私被拒后，循环不能崩，模型要能拿到拒绝原因继续走。"""
    model = ScriptedModel()
    model.load_script(
        [
            {
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "echo",
                            "arguments": '{"text":"x","approved":true}',
                        },
                    }
                ]
            },
            {"content": "我换个参数再试"},
        ]
    )
    result = await Agent(model=model, registry=make_registry()).run("q")

    assert result.answer == "我换个参数再试"
    assert result.stopped_reason == "completed"
    rejected = [s for s in result.steps if s.status == "rejected"]
    assert len(rejected) == 1
    assert any("approved" in e for e in rejected[0].detail["errors"])


@pytest.mark.asyncio
async def test_unknown_tool_is_rejected_and_fed_back() -> None:
    model = ScriptedModel()
    model.load_script(
        [
            {
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "rm_rf", "arguments": "{}"},
                    }
                ]
            },
            {"content": "没有这个工具"},
        ]
    )
    result = await Agent(model=model, registry=make_registry()).run("q")
    rejected = [s for s in result.steps if s.status == "rejected"]
    assert len(rejected) == 1
    assert rejected[0].name == "rm_rf"


@pytest.mark.asyncio
async def test_invalid_json_arguments_are_rejected_not_crashed() -> None:
    model = ScriptedModel()
    model.load_script(
        [
            {
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": "{not json"},
                    }
                ]
            },
            {"content": "参数格式错了"},
        ]
    )
    result = await Agent(model=model, registry=make_registry()).run("q")
    assert result.stopped_reason == "completed"
    rejected = [s for s in result.steps if s.status == "rejected"]
    assert len(rejected) == 1
    assert "not valid JSON" in rejected[0].detail["error"]


# ----------------------------------------------------------------------
# 人工确认
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dangerous_tool_denied_by_default() -> None:
    """没有配置审批人时，写操作必须默认拒绝。"""
    model = ScriptedModel()
    model.load_script(
        [
            {
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "write_thing", "arguments": '{"name":"t"}'},
                    }
                ]
            },
            {"content": "未获批准"},
        ]
    )
    result = await Agent(model=model, registry=make_registry()).run("q")
    approvals = [s for s in result.steps if s.kind == "approval"]
    assert len(approvals) == 1
    assert approvals[0].status == "rejected"
    assert "no approver" in approvals[0].detail["note"]


@pytest.mark.asyncio
async def test_dangerous_tool_runs_when_approved() -> None:
    model = ScriptedModel()
    model.load_script(
        [
            {
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "write_thing", "arguments": '{"name":"t"}'},
                    }
                ]
            },
            {"content": "已写入"},
        ]
    )
    agent = Agent(
        model=model,
        registry=make_registry(),
        approval_fn=lambda tool, args: (True, "ok"),
    )
    result = await agent.run("q")
    tool_steps = [s for s in result.steps if s.kind == "tool"]
    assert tool_steps and tool_steps[0].status == "ok"
    assert tool_steps[0].detail["observation"]["result"] == {"written": "t"}


@pytest.mark.asyncio
async def test_read_only_tool_does_not_need_approval() -> None:
    model = ScriptedModel()
    model.load_script(
        [
            {
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "noop", "arguments": "{}"},
                    }
                ]
            },
            {"content": "done"},
        ]
    )
    agent = Agent(
        model=model,
        registry=make_registry(),
        approval_fn=lambda tool, args: pytest.fail("只读工具不该触发审批"),
    )
    result = await agent.run("q")
    assert result.stopped_reason == "completed"
    assert not [s for s in result.steps if s.kind == "approval"]


# ----------------------------------------------------------------------
# 取消与模型错误
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_stops_the_loop() -> None:
    agent = Agent(
        model=AlwaysToolModel("noop"),
        registry=make_registry(),
        budget=Budget(max_steps=99, max_model_turns=99, max_tool_calls=99),
        cancel_fn=lambda: True,
    )
    result = await agent.run("q")
    assert result.stopped_reason == "cancelled"
    assert result.model_turns == 0


@pytest.mark.asyncio
async def test_model_error_is_reported_structurally() -> None:
    result = await Agent(model=ErrorModel(), registry=make_registry()).run("q")
    assert result.stopped_reason == "model_error"
    assert result.stop_detail["code"] == "model_error"
    err_steps = [s for s in result.steps if s.status == "error"]
    assert len(err_steps) == 1


# ----------------------------------------------------------------------
# 历史隔离与截断
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_history_is_passed_to_model() -> None:
    model = ScriptedModel()
    model.load_script([{"content": "ok"}])
    agent = Agent(model=model, registry=make_registry())
    await agent.run("第二个问题", history=[Message(role="user", content="第一个问题")])
    # 脚本模型按 assistant 消息数决定步进，这里只验证不抛异常且能收敛
    assert True


@pytest.mark.asyncio
async def test_long_history_is_truncated() -> None:
    """历史必须在第一次调用模型之前就被截断，否则长会话直接撑爆上下文。"""
    captured: dict = {}

    class CapturingModel(ModelClient):
        name = "capture"

        async def complete(self, messages, tools=None, *, temperature: float = 0.0):
            captured.setdefault("first_call", list(messages))
            return ModelResponse(content="ok", backend=self.name)

    history = [Message(role="user", content="x" * 2000) for _ in range(40)]
    agent = Agent(
        model=CapturingModel(),
        registry=make_registry(),
        max_history_chars=4000,
    )
    await agent.run("当前问题", history=history)

    first = captured["first_call"]
    total = sum(len(m.content or "") for m in first)
    assert total <= 4200, f"第一次调用就没有截断，总字符 {total}"
    assert first[0].role == "system"
    assert first[-1].content == "当前问题", "当前问题必须保留"


@pytest.mark.asyncio
async def test_truncation_does_not_orphan_tool_messages() -> None:
    """截断后不能出现无主的 tool 消息：assistant(tool_calls) 与 tool 必须同进同出。"""
    from agentloom.core.loop import _truncate

    msgs = [
        Message(role="system", content="sys"),
        Message(role="user", content="u" * 500),
        Message(role="assistant", content="", tool_calls=[{"id": "c1"}]),
        Message(role="tool", content="obs" * 200, tool_call_id="c1", name="t"),
    ]
    out = _truncate(msgs, max_chars=900)
    assert out[0].role == "system"
    # 要么 assistant+tool 都在，要么都不在；绝不允许只有 tool
    for i, m in enumerate(out):
        if m.role == "tool":
            assert i > 0 and out[i - 1].role in ("assistant", "tool"), "出现了无主的 tool 消息"


def test_truncation_keeps_last_message_even_if_oversized() -> None:
    from agentloom.core.loop import _truncate

    msgs = [
        Message(role="system", content="sys"),
        Message(role="user", content="old" * 100),
        Message(role="user", content="q" * 5000),
    ]
    out = _truncate(msgs, max_chars=1000)
    assert out[-1].content, "最后一条消息不能被丢成空"
    assert len(out[-1].content) <= 1000


# ----------------------------------------------------------------------
# 流式
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_emits_expected_event_sequence() -> None:
    model = ScriptedModel()
    model.load_script(
        [
            {
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "noop", "arguments": "{}"},
                    }
                ]
            },
            {"content": "最终答案"},
        ]
    )
    agent = Agent(model=model, registry=make_registry())
    events = [e async for e in agent.run_stream("q")]

    names = [e["event"] for e in events]
    assert names[0] == "start"
    assert names[-1] == "done"
    assert "step" in names

    done = events[-1]
    assert done["answer"] == "最终答案"
    assert done["stopped_reason"] == "completed"
    assert done["truncated"] is False
    assert done["tool_calls"] == 1
