"""可观测性测试：指标累计、延迟分位、轨迹持久化与脱敏。

延迟分位是手算的，这里用手工构造的样本验证算法本身正确，
而不是只验证「跑起来不报错」。
"""

from __future__ import annotations

import json
import time

import pytest

from agentloom.core.budget import Budget
from agentloom.core.loop import Agent
from agentloom.models.client import ScriptedModel
from agentloom.observability.metrics import Metrics, TraceStore, latency_percentiles
from agentloom.tools.registry import ParamSpec, ToolRegistry


# ---------------------------------------------------------------- 分位数

def test_percentiles_empty() -> None:
    assert latency_percentiles([]) == {"p50": 0.0, "p95": 0.0, "p99": 0.0, "count": 0}


def test_percentiles_single_sample() -> None:
    out = latency_percentiles([42.0])
    assert out["p50"] == 42.0
    assert out["p99"] == 42.0
    assert out["count"] == 1


def test_percentiles_known_values() -> None:
    """1..100 的样本，线性插值下 p50≈50.5、p95≈95.05。"""
    out = latency_percentiles(range(1, 101))
    assert out["p50"] == pytest.approx(50.5, abs=0.01)
    assert out["p95"] == pytest.approx(95.05, abs=0.01)
    assert out["count"] == 100


def test_percentiles_are_monotonic() -> None:
    out = latency_percentiles([5, 1, 9, 3, 7, 2, 8, 4, 6, 100])
    assert out["p50"] <= out["p95"] <= out["p99"]


def test_percentiles_unsorted_input() -> None:
    a = latency_percentiles([3, 1, 2])
    b = latency_percentiles([1, 2, 3])
    assert a == b


# ---------------------------------------------------------------- 指标

def make_registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool("echo", "回显", [ParamSpec("text", "string")])
    def echo(text: str):
        return {"echo": text}

    return reg


def scripted(steps: list[dict]) -> ScriptedModel:
    m = ScriptedModel()
    m.load_script(steps)
    return m


@pytest.mark.asyncio
async def test_metrics_record_completed_run() -> None:
    model = scripted([{"content": "答案"}])
    result = await Agent(model=model, registry=make_registry()).run("q")

    m = Metrics()
    m.record_run(result)
    snap = m.snapshot()

    assert snap["runs_total"] == 1
    assert snap["runs_completed"] == 1
    assert snap["runs_truncated"] == 0
    assert snap["runs_failed"] == 0
    assert snap["latency_ms"]["count"] == 1


@pytest.mark.asyncio
async def test_metrics_record_truncated_run() -> None:
    from tests.test_loop import AlwaysToolModel

    agent = Agent(
        model=AlwaysToolModel("echo", {"text": "x"}),
        registry=make_registry(),
        budget=Budget(max_steps=2, max_model_turns=99, max_tool_calls=99),
    )
    result = await agent.run("q")

    m = Metrics()
    m.record_run(result)
    snap = m.snapshot()
    assert snap["runs_truncated"] == 1
    assert snap["runs_by_stop_reason"].get("max_steps") == 1


@pytest.mark.asyncio
async def test_metrics_count_tool_rejections() -> None:
    model = scripted(
        [
            {
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": '{"text":1}'},
                    }
                ]
            },
            {"content": "done"},
        ]
    )
    result = await Agent(model=model, registry=make_registry()).run("q")

    m = Metrics()
    m.record_run(result)
    snap = m.snapshot()
    assert snap["tool_rejections_total"] == 1
    assert snap["tool_calls_by_name"]["echo"] == 1


@pytest.mark.asyncio
async def test_metrics_accumulate_tokens_and_cost() -> None:
    model = scripted([{"content": "a", "prompt_tokens": 100, "completion_tokens": 50}])
    result = await Agent(model=model, registry=make_registry()).run("q")

    m = Metrics()
    m.record_run(result)
    snap = m.snapshot()
    assert snap["prompt_tokens_total"] == 100
    assert snap["completion_tokens_total"] == 50
    assert snap["estimated_cost_usd_total"] > 0


def test_metrics_reset() -> None:
    m = Metrics()
    m.record_injection(3)
    m.record_guardrail_block()
    assert m.snapshot()["injection_hits_total"] == 3
    m.reset()
    snap = m.snapshot()
    assert snap["injection_hits_total"] == 0
    assert snap["guardrail_blocks_total"] == 0


def test_metrics_record_injection_and_guardrail() -> None:
    m = Metrics()
    m.record_injection(2)
    m.record_guardrail_block()
    snap = m.snapshot()
    assert snap["injection_hits_total"] == 2
    assert snap["guardrail_blocks_total"] == 1


def test_metrics_tool_call_by_name() -> None:
    m = Metrics()
    m.record_tool_call("lookup_order")
    m.record_tool_call("lookup_order")
    m.record_tool_call("check_stock")
    snap = m.snapshot()
    assert snap["tool_calls_by_name"] == {"check_stock": 1, "lookup_order": 2}


# ---------------------------------------------------------------- 轨迹存储

@pytest.mark.asyncio
async def test_trace_store_saves_and_reads_back() -> None:
    model = scripted([{"content": "答案"}])
    result = await Agent(model=model, registry=make_registry()).run("问题")

    store = TraceStore(":memory:")
    run_id = store.save("问题", result)
    assert run_id == result.run_id

    got = store.get(run_id)
    assert got is not None
    assert got["question"] == "问题"
    assert got["answer"] == "答案"
    assert got["stopped_reason"] == "completed"
    assert isinstance(got["steps"], list)
    store.close()


@pytest.mark.asyncio
async def test_trace_store_recent_orders_newest_first() -> None:
    store = TraceStore(":memory:")
    for i in range(3):
        model = scripted([{"content": f"a{i}"}])
        result = await Agent(model=model, registry=make_registry()).run(f"q{i}")
        store.save(f"q{i}", result)
        time.sleep(0.01)

    recent = store.recent(limit=3)
    assert len(recent) == 3
    assert recent[0]["question"] == "q2"
    store.close()


@pytest.mark.asyncio
async def test_trace_store_stats() -> None:
    store = TraceStore(":memory:")
    model = scripted([{"content": "x"}])
    result = await Agent(model=model, registry=make_registry()).run("q")
    store.save("q", result)

    stats = store.stats()
    assert stats["runs"] == 1
    assert stats["by_stop_reason"] == {"completed": 1}
    assert stats["avg_elapsed_seconds"] >= 0
    store.close()


@pytest.mark.asyncio
async def test_trace_store_redacts_pii() -> None:
    """轨迹落盘前必须脱敏。"""
    model = scripted([{"content": "请联系 15925172896"}])
    result = await Agent(model=model, registry=make_registry()).run("我的手机号 15925172896")

    store = TraceStore(":memory:", redact=True)
    run_id = store.save("我的手机号 15925172896", result)
    got = store.get(run_id)

    blob = json.dumps(got, ensure_ascii=False)
    assert "15925172896" not in blob
    assert "159****2896" in blob
    store.close()


@pytest.mark.asyncio
async def test_trace_store_can_disable_redaction() -> None:
    model = scripted([{"content": "x"}])
    result = await Agent(model=model, registry=make_registry()).run("q")
    store = TraceStore(":memory:", redact=False)
    run_id = store.save("手机 15925172896", result)
    assert "15925172896" in store.get(run_id)["question"]
    store.close()


def test_trace_store_unknown_run_returns_none() -> None:
    store = TraceStore(":memory:")
    assert store.get("nope") is None
    store.close()


def test_trace_store_persists_to_file(tmp_path) -> None:
    db = tmp_path / "traces.db"
    store = TraceStore(str(db))
    store.close()
    assert db.exists()

    reopened = TraceStore(str(db))
    assert reopened.stats()["runs"] == 0
    reopened.close()
