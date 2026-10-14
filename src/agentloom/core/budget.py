"""预算守卫：防止 Agent 无限自问自答、无限调工具、烧穿 token 与成本。

设计要点：
- 五维预算：模型轮次、工具调用次数、总步数、墙钟时间、token 与成本。
- 超预算**不抛 500**，而是返回结构化 stopped_reason，让调用方区分
  「被截断」（truncated）与「失败」（failed）——这是可运维性的关键。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

StopReason = Literal[
    "completed",
    "max_steps",
    "max_model_turns",
    "max_tool_calls",
    "timeout",
    "max_tokens",
    "max_cost",
    "model_error",
    "guardrail",
    "cancelled",
]


@dataclass
class Budget:
    """一次 Agent 运行的预算上限。"""

    max_steps: int = 12
    max_model_turns: int = 8
    max_tool_calls: int = 12
    timeout_seconds: float = 60.0
    max_tokens: int = 60_000
    max_cost_usd: float = 0.50

    # 单价（美元 / 千 token），用于成本估算；默认按通用兼容后端量级给一个保守值
    price_per_1k_input: float = 0.0005
    price_per_1k_output: float = 0.0015


@dataclass
class BudgetState:
    """预算的实时消耗状态。"""

    steps: int = 0
    model_turns: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # 用 perf_counter 而不是 monotonic：Windows 上 monotonic 走 GetTickCount64，
    # 分辨率约 15.6ms，一个跑得快的循环整个生命周期都可能读到 0.0 秒，
    # 导致超时预算形同虚设。perf_counter 是高精度计时器，适合测时长。
    started_at: float = field(default_factory=time.perf_counter)
    stopped_reason: StopReason | None = None
    stop_detail: dict = field(default_factory=dict)

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.started_at

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class BudgetGuard:
    """在每一步前后检查预算，返回停止原因而不是抛异常。"""

    def __init__(self, budget: Budget) -> None:
        self.budget = budget
        self.state = BudgetState()

    # ---- 记录消耗 ----
    def count_step(self) -> None:
        self.state.steps += 1

    def count_model_turn(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        self.state.model_turns += 1
        self.state.prompt_tokens += max(0, prompt_tokens)
        self.state.completion_tokens += max(0, completion_tokens)

    def count_tool_call(self) -> None:
        self.state.tool_calls += 1

    # ---- 成本估算 ----
    @property
    def estimated_cost_usd(self) -> float:
        b = self.budget
        s = self.state
        return (
            s.prompt_tokens / 1000.0 * b.price_per_1k_input
            + s.completion_tokens / 1000.0 * b.price_per_1k_output
        )

    # ---- 检查 ----
    def check(self) -> StopReason | None:
        """返回停止原因；None 表示可以继续。"""
        b, s = self.budget, self.state

        if s.stopped_reason is not None:
            return s.stopped_reason

        reason: StopReason | None = None
        detail: dict = {}

        if s.elapsed_seconds > b.timeout_seconds:
            reason = "timeout"
            detail = {"elapsed_seconds": round(s.elapsed_seconds, 3), "limit": b.timeout_seconds}
        elif s.steps >= b.max_steps:
            reason = "max_steps"
            detail = {"steps": s.steps, "limit": b.max_steps}
        elif s.model_turns >= b.max_model_turns:
            reason = "max_model_turns"
            detail = {"model_turns": s.model_turns, "limit": b.max_model_turns}
        elif s.tool_calls >= b.max_tool_calls:
            reason = "max_tool_calls"
            detail = {"tool_calls": s.tool_calls, "limit": b.max_tool_calls}
        elif s.total_tokens >= b.max_tokens:
            reason = "max_tokens"
            detail = {"total_tokens": s.total_tokens, "limit": b.max_tokens}
        elif self.estimated_cost_usd >= b.max_cost_usd:
            reason = "max_cost"
            detail = {
                "estimated_cost_usd": round(self.estimated_cost_usd, 6),
                "limit": b.max_cost_usd,
            }

        if reason is not None:
            s.stopped_reason = reason
            s.stop_detail = detail
        return reason

    def stop(self, reason: StopReason, **detail) -> None:
        self.state.stopped_reason = reason
        self.state.stop_detail = dict(detail)

    def snapshot(self) -> dict:
        s = self.state
        return {
            "steps": s.steps,
            "model_turns": s.model_turns,
            "tool_calls": s.tool_calls,
            "prompt_tokens": s.prompt_tokens,
            "completion_tokens": s.completion_tokens,
            "total_tokens": s.total_tokens,
            "elapsed_seconds": round(s.elapsed_seconds, 3),
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "stopped_reason": s.stopped_reason or "completed",
            "stop_detail": s.stop_detail,
        }
