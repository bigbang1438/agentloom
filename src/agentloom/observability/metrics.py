"""可观测层：结构化轨迹、指标采集、运行记录持久化。

三个东西：
1. `Metrics`：进程内计数器（运行数、工具调用、拒绝数、token、延迟分位）。
   多副本部署下各自独立，没有聚合——这是已知边界，不假装是 Prometheus。
2. `TraceStore`：把每次运行的轨迹写进 SQLite，支持事后复盘与回归。
   写入前过 PII 脱敏，避免把用户隐私落到磁盘。
3. `latency_percentiles`：手算 P50/P95/P99，不引第三方库。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Iterable

from agentloom.guardrails.filters import redact_pii


def latency_percentiles(samples: Iterable[float]) -> dict[str, float]:
    """手算分位数。用最近邻法（nearest-rank），样本少时也不虚报。"""
    data = sorted(float(s) for s in samples)
    if not data:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "count": 0}

    def pick(p: float) -> float:
        if len(data) == 1:
            return data[0]
        rank = p * (len(data) - 1)
        lo = int(rank)
        hi = min(lo + 1, len(data) - 1)
        frac = rank - lo
        return data[lo] * (1 - frac) + data[hi] * frac

    return {
        "p50": round(pick(0.50), 3),
        "p95": round(pick(0.95), 3),
        "p99": round(pick(0.99), 3),
        "count": len(data),
    }


@dataclass
class Metrics:
    """线程安全的进程内指标。"""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    runs_total: int = 0
    runs_completed: int = 0
    runs_truncated: int = 0
    runs_failed: int = 0
    model_turns_total: int = 0
    tool_calls_total: int = 0
    tool_rejections_total: int = 0
    approval_denials_total: int = 0
    guardrail_blocks_total: int = 0
    injection_hits_total: int = 0
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0
    estimated_cost_usd_total: float = 0.0
    _latencies: deque[float] = field(default_factory=lambda: deque(maxlen=2000), repr=False)
    _by_tool: dict[str, int] = field(default_factory=lambda: defaultdict(int), repr=False)
    _by_stop_reason: dict[str, int] = field(default_factory=lambda: defaultdict(int), repr=False)

    def record_run(self, result: Any) -> None:
        """记录一次 Agent 运行。result 是 AgentResult 或等价对象。"""
        with self._lock:
            self.runs_total += 1
            reason = getattr(result, "stopped_reason", "completed")
            self._by_stop_reason[reason] += 1

            if reason == "completed":
                self.runs_completed += 1
            elif reason in ("model_error",):
                self.runs_failed += 1
            else:
                self.runs_truncated += 1

            self.model_turns_total += int(getattr(result, "model_turns", 0) or 0)
            self.tool_calls_total += int(getattr(result, "tool_calls", 0) or 0)
            self.prompt_tokens_total += int(getattr(result, "prompt_tokens", 0) or 0)
            self.completion_tokens_total += int(getattr(result, "completion_tokens", 0) or 0)
            self.estimated_cost_usd_total += float(getattr(result, "estimated_cost_usd", 0.0) or 0.0)

            elapsed = float(getattr(result, "elapsed_seconds", 0.0) or 0.0)
            self._latencies.append(elapsed * 1000.0)

            for step in getattr(result, "steps", []) or []:
                kind = getattr(step, "kind", None)
                status = getattr(step, "status", None)
                name = getattr(step, "name", "?")
                if kind == "tool":
                    self._by_tool[name] += 1
                    if status in ("rejected", "error"):
                        self.tool_rejections_total += 1
                elif kind == "approval" and status == "rejected":
                    self.approval_denials_total += 1

    def record_tool_call(self, tool: str) -> None:
        with self._lock:
            self.tool_calls_total += 1
            self._by_tool[tool] += 1

    def record_injection(self, count: int = 1) -> None:
        with self._lock:
            self.injection_hits_total += count

    def record_guardrail_block(self) -> None:
        with self._lock:
            self.guardrail_blocks_total += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            lat = latency_percentiles(self._latencies)
            return {
                "runs_total": self.runs_total,
                "runs_completed": self.runs_completed,
                "runs_truncated": self.runs_truncated,
                "runs_failed": self.runs_failed,
                "model_turns_total": self.model_turns_total,
                "tool_calls_total": self.tool_calls_total,
                "tool_rejections_total": self.tool_rejections_total,
                "approval_denials_total": self.approval_denials_total,
                "guardrail_blocks_total": self.guardrail_blocks_total,
                "injection_hits_total": self.injection_hits_total,
                "prompt_tokens_total": self.prompt_tokens_total,
                "completion_tokens_total": self.completion_tokens_total,
                "estimated_cost_usd_total": round(self.estimated_cost_usd_total, 6),
                "latency_ms": lat,
                "tool_calls_by_name": dict(sorted(self._by_tool.items())),
                "runs_by_stop_reason": dict(sorted(self._by_stop_reason.items())),
            }

    def reset(self) -> None:
        with self._lock:
            self.runs_total = 0
            self.runs_completed = 0
            self.runs_truncated = 0
            self.runs_failed = 0
            self.model_turns_total = 0
            self.tool_calls_total = 0
            self.tool_rejections_total = 0
            self.approval_denials_total = 0
            self.guardrail_blocks_total = 0
            self.injection_hits_total = 0
            self.prompt_tokens_total = 0
            self.completion_tokens_total = 0
            self.estimated_cost_usd_total = 0.0
            self._latencies.clear()
            self._by_tool.clear()
            self._by_stop_reason.clear()


class TraceStore:
    """把运行轨迹持久化到 SQLite，支持事后复盘。"""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY,
        created_at REAL NOT NULL,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        stopped_reason TEXT NOT NULL,
        tool_calls INTEGER NOT NULL,
        model_turns INTEGER NOT NULL,
        elapsed_seconds REAL NOT NULL,
        estimated_cost_usd REAL NOT NULL,
        model_backend TEXT NOT NULL,
        steps_json TEXT NOT NULL,
        citations_json TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_runs_reason ON runs(stopped_reason);
    """

    def __init__(self, path: str = ":memory:", *, redact: bool = True) -> None:
        self.path = path
        self.redact = redact
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(self.SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def save(self, question: str, result: Any) -> str:
        run_id = getattr(result, "run_id", "") or f"run_{int(time.time()*1000)}"
        steps = [s.to_dict() if hasattr(s, "to_dict") else s for s in getattr(result, "steps", [])]
        citations = list(getattr(result, "citations", []) or [])

        payload = {
            "question": question,
            "answer": getattr(result, "answer", ""),
            "steps": steps,
            "citations": citations,
        }
        if self.redact:
            payload = json.loads(redact_pii(json.dumps(payload, ensure_ascii=False)))

        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, created_at, question, answer, stopped_reason,
                    tool_calls, model_turns, elapsed_seconds, estimated_cost_usd,
                    model_backend, steps_json, citations_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    time.time(),
                    payload["question"],
                    payload["answer"],
                    getattr(result, "stopped_reason", "completed"),
                    int(getattr(result, "tool_calls", 0) or 0),
                    int(getattr(result, "model_turns", 0) or 0),
                    float(getattr(result, "elapsed_seconds", 0.0) or 0.0),
                    float(getattr(result, "estimated_cost_usd", 0.0) or 0.0),
                    getattr(result, "backend", "unknown"),
                    json.dumps(payload["steps"], ensure_ascii=False),
                    json.dumps(payload["citations"], ensure_ascii=False),
                ),
            )
            self._conn.commit()
        return run_id

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) AS c FROM runs").fetchone()["c"]
            by_reason = self._conn.execute(
                "SELECT stopped_reason, COUNT(*) AS c FROM runs GROUP BY stopped_reason"
            ).fetchall()
            agg = self._conn.execute(
                """
                SELECT AVG(elapsed_seconds) AS avg_elapsed,
                       SUM(tool_calls) AS total_tools,
                       SUM(estimated_cost_usd) AS total_cost
                FROM runs
                """
            ).fetchone()
        return {
            "runs": total,
            "by_stop_reason": {r["stopped_reason"]: r["c"] for r in by_reason},
            "avg_elapsed_seconds": round(agg["avg_elapsed"] or 0.0, 3),
            "total_tool_calls": agg["total_tools"] or 0,
            "total_estimated_cost_usd": round(agg["total_cost"] or 0.0, 6),
        }

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["steps"] = json.loads(d.pop("steps_json") or "[]")
        d["citations"] = json.loads(d.pop("citations_json") or "[]")
        return d
