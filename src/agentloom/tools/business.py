"""业务工具集：演示「只读工具 + 需人工确认的写工具」的完整边界。

关键设计：
- 读工具（查订单、查库存、检索知识、看统计）可以直接执行。
- 写工具（创建待办）**标记为 dangerous**，必须过人工确认闸门；
  而且模型侧**拿不到**审批工具——它只能提出请求，批不批由人决定。
  这是防止「模型自己批准自己」的结构性隔离，不是靠提示词约束。
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from agentloom.rag.retriever import KnowledgeBase, verify_citations, strip_invalid_citations
from agentloom.tools.registry import ParamSpec, ToolRegistry, ToolResult


# ----------------------------------------------------------------------
# 演示数据（真实项目里换成数据库）
# ----------------------------------------------------------------------

@dataclass
class Order:
    order_no: str
    sku: str
    status: str
    carrier: str
    eta_days: int
    amount: float


_DEMO_ORDERS: dict[str, Order] = {
    "A1024": Order("A1024", "SKU-1001", "shipped", "顺丰速运", 2, 299.00),
    "A1025": Order("A1025", "SKU-1002", "pending", "-", 5, 149.50),
    "B2048": Order("B2048", "SKU-2001", "delivered", "京东物流", 0, 899.00),
    "C3072": Order("C3072", "SKU-1001", "cancelled", "-", 0, 299.00),
}

_DEMO_STOCK: dict[str, dict[str, Any]] = {
    "SKU-1001": {"name": "机械键盘 87 键", "on_hand": 42, "reserved": 8, "reorder_point": 20},
    "SKU-1002": {"name": "人体工学鼠标", "on_hand": 7, "reserved": 3, "reorder_point": 15},
    "SKU-2001": {"name": "27 寸显示器", "on_hand": 0, "reserved": 0, "reorder_point": 5},
    "SKU-2002": {"name": "USB-C 扩展坞", "on_hand": 130, "reserved": 12, "reorder_point": 40},
}


# ----------------------------------------------------------------------
# 待确认写操作队列
# ----------------------------------------------------------------------

@dataclass
class PendingApproval:
    approval_id: str
    tool: str
    args: dict[str, Any]
    created_at: float
    status: str = "pending"  # pending | approved | rejected
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "tool": self.tool,
            "args": self.args,
            "status": self.status,
            "note": self.note,
            "created_at": self.created_at,
        }


class ApprovalQueue:
    """待确认队列。同一个 id 只能被消费一次（防重放）。"""

    def __init__(self) -> None:
        self._items: dict[str, PendingApproval] = {}
        self._consumed: set[str] = set()
        self._lock = threading.Lock()

    def propose(self, tool: str, args: dict[str, Any]) -> PendingApproval:
        with self._lock:
            item = PendingApproval(
                approval_id=uuid.uuid4().hex[:12],
                tool=tool,
                args=dict(args),
                created_at=time.time(),
            )
            self._items[item.approval_id] = item
            return item

    def list_pending(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                i.to_dict()
                for i in self._items.values()
                if i.status == "pending" and i.approval_id not in self._consumed
            ]

    def decide(self, approval_id: str, approved: bool, note: str = "") -> PendingApproval:
        with self._lock:
            item = self._items.get(approval_id)
            if item is None:
                raise KeyError(f"unknown approval id {approval_id!r}")
            if approval_id in self._consumed:
                raise ValueError(f"approval {approval_id!r} has already been consumed")
            if item.status != "pending":
                raise ValueError(f"approval {approval_id!r} is already {item.status}")
            item.status = "approved" if approved else "rejected"
            item.note = note
            self._consumed.add(approval_id)
            return item


class TaskStore:
    """已确认的待办（写操作真正落地的地方）。"""

    def __init__(self) -> None:
        self._tasks: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def add(self, title: str, priority: str, note: str = "") -> dict[str, Any]:
        with self._lock:
            task = {
                "task_id": uuid.uuid4().hex[:10],
                "title": title,
                "priority": priority,
                "note": note,
                "created_at": time.time(),
            }
            self._tasks.append(task)
            return task

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return list(reversed(self._tasks))[:limit]

    def count(self) -> int:
        with self._lock:
            return len(self._tasks)


# ----------------------------------------------------------------------
# 构建工具注册表
# ----------------------------------------------------------------------

@dataclass
class ToolContext:
    """工具执行所需的共享上下文。"""

    knowledge: KnowledgeBase
    approvals: ApprovalQueue
    tasks: TaskStore
    metrics: Any | None = None
    citation_state: dict[str, Any] = field(default_factory=dict)


def build_registry(ctx: ToolContext) -> ToolRegistry:
    reg = ToolRegistry()

    # ---------------- 读工具 ----------------

    @reg.tool(
        "lookup_order",
        "按订单号查询订单状态、承运商与预计到达时间。只读。",
        [
            ParamSpec(
                "order_no",
                "string",
                "订单号，形如 A1024",
                min_length=3,
                max_length=32,
            )
        ],
    )
    def lookup_order(order_no: str) -> ToolResult:
        order = _DEMO_ORDERS.get(order_no.upper())
        if order is None:
            # 业务失败用 ok=False 表达，不是系统异常
            return ToolResult(ok=False, error=f"order {order_no!r} not found")
        return ToolResult(
            ok=True,
            data={
                "order_no": order.order_no,
                "sku": order.sku,
                "status": order.status,
                "carrier": order.carrier,
                "eta_days": order.eta_days,
                "amount": order.amount,
            },
        )

    @reg.tool(
        "check_stock",
        "查询某个 SKU 的库存、已预留数量与补货点，并判断是否需要补货。只读。",
        [
            ParamSpec(
                "sku",
                "string",
                "商品编码，形如 SKU-1001",
                min_length=3,
                max_length=32,
            )
        ],
    )
    def check_stock(sku: str) -> ToolResult:
        item = _DEMO_STOCK.get(sku.upper())
        if item is None:
            return ToolResult(ok=False, error=f"sku {sku!r} not found")
        available = item["on_hand"] - item["reserved"]
        return ToolResult(
            ok=True,
            data={
                "sku": sku.upper(),
                "name": item["name"],
                "on_hand": item["on_hand"],
                "reserved": item["reserved"],
                "available": available,
                "reorder_point": item["reorder_point"],
                "needs_reorder": available < item["reorder_point"],
            },
        )

    @reg.tool(
        "sales_stats",
        "查看销售统计（订单数、金额、取消率）。只读。",
        [
            ParamSpec(
                "period",
                "string",
                "统计周期",
                enum=["today", "week", "month"],
                required=False,
                default="today",
            )
        ],
    )
    def sales_stats(period: str = "today") -> ToolResult:
        orders = list(_DEMO_ORDERS.values())
        total = len(orders)
        cancelled = sum(1 for o in orders if o.status == "cancelled")
        amount = sum(o.amount for o in orders if o.status != "cancelled")
        return ToolResult(
            ok=True,
            data={
                "period": period,
                "orders": total,
                "gross_amount": round(amount, 2),
                "cancelled": cancelled,
                "cancel_rate": round(cancelled / total, 4) if total else 0.0,
            },
        )

    @reg.tool(
        "knowledge_search",
        "在本机知识库中检索资料片段，返回带编号 [n] 的来源。回答事实性问题前必须先用它取依据。只读。",
        [
            ParamSpec(
                "query",
                "string",
                "检索问题，用自然语言描述你要查什么",
                min_length=1,
                max_length=200,
            ),
            ParamSpec(
                "top_k",
                "integer",
                "返回片段数，1 到 12",
                required=False,
                default=5,
                minimum=1,
                maximum=12,
            ),
        ],
    )
    def knowledge_search(query: str, top_k: int = 5) -> ToolResult:
        citations, blocks = ctx.knowledge.retrieve_with_citations(query, top_k=top_k)
        if not citations:
            # 无证据：明确返回空，让模型拒答而不是编造
            return ToolResult(
                ok=True,
                data={"found": 0, "message": "没有检索到相关片段", "citations": []},
            )
        return ToolResult(
            ok=True,
            # data 必须是纯 JSON：它会被序列化后作为观察结果回灌给模型。
            data={
                "found": len(citations),
                "context": blocks,
                "citations": [c.to_dict() for c in citations],
            },
            # 原始 Citation 对象只放在 meta 里：供循环做引用校验，不进模型上下文。
            # 关键：**不使用跨请求的共享状态**，否则上一个请求的引用会泄漏到
            # 下一个请求（无答案的问题会继承上一轮的引用）。
            meta={"citation_objects": citations},
        )

    # ---------------- 写工具（需人工确认） ----------------

    @reg.tool(
        "create_task",
        "创建一个待办任务。这是写操作，只会产生一条待确认请求，需要人工批准后才真正写入。",
        [
            ParamSpec("title", "string", "任务标题", min_length=1, max_length=120),
            ParamSpec(
                "priority",
                "string",
                "优先级",
                enum=["low", "normal", "high"],
                required=False,
                default="normal",
            ),
        ],
        dangerous=True,
        side_effect=True,
    )
    def create_task(title: str, priority: str = "normal") -> ToolResult:
        item = ctx.approvals.propose("create_task", {"title": title, "priority": priority})
        return ToolResult(
            ok=True,
            data={
                "pending_approval": item.to_dict(),
                "message": "已提交待确认请求，尚未写入任务列表。",
            },
        )

    return reg


def finalize_answer(answer: str, citations: list[Any]) -> tuple[str, dict[str, Any]]:
    """对模型答案做引用校验，返回 (清洗后答案, 校验结果)。

    注意：追加的说明文字里**不能**再出现 `[n]` 形式的编号。
    否则下游再次做引用抽取时，会把这条说明当成一个新的引用，
    等于把刚剔除的假引用又放回去了。
    """
    check = verify_citations(answer, citations)
    if check.ok:
        return answer, check.to_dict()

    cleaned = strip_invalid_citations(answer, citations)
    if check.invalid:
        bad = "、".join(f"编号 {i}" for i in check.invalid)
        notice = f"\n\n（注：原回答引用了不存在的来源（{bad}），已移除该引用。）"
    else:
        notice = "\n\n（注：原回答未提供来源编号，结论缺少依据。）"
    return cleaned + notice, check.to_dict()
