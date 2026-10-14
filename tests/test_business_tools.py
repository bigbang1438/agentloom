"""业务工具与审批队列测试。

重点：写操作只能提出请求、不能自己批准；确认编号只能消费一次。
"""

from __future__ import annotations

import pytest

from agentloom.rag.retriever import KnowledgeBase
from agentloom.tools.business import (
    ApprovalQueue,
    TaskStore,
    ToolContext,
    build_registry,
    finalize_answer,
)


@pytest.fixture
def ctx() -> ToolContext:
    return ToolContext(
        knowledge=KnowledgeBase(),
        approvals=ApprovalQueue(),
        tasks=TaskStore(),
    )


@pytest.fixture
def reg(ctx: ToolContext):
    return build_registry(ctx)


# ---------------------------------------------------------------- 只读工具

def test_lookup_order_found(reg) -> None:
    result = reg.invoke("lookup_order", {"order_no": "A1024"})
    assert result.ok is True
    assert result.data["status"] == "shipped"
    assert result.data["carrier"] == "顺丰速运"


def test_lookup_order_is_case_insensitive(reg) -> None:
    assert reg.invoke("lookup_order", {"order_no": "a1024"}).ok is True


def test_lookup_order_not_found_is_business_failure(reg) -> None:
    result = reg.invoke("lookup_order", {"order_no": "Z9999"})
    assert result.ok is False
    assert "not found" in result.error
    assert result.meta.get("rejected") is not True


def test_check_stock_computes_available_and_reorder(reg) -> None:
    result = reg.invoke("check_stock", {"sku": "SKU-1002"})
    assert result.ok is True
    # on_hand 7, reserved 3 -> available 4, reorder_point 15
    assert result.data["available"] == 4
    assert result.data["needs_reorder"] is True


def test_check_stock_no_reorder_needed(reg) -> None:
    result = reg.invoke("check_stock", {"sku": "SKU-1001"})
    # on_hand 42, reserved 8 -> available 34 >= 20
    assert result.data["needs_reorder"] is False


def test_check_stock_unknown_sku(reg) -> None:
    assert reg.invoke("check_stock", {"sku": "SKU-9999"}).ok is False


def test_sales_stats(reg) -> None:
    result = reg.invoke("sales_stats", {})
    assert result.ok is True
    assert result.data["orders"] == 4
    assert result.data["cancelled"] == 1
    assert result.data["cancel_rate"] == 0.25


def test_sales_stats_period_enum_enforced(reg) -> None:
    assert reg.invoke("sales_stats", {"period": "year"}).ok is False


# ---------------------------------------------------------------- 知识检索工具

def test_knowledge_search_without_documents_reports_zero(ctx: ToolContext, reg) -> None:
    result = reg.invoke("knowledge_search", {"query": "工具调用为什么要校验"})
    assert result.ok is True
    assert result.data["found"] == 0
    assert result.data["citations"] == []


def test_knowledge_search_returns_numbered_citations(ctx: ToolContext, reg) -> None:
    ctx.knowledge.add_document(
        "s.md", "# 安全\n\n工具调用必须做参数校验，防止模型塞入未声明字段。\n"
    )
    result = reg.invoke("knowledge_search", {"query": "工具调用参数校验未声明字段"})
    assert result.ok is True
    assert result.data["found"] >= 1
    assert result.data["citations"][0]["index"] == 1
    assert "[1]" in result.data["context"]


def test_knowledge_search_top_k_is_bounded(reg) -> None:
    assert reg.invoke("knowledge_search", {"query": "x", "top_k": 99}).ok is False


# ---------------------------------------------------------------- 写工具

def test_create_task_is_marked_dangerous(reg) -> None:
    assert reg.get("create_task").dangerous is True


def test_create_task_only_proposes(ctx: ToolContext, reg) -> None:
    """写工具执行后不能直接落库，只能产生待确认请求。"""
    result = reg.invoke("create_task", {"title": "补货", "priority": "high"})
    assert result.ok is True
    assert result.data["pending_approval"]["status"] == "pending"
    assert ctx.tasks.count() == 0, "未批准前不该有任务"


def test_model_cannot_reach_an_approval_tool(reg) -> None:
    """结构性隔离：工具白名单里不存在审批工具。"""
    assert "approve" not in reg.names()
    assert not any("approv" in n for n in reg.names())


def test_undeclared_field_on_write_tool_is_rejected(reg) -> None:
    result = reg.invoke(
        "create_task", {"title": "t", "priority": "normal", "approved": True}
    )
    assert result.ok is False
    assert any("approved" in e for e in result.meta["errors"])


# ---------------------------------------------------------------- 审批队列

def test_approval_queue_propose_and_list() -> None:
    q = ApprovalQueue()
    item = q.propose("create_task", {"title": "t"})
    pending = q.list_pending()
    assert len(pending) == 1
    assert pending[0]["approval_id"] == item.approval_id


def test_approval_queue_decide_approve() -> None:
    q = ApprovalQueue()
    item = q.propose("create_task", {"title": "t"})
    decided = q.decide(item.approval_id, True, "同意")
    assert decided.status == "approved"
    assert q.list_pending() == []


def test_approval_queue_decide_reject() -> None:
    q = ApprovalQueue()
    item = q.propose("create_task", {"title": "t"})
    decided = q.decide(item.approval_id, False, "不需要")
    assert decided.status == "rejected"


def test_approval_cannot_be_consumed_twice() -> None:
    q = ApprovalQueue()
    item = q.propose("create_task", {"title": "t"})
    q.decide(item.approval_id, True)
    with pytest.raises(ValueError, match="already been consumed"):
        q.decide(item.approval_id, True)


def test_unknown_approval_id_raises() -> None:
    with pytest.raises(KeyError):
        ApprovalQueue().decide("nope", True)


def test_approval_ids_are_unique() -> None:
    q = ApprovalQueue()
    ids = {q.propose("create_task", {"title": str(i)}).approval_id for i in range(50)}
    assert len(ids) == 50


# ---------------------------------------------------------------- 任务存储

def test_task_store_add_and_list_newest_first() -> None:
    store = TaskStore()
    store.add("第一个", "low")
    store.add("第二个", "high")
    tasks = store.list()
    assert [t["title"] for t in tasks] == ["第二个", "第一个"]
    assert store.count() == 2


def test_task_store_limit() -> None:
    store = TaskStore()
    for i in range(10):
        store.add(f"t{i}", "normal")
    assert len(store.list(limit=3)) == 3


# ---------------------------------------------------------------- 引用收尾

def test_finalize_answer_accepts_valid_citation(ctx: ToolContext) -> None:
    ctx.knowledge.add_document("s.md", "# S\n\n库存扣减使用乐观锁防止超卖问题。\n")
    cites, _ = ctx.knowledge.retrieve_with_citations("库存 乐观锁 超卖")
    answer, check = finalize_answer("库存要用乐观锁 [1]。", cites)
    assert check["ok"] is True
    assert answer == "库存要用乐观锁 [1]。"


def test_finalize_answer_strips_fabricated_citation(ctx: ToolContext) -> None:
    ctx.knowledge.add_document("s.md", "# S\n\n库存扣减使用乐观锁防止超卖问题。\n")
    cites, _ = ctx.knowledge.retrieve_with_citations("库存 乐观锁 超卖")
    answer, check = finalize_answer("库存要用乐观锁 [9]。", cites)
    assert check["ok"] is False
    assert "[9]" not in answer, "假引用必须被剔除"
    assert "编号 9" in answer, "说明里要指出是哪个编号"


def test_finalize_answer_notice_contains_no_new_citation(ctx: ToolContext) -> None:
    """追加的说明不能引入新的 [n]，否则下游再抽取会把假引用放回去。"""
    import re

    ctx.knowledge.add_document("s.md", "# S\n\n库存扣减使用乐观锁防止超卖问题。\n")
    cites, _ = ctx.knowledge.retrieve_with_citations("库存 乐观锁 超卖")
    answer, _ = finalize_answer("库存要用乐观锁 [9]。", cites)
    assert re.findall(r"\[\d+\]", answer) == [], f"仍残留引用编号: {answer}"


def test_finalize_answer_flags_missing_citation(ctx: ToolContext) -> None:
    ctx.knowledge.add_document("s.md", "# S\n\n库存扣减使用乐观锁防止超卖问题。\n")
    cites, _ = ctx.knowledge.retrieve_with_citations("库存 乐观锁 超卖")
    answer, check = finalize_answer("库存要用乐观锁。", cites)
    assert check["ok"] is False
    assert "缺少依据" in answer
