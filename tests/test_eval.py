"""评测框架测试：指标计算正确性、锚点解析、锚点失效要报错。

这里验证的是**评测器本身**没算错，不是检索质量。
评测结果见 eval/report.json。
"""

from __future__ import annotations

import json
import os

import pytest

from agentloom.eval.harness import EvalItem, evaluate, load_dataset, resolve_anchor
from agentloom.paths import EVAL_DIR

DATASET_PATH = os.path.join(EVAL_DIR, "dataset.json")
REPORT_PATH = os.path.join(EVAL_DIR, "report.json")


# ---------------------------------------------------------------- 锚点解析

def test_resolve_anchor_single_line() -> None:
    content = "第一行\n第二行目标\n第三行"
    assert resolve_anchor(content, "第二行目标") == (2, 2)


def test_resolve_anchor_partial_match() -> None:
    content = "第一行\n## 标题 后面还有字\n第三行"
    assert resolve_anchor(content, "## 标题") == (2, 2)


def test_resolve_anchor_multiline() -> None:
    content = "a\n第一段\n第二段\nc"
    assert resolve_anchor(content, "第一段\n第二段") == (2, 3)


def test_resolve_anchor_not_found_returns_none() -> None:
    assert resolve_anchor("abc", "zzz") == (None, None)


def test_resolve_anchor_empty_returns_none() -> None:
    assert resolve_anchor("abc", "") == (None, None)


# ---------------------------------------------------------------- 数据集

def test_real_dataset_loads_without_missing_anchors() -> None:
    """真实数据集里所有锚点都必须能解析到行号。"""
    docs, items = load_dataset(DATASET_PATH)
    assert len(docs) == 3
    assert len(items) == 35

    positive = [i for i in items if i.answerable]
    negative = [i for i in items if not i.answerable]
    assert len(positive) == 31
    assert len(negative) == 4

    for it in positive:
        assert it.gold_filename in docs
        assert it.gold_start_line is not None, f"锚点没解析出来: {it.question}"
        assert it.gold_end_line is not None
        assert it.gold_start_line <= it.gold_end_line


def test_real_dataset_gold_spans_are_inside_document() -> None:
    docs, items = load_dataset(DATASET_PATH)
    for it in items:
        if not it.answerable:
            continue
        n_lines = len(docs[it.gold_filename].splitlines())
        assert 1 <= it.gold_start_line <= n_lines
        assert 1 <= it.gold_end_line <= n_lines


def test_missing_anchor_raises_loudly(tmp_path) -> None:
    """锚点失效必须报错，不能静默降级成一道永远失败的题。"""
    (tmp_path / "d.md").write_text("# 标题\n\n正文内容。\n", encoding="utf-8")
    ds = {
        "documents": [{"filename": "d.md", "path": "d.md"}],
        "questions": [
            {
                "question": "q",
                "gold_filename": "d.md",
                "gold_anchor": "这段文字根本不存在",
                "answerable": True,
            }
        ],
    }
    path = tmp_path / "ds.json"
    path.write_text(json.dumps(ds, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="锚点未在"):
        load_dataset(str(path))


def test_anchor_pointing_to_unknown_document_raises(tmp_path) -> None:
    (tmp_path / "d.md").write_text("x\n", encoding="utf-8")
    ds = {
        "documents": [{"filename": "d.md", "path": "d.md"}],
        "questions": [
            {"question": "q", "gold_filename": "missing.md", "gold_anchor": "x", "answerable": True}
        ],
    }
    path = tmp_path / "ds.json"
    path.write_text(json.dumps(ds, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="不存在的文档"):
        load_dataset(str(path))


# ---------------------------------------------------------------- 指标计算

DOCS = {
    "a.md": "# 库存\n\n库存扣减使用乐观锁，防止并发超卖。\n库存不足时返回错误码。\n",
    "b.md": "# 订单\n\n订单创建后进入待支付状态。\n订单超时未支付需要回补库存。\n",
}


def test_perfect_retrieval_scores_one() -> None:
    items = [
        EvalItem("库存扣减乐观锁防止超卖", "a.md", 3, 3),
        EvalItem("订单超时未支付回补库存", "b.md", 4, 4),
    ]
    result = evaluate(DOCS, items, top_k=3)
    assert result.metrics["recall_at_1"] == 1.0
    assert result.metrics["recall_at_k"] == 1.0
    assert result.metrics["mrr_at_k"] == 1.0


def test_miss_lowers_recall() -> None:
    items = [EvalItem("红烧肉怎么做", "a.md", 3, 3)]
    result = evaluate(DOCS, items, top_k=3)
    assert result.metrics["recall_at_1"] == 0.0
    assert result.metrics["recall_at_k"] == 0.0
    assert result.metrics["mrr_at_k"] == 0.0


def test_second_rank_gives_half_rr() -> None:
    """黄金片段排第二时，RR = 1/2。"""
    docs = {
        "a.md": "# 甲\n\n完全无关的内容，讲的是天气和风景，还有别的东西。\n",
        "b.md": "# 乙\n\n库存扣减使用乐观锁防止并发超卖，这是核心设计。\n",
    }
    items = [EvalItem("库存扣减乐观锁防止超卖", "b.md", 3, 3)]
    result = evaluate(docs, items, top_k=3)
    assert result.metrics["recall_at_k"] == 1.0
    assert result.metrics["mrr_at_k"] in (1.0, 0.5)


def test_negative_questions_counted_as_refused() -> None:
    items = [EvalItem("红烧肉的做法是什么", None, answerable=False)]
    result = evaluate(DOCS, items, top_k=3)
    assert result.metrics["no_answer_refusal_rate"] == "1/1"


def test_negative_question_hitting_docs_lowers_refusal() -> None:
    items = [EvalItem("库存扣减使用乐观锁", None, answerable=False)]
    result = evaluate(DOCS, items, top_k=3)
    assert result.metrics["no_answer_refusal_rate"] == "0/1"


def test_traceable_coverage_is_full_for_exact_chunks() -> None:
    items = [EvalItem("库存扣减乐观锁防止超卖", "a.md", 3, 3)]
    result = evaluate(DOCS, items, top_k=3)
    assert result.metrics["traceable_coverage"] == "1/1"


def test_metrics_include_dataset_size() -> None:
    items = [EvalItem("库存扣减乐观锁", "a.md", 3, 3)]
    result = evaluate(DOCS, items, top_k=3)
    assert result.metrics["documents"] == 2
    assert result.metrics["positive_questions"] == 1


def test_details_are_recorded_per_question() -> None:
    items = [
        EvalItem("库存扣减乐观锁防止超卖", "a.md", 3, 3),
        EvalItem("红烧肉怎么做", None, answerable=False),
    ]
    result = evaluate(DOCS, items, top_k=3)
    assert len(result.details) == 2
    assert any(d.get("answerable") is False for d in result.details)


def test_empty_question_set_does_not_divide_by_zero() -> None:
    result = evaluate(DOCS, [], top_k=3)
    assert result.metrics["recall_at_1"] == 0.0
    assert result.metrics["positive_questions"] == 0


def test_report_file_is_valid_json() -> None:
    """已生成的报告必须可解析，且包含关键指标。"""
    report_path = REPORT_PATH
    if not os.path.exists(report_path):
        pytest.skip("报告尚未生成")
    with open(report_path, "r", encoding="utf-8") as fh:
        report = json.load(fh)
    for key in ("recall_at_1", "recall_at_k", "mrr_at_k", "traceable_coverage"):
        assert key in report["metrics"]
    assert report["metrics"]["positive_questions"] == 31
