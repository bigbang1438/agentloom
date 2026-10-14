"""RAG 测试：分片、行号溯源、分词、BM25 打分、引用校验、拒答。

引用校验是重点：假引用比没引用更危险，必须被拦住。
"""

from __future__ import annotations

import pytest

from agentloom.rag.retriever import (
    BM25Index,
    Citation,
    KnowledgeBase,
    assign_citations,
    chunk_document,
    content_terms,
    strip_invalid_citations,
    tokenize,
    tokenize_content,
    verify_citations,
)

DOC = """# 标题

第一段内容，讲的是库存管理。
第二段内容，讲的是订单处理。

## 小节

第三段内容，讲的是并发控制。
"""


# ---------------------------------------------------------------- 分片

def test_chunk_preserves_line_numbers() -> None:
    chunks = chunk_document("t.md", DOC, max_chars=1000)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.start_line == 1
    assert c.end_line == len(DOC.splitlines())


def test_chunk_text_matches_source_lines_exactly() -> None:
    """行号必须能对回原文，否则「来源：第 x-y 行」就是假的。"""
    chunks = chunk_document("t.md", DOC, max_chars=1000)
    src = DOC.splitlines()
    c = chunks[0]
    expected = "\n".join(src[c.start_line - 1 : c.end_line])
    assert expected.strip() == c.text.strip()


def test_chunk_splits_when_exceeding_max_chars() -> None:
    long_doc = "\n".join(f"第 {i} 行内容，填充一些文字。" for i in range(1, 101))
    chunks = chunk_document("t.md", long_doc, max_chars=200)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 400, f"片段过长: {len(c.text)}"


def test_chunk_has_overlap_between_neighbors() -> None:
    long_doc = "\n".join(f"第 {i} 行内容。" for i in range(1, 61))
    chunks = chunk_document("t.md", long_doc, max_chars=120, overlap_lines=1)
    assert len(chunks) >= 2
    # 相邻片段的行区间应当有重叠
    overlapped = any(
        chunks[i].end_line >= chunks[i + 1].start_line for i in range(len(chunks) - 1)
    )
    assert overlapped


def test_single_long_line_is_hard_split() -> None:
    doc = "x" * 2500
    chunks = chunk_document("t.md", doc, max_chars=600)
    assert len(chunks) >= 4
    for c in chunks:
        assert c.start_line == 1 and c.end_line == 1


def test_empty_document_yields_no_chunks() -> None:
    assert chunk_document("t.md", "") == []
    assert chunk_document("t.md", "\n\n\n") == []


# ---------------------------------------------------------------- 分词

def test_chinese_bigram_tokenization() -> None:
    tokens = tokenize("工具调用")
    assert "工具" in tokens
    assert "具调" in tokens
    assert "调用" in tokens


def test_english_tokens_are_lowercased() -> None:
    assert "bm25" in tokenize("BM25 retrieval")


def test_stopwords_are_removed_from_content_tokens() -> None:
    tokens = tokenize_content("这个项目怎么样")
    assert "怎么" not in tokens
    assert "这个" not in tokens


def test_content_terms_drops_single_cjk_chars() -> None:
    terms = content_terms("库")
    assert terms == set(), "单字不进覆盖率分母"


def test_content_terms_keeps_meaningful_bigrams() -> None:
    terms = content_terms("库存管理")
    assert "库存" in terms
    assert "存管" in terms


# ---------------------------------------------------------------- BM25

@pytest.fixture
def kb() -> KnowledgeBase:
    k = KnowledgeBase()
    k.add_document(
        "stock.md",
        "# 库存\n\n库存扣减要用乐观锁，防止超卖。\n库存不足时返回错误。\n",
    )
    k.add_document(
        "order.md",
        "# 订单\n\n订单创建后进入待支付状态。\n订单超时未支付要回补库存。\n",
    )
    return k


def test_search_returns_relevant_chunk(kb: KnowledgeBase) -> None:
    results = kb.search("乐观锁防止超卖", top_k=3)
    assert results
    assert results[0].chunk.filename == "stock.md"


def test_search_respects_top_k(kb: KnowledgeBase) -> None:
    assert len(kb.search("库存 订单", top_k=1)) <= 1


def test_search_on_empty_index_returns_nothing() -> None:
    assert BM25Index().search("任何问题") == []


def test_search_with_only_stopwords_returns_nothing(kb: KnowledgeBase) -> None:
    assert kb.search("的 了 是 怎么") == []


def test_unrelated_query_is_refused(kb: KnowledgeBase) -> None:
    """无答案拒答：问红烧肉不该命中工程文档。"""
    assert kb.search("红烧肉的做法是什么，需要哪些调料") == []


def test_single_char_query_does_not_produce_evidence(kb: KnowledgeBase) -> None:
    """只有单字重合不算证据。"""
    assert kb.search("锁") == []


def test_relevance_ordering_is_stable(kb: KnowledgeBase) -> None:
    a = kb.search("订单 库存", top_k=3)
    b = kb.search("订单 库存", top_k=3)
    assert [x.chunk.chunk_id for x in a] == [x.chunk.chunk_id for x in b]


def test_duplicate_document_replaces_previous(kb: KnowledgeBase) -> None:
    before = kb.chunk_count
    kb.add_document("stock.md", "# 库存\n\n完全不同的内容，讲的是缓存。\n")
    assert kb.doc_count == 2
    assert kb.chunk_count <= before + 1
    assert kb.search("乐观锁防止超卖") == []


# ---------------------------------------------------------------- 引用

def _cites(*indexes: int) -> list[Citation]:
    return [
        Citation(
            index=i,
            chunk_id=f"c{i}",
            filename="f.md",
            start_line=i,
            end_line=i + 1,
            text="t",
            score=1.0,
        )
        for i in indexes
    ]


def test_valid_citations_pass() -> None:
    check = verify_citations("结论是这样的 [1]，另外 [2]。", _cites(1, 2, 3))
    assert check.ok is True
    assert check.cited == [1, 2]


def test_fabricated_citation_is_caught() -> None:
    """模型编了一个不存在的编号，必须被拦。"""
    check = verify_citations("根据资料 [7]，答案是 42。", _cites(1, 2))
    assert check.ok is False
    assert check.invalid == [7]


def test_answer_without_citation_is_rejected() -> None:
    check = verify_citations("我认为答案是 42。", _cites(1, 2))
    assert check.ok is False
    assert "no citation" in check.reason


def test_out_of_range_high_index_is_caught() -> None:
    check = verify_citations("见 [99]", _cites(1))
    assert check.ok is False
    assert 99 in check.invalid


def test_strip_removes_only_invalid_citations() -> None:
    cleaned = strip_invalid_citations("有效 [1] 无效 [9]", _cites(1))
    assert "[1]" in cleaned
    assert "[9]" not in cleaned


def test_assign_citations_numbers_from_one() -> None:
    kb = KnowledgeBase()
    kb.add_document("a.md", "# A\n\n库存扣减使用乐观锁防止超卖问题。\n")
    scored = kb.search("库存 乐观锁 超卖", top_k=3)
    cites = assign_citations(scored)
    assert [c.index for c in cites] == list(range(1, len(cites) + 1))


def test_retrieve_with_citations_returns_numbered_context() -> None:
    kb = KnowledgeBase()
    kb.add_document("a.md", "# A\n\n库存扣减使用乐观锁防止超卖问题。\n")
    cites, context = kb.retrieve_with_citations("库存 乐观锁 超卖", top_k=2)
    assert cites
    assert "[1]" in context
    assert "来源：" in context


def test_retrieve_with_citations_on_no_match_returns_empty() -> None:
    kb = KnowledgeBase()
    kb.add_document("a.md", "# A\n\n库存扣减使用乐观锁。\n")
    cites, context = kb.retrieve_with_citations("红烧肉怎么做", top_k=2)
    assert cites == []
    assert context == ""


# ---------------------------------------------------------------- 文档管理

def test_documents_listing() -> None:
    kb = KnowledgeBase()
    kb.add_document("a.md", "内容 A")
    kb.add_document("b.md", "内容 B")
    names = [d["filename"] for d in kb.documents()]
    assert names == ["a.md", "b.md"]


def test_remove_document_rebuilds_index() -> None:
    kb = KnowledgeBase()
    kb.add_document("a.md", "# A\n\n库存扣减使用乐观锁防止超卖。\n")
    assert kb.search("乐观锁 超卖")
    kb.remove_document("a.md")
    assert kb.search("乐观锁 超卖") == []
    assert kb.doc_count == 0
