"""离线评测：检索质量 + 引用正确性 + 拒答行为。

为什么要有评测：
「Agent 效果好不好」不能靠形容词。这里用一份**固定数据集**跑出可复现的数字：
Recall@K、MRR@K、行号可追溯覆盖率、无答案拒答率。

诚实的边界（写进 README）：
- 数据集小且公开，题目与资料同源，**不能**代表陌生领域或规模扩张后的表现。
- 这是**检索指标**，不是 LLM 问答准确率。
- 行号可追溯 ≠ 引用语义支持：编号对得上只证明指向了真实片段。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from agentloom.paths import EVAL_DIR, PROJECT_ROOT
from agentloom.rag.retriever import BM25Index, KnowledgeBase, chunk_document


@dataclass
class EvalItem:
    """一道评测题。gold 用 (文件名, 行区间) 标识黄金片段。"""

    question: str
    gold_filename: str | None
    gold_start_line: int | None = None
    gold_end_line: int | None = None
    answerable: bool = True


@dataclass
class EvalResult:
    metrics: dict[str, Any] = field(default_factory=dict)
    details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"metrics": self.metrics, "details": self.details}


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return not (a_end < b_start or b_end < a_start)


def evaluate(
    documents: dict[str, str],
    items: Sequence[EvalItem],
    *,
    top_k: int = 3,
    max_chars_per_chunk: int = 600,
) -> EvalResult:
    """跑完整评测。"""
    kb = KnowledgeBase(max_chars_per_chunk=max_chars_per_chunk)
    for name, content in documents.items():
        kb.add_document(name, content)

    positive = [it for it in items if it.answerable]
    negative = [it for it in items if not it.answerable]

    hit_at_1 = 0
    hit_at_k = 0
    rr_sum = 0.0
    traceable_ok = 0
    traceable_total = 0
    details: list[dict[str, Any]] = []

    for it in positive:
        scored = kb.search(it.question, top_k=top_k)
        ranks: list[int] = []
        for rank, s in enumerate(scored, start=1):
            c = s.chunk
            if c.filename != it.gold_filename:
                continue
            if it.gold_start_line is None:
                ranks.append(rank)
                break
            if _overlaps(c.start_line, c.end_line, it.gold_start_line, it.gold_end_line or it.gold_start_line):
                ranks.append(rank)
                break

        first = ranks[0] if ranks else 0
        if first == 1:
            hit_at_1 += 1
        if first >= 1:
            hit_at_k += 1
            rr_sum += 1.0 / first

        # 行号可追溯：返回片段的文本必须与源文件对应行完全一致
        if scored:
            traceable_total += 1
            c = scored[0].chunk
            src_lines = documents.get(c.filename, "").splitlines()
            if 1 <= c.start_line <= len(src_lines):
                expected = "\n".join(src_lines[c.start_line - 1 : c.end_line])
                if expected.strip() == c.text.strip():
                    traceable_ok += 1

        details.append(
            {
                "question": it.question,
                "gold": f"{it.gold_filename}:{it.gold_start_line}-{it.gold_end_line}",
                "first_hit_rank": first,
                "top": [
                    {
                        "rank": i,
                        "chunk": s.chunk.chunk_id,
                        "span": s.chunk.line_span,
                        "score": round(s.score, 4),
                    }
                    for i, s in enumerate(scored, start=1)
                ],
            }
        )

    # 无答案题：不应该返回任何词法匹配
    refused = 0
    for it in negative:
        scored = kb.search(it.question, top_k=top_k)
        if not scored:
            refused += 1
        details.append(
            {
                "question": it.question,
                "answerable": False,
                "returned": len(scored),
                "refused": not scored,
            }
        )

    n_pos = max(1, len(positive))
    metrics = {
        "documents": len(documents),
        "chunks": kb.chunk_count,
        "positive_questions": len(positive),
        "negative_questions": len(negative),
        "top_k": top_k,
        "recall_at_1": round(hit_at_1 / n_pos, 4),
        "recall_at_k": round(hit_at_k / n_pos, 4),
        "mrr_at_k": round(rr_sum / n_pos, 4),
        "traceable_coverage": f"{traceable_ok}/{traceable_total}",
        "no_answer_refusal_rate": f"{refused}/{max(1, len(negative))}",
    }
    return EvalResult(metrics=metrics, details=details)


def load_dataset(path: str) -> tuple[dict[str, str], list[EvalItem]]:
    """加载数据集，并把内容锚点解析成行号。

    用锚点而不是硬编码行号：资料文档增删一行不会让整个数据集失效，
    锚点找不到时会**直接报错**，而不是静默降级成一道永远失败的题。
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    documents: dict[str, str] = {}
    base = os.path.dirname(os.path.abspath(path))
    for doc in data.get("documents", []):
        if "content" in doc:
            documents[doc["filename"]] = doc["content"]
        else:
            with open(os.path.join(base, doc["path"]), "r", encoding="utf-8") as fh:
                documents[doc["filename"]] = fh.read()

    items: list[EvalItem] = []
    for q in data.get("questions", []):
        anchor = q.get("gold_anchor")
        filename = q.get("gold_filename")
        start = end = None

        if anchor and filename:
            content = documents.get(filename)
            if content is None:
                raise ValueError(f"锚点指向不存在的文档: {filename!r}")
            start, end = resolve_anchor(content, anchor)
            if start is None:
                raise ValueError(
                    f"锚点未在 {filename!r} 中找到: {anchor!r}\n"
                    f"资料文档改动后请同步更新 eval/dataset.json 的 gold_anchor。"
                )

        items.append(
            EvalItem(
                question=q["question"],
                gold_filename=filename,
                gold_start_line=start,
                gold_end_line=end,
                answerable=q.get("answerable", True),
            )
        )
    return documents, items


def resolve_anchor(content: str, anchor: str) -> tuple[int | None, int | None]:
    """把一段文本锚点解析为 (起始行, 结束行)，1-based，闭区间。

    锚点可以跨行：把锚点按行拆开，在文档里逐行匹配第一行，
    然后校验后续行是否连续匹配。
    """
    lines = content.splitlines()
    anchor_lines = anchor.splitlines() or [anchor]
    first = anchor_lines[0].strip()
    if not first:
        return None, None

    for i, line in enumerate(lines):
        if first not in line:
            continue
        # 校验锚点的后续行是否紧随其后
        ok = True
        for j, extra in enumerate(anchor_lines[1:], start=1):
            if i + j >= len(lines) or extra.strip() not in lines[i + j]:
                ok = False
                break
        if ok:
            return i + 1, i + len(anchor_lines)
    return None, None


def main() -> int:
    parser = argparse.ArgumentParser(description="AgentLoom 离线检索评测")
    here = PROJECT_ROOT
    parser.add_argument(
        "--dataset",
        default=os.path.join(EVAL_DIR, "dataset.json"),
        help="评测数据集路径",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(EVAL_DIR, "report.json"),
        help="报告输出路径",
    )
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    documents, items = load_dataset(args.dataset)
    t0 = time.perf_counter()
    result = evaluate(documents, items, top_k=args.top_k)
    elapsed = time.perf_counter() - t0

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": round(elapsed, 4),
        **result.to_dict(),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"\n报告已写入 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
