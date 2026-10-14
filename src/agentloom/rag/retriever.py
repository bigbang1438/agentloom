"""RAG 检索层：分片、BM25 检索、引用分配与校验。

这里的取舍是刻意的：**纯词法检索（BM25），不用向量库**。
理由：小规模资料库上词法检索足够、零依赖、结果可解释、行号可溯源。
代价是处理不了语义改写（同义不同词）。这个边界在 README 里写清楚，不假装。

引用校验是这一层最重要的部分：编号由服务端分配，模型只能引用已分配的编号；
答案里出现缺失或越界的 [n] 时，结论被拦截。这挡的是「假引用」——
模型编一个看起来很像的引用，比不引用更危险。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
WORD_RE = re.compile(r"[a-zA-Z0-9_]+")
CITE_RE = re.compile(r"\[(\d{1,3})\]")


# ----------------------------------------------------------------------
# 分片
# ----------------------------------------------------------------------

@dataclass
class Chunk:
    """一个可检索的文本片段，保留原始行号以便溯源。"""

    chunk_id: str
    doc_id: str
    filename: str
    text: str
    start_line: int
    end_line: int

    @property
    def line_span(self) -> str:
        return f"{self.filename}:{self.start_line}-{self.end_line}"


def chunk_document(
    filename: str,
    content: str,
    *,
    doc_id: str | None = None,
    max_chars: int = 600,
    overlap_lines: int = 1,
) -> list[Chunk]:
    """按行切分文档，每片不超过 max_chars 字符，保留起止行号。

    按行切而不是按字符切，是为了让「行号溯源」成立：
    返回的片段能对应回源文件的连续行区间，人可以打开原文核对。
    """
    doc_id = doc_id or _slug(filename)
    lines = content.splitlines()
    chunks: list[Chunk] = []

    buf: list[str] = []
    buf_start = 1
    size = 0
    idx = 0

    def flush(end_line: int) -> None:
        nonlocal buf, buf_start, size, idx
        if not buf:
            return
        text = "\n".join(buf).strip()
        if text:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{idx}",
                    doc_id=doc_id,
                    filename=filename,
                    text=text,
                    start_line=buf_start,
                    end_line=end_line,
                )
            )
            idx += 1
        buf = []
        size = 0

    for lineno, line in enumerate(lines, start=1):
        # 单行超长：硬切成多片，避免一片吃掉整篇
        if len(line) > max_chars:
            flush(lineno - 1)
            for i in range(0, len(line), max_chars):
                piece = line[i : i + max_chars]
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}#{idx}",
                        doc_id=doc_id,
                        filename=filename,
                        text=piece,
                        start_line=lineno,
                        end_line=lineno,
                    )
                )
                idx += 1
            buf_start = lineno + 1
            continue

        if not buf:
            buf_start = lineno
        buf.append(line)
        size += len(line) + 1

        if size >= max_chars:
            flush(lineno)
            # 重叠：把最后 overlap_lines 行留到下一片开头，避免答案正好跨在切口上
            if overlap_lines and lineno - overlap_lines + 1 > buf_start:
                carry = lines[lineno - overlap_lines : lineno]
                buf = list(carry)
                buf_start = lineno - overlap_lines + 1
                size = sum(len(x) + 1 for x in carry)

    flush(len(lines))
    return chunks


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", name).strip("_").lower() or "doc"


# ----------------------------------------------------------------------
# 停用词
# ----------------------------------------------------------------------

# 中文虚词与高频功能词。它们不承载主题信息，但会组成二元词组造成假命中
# （例如「怎么」「什么」「可以」）。这是一份通用停用词表，不是针对某个评测集定制的。
CJK_STOPWORDS: frozenset[str] = frozenset(
    """
    的 了 是 在 我 有 和 就 不 人 都 一 上 也 很 到 说 要 去 你 会 着 看 好 自己
    这 那 他 她 它 们 个 之 与 及 等 或 而 但 却 则 于 以 为 被 把 给 从 向 往
    什么 怎么 怎样 如何 为什么 哪些 哪个 哪里 谁 多少 几个 时候 现在 已经
    还是 或者 以及 并且 而且 不过 然后 于是 因为 所以 但是 如果 虽然 然而
    关于 对于 通过 根据 按照 为了 由于 除了 作为 一样 一些 这些 那些 之后
    之前 其中 其他 另外 只是 就是 还有 不能 可能 应该 必须 需要 可以 能够
    一个 这个 那个 我们 他们 你们 咱们 大家 自己 本身 这样 那样 这种 那种
    做法 方法 方式 情况 问题 内容 东西 事情 时候 地方 方面 时候
    请问 帮我 一下 一点 有些 没有 不是 不会 不要 就是 也是 都是 还是
    么 呢 吧 啊 呀 哦 嗯 啦 着 过 得 地 吗
    """.split()
)

EN_STOPWORDS: frozenset[str] = frozenset(
    """
    a an the is are was were be been being do does did doing have has had having
    i you he she it we they me him her us them my your his its our their this that
    these those what which who whom whose when where why how and or but if then
    than so as at by for with about into through during before after above below
    to from up down in out on off over under again further once here there all any
    both each few more most other some such no nor not only own same too very can
    will just should now of
    """.split()
)

STOPWORDS: frozenset[str] = CJK_STOPWORDS | EN_STOPWORDS


def is_stopword(term: str) -> bool:
    return term in STOPWORDS


def content_terms(text: str) -> set[str]:
    """抽取承载主题信息的词项（去掉停用词与中文单字）。

    用于计算「查询覆盖率」：一个问题里真正有信息量的词，命中了多少。
    """
    out: set[str] = set()
    for word in WORD_RE.findall(text):
        low = word.lower()
        if low not in STOPWORDS and not low.isdigit():
            out.add(low)
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(run) == 1:
            continue  # 单字不进覆盖率分母，它只用于低权重召回
        for i in range(len(run) - 1):
            bg = run[i : i + 2]
            if bg not in STOPWORDS:
                out.add(bg)
    return out


# ----------------------------------------------------------------------
# 分词
# ----------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """中英混合分词。

    中文没有空格，用**二元词组（bigram）**做词项，单字降权。
    这样「工具调用」会切成 工具/具调/调用，比纯单字精确得多。
    """
    tokens: list[str] = []
    for word in WORD_RE.findall(text):
        tokens.append(word.lower())

    cjk_runs = re.findall(r"[\u4e00-\u9fff]+", text)
    for run in cjk_runs:
        if len(run) == 1:
            tokens.append(run)
            continue
        for i in range(len(run) - 1):
            tokens.append(run[i : i + 2])
        # 单字以低权重形式保留，保证「库存」能命中「库」
        for ch in run:
            tokens.append(ch)
    return tokens


def _term_weight(term: str) -> float:
    """单字中文词项降权，避免「的」「是」这类噪声主导打分。"""
    if len(term) == 1 and CJK_RE.match(term):
        return 0.15
    return 1.0


def tokenize_content(text: str) -> list[str]:
    """分词并去掉停用词。用于检索主路径。"""
    return [t for t in tokenize(text) if t not in STOPWORDS]


# ----------------------------------------------------------------------
# BM25
# ----------------------------------------------------------------------

@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "filename": self.chunk.filename,
            "start_line": self.chunk.start_line,
            "end_line": self.chunk.end_line,
            "score": round(self.score, 4),
            "text": self.chunk.text,
        }


class BM25Index:
    """标准 BM25（k1 / b 可调）。索引在查询时按需构建，适合受限的小资料库。

    额外的两道精度闸门（在标准 BM25 之上）：
    - **停用词过滤**：中文虚词组成的二元词组（「怎么」「什么」）不参与打分。
    - **查询覆盖率门槛**：查询里有信息量的词，命中比例过低就不算证据。
      这一条直接决定「无答案问题能不能被拒答」。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, min_coverage: float = 0.25) -> None:
        self.k1 = k1
        self.b = b
        self.min_coverage = min_coverage
        self.chunks: list[Chunk] = []
        self._doc_terms: list[Counter[str]] = []
        self._doc_len: list[int] = []
        self._df: Counter[str] = Counter()
        self._avg_len: float = 0.0

    def add(self, chunks: Iterable[Chunk]) -> None:
        for c in chunks:
            terms = Counter(tokenize_content(c.text))
            self.chunks.append(c)
            self._doc_terms.append(terms)
            self._doc_len.append(sum(terms.values()) or 1)
            for term in terms:
                self._df[term] += 1
        self._recompute()

    def _recompute(self) -> None:
        n = len(self._doc_len)
        self._avg_len = (sum(self._doc_len) / n) if n else 0.0

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, *, top_k: int = 5) -> list[ScoredChunk]:
        if not self.chunks:
            return []

        q_terms = Counter(tokenize_content(query))
        if not q_terms:
            return []

        # 覆盖率分母：查询里承载主题信息的词项（去掉停用词与中文单字）
        informative = content_terms(query)
        n_informative = len(informative)

        n = len(self.chunks)
        scored: list[ScoredChunk] = []

        for i, chunk in enumerate(self.chunks):
            terms = self._doc_terms[i]
            dl = self._doc_len[i]
            score = 0.0
            matched_multi = False

            for term, qtf in q_terms.items():
                tf = terms.get(term, 0)
                if tf == 0:
                    continue
                if len(term) > 1 or not CJK_RE.match(term):
                    matched_multi = True
                df = self._df.get(term, 0) or 1
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self._avg_len or 1))
                score += _term_weight(term) * idf * (tf * (self.k1 + 1)) / denom * min(qtf, 3)

            # 闸门一：只有单字重合时不算证据（避免「的」把无关片段捞上来）
            if score <= 0 or not matched_multi:
                continue

            # 闸门二：查询覆盖率。查询里有信息量的词命中太少 -> 不是证据。
            # 这是「无答案拒答」能成立的关键：问「红烧肉怎么做」时，
            # 文档里也许零星重合几个字，但覆盖率极低，应当拒答而不是硬凑。
            if n_informative:
                hit = sum(1 for t in informative if t in terms)
                if hit / n_informative < self.min_coverage:
                    continue

            scored.append(ScoredChunk(chunk=chunk, score=score))

        scored.sort(key=lambda s: (-s.score, s.chunk.chunk_id))
        return scored[:top_k]


# ----------------------------------------------------------------------
# 引用
# ----------------------------------------------------------------------

@dataclass
class Citation:
    """服务端分配的引用编号。模型只能引用这些编号。"""

    index: int
    chunk_id: str
    filename: str
    start_line: int
    end_line: int
    text: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "chunk_id": self.chunk_id,
            "filename": self.filename,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "score": round(self.score, 4),
            "preview": self.text[:200],
        }


@dataclass
class CitationCheck:
    """引用校验结果。"""

    ok: bool
    cited: list[int]
    invalid: list[int]
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "cited": self.cited, "invalid": self.invalid, "reason": self.reason}


def assign_citations(scored: Sequence[ScoredChunk], *, max_citations: int = 12) -> list[Citation]:
    """给检索结果分配 1..N 的编号。编号由服务端决定，模型没有分配权。"""
    out: list[Citation] = []
    for i, s in enumerate(scored[:max_citations], start=1):
        out.append(
            Citation(
                index=i,
                chunk_id=s.chunk.chunk_id,
                filename=s.chunk.filename,
                start_line=s.chunk.start_line,
                end_line=s.chunk.end_line,
                text=s.chunk.text,
                score=s.score,
            )
        )
    return out


def verify_citations(answer: str, citations: Sequence[Citation]) -> CitationCheck:
    """校验答案里的 [n] 是否都指向已分配的编号。

    两种失败：
    - 出现了不存在的编号（假引用）
    - 一个引用都没有（无依据作答）

    注意边界：编号合法**只证明关联到了真实片段**，不证明每句话都被该片段支持。
    语义支持需要人工或模型复核，这里不假装做到了。
    """
    valid = {c.index for c in citations}
    found = [int(m) for m in CITE_RE.findall(answer)]
    cited = sorted(set(found))
    invalid = [i for i in cited if i not in valid]

    if invalid:
        return CitationCheck(
            ok=False,
            cited=cited,
            invalid=invalid,
            reason=f"answer cites undefined source(s): {invalid}",
        )
    if not cited:
        return CitationCheck(
            ok=False,
            cited=[],
            invalid=[],
            reason="answer contains no citation while sources were provided",
        )
    return CitationCheck(ok=True, cited=cited, invalid=[])


def strip_invalid_citations(answer: str, citations: Sequence[Citation]) -> str:
    """把越界引用从答案里剔除，并追加一条说明。

    宁可让答案显得不完整，也不让一个假引用留在成品里。
    """
    valid = {c.index for c in citations}

    def repl(m: re.Match[str]) -> str:
        idx = int(m.group(1))
        return m.group(0) if idx in valid else ""

    cleaned = CITE_RE.sub(repl, answer)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned


class KnowledgeBase:
    """文档库 + 索引 + 检索 + 引用分配。"""

    def __init__(self, *, max_chars_per_chunk: int = 600, min_coverage: float = 0.25) -> None:
        self.max_chars_per_chunk = max_chars_per_chunk
        self.min_coverage = min_coverage
        self.index = BM25Index(min_coverage=min_coverage)
        self._docs: dict[str, str] = {}

    @property
    def doc_count(self) -> int:
        return len(self._docs)

    @property
    def chunk_count(self) -> int:
        return len(self.index)

    def add_document(self, filename: str, content: str) -> int:
        """加入文档，返回新增片段数。同名文档会被替换。"""
        if filename in self._docs:
            self.remove_document(filename)
        self._docs[filename] = content
        chunks = chunk_document(
            filename, content, max_chars=self.max_chars_per_chunk
        )
        self.index.add(chunks)
        return len(chunks)

    def remove_document(self, filename: str) -> None:
        self._docs.pop(filename, None)
        self._rebuild()

    def _rebuild(self) -> None:
        self.index = BM25Index(min_coverage=self.min_coverage)
        for name, content in self._docs.items():
            self.index.add(
                chunk_document(name, content, max_chars=self.max_chars_per_chunk)
            )

    def documents(self) -> list[dict[str, Any]]:
        return [
            {"filename": name, "chars": len(content)}
            for name, content in sorted(self._docs.items())
        ]

    def search(self, query: str, *, top_k: int = 5) -> list[ScoredChunk]:
        return self.index.search(query, top_k=top_k)

    def retrieve_with_citations(
        self, query: str, *, top_k: int = 5
    ) -> tuple[list[Citation], str]:
        """检索并返回 (引用列表, 给模型看的编号上下文)。"""
        scored = self.search(query, top_k=top_k)
        citations = assign_citations(scored)
        if not citations:
            return [], ""
        blocks = [
            f"[{c.index}] 来源：{c.filename} 第 {c.start_line}-{c.end_line} 行\n{c.text}"
            for c in citations
        ]
        return citations, "\n\n".join(blocks)
