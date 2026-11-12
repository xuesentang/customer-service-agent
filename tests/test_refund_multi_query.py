"""退款子流程的多路检索:合并成一个池只精排一次。

背景:三路扩写查询若各排各的,同一批候选要被 cross-encoder 打分三遍。
CPU 上精排是最贵的一步,三倍延迟换不来任何召回增量——候选池并没有变大。
"""

from __future__ import annotations

import pytest

from app.agent import refund_flow
from app.rag.retriever import Hit


class _Chunk:
    def __init__(self, cid: int, text: str):
        self.id = cid
        self.text = text

    def __str__(self) -> str:
        return self.text


class _Reranker:
    def __init__(self):
        self.calls = 0
        self.scored = 0

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        self.calls += 1
        self.scored += len(docs)
        return [0.5 + i / 100 for i in range(len(docs))]


class _Retriever:
    def __init__(self):
        self.reranker = _Reranker()


@pytest.fixture
def rerank_mode(monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("RAG_STRATEGY", "hybrid_rerank")
    get_settings.cache_clear()
    yield
    monkeypatch.delenv("RAG_STRATEGY", raising=False)
    get_settings.cache_clear()


async def test_coarse_per_query_then_single_rerank(monkeypatch, rerank_mode):
    seen: list[tuple[str, str]] = []

    async def fake_retrieve(**kwargs):
        seen.append((kwargs["query"], kwargs["strategy"]))
        base = {"a": 0.9, "b": 0.8, "c": 0.7}[kwargs["query"]]
        return [
            Hit(chunk=_Chunk(1, "退货政策 第七条"), score=base),
            Hit(chunk=_Chunk(2, "运费承担规则"), score=base - 0.1),
        ]

    monkeypatch.setattr(refund_flow, "retrieve", fake_retrieve)
    retriever = _Retriever()

    hits = await refund_flow.multi_query_retrieve(
        ["a", "b", "c"], session_factory=None, retriever=retriever, question="订单 1003 能退吗"
    )

    assert [s for _, s in seen] == ["hybrid"] * 3, "每路应当只做混合粗排,不各自精排"
    assert retriever.reranker.calls == 1, "合并后的池只能精排一次"
    assert len(hits) == 2 and len({h.chunk.id for h in hits}) == 2, "同一 chunk 多路命中要按 id 去重"


async def test_dense_mode_skips_rerank(monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("RAG_STRATEGY", "dense")
    get_settings.cache_clear()

    async def fake_retrieve(**kwargs):
        assert kwargs["strategy"] == "dense"
        return [Hit(chunk=_Chunk(1, "退货政策"), score=0.7)]

    monkeypatch.setattr(refund_flow, "retrieve", fake_retrieve)
    retriever = _Retriever()

    hits = await refund_flow.multi_query_retrieve(
        ["退货怎么退"], session_factory=None, retriever=retriever, question="退货怎么退"
    )

    assert hits[0].score == 0.7
    assert retriever.reranker.calls == 0, "dense 档不该加载/调用重排"

    monkeypatch.delenv("RAG_STRATEGY", raising=False)
    get_settings.cache_clear()
