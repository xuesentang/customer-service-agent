"""退款子流程的多路检索:粗排按路走,精排按「这块由哪一路召回」打分,多路取最高。

背景:退款场景的用户原话里带订单号和具体商品(「订单1003的大容量双肩通勤包能退吗」)。
若合并后统一拿原话精排,通用条款(七日无理由退货)会被判成弱相关,分数压到 0.1 量级,
置信度闸据此拒答——可库里明明有条款。改按来源查询打分后,召回它的那一问才是相关的那一问。

代价控制:每路只排自己召回的 6 条,总打分对数 = 各路条数之和(3×6=18),
不比「合并池按原话排一遍(20 条)」更贵。
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
    """按查询给不同基础分,方便断言「多路取最高」。"""

    BASE = {"a": 0.2, "b": 0.4, "c": 0.6}

    def __init__(self):
        self.calls = 0
        self.scored = 0
        self.batches: list[list[int]] = []

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        self.calls += 1
        self.scored += len(docs)
        self.batches.append(list(range(len(docs))))
        base = self.BASE.get(query, 0.0)
        return [base + i / 100 for i in range(len(docs))]


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


async def test_rerank_by_source_query_and_take_max(monkeypatch, rerank_mode):
    seen: list[tuple[str, str]] = []
    # 三路各自召回两条,块 2 / 3 被两路共享
    by_query = {
        "a": [(1, 0.9), (2, 0.8)],
        "b": [(2, 0.75), (3, 0.7)],
        "c": [(3, 0.65), (4, 0.6)],
    }

    async def fake_retrieve(**kwargs):
        seen.append((kwargs["query"], kwargs["strategy"]))
        return [Hit(chunk=_Chunk(cid, f"文本{cid}"), score=s) for cid, s in by_query[kwargs["query"]]]

    monkeypatch.setattr(refund_flow, "retrieve", fake_retrieve)
    retriever = _Retriever()

    hits = await refund_flow.multi_query_retrieve(
        ["a", "b", "c"], session_factory=None, retriever=retriever, question="订单 1003 能退吗"
    )

    assert [s for _, s in seen] == ["hybrid"] * 3, "每路应当只做混合粗排,不各自精排"
    assert retriever.reranker.calls == 3, "按来源查询分三批打分"
    assert retriever.reranker.scored == 6, "总打分对数 = 各路条数之和,不比合并池多"

    by_id = {h.chunk.id: h.score for h in hits}
    assert len(by_id) == 4, "同一 chunk 多路命中要按 id 去重"
    # 块 1 只被 a 召回 → 0.2;块 2 被 a(0.21) / b(0.40) 召回 → 取高 0.40;
    # 块 3 被 b(0.41) / c(0.60) 召回 → 0.60;块 4 只被 c 召回 → 0.61。
    assert by_id[1] == pytest.approx(0.20)
    assert by_id[2] == pytest.approx(0.40)
    assert by_id[3] == pytest.approx(0.60)


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
