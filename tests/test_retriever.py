"""retriever 单测:四策略(dense/bm25/hybrid/hybrid_rerank)、元数据过滤、
回表分序、脏主键跳过、异常上抛(SQLite + FakeMilvus + FakeEmbedder + FakeReranker)。

降级策略在 query_faq 层,不在这层;LLM 问法改写默认关闭(rewrite=False),
只测同义词扩展与检索本身。
"""

import pytest

from app.models.knowledge_chunk import KnowledgeChunk
from app.rag.embeddings import FakeEmbedder
from app.rag.query_rewrite import expand_synonyms, rewrite_query
from app.rag.reranker import FakeReranker
from app.rag.retriever import Retriever, retrieve
from app.rag.vectorstore import FakeMilvus, MilvusStore, VectorStoreError
from app.repositories import knowledge_repo


async def _seed(session_factory, rows: list[tuple[str, str, str]]) -> list[KnowledgeChunk]:
    """灌知识块:(category, questions, answer),返回带 id 的实例。"""
    async with session_factory() as session:
        chunks = [
            KnowledgeChunk(category=cat, questions=q, answer=a) for cat, q, a in rows
        ]
        await knowledge_repo.add_chunks(session, chunks)
        return chunks


def _retriever(chunks: list[KnowledgeChunk], *, reranker=None, client=None) -> Retriever:
    """FakeMilvus + FakeEmbedder 检索器;chunks 一并写进向量库(text 供 BM25)。"""
    client = client or FakeMilvus()
    store = MilvusStore(client=client)
    embedder = FakeEmbedder()
    if chunks:
        store.upsert(
            [
                {
                    "chunk_id": c.id,
                    "embedding": embedder.embed_texts([c.embedding_text()])[0],
                    "text": c.embedding_text(),
                    "category": c.category,
                }
                for c in chunks
            ]
        )
    return Retriever(store=store, embedder=embedder, reranker=reranker)


async def _ask(session_factory, query, retriever, **kwargs):
    return await retrieve(
        session_factory=session_factory,
        query=query,
        retriever=retriever,
        rewrite=False,
        **kwargs,
    )


# ---------------------------------------------------------------- 四策略


async def test_dense_strategy_ranks_same_text_first(session_factory):
    chunks = await _seed(
        session_factory,
        [("售后", "运费多少", "满 99 包邮,否则 8 元"), ("售后", "退货几天", "签收 7 天内可退")],
    )
    hits = await _ask(
        session_factory, chunks[1].embedding_text(), _retriever(chunks), strategy="dense"
    )
    # FakeEmbedder 无语义:只有同文本才同向量,故 Top-1 必是目标块
    assert hits[0].chunk.id == chunks[1].id
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))


async def test_bm25_strategy_hits_keyword_even_when_vector_gap(session_factory):
    """关键词路的意义:问句里的实体词(型号/专有名词)向量路容易漏,BM25 能捞回来。"""
    chunks = await _seed(
        session_factory,
        [
            ("商品", "轻弹缓震跑步鞋适合什么脚型", "适合正常足弓,宽楦可选"),
            ("售后", "运费多少", "满 99 包邮,否则 8 元"),
        ],
    )
    hits = await _ask(session_factory, "缓震跑步鞋", _retriever(chunks), strategy="bm25")
    assert hits and hits[0].chunk.id == chunks[0].id


async def test_hybrid_strategy_keeps_both_routes(session_factory):
    chunks = await _seed(
        session_factory,
        [("售后", "运费多少", "满 99 包邮,否则 8 元"), ("售后", "退货几天", "签收 7 天内可退")],
    )
    # 向量路喂第 0 块的文本,关键词路喂只出现在第 1 块的词 → 两路各中一块
    hits = await _ask(
        session_factory, chunks[1].embedding_text(), _retriever(chunks), strategy="hybrid"
    )
    assert len(hits) == 2


async def test_hybrid_rerank_promotes_keyword_overlap(session_factory):
    """重排覆盖粗排:FakeReranker 按 2-gram 覆盖率打分,含关键词的块被顶到第一。"""
    chunks = await _seed(
        session_factory,
        [("售后", "运费多少", "满 99 包邮,否则 8 元"), ("售后", "退货几天", "签收 7 天内可退")],
    )
    reranker = FakeReranker()
    hits = await _ask(
        session_factory,
        "退货",
        _retriever(chunks, reranker=reranker),
        strategy="hybrid_rerank",
    )
    assert reranker.calls == 1
    assert hits[0].chunk.id == chunks[1].id
    assert hits[0].score >= hits[-1].score


async def test_unknown_strategy_raises(session_factory):
    retriever = _retriever([])
    with pytest.raises(ValueError, match="未知检索策略"):
        await _ask(session_factory, "x", retriever, strategy="semantic_magic")


# ---------------------------------------------------------------- 元数据过滤


async def test_category_filter_excludes_other_categories(session_factory):
    chunks = await _seed(
        session_factory,
        [("售后", "运费多少", "满 99 包邮"), ("开票", "能开发票吗", "支持电子发票")],
    )
    hits = await _ask(
        session_factory, "发票", _retriever(chunks), strategy="bm25", category="开票"
    )
    assert [h.chunk.id for h in hits] == [chunks[1].id]


async def test_category_with_expression_metacharacters_rejected(session_factory):
    """类目名要拼进 Milvus 表达式,带引号/运算符的一律拒绝,不做转义放行。"""
    retriever = _retriever([])
    with pytest.raises(ValueError, match="非法类目名"):
        await _ask(session_factory, "x", retriever, strategy="bm25", category='a" or id > 0 or "')


# ---------------------------------------------------------------- 边界与异常


async def test_empty_store_returns_empty(session_factory):
    assert await _ask(session_factory, "任意问题", _retriever([])) == []


async def test_top_k_limits_results(session_factory):
    chunks = await _seed(
        session_factory, [("售后", f"问题{i}", f"答案{i}") for i in range(3)]
    )
    hits = await _ask(session_factory, chunks[0].embedding_text(), _retriever(chunks), top_k=2)
    assert len(hits) == 2


async def test_skips_ids_missing_in_mysql(session_factory):
    """Milvus 里有、MySQL 权威源没有的主键跳过,不撑翻整条检索。"""
    chunks = await _seed(session_factory, [("售后", "真实问题", "真实答案")])
    retriever = _retriever(chunks)
    retriever.store.upsert(
        [{"chunk_id": 9999, "embedding": FakeEmbedder().embed_texts(["幽灵块"])[0], "text": "幽灵块"}]
    )
    hits = await _ask(session_factory, "幽灵块", retriever, strategy="dense")
    assert all(h.chunk.id != 9999 for h in hits)


async def test_embedder_error_propagates(session_factory):
    class BoomEmbedder:
        def embed_texts(self, texts):
            raise RuntimeError("model crashed")

    retriever = Retriever(store=MilvusStore(client=FakeMilvus()), embedder=BoomEmbedder())
    with pytest.raises(RuntimeError, match="model crashed"):
        await _ask(session_factory, "x", retriever, strategy="dense")


async def test_store_error_propagates(session_factory):
    class BoomClient(FakeMilvus):
        def search(self, *args, **kwargs):
            raise VectorStoreError("milvus down")

    retriever = Retriever(store=MilvusStore(client=BoomClient()), embedder=FakeEmbedder())
    with pytest.raises(VectorStoreError, match="milvus down"):
        await _ask(session_factory, "x", retriever, strategy="dense")


# ---------------------------------------------------------------- 理解层(检索侧)


def test_synonyms_expand_on_query_side():
    """「邮费」在知识库里一个字都没有,靠扩展才能被 BM25 捞到。"""
    kw = expand_synonyms("邮费怎么算")
    assert "运费" in kw
    assert expand_synonyms("完全无关的问题") == []


async def test_rewrite_query_without_model_keeps_raw_text():
    rq = await rewrite_query("邮费怎么算", model=None)
    assert rq.canonical == "邮费怎么算"
    assert "运费" in rq.keywords
    # BM25 串 = 原句 + 扩展词,且不重复
    assert "运费" in rq.bm25_query and rq.bm25_query.count("邮费怎么算") == 1


async def test_rewrite_query_falls_back_when_model_fails():
    class BoomModel:
        async def ainvoke(self, *args, **kwargs):
            raise RuntimeError("upstream 500")

    rq = await rewrite_query("退货要几天", model=BoomModel())
    assert rq.canonical == "退货要几天"  # 理解层挂了,T+检索不能跟着挂


def test_online_strategy_comes_from_config(monkeypatch):
    """线上策略只有一个开关:RAG_STRATEGY。代码里不再留第二份默认值。

    四策略在 357 块库上的对照见 reports/ch04_eval_v2.md(hybrid R@5 0.84 >
    dense 0.81,hybrid_rerank 把 hybrid 的 MRR 从 0.554 拉回 0.664)。
    这条改配置就能翻转,证明策略真的由配置决定而不是写死在代码里。
    """
    from app.config import get_settings
    from app.rag.retriever import online_strategy

    monkeypatch.setenv("RAG_STRATEGY", "hybrid")
    get_settings.cache_clear()
    try:
        assert online_strategy() == "hybrid"
    finally:
        monkeypatch.delenv("RAG_STRATEGY", raising=False)
        get_settings.cache_clear()
