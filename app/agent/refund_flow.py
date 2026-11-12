"""退款/售后子流程:先拿订单、再查政策,最后只问 Agent 一句话(第六章第 5 点)。

顺序是死的,不由模型决定:

    订单号 → query_order 拿这一单的事实 → 扩写多路检索政策 → 置信度闸 → Agent 判断「能不能退」

为什么这一步要从「让 Agent 自己决定查什么」里拿出来:退款判断错一次的代价是赔钱和
纠纷,而它需要的信息是固定的两样(这单是什么、政策怎么规定)。确定的事交给确定
的编排,Agent 只负责最后那句需要理解力的判断。

「缺订单号就弹选择器」也放在这里:模型不许猜订单号,猜错等于把别人的单子退了。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.agent.gate import GateResult, check_gate
from app.agent.query_understanding import expand_queries
from app.core.tools.executor import ToolExecutor
from app.core.tools.registry import ToolRegistry
from app.rag.citations import Citation, build_citations
from app.rag.retriever import online_strategy, retrieve

logger = logging.getLogger(__name__)

# 退款子流程里每一步检索取多少条(扩写后多路合并再截断)
POLICY_TOP_K = 6
MERGED_TOP_K = 10

# 退款原因固定类目:不追问用户「为什么退」,提交退款单时下拉选(第六第 6 点)
REFUND_REASONS = (
    "质量问题",
    "发错或漏发",
    "商品与描述不符",
    "不喜欢或不想要",
    "尺码规格不符",
    "物流破损",
    "价格问题",
    "其他",
)

NEED_ORDER_REPLY = (
    "要判断这一单能不能退、怎么退,我需要先确认是哪笔订单。"
    "下面是我这边能查到的近期订单,您点一笔我来核对。"
)


@dataclass
class RefundPrep:
    """退款子流程的产物:要么缺订单号,要么把两样东西都备齐。"""

    need_order: bool = False
    order_facts: str = ""
    citations: list[Citation] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    gate: GateResult | None = None

    @property
    def evidence(self) -> str:
        """喂给 Agent 的证据:订单事实在前,政策条款在后。"""
        from app.rag.citations import format_evidence

        parts: list[str] = []
        if self.order_facts:
            parts.append(f"【订单事实】\n{self.order_facts}")
        if self.citations:
            parts.append(f"【政策条款】\n{format_evidence(self.citations)}")
        return "\n\n".join(parts)


async def _order_facts(order_id: str, *, registry: ToolRegistry, executor: ToolExecutor) -> str:
    """走执行引擎查订单:超时、参数校验、错误分诊都复用同一套,不另开一条路。"""
    result = await executor.execute(
        {
            "name": "query_order",
            "args": {"order_id": order_id},
            "id": f"refund-order-{order_id}",
            "type": "tool_call",
        }
    )
    return result.content if result.ok else f"订单 {order_id} 查询失败:{result.content}"


async def multi_query_retrieve(
    queries: list[str],
    *,
    session_factory,
    retriever=None,
    top_k: int = POLICY_TOP_K,
    merged_top_k: int = MERGED_TOP_K,
    question: str | None = None,
) -> list:
    """多路检索后按 chunk 去重、取各块最高分,再按分数截断。

    去重按 chunk id 而不是按文本:同一块被两条查询召回时,分数取高的那条——
    分数代表「最贴合的那一问有多贴合」,平均会把强命中摊薄。

    精排只做一次:每路先只跑便宜的召回(向量 / 混合粗排),合并成一个池之后拿
    **用户原话**排一遍。三路各排各的会把候选打分三遍,CPU 上就是三倍延迟,而
    候选池并没有变大——这是精排最容易被忽略的一笔浪费。
    """
    strategy = online_strategy()
    # 精排档:各路只做混合粗排,把精排留给合并后的池(见下)
    coarse = "hybrid" if strategy == "hybrid_rerank" else strategy

    best: dict[int, object] = {}
    for query in queries:
        try:
            hits = await retrieve(
                session_factory=session_factory,
                query=query,
                top_k=top_k,
                strategy=coarse,
                retriever=retriever,
                rewrite=False,  # 改写/扩写已在理解层做过,这里不再花一次模型调用
            )
        except Exception as exc:  # noqa: BLE001 - 某一路挂了不影响其它路
            logger.warning("扩写查询检索失败(%s):%s", query, exc)
            continue
        for hit in hits:
            cid = int(hit.chunk.id)
            if cid not in best or hit.score > best[cid].score:
                best[cid] = hit

    merged = sorted(best.values(), key=lambda h: h.score, reverse=True)
    if strategy != "hybrid_rerank":
        return merged[:merged_top_k]

    # 合并池再精排一次,用用户原话问;池子按精排的常规深度截断
    from app.rag.retriever import RERANK_POOL, get_default_retriever, rerank_hits

    rerank_query = question or (queries[0] if queries else "")
    pool = merged[: max(merged_top_k, RERANK_POOL)]
    return rerank_hits(
        rerank_query, pool, retriever=retriever or get_default_retriever(), top_k=merged_top_k
    )


async def prepare_refund(
    question: str,
    *,
    order_id: str | None,
    registry: ToolRegistry,
    executor: ToolExecutor,
    session_factory,
    retriever=None,
    expand_model=None,
    check_model=None,
    min_score: float = 0.45,
) -> RefundPrep:
    """备齐「这一单能不能退」所需的全部材料。缺订单号就到此为止。"""
    if not order_id:
        return RefundPrep(need_order=True)

    facts = await _order_facts(order_id, registry=registry, executor=executor)
    queries = await expand_queries(question, model=expand_model)
    hits = await multi_query_retrieve(
        queries, session_factory=session_factory, retriever=retriever, question=question
    )
    citations = build_citations(hits)
    gate = await check_gate(question, citations, min_score=min_score, model=check_model)
    return RefundPrep(
        order_facts=facts,
        citations=citations,
        queries=queries,
        gate=gate,
    )
