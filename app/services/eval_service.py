"""评估流水线：跑一轮检索评估并落 `eval_runs`（ch09 需求 6 的本地版）。

为什么要有这张表：ch04 那次评估是「跑一次、看一眼报告」，看完就散——没人知道
上周改完检索之后指标是升了还是降了。要回答这个问题，每轮评估必须落一行能比的数据。

## 只跑检索，生成段默认关

检索段（R@5 / R@10 / MRR）一轮几十次嵌入 + 重排，几分钟；生成段要多花 31 次
LLM 调用判忠实度。定时任务里跑生成段等于每天烧一遍钱，所以默认不跑，
`metrics.with_generation` 记下这轮跑没跑，趋势图上两段分开画，不混在一条线里。
"""

from __future__ import annotations

import logging

from app.eval.metrics import CaseResult, load_cases, rank_of, summarize
from app.repositories import eval_run_repo

logger = logging.getLogger(__name__)

# 默认跟随在线配置:定时评估量的是用户实际拿到的检索,不是另一条链路
def _default_strategy() -> str:
    from app.config import get_settings

    return get_settings().rag_strategy


async def run_round(
    session_factory,
    *,
    cases_path,
    retriever=None,
    strategy: str | None = None,
    triggered_by: str = "手动",
    top_k: int = 10,
    limit: int | None = None,
    with_generation: bool = False,
) -> dict:
    """跑一轮：逐题检索 → 记排名 → 汇总指标 → 落表。

    `retriever` 可注入（单测用假检索器），不传就用真实现——真实现要加载嵌入
    与重排模型，所以延迟到真正要跑的时候才建。
    """
    from app.rag.retriever import retrieve

    strategy = strategy or _default_strategy()
    cases = [c for c in load_cases(cases_path) if c.get("answerable")]
    if limit:
        cases = cases[:limit]

    if retriever is None:
        from app.rag.embeddings import LocalBGEEmbedder
        from app.rag.reranker import LocalBGEReranker
        from app.rag.retriever import Retriever
        from app.rag.vectorstore import MilvusStore

        retriever = Retriever(
            store=MilvusStore(), embedder=LocalBGEEmbedder(), reranker=LocalBGEReranker()
        )

    results: list[CaseResult] = []
    for case in cases:
        hits = await retrieve(
            session_factory=session_factory,
            query=case["query"],
            top_k=top_k,
            strategy=strategy,
            retriever=retriever,
        )
        results.append(
            CaseResult(
                case["id"],
                case.get("bucket") or "",
                rank_of(hits, case.get("expected_section") or ""),
                weight=int(case.get("weight") or 1),
                noise=bool(case.get("noise")),
            )
        )

    metrics = summarize(results, strategy)
    metrics["with_generation"] = bool(with_generation)
    metrics["top_k"] = top_k

    async with session_factory() as session:
        row = await eval_run_repo.add_eval_run(
            session,
            triggered_by=triggered_by,
            dataset_size=len(results),
            metrics=metrics,
        )
    logger.info(
        "评估一轮:%s | R@5=%.3f R@10=%.3f MRR=%.3f | %s 条 | 触发=%s",
        strategy,
        metrics["r5"],
        metrics["r10"],
        metrics["mrr"],
        metrics["cases"],
        triggered_by,
    )
    return {"id": row.id, "metrics": metrics, "dataset_size": len(results)}


def rows_to_trend(rows) -> list[dict]:
    """eval_runs 行 → 曲线数据（旧→新），只挑能画的字段。

    单独抽出来是因为接口那边手里已经有一个 session，不必为了查趋势再开一个；
    开新 session 换不来什么，反而在测试里要多绕一圈依赖覆盖。
    """
    out = []
    for row in rows:
        m = row.metrics or {}
        out.append(
            {
                "id": row.id,
                "created_at": row.created_at.isoformat(sep=" ", timespec="minutes")
                if row.created_at
                else None,
                "triggered_by": row.triggered_by,
                "dataset_size": row.dataset_size,
                "strategy": m.get("strategy"),
                "r5": m.get("r5"),
                "r10": m.get("r10"),
                "mrr": m.get("mrr"),
                "with_generation": bool(m.get("with_generation")),
            }
        )
    return out


async def trend(session_factory, limit: int = 10) -> list[dict]:
    """最近 N 轮的曲线数据（旧→新）。"""
    async with session_factory() as session:
        rows = await eval_run_repo.list_eval_runs(session, limit=limit)
    return rows_to_trend(rows)
