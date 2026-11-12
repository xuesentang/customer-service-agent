"""数据飞轮服务（ch09 需求 3/4）。

三件事，按流水线顺序：

1. **入池**（`enqueue`）：三个入口（闸拦下 / 自评不够答 / 用户 👎）都走这一个函数，
   顺带把当轮召回片段裁成快照存下来。裁剪在服务层做，不在调用方——前端回传的东西
   不能直接进库。
2. **标准化 + 查重 + 入队**（`normalize` / `dedupe` / `run_batch`）：把池里的原话变成
   FAQ 式问题，跟待审队列里的已有缺口比对，命中就累加次数，不新建行。
3. **审核**（`approve` / `reject`）：通过才走 ch03 的双写流程写知识库。

## 快照为什么要在服务层裁

`retrieved_chunks` 是审核人判断「知识库真缺」还是「有但没检到」的唯一依据。前端回传
的 citations 里带着整段 answer（一条能到几百字），五条就是几 KB，写进池表不合适；
模型看到的也只需要够判断的片段。所以统一裁到 5 条、每条 120 字，其余字段丢掉。

## 为什么规则查重要在模型查重之前

池子里同一句话重复入池是常态（CH10 实测：39 行只有 13 个不同问题）。这些重复用
「规范化后相等/互相包含」就能认出来，不该每一条都花一次模型调用。模型查重留给
「说法不同但意思一样」——那是真正需要语义判断的部分。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.models.low_confidence_question import LowConfidenceQuestion
from app.models.review_queue import ReviewQueue
from app.repositories import low_confidence_repo, review_repo

logger = logging.getLogger(__name__)

SNAPSHOT_LIMIT = 5
SNIPPET_CHARS = 120

# 查重口径(D5):待审队列取前 20 条比;规则匹配容忍的长度差上限
DEDUPE_CANDIDATES = 20
RULE_LEN_TOLERANCE = 0.40
NORMALIZED_MAX = 512  # review_queue.normalized_question 的列宽

# 入池入口（与 low_confidence_repo.SOURCES 同源，这里只给出语义名）
SOURCE_RETRIEVAL = "retrieval_low_conf"
SOURCE_SELF_CHECK = "self_check"
SOURCE_USER_FEEDBACK = "user_feedback"

# 归并方式（进日志与统计，不进库）
VIA_RULE = "rule"
VIA_MODEL = "model"
VIA_NEW = "new"


def _snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def snapshot_from_citations(citations, limit: int = SNAPSHOT_LIMIT) -> list[dict]:
    """Citation 对象 → 快照。入参可以是 dataclass，也可以是前端回传的 dict。"""
    out: list[dict] = []
    for item in list(citations or [])[:limit]:
        if isinstance(item, dict):
            out.append(
                {
                    "n": int(item.get("n") or 0),
                    "chunk_id": int(item.get("chunk_id") or 0),
                    "section_path": str(item.get("section_path") or ""),
                    "score": round(float(item.get("score") or 0.0), 4),
                    "snippet": _snippet(item.get("snippet") or item.get("answer") or ""),
                }
            )
        else:
            out.append(
                {
                    "n": int(getattr(item, "n", 0)),
                    "chunk_id": int(getattr(item, "chunk_id", 0)),
                    "section_path": str(getattr(item, "section_path", "") or ""),
                    "score": round(float(getattr(item, "score", 0.0)), 4),
                    "snippet": _snippet(getattr(item, "answer", "") or ""),
                }
            )
    return out


def sanitize_snapshot(raw, limit: int = SNAPSHOT_LIMIT) -> list[dict] | None:
    """外部（前端）回传的快照做一遍清洗。

    不信任字段：非 list 直接当没有、非法项丢掉、只留五个键、分数截到 [0,1]。
    返回 None 表示「这一轮没有可用的召回快照」——审核页会显示「该轮未走检索」。
    """
    if not isinstance(raw, list) or not raw:
        return None
    cleaned = []
    for item in raw[:limit]:
        if not isinstance(item, dict):
            continue
        try:
            score = float(item.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        cleaned.append(
            {
                "n": int(item.get("n") or 0),
                "chunk_id": int(item.get("chunk_id") or 0),
                "section_path": str(item.get("section_path") or "")[:200],
                "score": round(min(max(score, 0.0), 1.0), 4),
                "snippet": _snippet(item.get("snippet") or item.get("answer") or ""),
            }
        )
    return cleaned or None


async def enqueue(
    session,
    *,
    raw_question: str,
    source: str,
    reason: str | None = None,
    conversation_id: int | None = None,
    citations=None,
    snapshot: list[dict] | None = None,
) -> LowConfidenceQuestion | None:
    """统一入池入口。原话为空直接返回 None（空问题进池只会污染审核队列）。"""
    question = " ".join(str(raw_question or "").split())
    if not question:
        return None
    if snapshot is None and citations:
        snapshot = snapshot_from_citations(citations)
    row = await low_confidence_repo.add_question(
        session,
        raw_question=question,
        source=source,
        reason=(reason or None),
        conversation_id=conversation_id,
        retrieved_chunks=snapshot,
    )
    logger.info(
        "入池:%s | source=%s | 快照=%s 条", question[:40], source, len(snapshot or [])
    )
    return row


# ---------------------------------------------------------------------------
# 标准化：把用户原话改写成 FAQ 式问题，并顺手给一版示例答案
# ---------------------------------------------------------------------------

NORMALIZE_PROMPT = """你是电商客服知识库的编辑。下面是一条用户没得到答案的原话（可能口语、有错别字、夹带无关信息）。

请把它改写成一句知识库 FAQ 式问题：
- 一句问句，说清问的是什么（商品/政策/流程/时效）；
- 保留原话里的关键限定（型号、场景、时间），不要补原话里没有的限定；
- 不要改成「如何提升用户体验」这种泛化问法，也不要拆成两句。

同时给出一版可直接采用的示例答案：
- 只根据原话能确定的事写，不确定的地方用「需人工核实」标出，不要编造具体数字；
- 2~3 句，客服口吻。

只输出 JSON：{{"normalized_question": "...", "ai_suggested_answer": "..."}}

原话：{raw_question}"""

DEDUPE_PROMPT = """你是客服知识库的编辑。下面有一条新问题，以及当前待审的缺口列表（编号 + 问题）。

判断新问题是否和其中某一条**说的是同一件事**：
- 同一个政策/流程/商品属性，只是问法不同，算同一条；
- 主题相近但问的不是同一件事（一个问运费谁出、一个问运费多少）不算。

只输出 JSON：{{"matched_id": 编号或 null, "reason": "一句话说明"}}
只能选列表里出现过的编号，没有合适的就输出 null。

新问题：{question}

待审缺口：
{candidates}"""


@dataclass
class Normalization:
    """标准化结果。failed=True 表示这条没改写成，留在池里等下一轮。"""

    question: str
    answer: str = ""
    failed: bool = False
    detail: str = ""


def _parse_json(text: str) -> dict:
    """宽松 JSON 解析：模型爱在 JSON 前后加解释或 ``` 围栏。"""
    from app.rag.query_rewrite import _parse_json as parse

    return parse(text)


async def normalize(raw_question: str, *, model=None) -> Normalization:
    """一次模型调用：原话 → FAQ 式问题 + 示例答案。

    失败不抛异常而是返回 `failed=True`：一条改写失败不该让整批停下，
    也不该被悄悄丢掉（丢掉的后果是这条问题永远没人看到）。
    """
    question = " ".join(str(raw_question or "").split())
    if not question:
        return Normalization("", failed=True, detail="原话为空")

    if model is None:
        try:
            from app.core.llm import get_extract_model

            model = get_extract_model()
        except Exception as exc:  # noqa: BLE001
            return Normalization(question, failed=True, detail=f"模型不可用:{exc}")

    prompt = NORMALIZE_PROMPT.format(raw_question=question)
    try:
        try:
            resp = await model.ainvoke([("human", prompt)], response_format={"type": "json_object"})
        except TypeError:
            resp = await model.ainvoke([("human", prompt)])
        payload = _parse_json(getattr(resp, "content", "") or "")
    except Exception as exc:  # noqa: BLE001
        logger.warning("标准化失败:%s | %s", question[:40], exc)
        return Normalization(question, failed=True, detail=f"调用失败:{exc}")

    normalized = str(payload.get("normalized_question") or "").strip()
    if not normalized:
        return Normalization(question, failed=True, detail="缺少 normalized_question")
    return Normalization(
        normalized[:NORMALIZED_MAX],
        answer=str(payload.get("ai_suggested_answer") or "").strip(),
    )


# ---------------------------------------------------------------------------
# 查重：先规则、后模型
# ---------------------------------------------------------------------------

_PUNCT = re.compile(r"[\s\u3000·、，。！？；：,.!?;:\"'“”‘’（）()\[\]【】<>《》~～\-—_/\\|]+")


def _squash(text: str) -> str:
    """比对用的最小化形式：去标点空白、统一小写。只用于规则比较，不进库。"""
    return _PUNCT.sub("", str(text or "")).lower()


def rule_match(question: str, candidates: list[ReviewQueue]) -> ReviewQueue | None:
    """规范化后相等，或互相包含且长度差 ≤ 40%。

    只解决「同一句话反复入池」这一种重复——池子里这类占大头，
    且判断绝对可靠（不会把两个不同问题误判成一个）。
    """
    key = _squash(question)
    if not key:
        return None
    for row in candidates:
        other = _squash(row.normalized_question)
        if not other:
            continue
        if key == other:
            return row
        short, long_ = (key, other) if len(key) <= len(other) else (other, key)
        if short in long_ and (len(long_) - len(short)) / len(long_) <= RULE_LEN_TOLERANCE:
            return row
    return None


async def model_match(question: str, candidates: list[ReviewQueue], *, model) -> tuple[int | None, bool]:
    """语义查重。返回 (matched_id, degraded)。

    degraded=True 表示「没判成」（不是判成不重复）：调用方据此把这条计入 failed，
    免得一次上游抖动被当成「确认是新缺口」。
    """
    if not candidates:
        return None, False
    listing = "\n".join(f"{row.id}. {row.normalized_question}" for row in candidates)
    prompt = DEDUPE_PROMPT.format(question=question[:NORMALIZED_MAX], candidates=listing)
    try:
        try:
            resp = await model.ainvoke([("human", prompt)], response_format={"type": "json_object"})
        except TypeError:
            resp = await model.ainvoke([("human", prompt)])
        payload = _parse_json(getattr(resp, "content", "") or "")
    except Exception as exc:  # noqa: BLE001
        logger.warning("模型查重失败:%s | %s", question[:40], exc)
        return None, True

    if "matched_id" not in payload:
        return None, True
    raw_id = payload.get("matched_id")
    if raw_id in (None, "", "null"):
        return None, False
    try:
        matched_id = int(raw_id)
    except (TypeError, ValueError):
        return None, True
    # 只认列表里的编号：模型编一个不存在的 id 会让归并指到空处
    if matched_id not in {row.id for row in candidates}:
        logger.warning("模型给了不存在的缺口编号 %s，按未命中处理", matched_id)
        return None, True
    return matched_id, False


async def dedupe(
    session,
    question: str,
    *,
    model=None,
    candidates: list[ReviewQueue] | None = None,
    limit: int = DEDUPE_CANDIDATES,
    dry_run: bool = False,
) -> dict:
    """把一条标准化后的问题归到某个缺口，或新建一条。

    返回 `{"review": row, "via": rule|model|new, "degraded": bool}`。
    规则命中时不碰模型——这是流水线里最省的一步，也是故意排在前面的原因。

    `dry_run=True` 时「该新建」的那条不落库，`review` 返回 None——
    预览要的是数量，不是真写一行再去删。
    """
    if candidates is None:
        candidates = await review_repo.list_for_dedupe(session, limit=limit)

    hit = rule_match(question, candidates)
    if hit is not None:
        return {"review": hit, "via": VIA_RULE, "degraded": False}

    degraded = False
    if candidates and model is not None:
        matched_id, degraded = await model_match(question, candidates, model=model)
        if matched_id is not None:
            row = await review_repo.get(session, matched_id)
            if row is not None:
                return {"review": row, "via": VIA_MODEL, "degraded": False}

    if dry_run:
        return {"review": None, "via": VIA_NEW, "degraded": degraded}
    row = await review_repo.create(session, normalized_question=question)
    return {"review": row, "via": VIA_NEW, "degraded": degraded}


# ---------------------------------------------------------------------------
# 批量跑：池里未归并的 → 标准化 → 查重 → 累加或新建 → 写回归并落点
# ---------------------------------------------------------------------------


async def run_batch(
    session_factory,
    *,
    limit: int = 50,
    model=None,
    dry_run: bool = False,
    since_id: int = 0,
) -> dict:
    """处理一批未归并的池内问题。

    `dry_run` 只算不写：不建缺口、不累加次数、不写 `matched_review_id`，
    但照常调模型——否则「这一批大概会新增几条缺口」这个预览就没意义了。
    """
    stats = {
        "processed": 0,
        "created": 0,
        "merged": 0,
        "failed": 0,
        "dry_run": dry_run,
        "last_id": since_id,
        "items": [],
    }

    async with session_factory() as session:
        rows = await low_confidence_repo.list_unmerged(session, limit=limit, since_id=since_id)
        if not rows:
            return stats

        # 本轮新建的缺口。必须自己留着:候选池按出现次数排序取前 20,而新建行的
        # 次数是 1,只要待审队列已经排满,它就会被挤出窗口——于是同一个问题在一次
        # 批量里开出两条缺口(2026-11-12 彩排实测:三行一模一样的待审)。
        created_here: list[ReviewQueue] = []

        for item in rows:
            stats["processed"] += 1
            stats["last_id"] = item.id
            norm = await normalize(item.raw_question, model=model)
            if norm.failed:
                # 留在池里等下一轮：不写 matched_review_id 就是「没处理过」
                stats["failed"] += 1
                stats["items"].append(
                    {"id": item.id, "question": item.raw_question[:40], "via": "failed", "detail": norm.detail}
                )
                continue

            # 每轮重新取候选：上一条刚新建的缺口，下一条就该能并进去
            candidates = await review_repo.list_for_dedupe(session)
            if created_here:
                seen = {c.id for c in candidates}
                candidates = created_here + [c for c in candidates if c.id not in seen]
            plan = await dedupe(
                session, norm.question, model=model, candidates=candidates, dry_run=dry_run
            )
            review = plan["review"]
            if plan["degraded"]:
                stats["failed"] += 1

            if dry_run:
                stats["created" if plan["via"] == VIA_NEW else "merged"] += 1
                stats["items"].append(
                    {"id": item.id, "question": norm.question[:40], "via": plan["via"], "dry_run": True}
                )
                continue

            if plan["via"] == VIA_NEW:
                stats["created"] += 1
                if plan["review"] is not None:
                    created_here.append(plan["review"])
            else:
                await review_repo.bump(session, review.id)
                stats["merged"] += 1
            await low_confidence_repo.set_matched_review(session, item.id, review.id)
            stats["items"].append(
                {"id": item.id, "question": norm.question[:40], "via": plan["via"], "review_id": review.id}
            )

    logger.info(
        "飞轮批处理:处理 %s 条 / 新建 %s / 累加 %s / 失败 %s%s",
        stats["processed"],
        stats["created"],
        stats["merged"],
        stats["failed"],
        "（dry-run 未落库）" if dry_run else "",
    )
    return stats


# ---------------------------------------------------------------------------
# 审核：通过 = 走 ch03 的双写流程写知识库；驳回 = 只改状态
# ---------------------------------------------------------------------------

# 飞轮补进去的块统一挂这个分类，检索/复盘时能和建库文档区分开
FLYWHEEL_CATEGORY = "飞轮补充"
FLYWHEEL_SECTION = "飞轮补充/人工审核"
FLYWHEEL_CONTENT_TYPE = "qa"


async def approve(
    session,
    review_id: int,
    approved_answer: str | None = None,
    *,
    store,
    embedder,
    batch_size: int = 32,
) -> dict | None:
    """审核通过：建块 → 向量化 → 状态置通过。

    不自己写 embedding / upsert，调 `vectorize_pending`——知识库只有一条
    正确的写入路径（pending → 向量化 → done），绕过去迟早出现「MySQL 有行、
    向量库里没有」的块，检索时等于不存在。

    二次审核同一个缺口时按 (分类, 问法) 找已有块改答案，不再插一行。
    """
    from app.models.knowledge_chunk import KnowledgeChunk
    from app.rag.indexer import vectorize_pending
    from app.repositories import knowledge_repo

    row = await review_repo.get(session, review_id)
    if row is None:
        return None

    answer = " ".join(str(approved_answer or "").split())
    if not answer:
        # 没答案就通过，等于往知识库塞一条空壳；让调用方去要答案
        raise ValueError("核准答案不能为空")

    chunk = await knowledge_repo.find_by_category_and_questions(
        session, FLYWHEEL_CATEGORY, row.normalized_question
    )
    created = chunk is None
    if created:
        chunk = (
            await knowledge_repo.add_chunks(
                session,
                [
                    KnowledgeChunk(
                        category=FLYWHEEL_CATEGORY,
                        questions=row.normalized_question,
                        answer=answer,
                        section_path=FLYWHEEL_SECTION,
                        content_type=FLYWHEEL_CONTENT_TYPE,
                        is_key_clause=False,
                    )
                ],
            )
        )[0]
    else:
        await knowledge_repo.replace_answer(session, chunk.id, answer)

    vr = await vectorize_pending(session, store, embedder, batch_size=batch_size)
    await review_repo.mark_approved(session, review_id, answer)

    logger.info(
        "审核通过:#%s → 块 #%s（%s）| 本轮补向量化 %s 条",
        review_id,
        chunk.id,
        "新建" if created else "改答案",
        vr.vectorized,
    )
    return {
        "review_id": review_id,
        "chunk_id": chunk.id,
        "created": created,
        "vectorized": vr.vectorized,
    }


async def reject(session, review_id: int) -> ReviewQueue | None:
    """驳回：只改状态。驳回通常意味着「这不是知识缺口」，写进库里只会污染检索。"""
    row = await review_repo.mark_rejected(session, review_id)
    if row is not None:
        logger.info("审核驳回:#%s %s", review_id, row.normalized_question[:40])
    return row
