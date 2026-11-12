"""CH09 飞轮流水线：标准化、规则查重优先、模型查重兜底、归并落点。

模型一律假实现（判断提示词里出现「待审缺口」还是「FAQ 式问题」决定答哪段），
真模型只在 T9 的验收圈里跑。
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage

from app.models.review_queue import ReviewQueue
from app.repositories import low_confidence_repo, review_repo
from app.services import flywheel


class FakeModel:
    """两段提示词各答一份。fail_on 里的原话会让标准化抛错。"""

    def __init__(
        self,
        *,
        mapping: dict[str, str] | None = None,
        matched_id=None,
        answer: str = "示例答案：签收后 7 天内可申请退货。",
        fail_on: set[str] | None = None,
        dedupe_error: Exception | None = None,
        normalize_json: str | None = None,
    ):
        self.mapping = mapping or {}
        self.matched_id = matched_id
        self.answer = answer
        self.fail_on = fail_on or set()
        self.dedupe_error = dedupe_error
        self.normalize_json = normalize_json
        self.normalize_calls = 0
        self.dedupe_calls = 0

    async def ainvoke(self, messages, **kwargs):
        prompt = messages[0][1] if isinstance(messages[0], tuple) else getattr(messages[0], "content", "")
        if "待审缺口" in prompt:
            self.dedupe_calls += 1
            if self.dedupe_error is not None:
                raise self.dedupe_error
            return AIMessage(content=json.dumps({"matched_id": self.matched_id, "reason": "同义"}))

        self.normalize_calls += 1
        raw = prompt.rsplit("原话：", 1)[-1].strip()
        if any(bad in raw for bad in self.fail_on):
            raise RuntimeError("upstream 502")
        if self.normalize_json is not None:
            return AIMessage(content=self.normalize_json)
        return AIMessage(
            content=json.dumps(
                {"normalized_question": self.mapping.get(raw, raw), "ai_suggested_answer": self.answer},
                ensure_ascii=False,
            )
        )


def _row(qid: int, question: str) -> ReviewQueue:
    """只造规则比对要用的两个字段，不入库。"""
    return ReviewQueue(id=qid, normalized_question=question, occurrence_count=1)


# ------------------------------------------------------------------ 规则查重


def test_rule_match_ignores_punctuation_and_spacing():
    hit = flywheel.rule_match("退货运费谁出？", [_row(1, "退货运费谁出")])
    assert hit is not None and hit.id == 1


def test_rule_match_accepts_containment_within_tolerance():
    """长一点的问法包含短的：同一件事的两种说法。"""
    hit = flywheel.rule_match("七天无理由退货的运费谁承担", [_row(1, "退货的运费谁承担")])
    assert hit is not None


def test_rule_match_rejects_containment_that_is_too_long():
    """包含关系但长度差太大：多半是拿短问题套了个长场景，不是同一件事。"""
    long_q = "退货运费谁承担啊我买的是那个加绒款还用了优惠券到底怎么算"
    assert flywheel.rule_match(long_q, [_row(1, "退货运费谁承担")]) is None


def test_rule_match_returns_none_for_blank_question():
    assert flywheel.rule_match("  ", [_row(1, "退货运费谁承担")]) is None


# ------------------------------------------------------------------ 标准化


@pytest.mark.asyncio
async def test_normalize_returns_question_and_suggested_answer():
    model = FakeModel(mapping={"那个退货运费谁出啊": "退货运费由谁承担？"})
    norm = await flywheel.normalize("那个退货运费谁出啊", model=model)
    assert norm.failed is False
    assert norm.question == "退货运费由谁承担？"
    assert norm.answer.startswith("示例答案")


@pytest.mark.asyncio
async def test_normalize_marks_failed_instead_of_raising():
    model = FakeModel(fail_on={"退货运费"})
    norm = await flywheel.normalize("退货运费谁出", model=model)
    assert norm.failed is True
    assert norm.question == "退货运费谁出"  # 原话留着，还在池里
    assert "502" in norm.detail


@pytest.mark.asyncio
async def test_normalize_rejects_response_without_question():
    model = FakeModel(normalize_json='{"ai_suggested_answer": "只有答案"}')
    norm = await flywheel.normalize("退货运费谁出", model=model)
    assert norm.failed is True
    assert "normalized_question" in norm.detail


# ------------------------------------------------------------------ 查重两级


@pytest.mark.asyncio
async def test_rule_hit_never_calls_the_model(session):
    """规则能认出来的重复不该花模型调用——这是把规则排在前面的全部理由。"""
    model = FakeModel(matched_id=1)
    existing = await review_repo.create(session, normalized_question="退货运费谁出")
    plan = await flywheel.dedupe(session, "退货运费谁出？", model=model)
    assert plan["via"] == flywheel.VIA_RULE
    assert plan["review"].id == existing.id
    assert model.dedupe_calls == 0


@pytest.mark.asyncio
async def test_model_hit_returns_existing_row(session):
    model = FakeModel(matched_id=1)
    existing = await review_repo.create(session, normalized_question="退货运费由谁承担")
    plan = await flywheel.dedupe(session, "寄回去的邮费谁付", model=model)
    assert plan["via"] == flywheel.VIA_MODEL
    assert plan["review"].id == existing.id
    assert model.dedupe_calls == 1


@pytest.mark.asyncio
async def test_model_hallucinated_id_is_treated_as_new(session):
    """模型编一个不存在的编号，归并就会指到空处——只认列表里出现过的 id。"""
    model = FakeModel(matched_id=999)
    await review_repo.create(session, normalized_question="退货运费由谁承担")
    plan = await flywheel.dedupe(session, "寄回去的邮费谁付", model=model)
    assert plan["via"] == flywheel.VIA_NEW
    assert plan["degraded"] is True


@pytest.mark.asyncio
async def test_dedupe_model_failure_falls_back_to_new_and_flags_degraded(session):
    model = FakeModel(dedupe_error=RuntimeError("upstream 502"))
    await review_repo.create(session, normalized_question="退货运费由谁承担")
    plan = await flywheel.dedupe(session, "寄回去的邮费谁付", model=model)
    assert plan["via"] == flywheel.VIA_NEW
    assert plan["degraded"] is True  # 计入 failed，别当成「确认是新缺口」


@pytest.mark.asyncio
async def test_dedupe_without_candidates_skips_model(session):
    """待审队列空的时候没有可比的，不必空跑一次模型。"""
    model = FakeModel(matched_id=1)
    plan = await flywheel.dedupe(session, "退货运费谁出", model=model)
    assert plan["via"] == flywheel.VIA_NEW
    assert model.dedupe_calls == 0


# ------------------------------------------------------------------ 批处理


async def _pool(session, *questions: str):
    for q in questions:
        await flywheel.enqueue(session, raw_question=q, source=flywheel.SOURCE_RETRIEVAL)


@pytest.mark.asyncio
async def test_run_batch_merges_repeats_and_backfills_match(session, session_factory):
    await _pool(session, "退货运费谁出", "退货运费谁出？", "发票怎么开")
    model = FakeModel()

    stats = await flywheel.run_batch(session_factory, model=model)

    assert stats["processed"] == 3
    assert stats["created"] == 2  # 退货运费 + 发票，重复的那条并进去
    assert stats["merged"] == 1
    assert stats["failed"] == 0

    pending = await review_repo.list_for_dedupe(session)
    by_q = {r.normalized_question: r for r in pending}
    assert by_q["退货运费谁出"].occurrence_count == 2
    assert by_q["发票怎么开"].occurrence_count == 1

    # 三条原话都要指向某个缺口（原话不删，靠指针反查）
    questions = await low_confidence_repo.list_by_review(session, by_q["退货运费谁出"].id)
    assert len(questions) == 2


@pytest.mark.asyncio
async def test_run_batch_dedupes_within_batch_when_queue_is_full(session, session_factory):
    """待审队列排满候选窗口后，本轮刚新建的缺口仍要被下一条原话认出来。

    候选池按出现次数排序取前 DEDUPE_CANDIDATES 条，新建行的次数是 1，天然排最后、
    被截断。2026-11-12 彩排实测到的后果：同一句「你们卖宠物驱虫药吗」在一次批量里
    开出三条一模一样的待审缺口。
    """
    for i in range(flywheel.DEDUPE_CANDIDATES):
        row = await review_repo.create(session, normalized_question=f"占窗口的老问题 {i}")
        row.occurrence_count = 5
    await session.commit()

    await _pool(session, "退款要几天到账", "退款要几天到账？")
    stats = await flywheel.run_batch(session_factory, model=FakeModel())

    assert stats["created"] == 1
    assert stats["merged"] == 1
    # 查全量待审（list_for_dedupe 只给前 20 条，新的这条本来就在窗口外）
    pending = await review_repo.list_by_status(session, "待审", limit=200)
    hits = [r for r in pending if r.normalized_question.startswith("退款要几天到账")]
    assert len(hits) == 1
    assert hits[0].occurrence_count == 2


@pytest.mark.asyncio
async def test_run_batch_dry_run_writes_nothing(session, session_factory):
    await _pool(session, "退货运费谁出", "发票怎么开")
    model = FakeModel()

    stats = await flywheel.run_batch(session_factory, model=model, dry_run=True)

    assert stats["dry_run"] is True
    assert stats["created"] == 2
    assert await review_repo.list_for_dedupe(session) == []
    assert len(await low_confidence_repo.list_unmerged(session)) == 2  # 仍然是「没处理过」


@pytest.mark.asyncio
async def test_run_batch_leaves_failed_rows_unmerged(session, session_factory):
    """标准化失败的那条不能写 matched_review_id，否则它永远没人再看一眼。"""
    await _pool(session, "退货运费谁出", "这个瞎问题")
    model = FakeModel(fail_on={"瞎问题"})

    stats = await flywheel.run_batch(session_factory, model=model)

    assert stats["failed"] == 1
    assert stats["created"] == 1
    leftover = await low_confidence_repo.list_unmerged(session)
    assert [r.raw_question for r in leftover] == ["这个瞎问题"]


@pytest.mark.asyncio
async def test_run_batch_respects_since_id_for_resume(session, session_factory):
    await _pool(session, "问题一", "问题二", "问题三")
    first = await low_confidence_repo.list_unmerged(session, limit=2)

    stats = await flywheel.run_batch(session_factory, model=FakeModel(), since_id=first[-1].id)

    assert stats["processed"] == 1
    assert stats["last_id"] == 3


@pytest.mark.asyncio
async def test_run_batch_noop_on_empty_pool(session_factory):
    stats = await flywheel.run_batch(session_factory, model=FakeModel())
    assert stats["processed"] == 0
    assert stats["last_id"] == 0
