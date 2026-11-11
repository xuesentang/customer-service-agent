"""后台摘要:层 2 攒到超预算,就把这一批压成一段梗概追加进摘要表。

三条硬规矩(需求 2):

1. **不阻塞。** 摘要跑在 `asyncio.create_task` 里,失败只记日志。用户这一轮的回复
   已经在路上了,不能因为「后台整理历史」慢了一秒就把他卡住。
2. **只提炼事实与诉求。** 问过哪款商品、报过的订单号、明确的诉求、还没解决的问题。
   对话里没出现的一个字都不许编——梗概会被当作事实喂回模型,编一条就等于给系统
   植入一条假记忆,而且它会一直躺在库里。
3. **一段一条、不回头重写。** 压完的梗概不再参与下一轮合并:第二次压缩读的是第一次
   的摘要,错一次就再也纠不回来。

触发看**用量**不看条数:工具结果不落消息表,数条数看不见层 2 涨了多少。
"""

from __future__ import annotations

import asyncio
import logging
import time

from app.core.token_budget import estimate_tokens
from app.models.conversation import Conversation
from app.repositories import conversation_summary_repo, message_repo

logger = logging.getLogger(__name__)

# 会话 id -> 还在跑的摘要任务。压一段要一两秒,这期间用户完全可能再发一句;
# 不挡的话第二次用的还是旧锚点,同一段对话被压两遍、落两段重复梗概。
# 与 LangGraph 的并发无关:这是纯 asyncio 单进程内的去重。
_running: dict[int, asyncio.Task] = {}

SUMMARY_PROMPT = """你在整理一段电商客服对话,供后续轮次当背景参考。只输出梗概正文,不要任何解释、不要 markdown。

只提炼这四类事实:
1. 问过/买过的商品(型号、品类);
2. 用户报过的订单号、手机号等标识;
3. 用户的明确诉求(要退货、要换货、要赔偿、要查物流……)和原因;
4. 还没解决的问题。

硬约束:
- 对话里没出现的内容一个字都不许加,不许推测、不许补全;
- 寒暄、道谢、客套全部丢掉;
- 保留用户原话里的关键数字与型号,不要改写成近义词;
- 长度控制在几十到两百字之间,写成一段连续的话。

对话内容:
{conversation}"""


def should_summarize(layer2_tokens: int, layer2_budget: int) -> bool:
    """层 2 用量超预算才压。装得下就不压——压缩是成本不是美德。"""
    return layer2_budget > 0 and layer2_tokens > layer2_budget


def _render_dialogue(messages) -> str:
    lines: list[str] = []
    for m in messages:
        role = {"user": "用户", "assistant": "客服", "tool": "工具"}.get(m.role, m.role)
        content = (m.content or "").strip()
        if not content and m.tool_calls:
            content = "[调用工具]"
        if content:
            lines.append(f"{role}:{content[:400]}")
    return "\n".join(lines)


async def _compress(model, dialogue: str) -> str:
    # 不做重试:摘要失败下一轮还会再触发,没有必须救回来的理由。
    resp = await model.ainvoke([("human", SUMMARY_PROMPT.format(conversation=dialogue))])
    return str(getattr(resp, "content", "") or "").strip()


async def run_summary(
    *,
    conversation_id: int,
    session_factory,
    model=None,
    layer1_from_msg_id: int | None = None,
    upto_msg_id: int | None = None,
) -> dict:
    """真正压一次。自己开 session——借请求的 session 会在请求结束时被关掉。"""
    from app.core.llm import get_extract_model

    started = time.monotonic()
    model = model or get_extract_model()
    async with session_factory() as session:
        conv = await session.get(Conversation, conversation_id)
        if conv is None:
            logger.info("summary skip 会话 %s 不存在", conversation_id)
            return {"ok": False, "reason": "no_conversation"}
        upto = upto_msg_id if upto_msg_id is not None else int(conv.layer1_from_msg_id or 0) - 1
        start_id = int(conv.summary_upto_msg_id or 0)
        if layer1_from_msg_id is not None:
            upto = layer1_from_msg_id - 1
        msgs = [
            m
            for m in await message_repo.list_by_conversation(session, conversation_id)
            if start_id < m.id <= upto
        ]
        if not msgs:
            logger.info("summary skip 会话 %s 无待压消息(区间 %s~%s)", conversation_id, start_id, upto)
            return {"ok": False, "reason": "empty_range"}

        seq = await conversation_summary_repo.next_seq(session, conversation_id)
        logger.info(
            "summary start 会话 %s 第%s段 覆盖 msg %s~%s(%s 条)", conversation_id, seq, msgs[0].id, msgs[-1].id, len(msgs)
        )
        try:
            content = await _compress(model, _render_dialogue(msgs))
        except Exception as exc:  # noqa: BLE001 - 摘要失败不影响主链路
            logger.warning("summary fail 会话 %s 第%s段 耗时 %.0fms:%s", conversation_id, seq, (time.monotonic() - started) * 1000, exc)
            return {"ok": False, "reason": "model_error"}

        if not content:
            logger.warning("summary fail 会话 %s 第%s段 模型返回空", conversation_id, seq)
            return {"ok": False, "reason": "empty"}

        tokens = estimate_tokens(content)
        try:
            # 摘要行与锚点推进是一个事务:分批提交会留下「梗概已落、锚点没动」,
            # 下一轮会把同一段再压一遍——历史里多一段重复梗概,而且不会自愈。
            await conversation_summary_repo.add_summary(
                session,
                conversation_id,
                seq=seq,
                content=content,
                from_msg_id=msgs[0].id,
                to_msg_id=msgs[-1].id,
                tokens=tokens,
                commit=False,
            )
            # 边界追到层 1 起点:摘要覆盖到哪,层 1 就从那条之后起——两个锚点同步推进
            await conversation_summary_repo.advance_anchors(
                session,
                conv,
                summary_upto_msg_id=msgs[-1].id,
                layer1_from_msg_id=msgs[-1].id + 1,
                commit=False,
            )
            await session.commit()
        except Exception as exc:  # noqa: BLE001 - 摘要失败不影响主链路,但要整体回滚
            await session.rollback()
            logger.warning("summary db fail 会话 %s 第%s段:%s", conversation_id, seq, exc)
            return {"ok": False, "reason": "db_error"}
        elapsed = (time.monotonic() - started) * 1000
        logger.info(
            "summary done 会话 %s 第%s段 %s token,覆盖 msg %s~%s,耗时 %.0fms",
            conversation_id,
            seq,
            tokens,
            msgs[0].id,
            msgs[-1].id,
            elapsed,
        )
        return {"ok": True, "seq": seq, "tokens": tokens, "elapsed_ms": elapsed, "content": content}


def schedule_summary(
    *,
    conversation_id: int,
    session_factory,
    layer2_tokens: int,
    layer2_budget: int,
    layer1_from_msg_id: int,
    model=None,
) -> asyncio.Task | None:
    """超预算就扔到后台。返回 Task 只是为了单测能 await 它,生产代码不看返回值。"""
    from app.config import get_settings

    if not get_settings().context_summary_enabled:
        logger.info("summary skip 会话 %s:总开关关闭", conversation_id)
        return None
    if not should_summarize(layer2_tokens, layer2_budget):
        return None
    if conversation_id in _running:
        logger.info("summary skip 会话 %s:上一段还在压", conversation_id)
        return None
    logger.info(
        "summary trigger 会话 %s 层2 约 %s token > 预算 %s", conversation_id, layer2_tokens, layer2_budget
    )
    task = asyncio.create_task(
        run_summary(
            conversation_id=conversation_id,
            session_factory=session_factory,
            model=model,
            layer1_from_msg_id=layer1_from_msg_id,
        )
    )
    _running[conversation_id] = task

    def _clear(t: asyncio.Task) -> None:
        # 只清自己:任务结束时可能已经排了下一个,别把后来者抹掉
        if _running.get(conversation_id) is t:
            _running.pop(conversation_id, None)

    task.add_done_callback(_clear)
    return task
