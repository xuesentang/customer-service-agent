"""conversation_summaries 数据访问 + 会话锚点推进(第七章)。

锚点推进只有这一个入口:`advance_anchors`。散在各处改 `layer1_from_msg_id` 迟早
出现「摘要覆盖了 20 条、层 1 却从第 15 条起」这种前后打架的状态,而那两个 id 一打架,
三层切分就切出重叠或空洞,表现是模型看到重复内容或漏掉整段对话。
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import Conversation
from app.models.conversation_summary import ConversationSummary


async def list_by_conversation(session: AsyncSession, conversation_id: int) -> list[ConversationSummary]:
    """按 seq 升序取全部摘要段(拼接顺序即时间序)。"""
    stmt = (
        select(ConversationSummary)
        .where(ConversationSummary.conversation_id == conversation_id)
        .order_by(ConversationSummary.seq)
    )
    return list((await session.execute(stmt)).scalars().all())


async def next_seq(session: AsyncSession, conversation_id: int) -> int:
    existing = await list_by_conversation(session, conversation_id)
    return (existing[-1].seq + 1) if existing else 1


async def add_summary(
    session: AsyncSession,
    conversation_id: int,
    *,
    seq: int,
    content: str,
    from_msg_id: int,
    to_msg_id: int,
    tokens: int = 0,
    commit: bool = True,
) -> ConversationSummary:
    """落一段摘要。`commit=False` 用于与锚点推进同事务提交(见 run_summary)。"""
    row = ConversationSummary(
        conversation_id=conversation_id,
        seq=seq,
        content=content,
        from_msg_id=from_msg_id,
        to_msg_id=to_msg_id,
        tokens=tokens,
    )
    session.add(row)
    if commit:
        await session.commit()
    else:
        await session.flush()
    await session.refresh(row)
    return row


async def advance_anchors(
    session: AsyncSession,
    conversation: Conversation,
    *,
    summary_upto_msg_id: int,
    layer1_from_msg_id: int,
    commit: bool = True,
) -> Conversation:
    """同时推进两个锚点。摘要覆盖到哪,层 1 就从那条之后起——两者永远同步。

    摘要行和锚点是同一件事的两半:只落摘要不推锚点,下一轮会把同一段再压一次,
    历史里就多一段重复梗概。所以两者必须一个事务,`commit=False` 交给调用方收口。
    """
    conversation.summary_upto_msg_id = summary_upto_msg_id
    conversation.layer1_from_msg_id = layer1_from_msg_id
    if commit:
        await session.commit()
    else:
        await session.flush()
    await session.refresh(conversation)
    return conversation


async def joined_text(session: AsyncSession, conversation_id: int) -> str:
    """把各段摘要按顺序拼成一段背景。旧梗概只作背景,不参与合并。"""
    rows = await list_by_conversation(session, conversation_id)
    return "\n".join(r.content.strip() for r in rows if r.content.strip())
