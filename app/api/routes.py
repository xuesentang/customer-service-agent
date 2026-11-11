"""HTTP API routes (spec D4 protocol; CH02: tools on the same SSE entry; CH05: graph)."""

import logging
import re
import uuid
from collections.abc import AsyncIterable
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.refund_flow import REFUND_REASONS
from app.core.tools.business_tools import build_default_registry
from app.core.tools.context import current_conversation_id
from app.core.tools.executor import ToolExecutor
from app.db import get_session
from app.models.conversation import Conversation
from app.repositories import message_repo
from app.services.agent_chat import chat_with_agent
from app.services.extract import ExtractionError, extract_after_sales

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

# 「已转接人工客服」是本章的前端模拟,没有真人系统可接;这两句是前端要显示的东西,
# 由后端给出是为了文案只有一处来源(前端不自己拼话术)。
TRANSFER_NOTICE = "已转接人工客服"
TRANSFER_GREETING = "您好,我是客服小猫,请问有什么可以帮您的"

_TICKET_NO = re.compile(r"已创建工单\s+([A-Za-z0-9_-]+)")


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    # CH02: 不传则复用该用户最近进行中的会话;done 事件回传 conversation_id。
    # new_conversation=True 强制开新会话(前端「新会话」按钮,review I2:
    # 只把前端 conversation_id 清成 null 并不够——服务端 None 的语义是复用)。
    conversation_id: int | None = None
    new_conversation: bool = False
    # CH06: 前端点选订单卡片后回填的订单号;有它就不再弹选择器
    order_id: str | None = None

    @field_validator("session_id")
    @classmethod
    def session_id_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("session_id must not be blank")
        return value

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


class TransferRequest(BaseModel):
    """转人工:前端点按钮才发,后端不自动触发。"""

    conversation_id: int


class RefundApplyRequest(BaseModel):
    """退款单:用户在前端表单里选好原因才提交。原因必须来自固定类目。"""

    conversation_id: int
    order_id: str = Field(min_length=1)
    reason: str
    description: str | None = None

    @field_validator("reason")
    @classmethod
    def reason_must_be_known(cls, value: str) -> str:
        if value not in REFUND_REASONS:
            raise ValueError(f"退款原因必须在固定类目内:{'、'.join(REFUND_REASONS)}")
        return value


class TicketRequest(BaseModel):
    """建工单:用户点按钮才发。描述缺省取会话里最近一条用户消息。"""

    conversation_id: int
    description: str | None = None
    ticket_type: Literal["售后", "投诉", "咨询"] = "投诉"


class ResumeRequest(BaseModel):
    """工单预览卡片上的「确认提交 / 取消」。

    `decision` 只认 approve / reject 两个语义(中文几个常见说法也认),其余一律按
    拒绝处理——写操作宁可少做一次。
    """

    session_id: str = Field(min_length=1)
    conversation_id: int
    decision: Literal["approve", "reject"] = "reject"
    # 卡片上显示过的内容原样带回来:图恢复时会重跑节点,万一模型这次给的参数变了,
    # 后端靠这两项比对得出「不是用户看到的那张单」,按拒绝处理。
    ticket_type: str | None = None
    description: str | None = None


@router.post("/chat/stream", response_class=EventSourceResponse, response_model=None)
async def chat_stream(
    request: ChatRequest,
    session: AsyncSession = Depends(get_session),
) -> AsyncIterable[ServerSentEvent]:
    async for event in chat_with_agent(
        request.session_id,
        request.message,
        request.conversation_id,
        new_conversation=request.new_conversation,
        order_id=request.order_id,
        session=session,
    ):
        yield event


@router.post("/chat/resume", response_class=EventSourceResponse, response_model=None)
async def chat_resume(
    request: ResumeRequest,
    session: AsyncSession = Depends(get_session),
) -> AsyncIterable[ServerSentEvent]:
    """接着跑被 interrupt 挂住的那一轮(第八章需求 7)。

    与 `/chat/stream` 的区别只有一处:这里不传新消息,只把用户的决定递回去;
    图从 checkpoint 继续,把工具执行完、把答复吐完。
    """
    async for event in chat_with_agent(
        request.session_id,
        "",
        request.conversation_id,
        resume={
            "decision": request.decision,
            "ticket_type": request.ticket_type or "",
            "description": request.description or "",
        },
        session=session,
    ):
        yield event


@router.post("/chat/extract")
async def chat_extract(request: ExtractRequest) -> dict[str, Any]:
    try:
        return await extract_after_sales(request.text)
    except ExtractionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


class FeedbackRequest(BaseModel):
    """聊天页的 👍/👎（ch09 起接后端）。

    `question` 是用户当轮原话（前端 `turn.question`）；`citations` 是那一轮的召回结果，
    前端手上就有，直接回传胜过在库里加列——用户只能对刚看到的回答点 👎。
    """

    conversation_id: int | None = None
    question: str = Field(min_length=1)
    rating: Literal["up", "down"]
    citations: list[dict] | None = None

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """👎 把这条问题落进低置信度问题池（需求 3 的第三个入口）。

    👍 不入池——「答得好」不是知识缺口。但它也不该被丢掉，所以照常返回 ok，
    只是 `pooled=False`；将来要做满意度统计，加一张表就行，这里不假装做了。
    """
    from app.models.conversation import Conversation
    from app.services import flywheel

    if request.rating == "up":
        return {"ok": True, "pooled": False, "reason": "点赞不入池"}

    conversation_id = request.conversation_id
    if conversation_id is not None and await session.get(Conversation, conversation_id) is None:
        conversation_id = None  # 会话被清了也别让反馈报错

    row = await flywheel.enqueue(
        session,
        raw_question=request.question,
        source=flywheel.SOURCE_USER_FEEDBACK,
        reason="user_feedback",
        conversation_id=conversation_id,
        snapshot=flywheel.sanitize_snapshot(request.citations),
    )
    return {
        "ok": True,
        "pooled": row is not None,
        "question_id": row.id if row else None,
        "snapshot_size": len(row.retrieved_chunks or []) if row else 0,
    }


@router.post("/chat/actions/transfer")
async def transfer_to_human(
    request: TransferRequest,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """把会话标记为已转人工(前端模拟)。

    不做的事:不建工单、不调工具、不改会话状态机。转人工和建工单是两件事,
    这一路只回一句「已转接」并把这两句话记进消息流水,方便后续回放时上下文完整。
    """
    conv = await session.get(Conversation, request.conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    notice = f"{TRANSFER_NOTICE}。{TRANSFER_GREETING}。"
    try:
        await message_repo.add_message(
            session, conv.id, role="assistant", content=f"【{notice}】"
        )
    except Exception as exc:  # noqa: BLE001 - 留痕失败不该挡用户走人
        logger.warning("转人工留痕失败:%s", exc)

    logger.info("agent.transfer conversation_id=%s", conv.id)
    return {"status": "transferred", "notice": TRANSFER_NOTICE, "greeting": TRANSFER_GREETING}


@router.post("/chat/actions/create-ticket")
async def create_ticket_from_action(
    request: TicketRequest,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """用户点「建工单」才走这里:调第 2 章的工单工具写 tickets 表。

    刻意不直接写 repo:工具是唯一入口,这样「工具被谁调用、参数什么样」
    在日志/审计里只有一条路径可查(CH08 要在这里加权限与审计)。
    """
    conv = await session.get(Conversation, request.conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    description = (request.description or "").strip()
    if not description:
        msgs = await message_repo.list_by_conversation(session, conv.id)
        for msg in reversed(msgs):
            if msg.role == "user" and (msg.content or "").strip():
                description = (msg.content or "").strip()
                break
    if not description:
        raise HTTPException(status_code=422, detail="缺少问题描述,无法建单")

    registry = build_default_registry()
    executor = ToolExecutor(registry)
    token = current_conversation_id.set(conv.id)
    try:
        result = await executor.execute(
            {
                "name": "create_ticket",
                "args": {"description": description, "ticket_type": request.ticket_type},
                "id": f"action-{uuid.uuid4().hex[:8]}",
                # type 不能省:执行引擎按 ToolCall 协议拆 args,少了这个键会把整个
                # 字典当成参数去校验(报「description 缺失」,看着像参数没传)
                "type": "tool_call",
            },
            # 确认发生在前端:用户点了「建工单」按钮才发这个请求,表单提交本身就是
            # 确认动作。这里显式传 True 而不是靠执行引擎的默认值——默认值是关的,
            # 漏传就该被拒,不能靠「忘了传」放行。
            confirmed=True,
        )
    finally:
        current_conversation_id.reset(token)

    if not result.ok:
        raise HTTPException(status_code=502, detail=result.content)

    match = _TICKET_NO.search(result.content)
    return {
        "status": "created",
        "ticket_no": match.group(1) if match else None,
        "ticket_type": request.ticket_type,
        "description": description,
        "message": result.content,
    }


@router.get("/refund/reasons")
async def refund_reasons() -> dict[str, Any]:
    """固定退款原因类目:前端下拉用,后端是唯一来源(不写死在前端)。"""
    return {"reasons": list(REFUND_REASONS)}


@router.post("/refund/apply")
async def apply_refund(
    request: RefundApplyRequest,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """提交退款单:受理即建一张「售后」工单,工单号就是受理单号。

    本章不做退款状态机(提示词没要求,建表语句本章也是空的):真实系统里退款单要落
    单独的明细表并驱动审核流;这里用售后工单承接,保证「用户提交了、有单号可查」。
    """
    conv = await session.get(Conversation, request.conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    description = (request.description or "").strip() or (
        f"订单 {request.order_id} 申请退款,原因:{request.reason}"
    )
    registry = build_default_registry()
    executor = ToolExecutor(registry)
    token = current_conversation_id.set(conv.id)
    try:
        result = await executor.execute(
            {
                "name": "create_ticket",
                "args": {"description": description, "ticket_type": "售后"},
                "id": f"refund-{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            },
            # 同 create-ticket:退款表单的提交按钮就是确认动作,显式声明。
            confirmed=True,
        )
    finally:
        current_conversation_id.reset(token)

    if not result.ok:
        raise HTTPException(status_code=502, detail=result.content)

    match = _TICKET_NO.search(result.content)
    refund_no = match.group(1) if match else None
    reply = (
        f"已收到您的退款申请(受理单号 {refund_no or '已受理'}):订单 {request.order_id},原因「{request.reason}」。"
        "审核结果会同步给到您,期间不用重复提交。"
    )
    try:
        await message_repo.add_message(session, conv.id, role="assistant", content=reply)
    except Exception as exc:  # noqa: BLE001 - 留痕失败不影响受理
        logger.warning("退款受理留痕失败:%s", exc)

    logger.info("agent.refund_applied conversation_id=%s order_id=%s", conv.id, request.order_id)
    return {
        "status": "submitted",
        "refund_no": refund_no,
        "order_id": request.order_id,
        "reason": request.reason,
        "message": reply,
    }


# ---------------------------------------------------------------- CH07 只读会话接口


@router.get("/conversations")
async def list_conversations(
    user_key: str,
    limit: int = 50,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """侧栏用:该用户全部会话,新的在前,带首问预览与「已摘要」标记。

    `user_key` 就是聊天接口的 `session_id`(本项目的用户键)。只读、不带鉴权,
    与聊天接口的信任级别一致(README 已知限制:无用户体系)。
    """
    from app.repositories import conversation_repo

    rows = await conversation_repo.list_by_user(session, user_key, limit=limit)
    out: list[dict[str, Any]] = []
    for conv in rows:
        msgs = await message_repo.list_by_conversation(session, conv.id)
        first_user = next((m for m in msgs if m.role == "user"), None)
        preview = (first_user.content or "").strip().replace("\n", " ") if first_user else ""
        out.append(
            {
                "conversation_id": conv.id,
                "status": conv.status,
                "preview": preview[:40],
                "message_count": len(msgs),
                "summarized": int(conv.summary_upto_msg_id or 0) > 0,
                "created_at": conv.created_at.isoformat() if conv.created_at else None,
            }
        )
    return {"conversations": out}


@router.get("/conversations/{conversation_id}/messages")
async def conversation_messages(
    conversation_id: int,
    user_key: str,
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """切回旧会话时回载历史原文(不是梗概,原文永远在库里)。"""
    conv = await session.get(Conversation, conversation_id)
    if conv is None or conv.user_id != user_key:
        raise HTTPException(status_code=404, detail="会话不存在")
    msgs = await message_repo.list_by_conversation(session, conversation_id)
    return {
        "conversation_id": conversation_id,
        "summarized": int(conv.summary_upto_msg_id or 0) > 0,
        "messages": [
            {
                "role": m.role,
                "content": m.content or "",
                "tool_call_id": m.tool_call_id,
                "has_tool_calls": bool(m.tool_calls),
            }
            for m in msgs
        ],
    }
