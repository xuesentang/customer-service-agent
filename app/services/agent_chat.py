"""图编排版对话服务:图负责想清楚,这里负责说话和记账。

职责边界很清楚,不能混:
- **图**只读写 State,不碰数据库、不知道 SSE 是什么;
- **这一层**管会话(取/建)、历史(读/裁)、落库(user / 工具轨迹 / 终答)、
  把图吐出来的事件翻译成 SSE 帧。

这样图能单独 ainvoke 测,服务层能拿假图测,两边都不必背着对方跑。

事件契约(与 CH02 一致,只增不改):
  node → token… → tool_start → tool_end … → citations → done
新增:`node`(节点进出)、`intent`(判意图结果)、`gate`(闸的结论),
`done` 多带 actions / intent / route。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from fastapi.sse import ServerSentEvent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Command
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.gate import REASON_CONFIDENCE_LOW, REASON_NO_HITS, REASON_SCORE_LOW
from app.agent.graph import AgentDeps, build_agent_graph
from app.config import get_settings
from app.core.context import HistoryItem, LayerPlan, plan_layers, render_history_ctx
from app.core.prompts import build_agent_system_prompt
from app.core.token_budget import ContextBudget, budget_from_settings
from app.core.tools.business_tools import build_default_registry
from app.core.tools.context import CitationBucket, current_citations, current_conversation_id
from app.models.message import Message
from app.observability import tracing
from app.repositories import (
    conversation_repo,
    conversation_summary_repo,
    message_repo,
)
from app.services import flywheel, usage_service
from app.services.tool_chat import _get_lock, _sanitize, _to_langchain

logger = logging.getLogger(__name__)

MAX_TOOL_CONTENT_LENGTH = 2000

# 每轮开跑前显式重置的本轮字段。
#
# checkpointer 以 thread_id = conversation_id 保留 State,下一轮只更新输入那几项,
# 其余字段原样留在上一次的值上。于是上一轮的 `order_options`、`finish_reason`
# 会漏到本轮的 done 事件里——第六章联调时真实发生过:本轮闸没过走了拒答,
# 前端却既收到拒答话术又收到上一轮的订单卡片。
#
# 放在调用点而不是每个节点里:节点只对自己写的字段负责,"一轮从哪开始"是服务层
# 的知识。要新增本轮字段,加到这张表里,别指望节点挨个清。
_TURN_RESET: dict = {
    "canonical_query": "",
    "resolved_query": "",
    "understanding_changed": False,
    "understanding_fallback": False,
    "slot_order_id": "",
    "intent": "",
    "intent_confidence": 0.0,
    "intent_fallback": False,
    "route": "",
    "order_facts": "",
    "order_options": [],
    "route_queries": [],
    "citations": [],
    "top_score": 0.0,
    "gate_passed": None,
    "gate_reason": "",
    "gate_detail": "",
    "reply": "",
    "finish_reason": "",
    "actions": [],
    "trail": [],
    "steps": 0,
    "tokens": 0,
    "stopped_by": "",
    "trace": {},
    "layer2_count": 0,
}


def _sse(event: str, data: dict) -> ServerSentEvent:
    return ServerSentEvent(data=data, event=event)


def _unpack(chunk) -> tuple[str, dict] | None:
    """custom stream 的返回形态随调用方式/版本变,三种都认:
    单选一个 stream_mode → 裸 payload 字典;传列表 → ("custom", payload) 元组;
    部分版本 → {"type","data"} 字典。升一次库挂一次的成本太高,这里一次兜住。"""
    if isinstance(chunk, tuple) and len(chunk) == 2 and isinstance(chunk[1], dict):
        return chunk[0], chunk[1]
    if isinstance(chunk, dict):
        if "type" in chunk and isinstance(chunk.get("data"), dict):
            return chunk["type"], chunk["data"]
        return "custom", chunk
    return None


def _custom_to_sse(data: dict) -> ServerSentEvent | None:
    kind = data.get("kind")
    if kind == "token":
        return _sse("token", {"content": data.get("text", "")})
    if kind == "tool_start":
        return _sse(
            "tool_start",
            {"tool_call_id": data.get("tool_call_id"), "name": data.get("name"), "args": data.get("args") or {}},
        )
    if kind == "tool_end":
        return _sse(
            "tool_end",
            {
                "tool_call_id": data.get("tool_call_id"),
                "name": data.get("name"),
                "ok": data.get("ok"),
                "summary": str(data.get("summary") or "")[:MAX_TOOL_CONTENT_LENGTH],
            },
        )
    if kind == "node":
        return _sse("node", {"node": data.get("node"), "phase": data.get("phase")})
    if kind == "intent":
        return _sse(
            "intent",
            {"intent": data.get("intent"), "route": data.get("route"), "confidence": data.get("confidence")},
        )
    if kind == "order_options":
        return _sse("order_options", {"orders": data.get("orders") or []})
    if kind == "resolved":
        return _sse(
            "resolved",
            {
                "resolved": data.get("resolved", ""),
                "canonical": data.get("canonical", ""),
                "changed": bool(data.get("changed")),
                "order_id": data.get("order_id") or None,
            },
        )
    if kind == "gate":
        return _sse(
            "gate",
            {
                "passed": data.get("passed"),
                "reason": data.get("reason", ""),
                "top_score": data.get("top_score", 0.0),
                # CH09:三信号合成的置信度。top_score 是旧口径,两个都给便于对照
                "confidence": data.get("confidence", 0.0),
            },
        )
    if kind == "interrupt":
        # 写操作挂起:前端拿这个事件渲染确认卡片,点按钮再调 /chat/resume
        return _sse(
            "interrupt",
            {
                "type": data.get("type", ""),
                "tool": data.get("tool", ""),
                "tool_call_id": data.get("tool_call_id"),
                "ticket_type": data.get("ticket_type", ""),
                "description": data.get("description", ""),
            },
        )
    return None


async def _plan_context(
    session: AsyncSession, conv, message: str
) -> tuple[LayerPlan, str, ContextBudget]:
    """算出这一轮发给模型的精简版历史 + 梗概(需求 1/4)。

    **历史一律从 messages 表读**,不从 checkpoint 的 State 读:锚点比的是 `messages.id`,
    只有库里的行才有这个 id。State 里那份全量消息负责「完整历史落盘」和跨节点传话,
    两件事互不相干(需求 5)。
    """
    settings = get_settings()
    budget: ContextBudget = budget_from_settings(
        build_agent_system_prompt(max_steps=settings.agent_max_steps)
    )
    rows = await message_repo.list_by_conversation(session, conv.id)
    converted = _to_langchain(rows)
    sizes = len(converted)
    items = [
        HistoryItem(msg_id=rows[i].id, message=converted[i]) for i in range(min(len(rows), sizes))
    ]
    summary = await conversation_summary_repo.joined_text(session, conv.id)

    plan = plan_layers(
        items,
        summary_upto_msg_id=int(conv.summary_upto_msg_id or 0),
        layer1_from_msg_id=int(conv.layer1_from_msg_id or 0),
        layer1_budget=budget.layer1,
        layer2_budget=budget.layer2,
        assistant_head=settings.layer2_assistant_head,
    )
    if plan.demoted:
        old = int(conv.layer1_from_msg_id or 0)
        # 降级只挪锚点、不搬数据(需求 1)
        await conversation_summary_repo.advance_anchors(
            session,
            conv,
            summary_upto_msg_id=int(conv.summary_upto_msg_id or 0),
            layer1_from_msg_id=plan.layer1_from_msg_id,
        )
        logger.info(
            "layer1 降级 %s→%s(挪 %s 条进层2),层1 预算 %s",
            old,
            plan.layer1_from_msg_id,
            plan.demoted,
            budget.layer1,
        )
    # 每轮必打,闲聊兜底轮也打(需求 6)
    logger.info(render_history_ctx(plan, summary=summary, question=message))
    return plan, summary, budget


async def _schedule_summary(session_factory, conv, plan: LayerPlan, budget: ContextBudget) -> None:
    """层 2 超预算才起后台摘要;失败不影响本轮(摘要下一轮还会再触发)。

    模型槽位留空(用默认抽取模型);要换成别的模型就走
    `summary_service.schedule_summary` 传 `model=`。
    """
    from app.services.summary_service import schedule_summary

    schedule_summary(
        conversation_id=conv.id,
        session_factory=session_factory,
        layer2_tokens=plan.tokens_layer2_raw,
        layer2_budget=budget.layer2,
        layer1_from_msg_id=plan.layer1_from_msg_id,
    )


def build_default_deps(session_factory) -> AgentDeps:
    """生产依赖:真模型 + 默认工具注册表 + 默认检索器 + 内存 checkpointer。"""
    from app.agent.checkpointer import get_checkpointer
    from app.core.llm import get_chat_model, get_extract_model

    settings = get_settings()
    return AgentDeps(
        model=get_chat_model(),
        registry=build_default_registry(),
        session_factory=session_factory,
        intent_model=get_extract_model(),
        check_model=get_extract_model(),
        understand_model=get_extract_model(),
        expand_model=get_extract_model(),
        kb_min_score=settings.kb_min_score,
        min_confidence=settings.evidence_min_confidence,
        max_steps=settings.agent_max_steps,
        token_budget=settings.agent_token_budget,
        checkpointer=get_checkpointer(),
    )


async def _persist_trail(session: AsyncSession, conversation_id: int, trail: list, *, commit: bool = False) -> str | None:
    """把 Agent 的工具轨迹按协议落库:assistant(tool_calls) + tool 逐条对号。

    默认不提交:轨迹要与同一轮的 user/assistant 消息一起提交,由 _persist_turn 收口。
    """
    for msg in trail:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            await message_repo.add_message(
                session,
                conversation_id,
                role="assistant",
                content=str(msg.content or "") or None,
                tool_calls=[
                    {"name": tc["name"], "args": tc.get("args") or {}, "id": tc["id"], "type": "tool_call"}
                    for tc in msg.tool_calls
                ],
                commit=commit,
            )
        else:
            await message_repo.add_message(
                session,
                conversation_id,
                role="tool",
                content=str(msg.content or "")[:MAX_TOOL_CONTENT_LENGTH],
                tool_call_id=getattr(msg, "tool_call_id", None),
                commit=commit,
            )
    return None


async def chat_with_agent(
    session_id: str,
    message: str,
    conversation_id: int | None = None,
    *,
    new_conversation: bool = False,
    order_id: str | None = None,
    resume: str | dict | None = None,
    session: AsyncSession,
    deps: AgentDeps | None = None,
    graph=None,
) -> AsyncGenerator[ServerSentEvent, None]:
    """跑一轮:建/取会话 → 读历史 → 图执行(边跑边推事件)→ 落库 → 下发结果。

    `resume` 非空 = 用户刚在工单预览卡片上点了确认/取消,从 checkpoint 接着跑:
    不重新规划上下文(同一轮的历史没变),不重复落 user 消息,只把决定递回图里。
    """
    from app.db import get_sessionmaker

    session_factory = get_sessionmaker()
    deps = deps or build_default_deps(session_factory)
    app_graph = graph or build_agent_graph(deps)

    # MCP 工具按 TTL 刷新一次(通常命中缓存):Server 侧加了工具、只重启 Server,
    # 这边下一轮就能用上,不用重启客服服务(需求 1)
    settings_now = get_settings()
    if settings_now.mcp_enabled:
        try:
            from app.core.tools.mcp_source import get_mcp_source, refresh_registry

            await refresh_registry(deps.registry, get_mcp_source())
        except Exception as exc:  # noqa: BLE001 - MCP 挂了不能拖垮内置工具
            logger.warning("MCP 工具刷新失败(继续用上次清单):%s", exc)

    async with _get_lock(session_id):
        try:
            conv = await conversation_repo.get_or_create(
                session, conversation_id, session_id, force_new=new_conversation
            )
        except Exception as exc:
            yield _sse("error", {"code": "DB_ERROR", "message": _sanitize(str(exc))})
            return

        token_ctx = current_conversation_id.set(conv.id)
        citation_ctx = current_citations.set(CitationBucket())
        try:
            if resume:
                # 同一轮的续跑:历史与预算都没变,不用重算(重算还会把锚点再推一次)
                plan, summary, budget = LayerPlan(), "", None
            else:
                try:
                    plan, summary, budget = await _plan_context(session, conv, message)
                except Exception as exc:
                    yield _sse("error", {"code": "PROMPT_ERROR", "message": _sanitize(str(exc))})
                    return

            # 上游真跑起来之后才落 user:失败不留半截(CH01 语义)
            final_state: dict = {}
            graph_input = (
                Command(resume=resume)
                if resume
                else {
                    **_TURN_RESET,
                    "user_input": message,
                    "messages": [HumanMessage(content=message)],
                    "history": plan.layer2 + plan.layer1,
                    "layer2_count": len(plan.layer2),
                    "summary": summary,
                    "user_key": session_id,
                    "order_id": order_id or "",
                }
            )
            # CH09: 一轮对话 = 一条 trace。上下文包住 async for——astream 是
            # 异步生成器,图在迭代时才跑;关闭时 yield None,链路零感知。
            with tracing.open_chat_trace(
                conversation_id=conv.id, session_id=session_id
            ) as chat_span:
                run_config = {
                    "configurable": {"thread_id": str(conv.id)},
                    "callbacks": tracing.get_langfuse_callbacks(),
                    "metadata": tracing.build_trace_metadata(
                        conversation_id=conv.id, session_id=session_id
                    ),
                }
                try:
                    async for chunk in app_graph.astream(
                        graph_input,
                        run_config,
                        stream_mode="custom",
                        version="v2",
                    ):
                        unpacked = _unpack(chunk)
                        if unpacked is None:
                            continue
                        mode, data = unpacked
                        if mode != "custom":
                            continue
                        if data.get("kind") == "intent":
                            # 意图/路由是图跑出来的,到了就补写进 trace 元数据
                            tracing.record_intent_route(
                                chat_span,
                                intent=data.get("intent"),
                                route=data.get("route"),
                                confidence=data.get("confidence"),
                            )
                        frame = _custom_to_sse(data)
                        if frame is not None:
                            yield frame
                    snapshot = await app_graph.aget_state({"configurable": {"thread_id": str(conv.id)}})
                    final_state = dict(snapshot.values)
                    # 图停在 interrupt 上等确认:此时 next 非空,这一轮没跑完
                    pending = tuple(getattr(snapshot, "next", ()) or ())
                    if final_state.get("finish_reason") == "abstain":
                        tracing.mark_abstain(chat_span)
                except Exception as exc:
                    logger.exception("agent graph failed")
                    # token 可能已经推给前端了:这一轮必须留下痕迹,不能凭空消失
                    err = await _persist_interrupted(session, conv.id, message)
                    if err:
                        logger.warning("中断轮落库失败:%s", err)
                    yield _sse("error", {"code": "UPSTREAM_ERROR", "message": _sanitize(str(exc))})
                    return

            err = await _persist_turn(session, conv.id, message, final_state)
            if err:
                yield _sse("error", {"code": "DB_ERROR", "message": err})
                return

            if pending:
                # 挂起等确认:不落 done 的终态,前端凭 interrupted 保留卡片
                yield _sse(
                    "done",
                    {
                        "session_id": session_id,
                        "conversation_id": conv.id,
                        "finish_reason": "awaiting_confirm",
                        "interrupted": True,
                        "intent": final_state.get("intent"),
                        "route": final_state.get("route"),
                        "actions": [],
                        "order_options": [],
                    },
                )
                return

            # 层 2 攒到超预算就在后台压一段:不 await,不阻塞这一轮的回复(需求 2)
            if budget is not None:
                await _schedule_summary(session_factory, conv, plan, budget)

            # CH09:按意图记 token 估算。旁路——写失败只记日志,不影响这一轮的答复
            prompt_tokens, completion_tokens = usage_service.estimate_turn(
                final_state, message=message
            )
            await usage_service.record_turn(
                session,
                intent=final_state.get("intent"),
                route=final_state.get("route"),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                conversation_id=conv.id,
                model=get_settings().llm_model,
            )

            citations = final_state.get("citations") or []
            if not citations:
                # 业务数据类没走预检索,但 Agent 自己调了 query_faq:引用走旁路拿回来
                bucket = current_citations.get()
                if bucket is not None and bucket.items:
                    citations = [c.to_payload() for c in bucket.items]
            if citations:
                yield _sse("citations", {"citations": citations})

            yield _sse(
                "done",
                {
                    "session_id": session_id,
                    "conversation_id": conv.id,
                    "finish_reason": final_state.get("finish_reason") or "stop",
                    "intent": final_state.get("intent"),
                    "route": final_state.get("route"),
                    "actions": final_state.get("actions") or [],
                    "order_options": final_state.get("order_options") or [],
                    "gate": {
                        "passed": final_state.get("gate_passed"),
                        "reason": final_state.get("gate_reason") or "",
                    },
                },
            )
        finally:
            current_citations.reset(citation_ctx)
            current_conversation_id.reset(token_ctx)


async def _persist_turn(session: AsyncSession, conversation_id: int, message: str, state: dict) -> str | None:
    """user → 工具轨迹 → 终答,顺序与协议一致;DB 异常转成脱敏文本返回,不抛。

    一轮 = 一个事务:全部 flush 完再 commit,任何一步失败整体 rollback。
    拆成 N 次提交的写法会留下半截历史(可能带不配对的 tool_calls),而脏数据
    不会自愈——下一轮回灌 LLM 时就按这份错历史推上下文。

    resume(用户点了确认/取消之后接着跑)传空的 message:这一轮的 user 消息在
    发问那次就落过了,再落一条会让历史里出现两句一样的话。
    """
    try:
        if message:
            await message_repo.add_message(
                session, conversation_id, role="user", content=message, commit=False
            )
        trail = state.get("trail") or []
        if trail:
            await _persist_trail(session, conversation_id, trail)
        reply = state.get("reply") or ""
        if reply:
            await message_repo.add_message(
                session, conversation_id, role="assistant", content=reply, commit=False
            )
        await session.commit()
    except Exception as exc:
        await session.rollback()
        return _sanitize(str(exc))

    # 入池是旁路:放在本轮事务之后,失败只记日志,不能把已经答完的一轮回滚掉
    if state.get("finish_reason") == "abstain":
        reason = state.get("gate_reason") or ""
        source = "retrieval_low_conf" if reason in (REASON_NO_HITS, REASON_SCORE_LOW, REASON_CONFIDENCE_LOW) else "self_check"
        detail = state.get("gate_detail") or ""
        try:
            # CH09：入池统一走 flywheel.enqueue，并把当轮召回片段一起带上——
            # 审核人靠快照判断「知识库真缺」还是「有但没检到」
            await flywheel.enqueue(
                session,
                raw_question=state.get("resolved_query") or message,
                source=source,
                reason=(f"{reason}: {detail}"[:255] if detail else reason or None),
                conversation_id=conversation_id,
                snapshot=flywheel.sanitize_snapshot(state.get("citations") or []),
            )
        except Exception as exc:  # noqa: BLE001 - 入池失败不影响答复
            logger.warning("低置信度问题入池失败:%s", exc)
    return None


INTERRUPTED_REPLY = "（本轮回复因上游异常中断，没有给出完整答复，请重新提问）"


async def _persist_interrupted(session: AsyncSession, conversation_id: int, message: str) -> str | None:
    """图跑到一半抛异常时,把这一轮按「已发生但中断」落库。

    流的 token 已经推给前端了,用户看到过半截回答——这时库里如果什么都不落,
    下一轮读历史就会缺这一轮,用户说「刚才那个」系统完全接不上。
    落一条明确的占位而不是空:回灌 LLM 时它知道这轮没答成,不会当成正常上下文。
    """
    try:
        if message:
            await message_repo.add_message(
                session, conversation_id, role="user", content=message, commit=False
            )
        await message_repo.add_message(
            session, conversation_id, role="assistant", content=INTERRUPTED_REPLY, commit=False
        )
        await session.commit()
        return None
    except Exception as exc:
        await session.rollback()
        return _sanitize(str(exc))
