"""客服工作流的图骨架(第五章第 2 点)。

    START → 指代消解 → 意图识别 ─┬─ 知识类 → 检索 → 置信度闸 ─┬─ 过闸 → 主力 Agent ─┐
                                 │                              └─ 不过 → 兜底话术 ─┤
                                 ├─ 业务数据类 → 主力 Agent ───────────────────────┤
                                 ├─ 投诉 → 安抚 + 两个选项 ───────────────────────┤
                                 └─ 闲聊 → 固定话术 ───────────────────────────────┘
                                                                                   ↓
                                                                              日志记录 → END

分工很明确:**能确定的事写进图**(分流、检索、过闸、兜底),**不确定的事才交给模型**
(判意图、决定调哪个工具、组织语言)。图是骨架,Agent 是骨架里的一个节点——不是反过来。

CH06 在这张骨架上加了三处:
- 指代消解不再是透传,和问法归一合成一次模型调用(`query_understanding`);
- 退款退货与售后从「知识类」拆出独立出口 `refund_subflow`:订单事实 + 多路政策检索 + 闸,
  只把「这一单能不能退」交给 Agent;
- 缺订单号时流程停在 `order_select` 态,把候选订单卡片交前端点选,不让模型猜单号。

节点保持纯函数:不碰数据库、不碰 SSE。它们只读写 State,把要推给前端的东西扔进
`get_stream_writer()`,把要落库的东西留在 State 里让服务层去写。这样每个节点都能
单独 ainvoke 测,不必先起一个服务。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.agent.gate import check_gate, citations_from_payload
from app.agent.intent import (
    ROUTE_BUSINESS,
    ROUTE_CHITCHAT,
    ROUTE_COMPLAINT,
    ROUTE_KNOWLEDGE,
    ROUTE_REFUND,
    classify_intent,
)
from app.agent.query_understanding import understand_query
from app.agent.react import run_react
from app.agent.refund_flow import NEED_ORDER_REPLY, prepare_refund
from app.core.mock_orders import candidate_orders
from app.core.tools.executor import ToolExecutor
from app.rag.citations import build_citations, format_evidence
from app.rag.retriever import retrieve
from app.rag.selfcheck import ABSTAIN_REPLY

logger = logging.getLogger(__name__)

# 知识类检索取多少条证据送进 Agent(ch04 的 Top-10 口径)
EVIDENCE_TOP_K = 10

COMPLAINT_REPLY = (
    "非常抱歉给您带来了不好的体验,您说的情况我记下了。"
    "这个问题需要专人跟进,您可以选择转人工客服直接沟通,"
    "或者我帮您建一张工单转给售后团队处理——两个都可以选,也可以都选。"
)

CHITCHAT_REPLY = (
    "您好呀,我是小智,电商平台的在线客服。"
    "查订单、看物流、退换货、问商品参数都可以找我,您想了解什么?"
)

# 两个动作互不绑定:前端各渲染一个按钮,后端只给建议、不自动执行(第 8 点)
ACTION_TRANSFER = {"id": "transfer_human", "label": "转人工"}
ACTION_REFUND = {"id": "apply_refund", "label": "申请退款"}
ACTION_TICKET = {"id": "create_ticket", "label": "建工单"}
ESCALATION_ACTIONS = [ACTION_TRANSFER, ACTION_TICKET]


class AgentState(TypedDict, total=False):
    """一轮对话的完整状态。只装这一轮需要的东西,历史靠 MySQL。"""

    # 输入
    user_input: str
    # CH07:全量对话消息挂 add_messages reducer——各节点只管吐自己新产出的消息,
    # 框架按顺序并入,并由 checkpoint 落盘。喂给模型的是另一份裁过、带梗概的精简版
    # (`history` + `summary`),两者互不影响(需求 5)。
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str  # 早期对话梗概(空 = 还没压过)
    history: list[BaseMessage]  # 服务层注入的裁剪后历史(每轮覆盖,不用 reducer)
    layer2_count: int  # history 里前 n 条是层 2(只给日志分层用,不影响发给模型的顺序)
    user_key: str  # 用户键(session_id),用于取「这个人的近期订单」
    order_id: str  # 前端点选订单卡片回填的槽位
    # 理解层
    canonical_query: str
    understanding_changed: bool
    understanding_fallback: bool
    slot_order_id: str
    # 分流
    resolved_query: str
    intent: str
    intent_confidence: float
    intent_fallback: bool
    route: str
    # 退款子流程
    order_facts: str
    order_options: list[dict]
    route_queries: list[str]  # 扩写出来的检索查询(留痕用)
    # 检索与闸
    citations: list[dict]
    top_score: float
    gate_passed: bool
    gate_reason: str
    gate_detail: str
    gate_confidence: float  # CH09:三信号合成的置信度,与 top_score 一起留作复盘
    # 产物
    reply: str
    finish_reason: str
    actions: list[dict]
    trail: list[BaseMessage]
    steps: int
    tokens: int
    stopped_by: str
    trace: dict


@dataclass
class AgentDeps:
    """图要用到的外部依赖。全部可注入 = 单测能整张图跑假模型。"""

    model: object
    registry: object
    session_factory: object
    intent_model: object | None = None
    check_model: object | None = None
    understand_model: object | None = None  # 指代消解 + 问法归一
    expand_model: object | None = None  # 查询扩写
    retriever: object | None = None
    kb_min_score: float = 0.45
    min_confidence: float = 0.45  # CH09: 三信号合成的置信度阈值(校准结论见 config)
    max_steps: int = 4
    token_budget: int = 6000
    checkpointer: object = field(default_factory=InMemorySaver)


def _emit(payload: dict) -> None:
    """节点 → 外部的事件旁路。没有 writer(直接 ainvoke)时是空实现。"""
    writer = get_stream_writer()
    if writer is not None:
        writer(payload)


def _emit_node(name: str, phase: str) -> None:
    _emit({"kind": "node", "node": name, "phase": phase})


def _emit_token(text: str) -> None:
    if text:
        _emit({"kind": "token", "text": text})


# 用户点「确认提交」时前端回传的取值。给宽松一点:中英文几个常见说法都认
_APPROVE_WORDS = {"approve", "approved", "yes", "ok", "true", "1", "确认", "确认提交", "同意"}


async def _confirm_write(call: dict, writer):
    """写操作的确认闸:把预览推给前端,挂起图,等 resume 带决定回来。

    顺序不能反——先把 `interrupt` 事件推出去(前端要靠它渲染预览卡片),
    再调 `interrupt()` 挂起。反过来的话卡片永远出不来,用户没东西可点。

    没有 writer(直接 ainvoke)时也照样挂起:宁可挂住,也不能静默执行写操作。

    返回三种值,`run_react` 按它决定执行什么:
    - `False` 拒绝;
    - `dict` 同意,且**以用户看到的那份参数为准**——图恢复时节点会从头重跑
      (LangGraph 的 interrupt 语义),模型很可能把描述改了个措辞。用户点的是
      卡片上那句话,就提交那句话;拿重跑后的版本来写库等于替用户改了主意。
    - `True` 同意,沿用这次的参数(前端没回传内容时的退路)。
    """
    from langgraph.types import interrupt

    args = call.get("args") or {}
    preview = {
        "kind": "interrupt",
        "type": "ticket_confirm",
        "tool": call.get("name"),
        "tool_call_id": call.get("id"),
        # 这两项就是卡片上要显示的内容(需求 7:工单类型 + 问题描述)
        "ticket_type": args.get("ticket_type") or "",
        "description": args.get("description") or "",
    }
    if writer is not None:
        writer(preview)
    resolved = interrupt(preview)

    if isinstance(resolved, dict):
        if str(resolved.get("decision") or "").strip().lower() not in _APPROVE_WORDS:
            return False
        seen = {
            "ticket_type": resolved.get("ticket_type") or "",
            "description": resolved.get("description") or "",
        }
        return seen if all(seen.values()) else True

    approved = str(resolved or "").strip().lower() in _APPROVE_WORDS
    return True if approved else False


def build_agent_graph(deps: AgentDeps):
    """编译图。deps 用闭包捕获——节点保持纯函数,不必知道依赖从哪来。"""

    # ------------------------------------------------------------ 节点
    async def resolve_reference(state: AgentState) -> dict:
        """指代消解 + 问法归一(一次调用),顺带抽订单号槽位。

        问题已经完整时原样透传——改写不是目的,把一句好问题揉一遍只会让检索更差。
        """
        _emit_node("resolve_reference", "enter")
        raw = state.get("user_input", "")
        understood = await understand_query(
            raw,
            history=state.get("history") or [],
            summary=state.get("summary") or "",
            model=deps.understand_model,
        )
        # 前端点选订单卡片回填的槽位优先于模型抽出来的
        slot = state.get("order_id") or understood.order_id
        _emit_node("resolve_reference", "exit")
        _emit(
            {
                "kind": "resolved",
                "resolved": understood.resolved,
                "canonical": understood.canonical,
                "changed": understood.changed,
                "order_id": slot,
            }
        )
        return {
            "resolved_query": understood.resolved,
            "canonical_query": understood.canonical,
            "understanding_changed": understood.changed,
            "understanding_fallback": understood.fallback,
            "slot_order_id": slot or "",
        }

    async def classify(state: AgentState) -> dict:
        _emit_node("classify_intent", "enter")
        result = await classify_intent(
            state.get("resolved_query", ""), model=deps.intent_model, history=state.get("history")
        )
        _emit_node("classify_intent", "exit")
        _emit({"kind": "intent", "intent": result.intent, "route": result.route, "confidence": result.confidence})
        return {
            "intent": result.intent,
            "route": result.route,
            "intent_confidence": result.confidence,
            "intent_fallback": result.fallback,
        }

    async def retrieve_knowledge(state: AgentState) -> dict:
        """知识类强制检索:这一步是确定性编排的核心,绕过它就没有证据。"""
        _emit_node("retrieve_knowledge", "enter")
        query = state.get("canonical_query") or state.get("resolved_query", "")
        hits = []
        try:
            hits = await retrieve(
                session_factory=deps.session_factory,
                query=query,
                top_k=EVIDENCE_TOP_K,
                retriever=deps.retriever,
            )
        except Exception as exc:  # noqa: BLE001 - 检索挂了也要往下走,交给闸拒答
            logger.warning("知识类检索失败,按无证据处理:%s", exc)
        citations = build_citations(hits)
        _emit_node("retrieve_knowledge", "exit")
        return {
            "citations": [c.to_payload() for c in citations],
            "top_score": citations[0].score if citations else 0.0,
        }

    async def confidence_gate(state: AgentState) -> dict:
        """置信度闸:证据弱就到此为止,不进 Agent(答复是流式的,进去就晚了)。"""
        _emit_node("confidence_gate", "enter")
        citations = citations_from_payload(state.get("citations") or [])
        result = await check_gate(
            state.get("resolved_query", ""),
            citations,
            min_score=deps.kb_min_score,
            min_confidence=deps.min_confidence,
            model=deps.check_model,
        )
        _emit_node("confidence_gate", "exit")
        _emit(
            {
                "kind": "gate",
                "passed": result.passed,
                "reason": result.reason,
                "top_score": round(result.top_score, 4),
                "confidence": round(result.confidence, 4),
            }
        )
        return {
            "gate_passed": result.passed,
            "gate_confidence": round(result.confidence, 4),
            "gate_reason": result.reason,
            "gate_detail": result.detail,
        }

    async def refund_subflow(state: AgentState) -> dict:
        """退款/售后:订单事实 + 多路政策检索 + 闸。缺订单号就在这里停住出选择器。"""
        _emit_node("refund_subflow", "enter")
        prep = await prepare_refund(
            state.get("canonical_query") or state.get("resolved_query", ""),
            order_id=state.get("slot_order_id") or None,
            registry=deps.registry,
            executor=ToolExecutor(deps.registry, session_factory=deps.session_factory),
            session_factory=deps.session_factory,
            retriever=deps.retriever,
            expand_model=deps.expand_model,
            # 不传 min_score:退款路用自己的门槛(settings.refund_gate_min_score),
            # 两路的分数分布不同,共用 kb_min_score 会让退款请求全被判成证据不足。
            # 也不传 check_model:退款路不走自评(见 prepare_refund 文档)。
        )
        _emit_node("refund_subflow", "exit")

        if prep.need_order:
            options = [
                o.to_payload() for o in candidate_orders(state.get("user_key") or "anonymous", limit=3)
            ]
            _emit({"kind": "order_options", "orders": options})
            _emit_token(NEED_ORDER_REPLY)
            return {
                "order_options": options,
                "reply": NEED_ORDER_REPLY,
                "messages": [AIMessage(content=NEED_ORDER_REPLY)],
                "finish_reason": "need_order",
                "actions": [],
                "gate_passed": None,
            }

        citations = [c.to_payload() for c in prep.citations]
        gate = prep.gate
        _emit(
            {
                "kind": "gate",
                "passed": bool(gate and gate.passed),
                "reason": gate.reason if gate else "no_gate",
                "top_score": round(gate.top_score, 4) if gate else 0.0,
            }
        )
        return {
            "order_facts": prep.order_facts,
            "citations": citations,
            "top_score": citations[0]["score"] if citations else 0.0,
            "gate_passed": bool(gate and gate.passed),
            "gate_reason": gate.reason if gate else "",
            "gate_detail": gate.detail if gate else "",
            "route_queries": prep.queries,
        }

    async def agent(state: AgentState) -> dict:
        """主力 Agent:ReAct 循环,工具复用第 2 章那五个。"""
        _emit_node("agent", "enter")
        citations = citations_from_payload(state.get("citations") or [])
        policies = format_evidence(citations) if citations else ""
        order_facts = state.get("order_facts") or ""
        evidence = ""
        if order_facts:
            evidence += f"【订单事实】\n{order_facts}\n\n"
        if policies:
            evidence += f"【政策条款】\n{policies}"
        writer = get_stream_writer()
        outcome = await run_react(
            state.get("resolved_query", ""),
            model=deps.model,
            registry=deps.registry,
            # 带 session_factory 才会落审计:写操作被拒也要留痕(需求 5)
            executor=ToolExecutor(deps.registry, session_factory=deps.session_factory),
            history=state.get("history") or [],
            summary=state.get("summary") or "",
            evidence=evidence,
            layer2_count=int(state.get("layer2_count") or 0),
            max_steps=deps.max_steps,
            token_budget=deps.token_budget,
            writer=writer,
            confirm_write=lambda call: _confirm_write(call, writer),
        )
        _emit_node("agent", "exit")
        # 退款子流程里已经拿到订单事实 = 这一单能走到「申请退款」这一步,给个入口;
        # 是否真的可退由 Agent 的回答说明,按钮只是把动作递到用户手上
        actions = [ACTION_REFUND] if state.get("route") == ROUTE_REFUND and state.get("order_facts") else []
        new_messages = list(outcome.trail)
        if outcome.reply:
            new_messages.append(AIMessage(content=outcome.reply))
        return {
            "reply": outcome.reply,
            "finish_reason": "stop",
            "trail": outcome.trail,
            "messages": new_messages,
            "steps": outcome.model_calls,
            "tokens": outcome.tokens,
            "stopped_by": outcome.stopped_by,
            "actions": actions,
        }

    async def abstain(state: AgentState) -> dict:
        """证据不够:固定兜底话术 + 两个出口。拒答不给下一步等于把人晾在原地。"""
        _emit_node("abstain", "enter")
        _emit_token(ABSTAIN_REPLY)
        _emit_node("abstain", "exit")
        return {
            "reply": ABSTAIN_REPLY,
            "messages": [AIMessage(content=ABSTAIN_REPLY)],
            "finish_reason": "abstain",
            "actions": list(ESCALATION_ACTIONS),
            "gate_passed": False,
        }

    async def complaint_reply(state: AgentState) -> dict:
        """投诉:零模型调用。安抚 + 两个可选项,后端不替用户做决定。"""
        _emit_node("complaint_reply", "enter")
        _emit_token(COMPLAINT_REPLY)
        _emit_node("complaint_reply", "exit")
        return {
            "reply": COMPLAINT_REPLY,
            "messages": [AIMessage(content=COMPLAINT_REPLY)],
            "finish_reason": "complaint",
            "actions": list(ESCALATION_ACTIONS),
        }

    async def chitchat_reply(state: AgentState) -> dict:
        """闲聊:固定话术,一次模型都不调——确定性编排最直接的价值。"""
        _emit_node("chitchat_reply", "enter")
        _emit_token(CHITCHAT_REPLY)
        _emit_node("chitchat_reply", "exit")
        return {
            "reply": CHITCHAT_REPLY,
            "messages": [AIMessage(content=CHITCHAT_REPLY)],
            "finish_reason": "chitchat",
            "actions": [],
        }

    async def log_trace(state: AgentState) -> dict:
        """日志记录节点:一行结构化日志,不建表(本章无新表)。"""
        _emit_node("log_trace", "enter")
        trace = {
            "intent": state.get("intent"),
            "route": state.get("route"),
            "intent_confidence": round(float(state.get("intent_confidence") or 0.0), 3),
            "intent_fallback": bool(state.get("intent_fallback")),
            "gate_passed": state.get("gate_passed"),
            "gate_reason": state.get("gate_reason") or "",
            "top_score": round(float(state.get("top_score") or 0.0), 4),
            "steps": state.get("steps", 0),
            "tokens": state.get("tokens", 0),
            "stopped_by": state.get("stopped_by", ""),
            "finish_reason": state.get("finish_reason", ""),
            "actions": [a["id"] for a in state.get("actions") or []],
        }
        logger.info("agent.trace %s", trace)
        _emit_node("log_trace", "exit")
        return {"trace": trace}

    # ------------------------------------------------------------ 条件边
    def route_after_intent(state: AgentState) -> str:
        route = state.get("route") or ROUTE_BUSINESS
        return {
            ROUTE_KNOWLEDGE: "retrieve_knowledge",
            ROUTE_REFUND: "refund_subflow",
            ROUTE_COMPLAINT: "complaint_reply",
            ROUTE_CHITCHAT: "chitchat_reply",
            ROUTE_BUSINESS: "agent",
        }.get(route, "agent")

    def route_after_refund(state: AgentState) -> str:
        """缺订单号就停在这里等用户点选;材料备齐才放行给 Agent。"""
        if state.get("finish_reason") == "need_order":
            return "log_trace"
        return "agent" if state.get("gate_passed") else "abstain"

    def route_after_gate(state: AgentState) -> str:
        return "agent" if state.get("gate_passed") else "abstain"

    # ------------------------------------------------------------ 组装
    graph = StateGraph(AgentState)
    graph.add_node("resolve_reference", resolve_reference)
    graph.add_node("classify_intent", classify)
    graph.add_node("retrieve_knowledge", retrieve_knowledge)
    graph.add_node("refund_subflow", refund_subflow)
    graph.add_node("confidence_gate", confidence_gate)
    graph.add_node("agent", agent)
    graph.add_node("abstain", abstain)
    graph.add_node("complaint_reply", complaint_reply)
    graph.add_node("chitchat_reply", chitchat_reply)
    graph.add_node("log_trace", log_trace)

    graph.add_edge(START, "resolve_reference")
    graph.add_edge("resolve_reference", "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        ["retrieve_knowledge", "refund_subflow", "agent", "complaint_reply", "chitchat_reply"],
    )
    graph.add_edge("retrieve_knowledge", "confidence_gate")
    graph.add_conditional_edges("confidence_gate", route_after_gate, ["agent", "abstain"])
    graph.add_conditional_edges(
        "refund_subflow", route_after_refund, ["agent", "abstain", "log_trace"]
    )
    for terminal in ("agent", "abstain", "complaint_reply", "chitchat_reply"):
        graph.add_edge(terminal, "log_trace")
    graph.add_edge("log_trace", END)

    return graph.compile(checkpointer=deps.checkpointer)


@dataclass
class GraphRun:
    """一次图执行的结果 + 耗时,服务层拿它拼 SSE 的 done 事件。"""

    state: dict
    elapsed_ms: int = 0
    started: float = field(default_factory=time.perf_counter)

    def finish(self) -> "GraphRun":
        self.elapsed_ms = int((time.perf_counter() - self.started) * 1000)
        return self
