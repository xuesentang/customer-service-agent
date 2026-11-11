"""图版对话服务的单测:事件序、落库轨迹、拒答入池、动作下发。"""

from __future__ import annotations

from types import SimpleNamespace

from sqlalchemy import select

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.tools import tool

from app.agent import graph as graph_module
from app.agent.graph import AgentDeps
from app.core.tools.registry import ToolRegistry
from app.models.message import Message
from app.repositories import low_confidence_repo, message_repo
from app.services.agent_chat import chat_with_agent


class IntentModel:
    def __init__(self, intent: str):
        self.intent = intent

    async def ainvoke(self, messages, **kwargs):
        return AIMessage(content='{"intent": "%s", "confidence": 0.9}' % self.intent)


class CheckModel:
    def __init__(self, useful=True):
        self.useful = useful

    async def ainvoke(self, messages, **kwargs):
        return AIMessage(content='{"useful": %s, "reason": "r"}' % str(self.useful).lower())


class StreamAgentModel:
    """有工具就走一步,没有就直接收敛。"""

    def __init__(self, texts=("好的,", "已为您查到。"), with_tool=False):
        self.texts = list(texts)
        self.with_tool = with_tool
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def astream(self, messages):
        self.calls += 1
        if self.with_tool and self.calls == 1:
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[{"name": "fake_order", "args": '{"order_id": "1001"}', "id": "c1", "index": 0}],
            )
            return
        for t in self.texts:
            yield AIMessageChunk(content=t)


@tool
def fake_order(order_id: str) -> str:
    """查订单。order_id: 订单号"""
    return f"订单 {order_id} 已发货"


def _deps(intent="售后", with_tool=False, useful=True, scores=(0.7,)) -> AgentDeps:
    reg = ToolRegistry()
    reg.register(fake_order)
    return AgentDeps(
        model=StreamAgentModel(with_tool=with_tool),
        registry=reg,
        session_factory=None,
        intent_model=IntentModel(intent),
        check_model=CheckModel(useful),
        kb_min_score=0.45,
    )


def _patch_retrieve(monkeypatch, scores=(0.7,)):
    async def fake_retrieve(**kwargs):
        return [
            SimpleNamespace(
                chunk=SimpleNamespace(
                    id=1,
                    section_path="退货退款政策 > 运费规则",
                    questions="退货运费谁承担",
                    answer="七日无理由退货的寄回运费由消费者承担。",
                ),
                score=scores[0],
            )
        ]

    monkeypatch.setattr(graph_module, "retrieve", fake_retrieve)


async def _run(session, deps, text="订单1001的物流到哪了", **kw):
    return [
        e
        async for e in chat_with_agent("s1", text, session=session, deps=deps, **kw)
    ]


async def test_business_route_streams_tokens_and_done(session, monkeypatch):
    _patch_retrieve(monkeypatch)
    events = await _run(session, _deps("物流"))
    kinds = [e.event for e in events]
    assert "token" in kinds and kinds[-1] == "done"
    assert "node" in kinds  # 节点进出可观测
    done = events[-1].data
    assert done["intent"] == "物流"
    assert done["route"] == "business"
    assert done["actions"] == []
    assert done["finish_reason"] == "stop"


async def test_tool_events_carry_ids_and_persist_trail(session, monkeypatch):
    _patch_retrieve(monkeypatch)
    events = await _run(session, _deps("物流", with_tool=True))
    starts = [e for e in events if e.event == "tool_start"]
    ends = [e for e in events if e.event == "tool_end"]
    assert starts and ends
    assert starts[0].data["tool_call_id"] == "c1"
    assert ends[0].data["ok"] is True

    conv_id = events[-1].data["conversation_id"]
    msgs = await message_repo.list_by_conversation(session, conv_id)
    assert [m.role for m in msgs] == ["user", "assistant", "tool", "assistant"]
    assert msgs[1].tool_calls[0]["id"] == "c1"
    assert msgs[2].tool_call_id == "c1"


async def test_knowledge_route_emits_citations(session, monkeypatch):
    _patch_retrieve(monkeypatch)
    events = await _run(session, _deps("商品咨询"), text="退货运费谁出")
    cite = [e for e in events if e.event == "citations"]
    assert cite, "知识类命中必须下发引用"
    assert cite[0].data["citations"][0]["n"] == 1
    assert cite[0].data["citations"][0]["section_path"] == "退货退款政策 > 运费规则"


async def test_abstain_writes_pool_and_returns_actions(session, monkeypatch):
    _patch_retrieve(monkeypatch, scores=(0.3,))
    events = await _run(session, _deps("商品咨询"), text="帮我写首诗")
    done = events[-1].data
    assert done["finish_reason"] == "abstain"
    assert [a["id"] for a in done["actions"]] == ["transfer_human", "create_ticket"]
    assert done["gate"]["passed"] is False

    rows = await low_confidence_repo.list_by_source(session, "retrieval_low_conf")
    assert len(rows) == 1
    assert "confidence_low" in (rows[0].reason or "")


async def test_selfcheck_block_goes_to_self_check_entry(session, monkeypatch):
    _patch_retrieve(monkeypatch, scores=(0.7,))
    events = await _run(session, _deps("商品咨询", useful=False), text="W40 楦宽多少毫米")
    assert events[-1].data["gate"]["reason"] == "self_check"
    rows = await low_confidence_repo.list_by_source(session, "self_check")
    assert len(rows) == 1


async def test_complaint_returns_two_buttons_and_no_tool_calls(session, monkeypatch):
    _patch_retrieve(monkeypatch)
    events = await _run(session, _deps("投诉"), text="我要投诉")
    done = events[-1].data
    assert [a["id"] for a in done["actions"]] == ["transfer_human", "create_ticket"]
    assert not [e for e in events if e.event == "tool_start"]


async def test_chitchat_has_no_citations_and_no_tools(session, monkeypatch):
    _patch_retrieve(monkeypatch)
    events = await _run(session, _deps("闲聊"), text="你好")
    kinds = [e.event for e in events]
    assert "citations" not in kinds
    assert events[-1].data["route"] == "chitchat"


async def test_graph_failure_yields_error_event(session, monkeypatch):
    _patch_retrieve(monkeypatch)

    class Boom:
        def bind_tools(self, tools):
            return self

        async def astream(self, messages):
            raise RuntimeError("upstream down")
            yield  # pragma: no cover

    deps = _deps("物流")
    deps.model = Boom()
    events = await _run(session, deps)
    assert events[-1].event == "error"
    assert events[-1].data["code"] == "UPSTREAM_ERROR"
    # token 可能已经推给前端:这一轮必须留痕,否则下一轮读历史凭空少一轮
    rows = (await session.execute(select(Message).order_by(Message.id))).scalars().all()
    assert [r.role for r in rows] == ["user", "assistant"]
    assert "中断" in rows[-1].content


async def test_persist_turn_rolls_back_as_a_whole(session, monkeypatch):
    """一轮落库是一个事务:第 2 条写失败,第 1 条也不能留下。

    逐条 commit 的写法会留下「有 user 没 assistant」或「tool_calls 不配对」的
    半截历史,而脏数据不会自愈——下一轮回灌 LLM 就按错历史推上下文。
    """
    from app.repositories import conversation_repo
    from app.services import agent_chat as service

    conv = await conversation_repo.get_or_create(session, None, "s-tx")
    conv_id = conv.id  # rollback 会让 ORM 对象过期,先取出来
    real = message_repo.add_message
    calls = {"n": 0}

    async def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("database is locked")
        return await real(*args, **kwargs)

    monkeypatch.setattr(service.message_repo, "add_message", flaky)
    err = await service._persist_turn(
        session, conv_id, "订单1001到哪了", {"reply": "已发货", "finish_reason": "stop"}
    )

    assert err is not None  # 脱敏后的错误文本
    assert calls["n"] == 2
    rows = await message_repo.list_by_conversation(session, conv_id)
    assert rows == [], "第 2 条失败时第 1 条必须一起回滚"


async def test_second_turn_does_not_inherit_previous_turn_state(session, monkeypatch):
    """回归:checkpointer 按会话留 State,上一轮的字段会漏到本轮 done 里。

    第六章联调时真实发生:本轮闸没过走拒答,前端却同时收到上一轮的订单卡片。
    修法是在服务层每轮显式重置本轮字段(`_TURN_RESET`),这条锁住它。
    """
    _patch_retrieve(monkeypatch)
    monkeypatch.setattr(
        graph_module,
        "candidate_orders",
        lambda user_key, limit=3: [
            SimpleNamespace(to_payload=lambda: {"order_id": "1001", "product": "耳机", "quantity": 1, "amount": 199.0, "status": "已签收", "order_date": "2026-06-01"})
        ],
    )

    deps = _deps("退款退货")  # 第一轮:缺订单号 → 出订单卡片
    turn1 = await _run(session, deps, text="这个能退吗")
    done1 = turn1[-1].data
    assert done1["finish_reason"] == "need_order"
    assert len(done1["order_options"]) == 1

    deps.intent_model = IntentModel("物流")  # 第二轮:换成业务数据类
    conv_id = done1["conversation_id"]
    turn2 = await _run(session, deps, text="订单 1001 到哪了", conversation_id=conv_id)
    done2 = turn2[-1].data
    assert done2["order_options"] == [], "上一轮的订单卡片不许漏到本轮"
    assert done2["finish_reason"] == "stop"
    assert done2["route"] == "business"
