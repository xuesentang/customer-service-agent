"""演示日冒烟:一条命令回答"现在能不能演"。

    python scripts/demo_smoke.py            # 用现有演示库跑 8 条探针
    python scripts/demo_smoke.py --rebuild  # 先重建干净演示库,再跑
    python scripts/demo_smoke.py --port 18127

链路:
  1. (--rebuild)重建 data/demo.db + data/demo_milvus.db:建表 → 30 条 FAQ
     种子 → 15 篇知识文档切块嵌入(BGE-M3 + Milvus Lite)。演示库与开发库
     (ch07.db 等)隔离,反复跑冒烟不会污染任何历史数据。
  2. 起服务:uvicorn 进程内线程,devstack 参数(SQLite + Milvus Lite);
     Langfuse / MCP 强制关——冒烟只验聊天主链路,观测与 MCP 有各自验收。
  3. 8 条探针顺序打 /api/v1/chat/stream(真模型、真库、真 SSE),断言事件序列。
  4. 产出 reports/demo_smoke.md;有 FAIL 退出码非零。

探针是"能演"的最低标准:服务起得来、key 有效、五个出口的事件形态正确、
闸与写操作确认流在工作。答得好不好(人话质量、多轮自然度)不归它管——
那些只能人读,见 docs/测试与演示方案.md 第 3 节手动清单。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# 必须在 import app.* 之前进环境(settings 带 lru_cache)。演示库与 .env 里的
# 开发库(MySQL/ch07)完全隔离:冒烟用 SQLite + Milvus Lite 文件库,不依赖 Docker。
DEMO_DB_URL = "sqlite+aiosqlite:///./data/demo.db"
DEMO_VECTOR_URI = "./data/demo_milvus.db"
DEMO_DB_FILE = REPO_ROOT / "data" / "demo.db"
DEMO_VECTOR_FILE = REPO_ROOT / "data" / "demo_milvus.db"

os.environ["DATABASE_URL"] = DEMO_DB_URL
os.environ["VECTOR_STORE_URI"] = DEMO_VECTOR_URI
os.environ["LANGFUSE_ENABLED"] = "false"  # 冒烟不验观测;开着会往真实 Cloud 发测试 trace
os.environ["MCP_ENABLED"] = "false"  # MCP 要另起两个 Server,不进主链路冒烟

TURN_TIMEOUT = 240.0  # 单轮上限:真模型 + 多步 ReAct,慢一点比误杀好
READY_TIMEOUT = 300.0  # 首启上限:lifespan 要加载嵌入模型 + 向量库集合


# ------------------------------------------------------------------ 重建演示库


def _run_step(cmd: list[str]) -> None:
    print(f"  $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, env=os.environ.copy(), check=True)


def rebuild_demo_db() -> None:
    """重建干净演示库。只删本脚本自己产出的两个文件,不碰任何其他库。"""
    print("[rebuild] 清理旧演示文件(仅 demo.db / demo_milvus.db)", flush=True)
    DEMO_DB_FILE.unlink(missing_ok=True)
    DEMO_VECTOR_FILE.unlink(missing_ok=True)

    print("[rebuild] 建表(模型即 schema,SQLite 无需 DDL 脚本)", flush=True)

    async def _create_all() -> None:
        from app.db import get_engine
        from app.models import Base

        engine = get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await engine.dispose()

    asyncio.run(_create_all())

    print("[rebuild] FAQ 种子(30 条)", flush=True)
    _run_step([sys.executable, "scripts/seed_faq.py", "--database-url", DEMO_DB_URL])

    print("[rebuild] 知识文档落盘(已存在则跳过)", flush=True)
    _run_step([sys.executable, "scripts/seed_knowledge.py"])

    print("[rebuild] 切块 + 嵌入 + 向量库(BGE-M3,约 1-3 分钟)", flush=True)
    _run_step(
        [
            sys.executable,
            "scripts/build_knowledge.py",
            "--docs",
            "data/knowledge_docs",
            "--database-url",
            DEMO_DB_URL,
            "--vector-store-uri",
            DEMO_VECTOR_URI,
            "--reset-store",
        ]
    )
    print("[rebuild] done", flush=True)


# ------------------------------------------------------------------ 服务起停


def start_server(port: int):
    import uvicorn

    from app.main import app

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, name="demo-smoke-uvicorn", daemon=True).start()
    return server


def wait_ready(server, port: int, timeout: float = READY_TIMEOUT) -> bool:
    import httpx

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.should_exit:
            return False
        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/", timeout=5.0, trust_env=False)
            if resp.status_code == 200:
                return True
        except Exception:  # noqa: BLE001 - 没起来就继续等
            pass
        time.sleep(2.0)
    return False


# ------------------------------------------------------------------ SSE 收发


@dataclass
class Turn:
    events: list[tuple[str, dict]] = field(default_factory=list)

    def of(self, kind: str) -> list[dict]:
        return [d for k, d in self.events if k == kind]

    def last(self, kind: str) -> dict:
        got = self.of(kind)
        return got[-1] if got else {}

    def answer(self) -> str:
        return "".join(d.get("content", "") for d in self.of("token"))

    def tools_called(self) -> list[str]:
        return [d.get("name", "") for d in self.of("tool_start")]


async def _stream_turn(client, url: str, payload: dict) -> Turn:
    turn = Turn()
    event_name, data_lines = None, []

    def flush() -> None:
        nonlocal event_name, data_lines
        if event_name is not None:
            body = "\n".join(data_lines)
            try:
                data = json.loads(body) if body else {}
            except json.JSONDecodeError:
                data = {"_raw": body}
            turn.events.append((event_name, data))
        event_name, data_lines = None, []

    async with client.stream("POST", url, json=payload, timeout=TURN_TIMEOUT) as resp:
        resp.raise_for_status()
        async for raw in resp.aiter_lines():
            line = raw.strip()
            if not line:
                flush()
                continue
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
    flush()
    return turn


async def run_turn(
    client,
    port: int,
    session_id: str,
    message: str,
    conversation_id: int | None = None,
    new_conversation: bool = False,
) -> Turn:
    payload: dict = {"session_id": session_id, "message": message}
    if conversation_id is not None:
        payload["conversation_id"] = conversation_id
    if new_conversation:
        payload["new_conversation"] = True
    return await _stream_turn(
        client, f"http://127.0.0.1:{port}/api/v1/chat/stream", payload
    )


async def run_resume(
    client,
    port: int,
    session_id: str,
    conversation_id: int,
    decision: str,
    ticket_type: str,
    description: str,
) -> Turn:
    payload = {
        "session_id": session_id,
        "conversation_id": conversation_id,
        "decision": decision,
        "ticket_type": ticket_type,
        "description": description,
    }
    return await _stream_turn(
        client, f"http://127.0.0.1:{port}/api/v1/chat/resume", payload
    )


def count_tickets(conversation_id: int) -> int:
    import sqlite3

    con = sqlite3.connect(DEMO_DB_FILE)
    try:
        return con.execute(
            "select count(*) from tickets where conversation_id = ?", (conversation_id,)
        ).fetchone()[0]
    finally:
        con.close()


# ------------------------------------------------------------------ 断言框架


@dataclass
class ProbeResult:
    name: str
    passed: bool
    seconds: float
    checks: list[tuple[bool, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    conv_id: int | None = None


class Probe:
    """一条探针:若干轮对话 + 断言。check(ok, desc) 两条腿——报告里逐条列。"""

    def __init__(self, name: str):
        self.name = name
        self.checks: list[tuple[bool, str]] = []
        self.notes: list[str] = []
        self.conv_id: int | None = None

    def check(self, ok: bool, desc: str) -> bool:
        self.checks.append((bool(ok), desc))
        return bool(ok)

    def result(self, t0: float) -> ProbeResult:
        return ProbeResult(
            name=self.name,
            passed=all(ok for ok, _ in self.checks),
            seconds=time.monotonic() - t0,
            checks=list(self.checks),
            notes=list(self.notes),
            conv_id=self.conv_id,
        )


def _summary_line(turn: Turn) -> str:
    intent, gate, done = turn.last("intent"), turn.last("gate"), turn.last("done")
    parts = [f"intent={intent.get('intent')}/{intent.get('route')}"]
    if intent.get("confidence") is not None:
        parts.append(f"conf={intent.get('confidence')}")
    if gate:
        parts.append(f"gate={'过' if gate.get('passed') else '拦'}({gate.get('top_score')})")
    tools = turn.tools_called()
    if tools:
        parts.append("tools=" + ",".join(tools))
    if done:
        parts.append(f"finish={done.get('finish_reason')}")
    return " ".join(parts)


# ------------------------------------------------------------------ 8 条探针


async def probe_01_knowledge(client, port: int) -> ProbeResult:
    """商品咨询:hybrid 检索 → 过闸 → 引用。

    问句刻意选自商品规格(W20 宽楦),不用「邮费是多少」:后者在当前模型的
    意图分类下落「其他」→ business 路,Agent 会反问「哪种邮费」而不检索
    (2026-10-25 冒烟实测)。运费类问法的召回由 ch03 召回 eval 直接覆盖
    (14/14),这条探针验的是知识路端到端:意图→检索→闸→引用→答复。
    """
    p = Probe("P1 商品咨询(检索+闸+引用)")
    t0 = time.monotonic()
    turn = await run_turn(client, port, "smoke-1", "W20 有宽楦版本吗", new_conversation=True)
    intent, gate, done = turn.last("intent"), turn.last("gate"), turn.last("done")
    p.check(intent.get("route") == "knowledge", f"路由=knowledge(实际 {intent.get('route')})")
    p.check(
        any(d.get("node") == "retrieve_knowledge" and d.get("phase") == "exit" for d in turn.of("node")),
        "retrieve_knowledge 节点进出",
    )
    p.check(gate.get("passed") is True, f"置信度闸放过(top_score={gate.get('top_score')})")
    citations = turn.last("citations").get("citations", [])
    p.check(len(citations) > 0, f"带引用(实际 {len(citations)} 条)")
    p.check(len(turn.answer()) > 10, "答复非空")
    p.check(not turn.of("error"), "无 error 事件")
    p.conv_id = done.get("conversation_id")
    p.notes.append(_summary_line(turn))
    return p.result(t0)


async def probe_02_business(client, port: int, conv_id: int) -> ProbeResult:
    """业务查询:Agent 自己调物流工具,多步 ReAct。"""
    p = Probe("P2 查物流(多步工具)")
    t0 = time.monotonic()
    turn = await run_turn(client, port, "smoke-1", "订单1001的物流到哪了", conversation_id=conv_id)
    intent = turn.last("intent")
    p.check(intent.get("route") == "business", f"路由=business(实际 {intent.get('route')})")
    tools = turn.tools_called()
    p.check("query_logistics" in tools, f"调用 query_logistics(实际 {tools})")
    ends = turn.of("tool_end")
    p.check(len(ends) >= 1 and all(d.get("ok") for d in ends), "工具全部成功")
    p.check(not turn.of("error"), "无 error 事件")
    p.conv_id = turn.last("done").get("conversation_id") or conv_id
    p.notes.append(_summary_line(turn))
    return p.result(t0)


async def probe_03_reference(client, port: int, conv_id: int) -> ProbeResult:
    """指代消解 + 退款子流程:先聊 1003 的物流,再问'这个能退吗'。"""
    p = Probe("P3 指代消解+退光子流程")
    t0 = time.monotonic()
    turn = await run_turn(client, port, "smoke-1", "这个能退吗", conversation_id=conv_id)
    resolved, intent = turn.last("resolved"), turn.last("intent")
    p.check(resolved.get("changed") is True, f"指代被补全(实际 changed={resolved.get('changed')})")
    p.check("1003" in str(resolved.get("resolved", "")), f"补全结果含 1003({resolved.get('resolved')!r})")
    p.check(intent.get("route") == "refund", f"路由=refund(实际 {intent.get('route')})")
    p.check(
        any(d.get("node") == "refund_subflow" and d.get("phase") == "exit" for d in turn.of("node")),
        "refund_subflow 节点进出",
    )
    p.check(not turn.of("error"), "无 error 事件")
    p.conv_id = turn.last("done").get("conversation_id") or conv_id
    p.notes.append(_summary_line(turn))
    return p.result(t0)


async def probe_04_complaint(client, port: int) -> ProbeResult:
    """投诉:固定安抚 + 两个升级按钮(转人工/建工单),不进 Agent。"""
    p = Probe("P4 投诉(升级按钮)")
    t0 = time.monotonic()
    turn = await run_turn(
        client, port, "smoke-2", "我要投诉你们,订单1001的耳机收到就是坏的", new_conversation=True
    )
    intent, done = turn.last("intent"), turn.last("done")
    p.check(intent.get("route") == "complaint", f"路由=complaint(实际 {intent.get('route')})")
    action_ids = [a.get("id") for a in done.get("actions", [])]
    p.check("transfer_human" in action_ids, f"带转人工按钮(实际 {action_ids})")
    p.check("create_ticket" in action_ids, f"带建工单按钮(实际 {action_ids})")
    p.check(not turn.tools_called(), "投诉路不调工具")
    p.conv_id = done.get("conversation_id")
    p.notes.append(_summary_line(turn))
    return p.result(t0)


async def probe_05_chitchat(client, port: int, conv_id: int) -> ProbeResult:
    """闲聊:固定话术,零工具调用。"""
    p = Probe("P5 闲聊(零工具)")
    t0 = time.monotonic()
    turn = await run_turn(client, port, "smoke-2", "你好呀", conversation_id=conv_id)
    intent, done = turn.last("intent"), turn.last("done")
    p.check(intent.get("route") == "chitchat", f"路由=chitchat(实际 {intent.get('route')})")
    p.check(not turn.tools_called(), "零工具调用")
    p.check(done.get("finish_reason") == "chitchat", f"finish=chitchat(实际 {done.get('finish_reason')})")
    p.conv_id = done.get("conversation_id") or conv_id
    p.notes.append(_summary_line(turn))
    return p.result(t0)


async def probe_06_abstain(client, port: int) -> ProbeResult:
    """拒答:库里没有的知识点,闸必须在生成前拦下。"""
    p = Probe("P6 拒答(闸拦截)")
    t0 = time.monotonic()
    turn = await run_turn(client, port, "smoke-3", "W60 用的碳板是第几代", new_conversation=True)
    gate, done = turn.last("gate"), turn.last("done")
    p.check(bool(turn.of("gate")), "出现过 gate 事件")
    p.check(gate.get("passed") is False, f"闸拦下(top_score={gate.get('top_score')})")
    p.check(done.get("finish_reason") == "abstain", f"finish=abstain(实际 {done.get('finish_reason')})")
    p.notes.append(_summary_line(turn))
    return p.result(t0)


async def probe_07_need_order(client, port: int) -> ProbeResult:
    """缺单号:退款路先出订单选择器,不瞎猜。"""
    p = Probe("P7 缺单号(订单选择器)")
    t0 = time.monotonic()
    turn = await run_turn(client, port, "smoke-4", "我要退货", new_conversation=True)
    options = turn.last("order_options").get("orders", [])
    done = turn.last("done")
    p.check(len(options) == 3, f"3 笔候选订单(实际 {len(options)})")
    p.check(done.get("finish_reason") == "need_order", f"finish=need_order(实际 {done.get('finish_reason')})")
    p.notes.append(_summary_line(turn))
    return p.result(t0)


async def probe_08_confirm(client, port: int) -> ProbeResult:
    """写操作确认流:interrupt 卡片 → 用户确认 → 才落库。

    真模型可能第一轮不提建单(先追问细节),允许补一句再要;两轮都没有卡片
    才算失败——那说明演示脚本里这条路径不稳,得换问法。
    """
    p = Probe("P8 写操作确认流(interrupt→resume→落库)")
    t0 = time.monotonic()
    session, conv_id = "smoke-5", None
    turn = await run_turn(
        client,
        port,
        session,
        "订单1001显示已签收但我根本没收到货,帮我建个工单让专员核实",
        new_conversation=True,
    )
    cards = turn.of("interrupt")
    conv_id = turn.last("done").get("conversation_id", conv_id)
    if not cards:
        turn = await run_turn(
            client, port, session, "直接建工单吧,类型选投诉,就按刚才说的情况", conversation_id=conv_id
        )
        cards = turn.of("interrupt")
        conv_id = turn.last("done").get("conversation_id", conv_id)
    p.check(bool(cards), "出现 interrupt 确认卡片")
    if not cards:
        p.notes.append("两轮都没出卡片:" + _summary_line(turn))
        return p.result(t0)

    card = cards[-1]
    p.check(bool(card.get("description")), "卡片带描述")
    p.notes.append(f"卡片:{card.get('ticket_type')} | {str(card.get('description'))[:60]}")

    resumed = await run_resume(
        client,
        port,
        session,
        conv_id,
        "approve",
        card.get("ticket_type", ""),
        card.get("description", ""),
    )
    rdone = resumed.last("done")
    p.check(not resumed.of("error"), "resume 无 error 事件")
    p.check(
        rdone.get("finish_reason") != "awaiting_confirm",
        f"恢复后跑完(实际 {rdone.get('finish_reason')})",
    )
    tickets = count_tickets(conv_id)
    p.check(tickets == 1, f"tickets 表落 1 行(实际 {tickets})")
    p.notes.append(_summary_line(resumed))
    return p.result(t0)


# ------------------------------------------------------------------ 主流程


def write_report(results: list[ProbeResult], port: int, model: str, path: Path) -> None:
    passed = sum(1 for r in results if r.passed)
    lines = [
        "# 演示冒烟报告",
        "",
        f"- 时间:{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "- 演示库:`data/demo.db` + `data/demo_milvus.db`(SQLite + Milvus Lite)",
        f"- 服务:127.0.0.1:{port}(Langfuse / MCP 关闭)",
        f"- 模型:{model}",
        f"- 结果:**{passed}/{len(results)} PASS**",
        "",
        "| 探针 | 结果 | 耗时 | 断言与关键事件 |",
        "|---|---|---|---|",
    ]
    for r in results:
        mark = "PASS" if r.passed else "**FAIL**"
        cells = [f"{'✅' if ok else '❌'} {desc}" for ok, desc in r.checks] + r.notes
        lines.append(f"| {r.name} | {mark} | {r.seconds:.1f}s | {'<br>'.join(cells)} |")
    lines += [
        "",
        "判定:全 PASS = 现在能演。任何 FAIL 先修再演示——见 docs/测试与演示方案.md 第 4 节门禁。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


async def amain(port: int) -> int:
    import httpx

    from app.config import get_settings

    model = get_settings().llm_model

    print(f"[smoke] 起服务 127.0.0.1:{port}(预热嵌入模型,首启约 1-2 分钟)...", flush=True)
    server = start_server(port)
    if not wait_ready(server, port):
        print(
            "[smoke] 服务没起来:检查 DATABASE_URL / VECTOR_STORE_URI / LLM key,"
            "或手动起服务看日志",
            flush=True,
        )
        server.should_exit = True
        return 2

    results: list[ProbeResult] = []
    async with httpx.AsyncClient(trust_env=False) as client:
        r1 = await probe_01_knowledge(client, port)
        results.append(r1)
        # P2/P3 显式沿用 P1 的会话(指代消解要的就是跨轮上下文)
        r2 = await probe_02_business(client, port, r1.conv_id)
        results.append(r2)
        await run_turn(client, port, "smoke-1", "订单1003的物流到哪了", conversation_id=r1.conv_id)
        r3 = await probe_03_reference(client, port, r1.conv_id)
        results.append(r3)
        r4 = await probe_04_complaint(client, port)
        results.append(r4)
        r5 = await probe_05_chitchat(client, port, r4.conv_id)
        results.append(r5)
        r6 = await probe_06_abstain(client, port)
        results.append(r6)
        r7 = await probe_07_need_order(client, port)
        results.append(r7)
        r8 = await probe_08_confirm(client, port)
        results.append(r8)

    server.should_exit = True
    time.sleep(1.0)

    report = REPO_ROOT / "reports" / "demo_smoke.md"
    write_report(results, port, model, report)

    print()
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        print(f"  [{mark}] {r.name}  {r.seconds:.1f}s")
        for note in r.notes:
            print(f"         {note}")
        for ok, desc in r.checks:
            if not ok:
                print(f"         ❌ {desc}")
    passed = sum(1 for r in results if r.passed)
    print(f"\n[smoke] {passed}/{len(results)} PASS,报告:{report}")
    return 0 if passed == len(results) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="演示日冒烟:8 条探针验证现在能不能演")
    parser.add_argument("--rebuild", action="store_true", help="先重建干净演示库再跑")
    parser.add_argument("--port", type=int, default=18099)
    args = parser.parse_args()

    if args.rebuild:
        rebuild_demo_db()
    return asyncio.run(amain(args.port))


if __name__ == "__main__":
    raise SystemExit(main())
