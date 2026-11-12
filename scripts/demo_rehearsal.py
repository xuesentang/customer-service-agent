"""演示彩排:把演示脚本里每一幕按顺序打一遍真服务,记下耗时与事件。

    python scripts/demo_rehearsal.py                 # 打开发台库(986 块)+ Milvus Lite
    python scripts/demo_rehearsal.py --list          # 只列幕单,不起服务
    python scripts/demo_rehearsal.py --skip-flywheel # 不跑飞轮那几幕(不动库)
    python scripts/demo_rehearsal.py --langfuse      # 保留观测(默认关,别污染 dashboard)

和 demo_smoke.py 的分工:冒烟验「八个出口都还在」,彩排验「演示时我要说的这套词
能不能一条条走顺、各花多久」。真正的交付物是 reports/demo_rehearsal.md —— 演示脚本
里每一幕的耗时和现象都来自这里,不是估的。

飞轮那几幕会改库(入池 → 进待审 → 通过写知识库),这是演示本身要展示的效果;
反复跑时拒答那一幕会因为「这题已经补过了」直接答上而失败,换 FLYWHEEL_CANDIDATES
里的下一条再跑。
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEV_DB_URL = "sqlite+aiosqlite:///./data/ch07.db"
DEV_VECTOR_URI = "./data/ch07_milvus.db"

# 飞轮闭环用的题:库里没有、意图判成「商品咨询」所以会走检索、闸会拦下、业务上答得出。
# 「可以带上飞机吗」这类问法会被判成「其他」→ business 路,压根不检索、也不入池,
# 演不出闸和飞轮(2026-11-12 彩排踩过)。备选按离库程度排序,前一条被补过就换下一条。
FLYWHEEL_CANDIDATES = [
    "你们卖儿童安全座椅吗",
    "你们的睡袋能定制颜色吗",
    "你们的帐篷能印 logo 吗",
]
FLYWHEEL_KEYWORD = "安全座椅"  # 在待审队列里认那条新缺口用的(要够独特,别撞上老问题)

# 「人工」在审核台填的标准答案(演示时这段由人写,彩排里给一份等价的)
FLYWHEEL_ANSWER = (
    "我们目前不销售儿童安全座椅。经营品类只有鞋靴、户外装备、小家电与宠物用品,"
    "汽车用品不在范围内,也没有相关的安装或适配服务。"
    "需要安全座椅建议到母婴或汽车用品渠道购买,并按身高体重组别选择。"
    "已购商品如涉及儿童使用场景的尺码问题,可按 7 天无理由退换(吊牌完好)。"
)

_SMOKE = None
_DB_FILE: Path | None = None


def smoke():
    """复用 demo_smoke 的 SSE 收发与服务起停,不在两个脚本里各写一份。

    只在第一次调用时执行它的模块级代码(它会把环境切到演示库),之后由
    `_retarget` 改成彩排目标——放在函数里是因为 settings 带 lru_cache,
    谁先设环境谁说了算。
    """
    global _SMOKE
    if _SMOKE is None:
        spec = importlib.util.spec_from_file_location(
            "demo_smoke", REPO_ROOT / "scripts" / "demo_smoke.py"
        )
        module = importlib.util.module_from_spec(spec)
        # 必须先进 sys.modules:demo_smoke 里用了 @dataclass,
        # dataclasses 会回查 sys.modules[cls.__module__] 拿注解
        sys.modules["demo_smoke"] = module
        spec.loader.exec_module(module)
        _SMOKE = module
    return _SMOKE


def _reset_flywheel(db_url: str, vector_uri: str, keyword: str) -> dict:
    """调 scripts/flywheel_reset.py 的 reset(),不重复写一遍删库删向量的逻辑。"""
    spec = importlib.util.spec_from_file_location(
        "flywheel_reset", REPO_ROOT / "scripts" / "flywheel_reset.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["flywheel_reset"] = mod
    spec.loader.exec_module(mod)
    return mod.reset(db_url, vector_uri, keyword)


@dataclass
class Scene:
    """一幕 = 一句话(或一次确认) + 若干期望。期望只写「演的时候要指给人看」的那几条。"""

    sid: str
    title: str
    message: str
    session: str
    new_conv: bool = False
    follow: str | None = None  # 复用哪一幕的会话(指代消解要的就是这个)
    resume: bool = False  # 这一幕是点「确认」而不是发消息
    expect: dict = field(default_factory=dict)


SCENES: list[Scene] = [
    Scene(
        "A1",
        "知识库问型号差异(一次问两件事)",
        "LP500 和 LP300 差在哪？我一天出三十杯左右该选哪个",
        "rehearsal-a",
        new_conv=True,
        expect={"route": "knowledge", "gate": True, "citations_min": 1},
    ),
    Scene(
        "A2",
        "同话题深挖(除垢周期)",
        "咖啡机多久除一次垢",
        "rehearsal-a",
        follow="A1",
        expect={"gate": True, "citations_min": 1},
    ),
    Scene(
        "A3",
        "查物流(Agent 自己调工具)",
        "订单1003的物流到哪了",
        "rehearsal-a",
        follow="A2",
        expect={"route": "business", "tools": ["query_logistics"]},
    ),
    Scene(
        "A4",
        "指代消解 + 改址政策",
        "那能改地址吗",
        "rehearsal-a",
        follow="A3",
        expect={"resolved_changed": True, "resolved_has": "1003"},
    ),
    Scene(
        "B1",
        "退款判断:还没签收的订单",
        "订单1003能退吗",
        "rehearsal-b",
        new_conv=True,
        expect={"route": "refund", "gate": True, "citations_min": 1},
    ),
    Scene(
        "B2",
        "退款判断:签收很久的订单(要说不)",
        "那订单1007呢",
        "rehearsal-b",
        follow="B1",
        expect={"route": "refund"},
    ),
    Scene(
        "B3",
        "缺订单号 → 订单选择器(不许猜)",
        "我要退货",
        "rehearsal-c",
        new_conv=True,
        expect={"order_options": 3, "finish": "need_order"},
    ),
    Scene(
        "C1",
        "投诉 → 固定安抚 + 升级按钮",
        "我要投诉，订单1001的耳机拆开就是坏的",
        "rehearsal-d",
        new_conv=True,
        expect={"route": "complaint", "actions": ["transfer_human", "create_ticket"]},
    ),
    Scene(
        "C2",
        "建工单:确认卡片(要判成 business 才走 Agent)",
        "订单1001的包裹好像被放错地方了,帮我建个工单让专员查",
        "rehearsal-d2",
        new_conv=True,
        expect={"interrupt": True},
    ),
    Scene(
        "C3",
        "点确认 → 才落库",
        "",
        "rehearsal-d2",
        follow="C2",
        resume=True,
        expect={"no_error": True},
    ),
    Scene(
        "D1",
        "拒答:库里没有的知识点",
        FLYWHEEL_CANDIDATES[0],
        "rehearsal-e",
        new_conv=True,
        expect={"gate": False, "finish": "abstain"},
    ),
]

# 这一幕要等飞轮补完库才跑,所以不放在上面的顺序表里
CLOSING = Scene(
    "D4",
    "补库后换个问法再问(飞轮闭环)",
    "你们有卖儿童安全座椅吗",
    "rehearsal-e",
    follow="D1",
    expect={"gate": True, "citations_min": 1, "section_has": "飞轮补充"},
)


def _sqlite_file(url: str) -> Path:
    tail = url.split("///", 1)[-1]
    return Path(tail) if Path(tail).is_absolute() else REPO_ROOT / tail.lstrip("./")


def count_rows(table: str, where: str = "", params: tuple = ()) -> int:
    import sqlite3

    assert _DB_FILE is not None
    con = sqlite3.connect(_DB_FILE)
    try:
        sql = f"select count(*) from {table}" + (f" where {where}" if where else "")
        return con.execute(sql, params).fetchone()[0]
    finally:
        con.close()


# ------------------------------------------------------------------ 断言


def check_scene(scene: Scene, turn, conv_id: int | None) -> tuple[bool, list[str], dict]:
    """按 expect 逐条判,返回 (是否全过, 人读的检查结果, 报告用的小结)。"""
    intent, gate, done = turn.last("intent"), turn.last("gate"), turn.last("done")
    summary = {"route": intent.get("route"), "gate": gate.get("passed"),
               "finish": done.get("finish_reason")}
    out: list[str] = []
    ok_all = True

    def check(cond, text: str) -> None:
        nonlocal ok_all
        ok_all = ok_all and bool(cond)
        out.append(("✅ " if cond else "❌ ") + text)

    exp = scene.expect
    if "route" in exp:
        check(summary["route"] == exp["route"], f"路由={exp['route']}(实际 {summary['route']})")
    if "gate" in exp:
        check(bool(gate) and summary["gate"] is exp["gate"],
              f"闸{'放过' if exp['gate'] else '拦下'}(top1={gate.get('top_score')})")
    if "citations_min" in exp:
        n = len(turn.last("citations").get("citations", []))
        check(n >= exp["citations_min"], f"带引用 {n} 条")
    if "section_has" in exp:
        # 闭环要验的不是「答上了」,而是「答的是刚补进去的那条」——
        # 否则检索命中一条沾边的老块也会被判成成功
        paths = [c.get("section_path", "") for c in turn.last("citations").get("citations", [])]
        hit = [p for p in paths if exp["section_has"] in p]
        check(bool(hit), f"引用含「{exp['section_has']}」(实际 {(paths[:1] or ['无'])})")
    if "tools" in exp:
        called = turn.tools_called()
        for name in exp["tools"]:
            check(name in called, f"调用 {name}(实际 {called})")
        ends = turn.of("tool_end")
        check(bool(ends) and all(d.get("ok") for d in ends), "工具全部成功")
    if "resolved_changed" in exp:
        resolved = turn.last("resolved")
        check(resolved.get("changed") is True, f"指代被补全({resolved.get('resolved')!r})")
    if "resolved_has" in exp:
        resolved = turn.last("resolved")
        check(exp["resolved_has"] in str(resolved.get("resolved", "")), "补全结果含订单号")
    if "order_options" in exp:
        n = len(turn.last("order_options").get("orders", []))
        check(n == exp["order_options"], f"{n} 张订单卡")
    if "actions" in exp:
        ids = [a.get("id") for a in done.get("actions", [])]
        for aid in exp["actions"]:
            check(aid in ids, f"按钮 {aid}(实际 {ids})")
    if "interrupt" in exp:
        check(bool(turn.of("interrupt")), "出现确认卡片")
    if "finish" in exp:
        check(summary["finish"] == exp["finish"], f"finish={exp['finish']}(实际 {summary['finish']})")
    if exp.get("no_error") or "finish" not in exp:
        check(not turn.of("error"), "无 error 事件")
    if scene.resume and conv_id is not None:
        n = count_rows("tickets", "conversation_id = ?", (conv_id,))
        check(n >= 1, f"tickets 表落 {n} 行")
    return ok_all, out, summary


# ------------------------------------------------------------------ 飞轮


def run_flywheel_batch(db_url: str, vector_uri: str) -> tuple[bool, list[str]]:
    """入池 → 标准化/查重 → 待审。真跑一遍批量脚本,不模拟。"""
    before_review = count_rows("review_queue")
    before_low = count_rows("low_confidence_questions")
    proc = subprocess.run(
        [sys.executable, "scripts/run_flywheel.py", "--batch", "20"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "DATABASE_URL": db_url, "VECTOR_STORE_URI": vector_uri},
    )
    ok = proc.returncode == 0
    lines = [
        f"池内原话 {before_low} → {count_rows('low_confidence_questions')} 条;"
        f"待审 {before_review} → {count_rows('review_queue')} 条",
        *((proc.stdout or proc.stderr).strip().splitlines() or ["(无输出)"])[-4:],
    ]
    return ok, lines


async def approve_in_queue(port: int, keyword: str, answer: str) -> tuple[bool, list[str]]:
    """审核台点「通过并写入知识库」:挑出那条新缺口 → 填答案 → 建块 + 向量化。"""
    import httpx

    base = f"http://127.0.0.1:{port}/api/v1/review"
    async with httpx.AsyncClient(trust_env=False, timeout=300) as client:
        queue = (await client.get(f"{base}/queue", params={"limit": 100})).json()
        items = queue.get("items", [])
        if not items:
            return False, ["❌ 待审队列是空的,飞轮那一步没跑成?"]
        # 认不出这条就停:宁可失败,也不要把别人的缺口通过了塞进知识库
        target = next(
            (it for it in items if keyword and keyword in it["normalized_question"]), None
        )
        if target is None:
            names = "、".join(it["normalized_question"][:18] for it in items[:3])
            return False, [f"❌ 待审里找不到含「{keyword}」的那条;队列头部是:{names}"]
        lines = [f"挑中 #{target['id']}:{(target['normalized_question'] or '')[:40]}"]
        resp = await client.post(f"{base}/{target['id']}/approve", json={"answer": answer})
        if resp.status_code != 200:
            return False, [*lines, f"❌ 通过失败 {resp.status_code}:{resp.text[:160]}"]
        data = resp.json()
        lines += [
            f"状态 {data.get('review_status')},新块 {data.get('chunk_id')},"
            f"向量化 {data.get('vectorized') or data.get('vectorized_count') or 'ok'}",
            f"知识库已达 {count_rows('knowledge_chunks')} 块",
        ]
        return True, lines


# ------------------------------------------------------------------ 主流程


async def amain(args) -> int:
    global _DB_FILE

    use = smoke()  # 先加载(它会把环境指向演示库),后面再改成彩排目标
    use._apply_targets(args.database_url, args.vector_store_uri)
    _DB_FILE = _sqlite_file(args.database_url)

    if not args.langfuse:
        os.environ["LANGFUSE_ENABLED"] = "false"
    os.environ["MCP_ENABLED"] = "false"

    if args.list:
        for s in SCENES:
            print(f"  {s.sid}  {s.title}  ← {s.message or '(点确认)'}", flush=True)
        return 0

    print(f"[彩排] 目标 {args.database_url} + {args.vector_store_uri}", flush=True)
    print(
        f"[彩排] 知识库 {count_rows('knowledge_chunks')} 块,"
        f"待审 {count_rows('review_queue', 'review_status = ?', ('待审',))} 条,"
        f"池内 {count_rows('low_confidence_questions')} 条",
        flush=True,
    )

    print("[彩排] 起服务(预热嵌入 + 重排,首启 1-2 分钟)…", flush=True)
    server = use.start_server(args.port)
    if not use.wait_ready(server, args.port):
        print("[彩排] 服务没起来:检查 DATABASE_URL / VECTOR_STORE_URI / LLM key", flush=True)
        server.should_exit = True
        return 2

    results: list[dict] = []
    conv_of: dict[str, int | None] = {}
    card: dict = {}

    picked = {s.strip() for s in args.only.split(",") if s.strip()} if args.only else set()
    scenes = [s for s in SCENES if not picked or s.sid in picked]
    run_flywheel = not args.skip_flywheel and (not picked or "D3" in picked)
    run_closing = run_flywheel and (not picked or CLOSING.sid in picked)

    import httpx

    async def record(scene: Scene, turn, elapsed: float, tag: str = "") -> None:
        conv_id = turn.last("done").get("conversation_id") or conv_of.get(scene.follow or "")
        conv_of[scene.sid] = conv_id
        if turn.of("interrupt"):
            card.update(turn.of("interrupt")[-1])
        passed, checks, summary = check_scene(scene, turn, conv_id)
        row = {
            "sid": scene.sid + tag, "title": scene.title, "passed": passed,
            "seconds": elapsed, "checks": checks, "summary": summary,
            "answer": turn.answer().replace("\n", " ")[:80],
        }
        results.append(row)
        print(
            f"[{'PASS' if passed else 'FAIL'}] {row['sid']} {scene.title}  {elapsed:.1f}s  "
            f"route={summary['route']} gate={summary['gate']} finish={summary['finish']}",
            flush=True,
        )
        for line in checks:
            print(f"        {line}", flush=True)
        if row["answer"]:
            print(f"        答:{row['answer']}", flush=True)

    async with httpx.AsyncClient(trust_env=False) as client:
        async def do_turn(scene: Scene):
            if scene.resume:
                conv = conv_of.get(scene.follow or "")
                if conv is None or not card:
                    return None
                return await use.run_resume(
                    client, args.port, scene.session, conv, "approve",
                    card.get("ticket_type", ""), card.get("description", ""),
                )
            conv = conv_of.get(scene.follow) if scene.follow else None
            return await use.run_turn(
                client, args.port, scene.session, scene.message,
                conversation_id=conv, new_conversation=scene.new_conv,
            )

        for scene in scenes:
            t0 = time.monotonic()
            turn = None
            # 模型侧网络抖动(502 / APIConnectionError)会让这一轮直接吐 error 事件。
            # 那不是系统的毛病,重试一次比记一笔「失败」公平——演示当天也会遇到。
            for attempt in (1, 2):
                try:
                    turn = await do_turn(scene)
                except Exception as exc:  # noqa: BLE001 - 网络/SSE 异常统一重试
                    print(f"  [retry] {scene.sid} 第 {attempt} 次抛错:{type(exc).__name__}", flush=True)
                    turn = None
                if turn is not None and (not turn.of("error") or attempt == 2):
                    break
                if turn is not None:
                    print(f"  [retry] {scene.sid} 出现 error 事件,重试一次", flush=True)
                await asyncio.sleep(2.0)
            if turn is None:
                print(f"[SKIP] {scene.sid} 上一幕没出卡片,没法点确认", flush=True)
                continue
            await record(scene, turn, time.monotonic() - t0)

        if run_flywheel:
            # 飞轮三幕:归并进待审 → 审核通过写库 → 同题换个问法再问
            t0 = time.monotonic()
            ok_batch, lines = run_flywheel_batch(args.database_url, args.vector_store_uri)
            ok_ok, more = await approve_in_queue(args.port, FLYWHEEL_KEYWORD, args.answer)
            ok = ok_batch and ok_ok
            results.append({
                "sid": "D3", "title": "飞轮:归并并进待审 + 审核通过写库",
                "passed": ok, "seconds": time.monotonic() - t0,
                "checks": [("✅ " if ok else "❌ ") + x for x in [*lines, *more]],
                "summary": {"route": "", "gate": None, "finish": ""}, "answer": "",
            })
            print(f"[{'PASS' if ok else 'FAIL'}] D3 飞轮归并+通过  {results[-1]['seconds']:.1f}s",
                  flush=True)
            for line in [*lines, *more]:
                print(f"        {line}", flush=True)

            if run_closing:
                t0 = time.monotonic()
                turn = await use.run_turn(
                    client, args.port, CLOSING.session, CLOSING.message,
                    conversation_id=conv_of.get(CLOSING.follow or ""),
                )
                await record(CLOSING, turn, time.monotonic() - t0)

    server.should_exit = True
    time.sleep(1.0)

    # 回滚探针题:彩排把题补进了知识库,不回滚下次就演不出「拒答」。
    # 放在服务停掉之后做——Milvus Lite 是单进程文件库,服务还开着时另起一个
    # client 删实体容易卡住(2026-11-12 实测过)。
    if run_flywheel and not args.keep_flywheel:
        try:
            stats = _reset_flywheel(args.database_url, args.vector_store_uri, FLYWHEEL_KEYWORD)
            print(
                f"[彩排] 已回滚探针题「{FLYWHEEL_KEYWORD}」:块 {stats['chunks']},"
                f"审核行 {stats['reviews']},知识库剩 {stats.get('left', '?')} 块",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - 回滚失败不影响本轮结论
            print(f"[彩排] 回滚失败({exc}),录制前手动跑 scripts/flywheel_reset.py", flush=True)

    report = REPO_ROOT / "reports" / "demo_rehearsal.md"
    write_report(report, results, args)
    passed = sum(1 for r in results if r["passed"])
    print(f"\n[彩排] {passed}/{len(results)} 幕通过,报告:{report}", flush=True)
    return 0 if passed == len(results) else 1


def write_report(path: Path, results: list[dict], args) -> None:
    total = sum(r["seconds"] for r in results)
    lines = [
        "# 演示彩排报告",
        "",
        f"- 时间:{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 目标:`{args.database_url}` + `{args.vector_store_uri}`",
        f"- 结果:**{sum(1 for r in results if r['passed'])}/{len(results)} 幕通过**;"
        f"串起来 {total / 60:.1f} 分钟(不含人工填答案与口播)",
        "",
        "| 幕 | 内容 | 耗时 | 路由 / 闸 / 收尾 | 现象 |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        s = r["summary"]
        gate = {True: "过", False: "拦"}.get(s.get("gate"), "—")
        cells = "<br>".join(r["checks"])
        if r.get("answer"):
            cells += f"<br>答:{r['answer']}"
        mark = "" if r["passed"] else "**FAIL** "
        lines.append(
            f"| {r['sid']} | {mark}{r['title']} | {r['seconds']:.1f}s | "
            f"{s.get('route') or '—'} / {gate} / {s.get('finish') or '—'} | {cells} |"
        )
    lines += ["", "耗时含真模型往返;D3 含归并 + 建块向量化。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="演示彩排:逐幕打真服务,记耗时与现象")
    parser.add_argument("--database-url", default=DEV_DB_URL)
    parser.add_argument("--vector-store-uri", default=DEV_VECTOR_URI)
    parser.add_argument("--port", type=int, default=18111)
    parser.add_argument("--list", action="store_true", help="只列幕单")
    parser.add_argument("--skip-flywheel", action="store_true", help="不跑飞轮那几幕(不动库)")
    parser.add_argument("--only", default="", help="只跑这几幕,逗号分隔(如 C2,C3;含 D3 才会跑飞轮)")
    parser.add_argument("--langfuse", action="store_true", help="保留观测(默认关)")
    parser.add_argument(
        "--keep-flywheel",
        action="store_true",
        help="跑完不回滚探针题(默认回滚,便于反复彩排)",
    )
    parser.add_argument("--answer", default=FLYWHEEL_ANSWER, help="审核台填的标准答案")
    args = parser.parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
