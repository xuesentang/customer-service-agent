"""把出题器草稿收拾成可用的评估集(E4 收尾)。

这一步为什么必须有:
    `gen_eval_cases.py` 只看得到目录,看不到内容,所以它会把答案标到**旧版政策**上
    (「退换货政策(2024 版)」和现行「退货退款政策」在目录里长得一样),或者把睡袋
    SB200 标到「小家电型号总表」——它只能按字面猜。这类错标必须人工改,不能让脚本
    猜。噪声变体同理:模型把它们原样抄了一遍,只是贴了个 `typo` 标签,等于没做噪声。

    所以这里做三件事,全部显式记录:
    1. PATCH_SECTION:人工改对的小节标注(附原因);
    2. PATCH_QUERY:前提与库内事实冲突的题,改问句而不是改标注;
    3. 噪声变体改为**规则生成**——确定性、可复现、不花 token,而且规则改写更接近
       真人打字(漏标点、加错别字、把型号拆开写)。

产出:`tests/eval/ch04_cases.jsonl`(干净题 + 噪声变体 + D 桶负例)。

用法:
    python scripts/assemble_eval_set.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import kb_blueprint as blueprint
from gen_eval_cases import EXTRA_FACTS

DRAFT = Path("tests/eval/ch04_cases_draft.jsonl")
REFUSE = Path("tests/eval/ch04_cases_refuse_v2.jsonl")
OUT = Path("tests/eval/ch04_cases.jsonl")

# 出题器把答案标到了旧版政策上:旧版是脏数据,现行版才是正解。
PATCH_SECTION: dict[str, tuple[str, str]] = {
    "A02": ("退货退款政策 > 运费规则", "问的是现行退货运费,旧版 129 元包邮是脏数据"),
    "A08": ("配送与运费规则 > 发货时效", "旧版 72 小时已作废,现行 48 小时"),
    "C18": ("退货退款政策 > 换货规则", "问换大一码谁付邮费,归换货规则而非运费规则"),
    "C20": ("配送与运费规则 > 发货时效", "旧版 72 小时已作废"),
    "E27": ("退货退款政策 > 运费规则", "旧版运费规则已作废"),
    "B15": ("型号对照总表 > 户外装备型号总表", "SB200/SB400 是睡袋,不是小家电"),
}

# 题面前提与库内事实冲突:改问句,不改标注。
PATCH_QUERY: dict[str, tuple[str, str]] = {
    "E25": (
        "我买的LP300咖啡机显示故障代码E3,这个代码是什么意思?要修的话得寄回去吗,多久能修好?",
        "原题问「上门维修」,库里维修流程是寄回、无上门,原题不可答",
    ),
    "E30": (
        "W10这个型号是不是已经停产了?停产了以后维修配件还买得到吗?",
        "原题问 SB200 停产,库里停产的是 W10 和 LP50,SB200 在售",
    ),
}

# 骨架里没有、但正文明确写了的事实,手工补(只补真正在正文里的)。
EXTRA_KEYWORDS: dict[str, list[str]] = {
    "A06": ["3 个工作日内发出换货商品", "换尺码优先同一仓库调拨", "换不同商品走退货退款再下单"],
}

# 原 D 桶的题被证明是「可答」的:题库里没有 ≠ 库里没有,这一条要靠人读库才能发现。
# 判据是 payment_policy 明写「不支持货到付款」——答「不支持」就是正确答案,不该要求拒答。
EXTRA_CLEAN: list[dict] = [
    {
        "id": "A31",
        "query": "你们支持货到付款吗",
        "bucket": "A_policy",
        "difficulty": "easy",
        "expected_section": "支付与优惠规则 > 支持的支付方式",
        "expected_keywords": ["不支持货到付款", "不支持对公转账", "客服无法为个别订单单独开通"],
        "answerable": True,
        "note": "原为 D01(far),核对库内发现 payment_policy 明确写了不支持货到付款,是判断题不是无解题",
    },
    # --- 扩库(29 篇 / 约 1000 块)后新增:全部对应新增章节,不让新文档成为「没人问的死块」 ---
    {
        "id": "A32",
        "query": "我买的猫砂到货了，拆开用了一袋，剩下没拆的能退吗？",
        "bucket": "A_policy",
        "difficulty": "medium",
        "expected_section": "退货资格与例外边界 > 拆封即不支持无理由的类目",
        "expected_keywords": [
            "宠物食品、猫砂、化妆品一经拆封不支持无理由退货",
            "定制类商品一经生产不支持无理由退货",
            "虚拟商品与充值卡券一经兑换不支持退货",
        ],
        "answerable": True,
        "note": "拆封类目边界:问的是「没拆的那几袋」,考的是按商品而非按订单判定",
    },
    {
        "id": "A33",
        "query": "我这个订单发了两个包裹，一个前天签收一个今天才到，7 天无理由从哪天开始算？",
        "bucket": "A_policy",
        "difficulty": "hard",
        "expected_section": "时限计算与顺延规则 > 分批发货的时限计算",
        "expected_keywords": [
            "各包裹按各自签收时间分别计算售后时限",
            "部分包裹未签收的，整单退货时限自最后一件签收起算",
        ],
        "answerable": True,
        "note": "分批发货的时限口径:单件按各自签收算,整单按最后一件算",
    },
    {
        "id": "A34",
        "query": "寄到拉萨的订单运费怎么算？满 99 还包邮吗",
        "bucket": "A_policy",
        "difficulty": "easy",
        "expected_section": "地区服务边界与受限区域 > 新疆西藏的时效与运费",
        "expected_keywords": [
            "新疆、西藏首件运费 15 元",
            "每增加一件加收 5 元",
            "不参与 99 元包邮",
        ],
        "answerable": True,
        "note": "偏远地区运费:与 99 包邮互斥,最容易答错成「满 99 包邮」",
    },
    {
        "id": "A35",
        "query": "我凑到 110 用券后实付 87，还能包邮吗？",
        "bucket": "A_policy",
        "difficulty": "hard",
        "expected_section": "优惠叠加与门槛边界 > 门槛金额的计算口径",
        "expected_keywords": [
            "包邮门槛按实付金额计算，券后金额同样计入实付",
            "实付未满 99 元收取 8 元运费",
        ],
        "answerable": True,
        "note": "门槛口径:券后 87 < 99,不包邮;考的是「实付」而非「原价」",
    },
    {
        "id": "A36",
        "query": "满赠的赠品我用过了，现在退一半商品，赠品要退吗？",
        "bucket": "A_policy",
        "difficulty": "medium",
        "expected_section": "赠品与套装售后边界 > 满赠活动的门槛回退",
        "expected_keywords": [
            "剩余金额低于门槛，赠品需退回或按标价扣除",
            "赠品已使用的，按赠品标价的 50% 折算扣除",
        ],
        "answerable": True,
        "note": "满赠门槛回退:部分退货后金额跌破门槛的处理",
    },
    {
        "id": "B37",
        "query": "我脚宽 108 毫米，想买 W40，选常规楦还是宽楦？",
        "bucket": "B_model",
        "difficulty": "medium",
        "expected_section": "鞋靴选码与脚型适配指南 > 宽楦的判断与适用范围",
        "expected_keywords": [
            "脚宽 105 至 110 毫米建议 2E",
            "超过 110 毫米建议 4E",
            "W40 提供 2E 宽楦，W60 同时提供 2E 与 4E",
        ],
        "answerable": True,
        "note": "宽楦阈值:108 落在 2E 区间,W40 恰好有 2E",
    },
    {
        "id": "B38",
        "query": "想开个小咖啡店，一天出 30 多杯，家用机顶得住吗还是要上 LP500？",
        "bucket": "B_model",
        "difficulty": "medium",
        "expected_section": "咖啡机型号对照与选购 > 家用与商用的分界",
        "expected_keywords": [
            "日均出杯 30 杯以上、需要连续出杯的，选 LP500 商用双锅炉",
            "家用型号连续出杯 3 杯以上需要间隔等待，锅炉回温约需 40 秒",
            "商用场景使用家用型号不属保修范围",
        ],
        "answerable": True,
        "note": "家用商用分界:30 杯/天是分界线,且家用机在商用场景失去保修",
    },
    {
        "id": "B39",
        "query": "十月露营夜里大概 0 度，SB200 够用吗？",
        "bucket": "B_model",
        "difficulty": "easy",
        "expected_section": "露营装备季节适配指南 > 按气温选装备组合",
        "expected_keywords": [
            "夜间最低气温高于 5 摄氏度时选 SB200 即可",
            "5 至零下 5 摄氏度时选 SB400",
            "低于零下 5 摄氏度时在 SB400 内加抓绒内胆",
        ],
        "answerable": True,
        "note": "按气温选睡袋:0 度落在 SB400 区间,SB200 不够",
    },
    {
        "id": "B40",
        "query": "TN2 的帐杆能插到 TN4 上用吗？",
        "bucket": "B_model",
        "difficulty": "easy",
        "expected_section": "帐篷搭建与配件兼容 > 配件通用性",
        "expected_keywords": [
            "TN2 与 TN4 的帐杆长度与接口不同，不可通用",
            "地钉、风绳、地布为通用配件",
        ],
        "answerable": True,
        "note": "配件通用边界:帐杆不通用,地钉风绳通用",
    },
    {
        "id": "C41",
        "query": "鞋穿了半个月就开胶了，能换新的吗",
        "bucket": "C_colloquial",
        "difficulty": "medium",
        "expected_section": "鞋靴保养与寿命管理 > 开胶与脱线的处理",
        "expected_keywords": [
            "签收之日起 15 日内出现开胶，经核实属质量问题可换新",
            "超过 15 日的开胶按保外维修处理，维修费 60 元起",
        ],
        "answerable": True,
        "note": "开胶时限口径:15 日内换新、超期保外 60 元起",
    },
    {
        "id": "C42",
        "query": "咖啡机多久除一次垢啊，我家水质挺硬的",
        "bucket": "C_colloquial",
        "difficulty": "medium",
        "expected_section": "咖啡机清洁除垢与耗材管理 > 除垢周期与操作方法",
        "expected_keywords": [
            "水质硬度高（TDS 大于 200）的地区每 30 天除垢一次",
            "软水地区可延长到 90 天一次",
            "除垢程序耗时约 20 分钟",
        ],
        "answerable": True,
        "note": "除垢周期:硬水 30 天、软水 90 天,问的是周期不是操作步骤",
    },
    {
        "id": "C43",
        "query": "快递放柜子里三天才取，7 天无理由从哪天算呢",
        "bucket": "C_colloquial",
        "difficulty": "hard",
        "expected_section": "时限计算与顺延规则 > 签收时间的确定",
        "expected_keywords": [
            "代收点、快递柜以入柜时间视为签收时间",
            "不以取件时间为准",
        ],
        "answerable": True,
        "note": "签收时间口径:入柜即签收,取件时间不算",
    },
    {
        "id": "C44",
        "query": "退的钱啥时候能到我卡里啊",
        "bucket": "C_colloquial",
        "difficulty": "easy",
        "expected_section": "退款渠道与到账细则 > 各支付渠道的到账时效",
        "expected_keywords": [
            "微信与支付宝退款一般 2 小时内到账",
            "银行卡退款一般 3 至 7 个工作日到账",
            "不得承诺具体到账日期",
        ],
        "answerable": True,
        "note": "分渠道到账时效:银行卡 3 至 7 个工作日,且不许承诺具体日期",
    },
    {
        "id": "E45",
        "query": "鞋退回去你们说吊牌剪了不给退，那我还能换码吗？",
        "bucket": "E_multi",
        "difficulty": "hard",
        "expected_section": "退货操作细则 > 到仓验收与不合格处理",
        "expected_keywords": [
            "验收不通过的情形包括吊牌缺失、人为损坏、超期、缺件",
            "验收不通过须留存开箱视频并附照片说明理由",
            "用户对验收结论有异议的可申请复检",
        ],
        "answerable": True,
        "note": "验收不合格后的转路径:退货被驳不等于换码能过,同样要吊牌完好",
    },
    {
        "id": "E46",
        "query": "快递箱压扁了，里面咖啡机外壳凹了一块，怎么赔",
        "bucket": "E_multi",
        "difficulty": "medium",
        "expected_section": "责任划分与举证要求 > 物流破损的责任归属",
        "expected_keywords": [
            "签收后 48 小时内可报运输破损",
            "需提供外包装、商品、面单三张照片",
            "承运商确认责任的，由本店先行赔付再向承运商追偿",
        ],
        "answerable": True,
        "note": "破损责任:48 小时 + 三张照片,赔付由本店先行",
    },
    {
        "id": "E47",
        "query": "咖啡机过保了，你们说要先报价才修，报价不认可怎么办",
        "bucket": "E_multi",
        "difficulty": "hard",
        "expected_section": "换货与维修执行细则 > 保外报价与确认",
        "expected_keywords": [
            "保外维修先出具报价单，用户确认后再维修",
            "报价不认可的，原物退回，寄回运费由用户承担",
            "报价有效期 7 天",
        ],
        "answerable": True,
        "note": "保外维修:不认可报价则原物退回,运费用户承担",
    },
    {
        "id": "E48",
        "query": "TN2 能配你们家的天幕一起用吗",
        "bucket": "E_multi",
        "difficulty": "medium",
        "expected_section": "帐篷搭建与配件兼容 > 地布与天幕的选配",
        "expected_keywords": [
            "地布 129 元，天幕 399 元",
            "TN2 建议 3 米乘 3 米",
            "天幕杆不随机附赠，需单独购买",
        ],
        "answerable": True,
        "note": "原为 D14(near 负例),扩库后 outdoor_setup 明确写了 TN2/TN4 各配多大天幕,已转为可答题",
    },
]

# 噪声规则:同一道题只施加一种手法,施加不了就顺延到下一种。
TYPO = {
    "运费": "云费",
    "退款": "退宽",
    "保修": "包修",
    "尺码": "尺玛",
    "包邮": "包游",
    "发票": "发漂",
}
MODEL_RE = re.compile(r"\b(W|LP|TN|SB)(\d{2,3})\b")


def facts_for(doc_title: str, h2: str) -> list[str]:
    if (doc_title, h2) in EXTRA_FACTS:
        return EXTRA_FACTS[(doc_title, h2)]
    for d in blueprint.DOCS:
        if d["title"] != doc_title:
            continue
        for sec in d["sections"]:
            if sec["h2"] == h2:
                return sec["facts"][:3]
    return []


def split_section(path: str) -> tuple[str, str]:
    head, _, tail = path.partition(" > ")
    return head.strip(), tail.strip()


def _model_format(query: str) -> str:
    m = MODEL_RE.search(query)
    assert m
    alt = f"{m.group(1).lower()}-{m.group(2)}"
    return query[: m.start()] + alt + query[m.end() :]


def _typo(query: str) -> str:
    for word, typo in TYPO.items():
        if word in query:
            return query.replace(word, typo, 1)
    return query


def _context_drop(query: str) -> str:
    cut = max(query.find("?"), query.find("？"))
    return query[: cut + 1]


def _emotion(query: str) -> str:
    return "急!" + re.sub(r"[?？]", "", query) + ",快回一下"


def _punctuation(query: str) -> str:
    return re.sub(r"[，。？！,?！]", "", query)


def noise_variants(query: str, index: int) -> tuple[str, str]:
    """返回 (变体, 手法)。同一道题只施加一种手法,按题号轮转,让五种手法都用上。

    为什么不按「优先级」挑:第一版按优先级挑,结果 30 条里 11 条是加情绪前缀,
    噪声维度退化成「每条题都加一句脏话」。轮转才能把 typo / 型号写法 / 掉上下文
    这几种在报告里分开看。
    """
    pool: list[tuple[str, callable]] = []
    if MODEL_RE.search(query):
        pool.append(("model_format", _model_format))
    if any(word in query for word in TYPO):
        pool.append(("typo", _typo))
    if query.count("?") + query.count("？") >= 2:
        pool.append(("context_drop", _context_drop))
    pool.append(("emotion", _emotion))
    pool.append(("punctuation", _punctuation))

    for offset in range(len(pool)):
        how, fn = pool[(index + offset) % len(pool)]
        text = fn(query)
        if text != query:
            return text, how
    return query, "none"


def main() -> int:
    clean = [
        json.loads(line)
        for line in DRAFT.read_text(encoding="utf-8").splitlines()
        if line.strip() and not json.loads(line)["id"].endswith("n")
    ]

    for case in clean:
        cid = case["id"]
        if cid in PATCH_QUERY:
            case["query"], why = PATCH_QUERY[cid]
            case["note"] += f";改题:{why}"
        if cid in PATCH_SECTION:
            case["expected_section"], why = PATCH_SECTION[cid]
            case["note"] += f";改标:{why}"

    for case in clean:
        if case["id"] in EXTRA_KEYWORDS:
            case["expected_keywords"] = EXTRA_KEYWORDS[case["id"]]
        else:
            doc, h2 = split_section(case["expected_section"])
            points = facts_for(doc, h2)
            # 找不到就清空,让下面的「仍缺要点」报出来;留着旧要点等于悄悄用错标的要点。
            case["expected_keywords"] = points

    clean += [{**c, "source": "manual", "noise": [], "weight": 1} for c in EXTRA_CLEAN]

    variants: list[dict] = []
    for i, case in enumerate(clean):
        text, how = noise_variants(case["query"], i)
        variants.append(
            {
                **case,
                "id": f"{case['id']}n",
                "query": text,
                "noise": [how],
                "note": f"{case['note']};噪声变体({how})",
            }
        )

    refuse = [
        json.loads(line) for line in REFUSE.read_text(encoding="utf-8").splitlines() if line.strip()
    ]

    rows = clean + variants + refuse
    OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )

    no_points = [c["id"] for c in clean if not c["expected_keywords"]]
    print(f"written {OUT}:干净 {len(clean)} + 噪声 {len(variants)} + D 桶 {len(refuse)} = {len(rows)}")
    print("噪声手法:" + "、".join(sorted({v["noise"][0] for v in variants})))
    if no_points:
        print(f"!! 仍缺要点:{no_points}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
