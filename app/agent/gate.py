"""置信度闸(第五章第 7 点):知识类检索之后、进 Agent 之前。

为什么必须卡在这个位置:Agent 的答复是流式吐给用户的,等话说完再判「这条证据
够不够」已经晚了——用户已经读到。所以闸放在「检索完、还没开口」的空档里。

两层判定,顺序不能反:
1. **confidence 粗筛**(便宜):三个信号合成一个 0~1 的分数,低于阈值直接判弱,
   连自评的一次模型调用都省掉;
2. **证据自评**(贵但准):过了才值得花一次调用问「这几条够不够答这个问题」。

## 为什么从「单看 top1」升级成三信号合成(ch09)

ch04 的实现只用 top1 相似度,文档里就记着它的毛病:同一份库里,dense 分挤在
0.38~0.78 窄带,库外问题能拿到 0.62,比库内某条真答案的 0.49 还高——单值阈值
分不开「沾边」和「答得上」。ch09 按需求给它配上三个信号:

```
confidence = 0.60 * top1
           + 0.25 * min(有效证据数 / 3, 1)
           + 0.15 * min(max(top1 - top2, 0) / 0.30, 1)
```

权重不做拟合,按「谁更可信」排序给:单条分数最可信 → 有效证据条数(多路印证)→
Top1/Top2 分差(只有它一条好,说明别的都不沾边,反而更像答不了)。
`有效证据`沿用 `KB_MIN_SCORE` 当门槛——那个阈值的用途从「判够不够」降级成
「算不算一条有效证据」,它的老毛病在这个用途下不致命。

阈值不拍脑袋:`scripts/calibrate_gate.py` 用 ch04 的评估集(31 干净题当正例、
D 桶 14 条当负例)扫曲线定,报告在 `reports/ch09_gate_calibration.md`。

业务数据类不走这道闸:它没有检索证据,证据强弱无从谈起——拿订单工具查到的
订单号去判「证据够不够」是错的问法。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.rag.citations import Citation, format_evidence
from app.rag.selfcheck import check_evidence

logger = logging.getLogger(__name__)

# 入池原因分级(与 low_confidence_questions.reason 一起存,复盘时分得清)
REASON_NO_HITS = "no_hits"
REASON_SCORE_LOW = "score_low"  # ch04 的旧原因码:池里已有数据在用,保留不再产生
REASON_CONFIDENCE_LOW = "confidence_low"
REASON_SELF_CHECK = "self_check"
REASON_DEGRADED = "self_check_degraded"

# confidence 的三个权重与两个归一化尺度(见模块文档;不做拟合,别随手调)
CONF_WEIGHT_TOP1 = 0.60
CONF_WEIGHT_COUNT = 0.25
CONF_WEIGHT_GAP = 0.15
CONF_EFFECTIVE_TARGET = 3  # 有效证据到这个条数算「够多」,再多不加分
CONF_GAP_SCALE = 0.30  # Top1-Top2 差 0.30 以上算「分得很开」


@dataclass
class ConfidenceSignals:
    """三个信号与合成结果。留原始值是为了报告里能解释「为什么这条被判弱」。"""

    top1: float
    top2: float
    gap: float
    effective: int
    confidence: float

    def describe(self) -> str:
        return (
            f"conf={self.confidence:.3f}(top1={self.top1:.3f}, gap={self.gap:.3f}, "
            f"有效={self.effective})"
        )


def evidence_confidence(citations, *, min_score: float) -> ConfidenceSignals:
    """三个信号 → 一个 0~1 的置信度。纯函数,不调模型,单测能钉死边界。"""
    if not citations:
        return ConfidenceSignals(0.0, 0.0, 0.0, 0, 0.0)
    scores = [float(c.score) for c in citations]
    top1 = scores[0]
    top2 = scores[1] if len(scores) > 1 else 0.0
    gap = max(top1 - top2, 0.0)
    effective = sum(1 for s in scores if s >= min_score)
    confidence = (
        CONF_WEIGHT_TOP1 * top1
        + CONF_WEIGHT_COUNT * min(effective / CONF_EFFECTIVE_TARGET, 1.0)
        + CONF_WEIGHT_GAP * min(gap / CONF_GAP_SCALE, 1.0)
    )
    return ConfidenceSignals(top1, top2, gap, effective, min(max(confidence, 0.0), 1.0))


@dataclass
class GateResult:
    passed: bool
    reason: str = ""
    detail: str = ""
    top_score: float = 0.0
    degraded: bool = False
    confidence: float = 0.0

    def pool_source(self) -> str:
        """落到 low_confidence_questions 的 source 字段。"""
        if self.reason in (REASON_NO_HITS, REASON_SCORE_LOW, REASON_CONFIDENCE_LOW):
            return "retrieval_low_conf"
        return "self_check"

    def pool_reason(self) -> str:
        if self.detail:
            return f"{self.reason}: {self.detail}"[:255]
        return self.reason


def citations_from_payload(payloads: list[dict]) -> list[Citation]:
    """State 里只放 JSON 安全的 dict(checkpointer 要能序列化),取用时就地还原。"""
    return [Citation(**p) for p in payloads]


async def check_gate(
    question: str,
    citations: list[Citation],
    *,
    min_score: float,
    min_confidence: float | None = None,
    model=None,
    use_self_check: bool = True,
) -> GateResult:
    """证据够不够把问题交给 Agent。判弱即拒答,后面的节点不再执行。

    `min_confidence` 为 None 时退化成 ch04 的老行为(只比 top1),给还没升级配置的
    调用方留一条路;图里始终传真值。

    `use_self_check=False` 只留阈值、不调自评模型。给「答案的另一半不在证据里」的
    链路用:退款子流程要拿订单事实(签收时间/商品/金额)配上政策条款才能判,而自评
    模型只看得到「问句 + 条款」,永远回一句「未涉及该订单」——那是判据缺料,不是
    证据不足,它的否决在这里没有意义。
    """
    if not citations:
        return GateResult(False, REASON_NO_HITS, "检索零命中")

    signals = evidence_confidence(citations, min_score=min_score)
    top = signals.top1

    if min_confidence is not None:
        if signals.confidence < min_confidence:
            return GateResult(
                False,
                REASON_CONFIDENCE_LOW,
                f"{signals.describe()} < 阈值 {min_confidence:.2f}",
                top_score=top,
                confidence=signals.confidence,
            )
    elif top < min_score:
        return GateResult(
            False,
            REASON_SCORE_LOW,
            f"top1={top:.4f} < 阈值 {min_score:.2f}",
            top_score=top,
            confidence=signals.confidence,
        )

    if not use_self_check:
        return GateResult(
            True, "", "跳过自评(答案另有一半在工具事实里)", top_score=top,
            confidence=signals.confidence,
        )

    verdict = await check_evidence(question, format_evidence(citations), model=model, on_error="block")
    if verdict.useful:
        return GateResult(
            True, "", verdict.reason, top_score=top, degraded=verdict.degraded,
            confidence=signals.confidence,
        )

    reason = REASON_DEGRADED if verdict.degraded else REASON_SELF_CHECK
    return GateResult(
        False, reason, verdict.reason, top_score=top, degraded=verdict.degraded,
        confidence=signals.confidence,
    )
