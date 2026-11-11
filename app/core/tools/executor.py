"""工具执行引擎:所有工具调用从这里过(第八章需求 2、3、4)。

## 三道关,顺序不能换

1. **存在性**:没这个工具就把可用清单回给模型(它可能记错了名字);
2. **权限**:写操作(`read_only=False`)必须 `confirmed=True` 才执行。没确认直接
   拒绝,不给模型绕过路径。外部 MCP Server 自报的用途声明不参与这个判断——
   那是对方写的,不可信。
3. **参数 Schema 校验**:拦下不抛异常,把校验说明包成一条工具结果回灌——模型
   据此追问用户或重新组织调用,比抛 500 让它重试有用得多。

权限在校验之前:不该执行的操作,连参数对不对都不用告诉它。

## 重试规矩

只重试 `transient_errors`(网络抖动这类),而且**写操作永不重试**——超时未必没
执行,重复执行比失败更糟。业务空结果(`ok=True` 的「未找到」)不重试,重试还是空。

## 审计

每次调用落一条,包括被校验拦、被权限拒的。见 `audit_repo`。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass

from langchain_core.tools import BaseTool
from pydantic import ValidationError

from app.config import get_settings
from app.core.token_budget import estimate_tokens
from app.core.tools.registry import ToolRegistry
from app.models.tool_audit import (
    STATUS_DENIED,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_TIMEOUT,
    STATUS_VALIDATION,
)

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """一次工具调用的结局;content 即将回灌给模型的文本。"""

    tool_call_id: str
    name: str
    ok: bool
    content: str
    status: str = STATUS_OK
    retry_count: int = 0
    duration_ms: int = 0


def normalize_content(value):
    """MCP 工具返回的是内容块列表(`[{"type":"text","text":...}]`),内置工具返回字符串。

    统一成字符串再交给 `format_result`:否则一个纯文本的物流轨迹会被序列化成
    `[{"type": "text", "text": "..."}]`,模型得先解析 JSON 才看得到轨迹,白花 token
    还容易读错。非文本块(图片等)原样保留在列表里,由 format_result 序列化。
    """
    if isinstance(value, list) and value:
        texts: list[str] = []
        others = 0
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                texts.append(str(item["text"]))
            else:
                others += 1
        if texts and not others:
            return "\n".join(texts)
    return value


def format_result(value, *, max_tokens: int | None = None) -> str:
    """把工具返回值变成可以回灌的文本。

    `dict/list` 走 `json.dumps(ensure_ascii=False)`——中文不转义,模型读得懂,人也
    读得懂;`None` 给空串;超预算截断并注明,免得一条大结果把下一轮挤爆。
    「挑哪些字段」是工具自己的事(工具知道业务语义),引擎只保证「结构化的一定变成
    可读文本」——引擎替工具挑字段等于猜业务。
    """
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    elif isinstance(value, (dict, list, tuple)):
        try:
            text = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(value)
    else:
        text = str(value)

    limit = max_tokens if max_tokens is not None else get_settings().tool_result_max_tokens
    used = estimate_tokens(text)
    if limit > 0 and used > limit:
        keep = max(int(len(text) * limit / max(used, 1)), 80)
        text = text[:keep] + f"…(结果过长,已截断至约 {limit} token)"
    return text


def validate_args(tool: BaseTool, args: dict) -> str | None:
    """按工具的 Schema 校验入参,返回错误说明(合法则 None)。

    内置工具给的是 pydantic 模型,`model_validate` 一次到底。MCP 拉来的工具不一样:
    适配器只带了 Server 那边给的 **JSON Schema dict**,没有 pydantic 模型。这种就只查
    必填项——不为了补全校验去引一个 jsonschema 依赖,也不假装校验过。
    """
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return None
    validate = getattr(schema, "model_validate", None)
    if validate is None:
        if isinstance(schema, dict):
            required = schema.get("required") or []
            missing = [str(k) for k in required if k not in (args or {})]
            if missing:
                return "缺少必填参数:" + ", ".join(missing)
        return None
    try:
        validate(args)
    except ValidationError as exc:
        return "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
    return None


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        timeout: float | None = None,
        transient_errors: tuple[type[BaseException], ...] = (),
        session_factory=None,
        audit: bool = True,
    ) -> None:
        self._registry = registry
        self._timeout = timeout if timeout is not None else get_settings().tool_timeout_seconds
        self._transient_errors = transient_errors
        self._session_factory = session_factory
        self._audit = bool(audit and session_factory is not None)

    async def execute(self, tool_call: dict, *, confirmed: bool = False) -> ToolResult:
        """执行一次调用。`confirmed=False` 只影响写操作,只读工具不受影响。

        默认 False 是刻意的:写操作必须由调用方显式声明「用户已经确认过」,
        省略参数等于没确认、直接拒——权限闸的默认状态只能是关。
        """
        started = time.monotonic()
        name = tool_call.get("name", "")
        args = tool_call.get("args") or {}
        call_id = tool_call.get("id", "")
        meta = self._registry.meta(name)
        tool = self._registry.get(name)

        if tool is None:
            return await self._finish(
                ToolResult(
                    call_id,
                    name,
                    False,
                    f"工具 {name!r} 不存在。可用工具: {', '.join(self._registry.names()) or '(无)'}",
                    status=STATUS_FAILED,
                ),
                args=args,
                meta=meta,
                started=started,
            )

        # 1. 权限:写操作没确认就拒
        if meta is not None and meta.is_write and not confirmed:
            logger.info("tool %s denied: 写操作未经用户确认", name)
            return await self._finish(
                ToolResult(
                    call_id,
                    name,
                    False,
                    f"工具 {name} 是写操作,需要用户确认后才能执行;本次未确认,已拒绝。",
                    status=STATUS_DENIED,
                ),
                args=args,
                meta=meta,
                started=started,
            )

        # 2. 参数校验
        reasons = validate_args(tool, args)
        if reasons:
            logger.info("tool %s validation rejected: %s", name, reasons)
            return await self._finish(
                ToolResult(
                    call_id,
                    name,
                    False,
                    f"参数不合法({reasons}),请向用户追问或修正后重试",
                    status=STATUS_VALIDATION,
                ),
                args=args,
                meta=meta,
                started=started,
            )

        # 3. 执行(超时 + 只读工具的一次性重试)
        retry_count = 0
        try:
            content, retry_count = await self._invoke(
                tool, tool_call, is_write=bool(meta and meta.is_write)
            )
        except asyncio.TimeoutError:
            logger.warning("tool %s timed out after %ss", name, self._timeout)
            return await self._finish(
                ToolResult(
                    call_id,
                    name,
                    False,
                    f"工具 {name} 执行超时(>{self._timeout}s)",
                    status=STATUS_TIMEOUT,
                    retry_count=retry_count,
                ),
                args=args,
                meta=meta,
                started=started,
            )
        except Exception as exc:  # 真故障:如实回给模型,不静默吞
            logger.exception("tool %s failed", name)
            return await self._finish(
                ToolResult(
                    call_id,
                    name,
                    False,
                    f"工具 {name} 执行失败:{exc}",
                    status=STATUS_FAILED,
                    retry_count=retry_count,
                ),
                args=args,
                meta=meta,
                started=started,
            )

        return await self._finish(
            ToolResult(
                call_id, name, True, format_result(normalize_content(content)), retry_count=retry_count
            ),
            args=args,
            meta=meta,
            started=started,
        )

    async def _invoke(self, tool: BaseTool, tool_call: dict, *, is_write: bool) -> tuple[str, int]:
        """执行并返回 (内容, 重试次数)。写操作一次都不重试。"""
        attempts = 2 if (self._transient_errors and not is_write) else 1
        last_exc: BaseException | None = None
        # langchain-core 1.6 起,ToolCall dict 必须带 type="tool_call" 才会走「按
        # tool_call 执行」的分支;不带就会被当成普通入参去校验,报「字段缺失」。
        # 模型产出的 tool_calls 本来就带 type,补这一步是给手写的调用(测试、脚本)兜底。
        payload = dict(tool_call)
        payload.setdefault("type", "tool_call")
        for attempt in range(attempts):
            try:
                # StructuredTool.ainvoke(ToolCall) → ToolMessage(带 tool_call_id 对号)
                msg = await asyncio.wait_for(tool.ainvoke(payload), timeout=self._timeout)
                return msg.content, attempt
            except self._transient_errors as exc:  # 网络抖动类:再试一次
                last_exc = exc
                logger.warning(
                    "tool %s transient error (attempt %d/%d): %s",
                    tool_call.get("name"), attempt + 1, attempts, exc,
                )
        raise last_exc  # type: ignore[misc]

    async def _finish(
        self, result: ToolResult, *, args: dict, meta, started: float
    ) -> ToolResult:
        result.duration_ms = int((time.monotonic() - started) * 1000)
        if not self._audit:
            return result
        from app.repositories import audit_repo

        await audit_repo.record(
            self._session_factory,
            tool_name=result.name,
            tool_source=getattr(meta, "source", "builtin"),
            tool_call_id=result.tool_call_id or None,
            mcp_server=getattr(meta, "server", None),
            arguments=args,
            result=result.content,
            status=result.status,
            error_message=None if result.ok else result.content,
            retry_count=result.retry_count,
            duration_ms=result.duration_ms,
        )
        return result
