"""权限、重试规矩、结果格式化单测。"""

from __future__ import annotations

import asyncio

from langchain_core.tools import tool

from app.core.tools.executor import ToolExecutor, format_result
from app.core.tools.registry import ToolMeta, ToolRegistry


@tool
def echo(text: str) -> str:
    """原样返回。text: 任意文本"""
    return f"echo:{text}"


@tool
def create_thing(name: str) -> str:
    """写操作。name: 名字"""
    return f"created {name}"


class Flaky(Exception):
    """模拟网络抖动。"""


def _flaky_tool(counter: dict, *, fail_times: int):
    @tool
    async def flaky(query: str) -> str:
        """不稳定工具。query: 任意"""
        counter["calls"] += 1
        if counter["calls"] <= fail_times:
            raise Flaky("连接被重置")
        return f"ok:{query}"

    return flaky


def _registry(*extra) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(echo)
    reg.register(create_thing, ToolMeta(name="create_thing", read_only=False))
    for t, meta in extra:
        reg.register(t, meta)
    return reg


async def test_write_tool_requires_confirmation():
    executor = ToolExecutor(_registry())
    denied = await executor.execute({"name": "create_thing", "args": {"name": "x"}, "id": "1"}, confirmed=False)
    assert denied.ok is False
    assert denied.status == "权限拒绝"
    assert "未确认" in denied.content

    allowed = await executor.execute({"name": "create_thing", "args": {"name": "x"}, "id": "2"}, confirmed=True)
    assert allowed.ok is True


async def test_read_tool_ignores_confirmation_flag():
    """只读工具不受确认闸影响——不确认只是「这次没人点按钮」,查单又不会改数据。"""
    executor = ToolExecutor(_registry())
    result = await executor.execute({"name": "echo", "args": {"text": "hi"}, "id": "3"}, confirmed=False)
    assert result.ok is True


async def test_read_tool_retries_transient_error_once():
    counter = {"calls": 0}
    reg = _registry((_flaky_tool(counter, fail_times=1), None))
    executor = ToolExecutor(reg, transient_errors=(Flaky,))
    result = await executor.execute({"name": "flaky", "args": {"query": "q"}, "id": "4"})
    assert result.ok is True
    assert result.retry_count == 1
    assert counter["calls"] == 2


async def test_write_tool_never_retries():
    """超时未必没执行,重复执行比失败更糟。"""
    counter = {"calls": 0}

    @tool
    async def flaky_write(name: str) -> str:
        """不稳定的写操作。name: 名字"""
        counter["calls"] += 1
        raise Flaky("连接被重置")

    reg = ToolRegistry()
    reg.register(flaky_write, ToolMeta(name="flaky_write", read_only=False))
    executor = ToolExecutor(reg, transient_errors=(Flaky,))
    # 写操作默认拒(权限闸 fail-closed),这条测的是「重试」不是「权限」,显式确认
    result = await executor.execute(
        {"name": "flaky_write", "args": {"name": "x"}, "id": "5"}, confirmed=True
    )
    assert result.ok is False
    assert result.retry_count == 0
    assert counter["calls"] == 1, "写操作只许调一次"


async def test_timeout_is_reported_not_raised():
    @tool
    async def slow(query: str) -> str:
        """慢工具。query: 任意"""
        await asyncio.sleep(2)
        return "done"

    reg = ToolRegistry()
    reg.register(slow)
    executor = ToolExecutor(reg, timeout=0.05)
    result = await executor.execute({"name": "slow", "args": {"query": "q"}, "id": "6"})
    assert result.ok is False
    assert result.status == "超时"
    assert "超时" in result.content


def test_format_result_keeps_chinese_readable():
    """JSON 中文不转义:模型读得懂,人也读得懂。"""
    text = format_result({"order_id": "1001", "状态": "待发货"})
    assert "待发货" in text and "\\u" not in text


def test_format_result_handles_none_and_scalars():
    assert format_result(None) == ""
    assert format_result(42) == "42"
    assert format_result("已经是文本") == "已经是文本"


def test_format_result_truncates_over_budget():
    long_text = "订" * 5000
    out = format_result(long_text, max_tokens=200)
    assert "已截断" in out
    assert len(out) < len(long_text)


async def test_unknown_tool_lists_available_names():
    executor = ToolExecutor(_registry())
    result = await executor.execute({"name": "nope", "args": {}, "id": "7"})
    assert result.ok is False
    assert "echo" in result.content and "create_thing" in result.content


def test_mcp_content_blocks_flatten_to_text():
    """MCP 返回内容块列表,要摊平成文本,别让模型去解析 JSON。"""
    from app.core.tools.executor import normalize_content

    blocks = [{"type": "text", "text": "订单 1001 已发货"}, {"type": "text", "text": "- 06-01 广州"}]
    assert normalize_content(blocks) == "订单 1001 已发货\n- 06-01 广州"
    assert normalize_content("纯文本") == "纯文本"
    assert normalize_content({"a": 1}) == {"a": 1}
    # 混了非文本块就整块交给 format_result 序列化,不丢内容
    mixed = [{"type": "text", "text": "x"}, {"type": "image", "data": "..."}]
    assert normalize_content(mixed) == mixed


def test_mcp_dict_schema_checks_required_only():
    """MCP 工具带的是 JSON Schema dict,没有 pydantic 模型:只查必填项,不假装全验过。"""
    from app.core.tools.executor import validate_args

    class FakeMcpTool:
        args_schema = {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}

    assert validate_args(FakeMcpTool(), {"order_id": "1001"}) is None
    assert "order_id" in validate_args(FakeMcpTool(), {})


def test_builtin_schema_still_validates_types():
    from app.core.tools.executor import validate_args

    @tool
    def needs_int(count: int) -> str:
        """要整数。count: 数量"""
        return str(count)

    assert validate_args(needs_int, {"count": 3}) is None
    assert "count" in validate_args(needs_int, {"count": "不是数字"})
