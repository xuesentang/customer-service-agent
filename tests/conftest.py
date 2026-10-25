"""Shared pytest fixtures (CH02).

SQLite :memory: 每个连接是独立库,StaticPool 固定单连接共享同一内存库 —
这是单测库跑 SQLite 的标准手法(spec D8),MySQL 真库留给集成验收。
engine / session / session_factory 三个 fixture 共享同一 engine:
工具内部自开短会话(session_factory),与测试 fixture session 看到同一份数据。
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.config import get_settings
from app.models import Base


@pytest.fixture(autouse=True)
def _langfuse_off_by_default(monkeypatch):
    """单测默认不连 Langfuse。

    .env 可能开着开关(云端验证后的残留),走真实聊天链路的用例会按 .env
    建出真实 client 注册进 SDK 全局表且没人清理;之后 Langfuse 自己的
    用例里 keyless get_client 看到多实例,会被 SDK 的跨项目保护拒绝
    tracing,span 就丢了。要开着测的用例自己显式 setenv。
    """
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _clean_langfuse_clients():
    """用例结束后摘掉新建的 Langfuse client 实例,清工厂缓存。

    LangfuseResourceManager 按 public_key 全局复用实例;显式开着测的
    用例(如 tracing 文件)建完要摘,否则残留实例会让后续用例的 keyless
    get_client 落入"多实例"分支直接跳过 tracing。
    """
    from langfuse._client.resource_manager import LangfuseResourceManager

    before = set(LangfuseResourceManager._instances)
    yield
    for key in set(LangfuseResourceManager._instances) - before:
        LangfuseResourceManager._instances.pop(key, None)
    from app.observability import tracing

    tracing._callbacks_for.cache_clear()


@pytest.fixture
async def engine():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def session(engine) -> AsyncSession:
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as s:
        yield s


@pytest.fixture
def session_factory(engine) -> async_sessionmaker:
    return async_sessionmaker(engine, expire_on_commit=False)
