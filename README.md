# 电商智能客服 Agent

一个电商场景的智能客服后端 + 聊天页。以「能查单、能查政策、能办退款、答不上来会拒答并记下来」为目标，逐步从纯对话搭到带 RAG 知识库、工作流编排、工具系统、数据飞轮和主题分类器的完整系统。

**技术栈**：Python 3.13 · FastAPI · LangChain · LangGraph · SQLAlchemy(Async) · MySQL 8.4 / SQLite · Milvus（Lite）· BGE-M3 + bge-reranker · ONNX Runtime · pytest

模型接入走 OpenAI 协议，换供应商只改 `.env`（当前为 DeepSeek-V4.1-Flash）。

---

## 整体架构

```
        浏览器聊天页（原生 HTML/JS，SSE 流式渲染）
                    │  POST /api/v1/chat/stream
                    ▼
        ┌───────────────────────────────┐
        │  接入层  FastAPI + 路由        │
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────────────────────┐
        │  编排层  LangGraph                             │
        │                                               │
        │  指代消解 ─→ 意图识别 ─→ 分流                  │
        │     ├ 知识类 → 检索 → 置信度闸 → 主力 Agent    │
        │     ├ 退款类 → 退款子流程 → 闸 → 主力 Agent    │
        │     ├ 业务类 ────────────────→ 主力 Agent      │
        │     ├ 投诉   → 安抚话术 + 转人工/建工单出口    │
        │     └ 闲聊   → 固定话术（零模型调用）          │
        └───────────────┬───────────────────────────────┘
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
   工具系统（注册中心 ·            RAG（检索 · 重排 ·
   执行引擎 · 审计 · MCP）        引用溯源 · 拒答自评）
          │                           │
          └─────────────┬─────────────┘
                        ▼
        ┌───────────────────────────────────────────────┐
        │  数据层  MySQL（业务数据 / 消息 / 知识原文）    │
        │         Milvus（知识块向量 + BM25 索引）        │
        └───────────────────────────────────────────────┘

    旁路：数据飞轮（低置信度入池 → 查重 → 人工审核 → 回写知识库）
          主题分类器（17 类多标签，批量归类飞轮池）
          可观测性（Langfuse trace 树，默认关闭）
```

一句话概括分工：**能确定的事写在图里（节点、路由、退款顺序），只有不确定的（判意图、选工具、组织语言）交给模型。**

---

## 核心模块

| 模块 | 位置 | 做什么 |
|---|---|---|
| 工作流编排 | `app/agent/` | LangGraph 图骨架、意图八分类、置信度闸、手写 ReAct、退款子流程、会话状态 |
| 工具系统 | `app/core/tools/` | 工具注册中心、执行引擎（校验/超时/重试/权限）、调用审计、MCP 接入、写操作确认流 |
| RAG 知识库 | `app/rag/` | 结构感知切分、MySQL+Milvus 双写、混合检索（BM25 + 向量 + RRF）、重排、引用溯源 |
| 上下文管理 | `app/core/context.py` | 三层结构（原文 / 截短 / 梗概）、预算从窗口倒推、后台异步摘要 |
| 数据飞轮 | `app/services/flywheel.py` | 低置信度入池、标准化去重、审核回写、评估趋势、按意图成本 |
| 主题分类 | `app/ml/` `app/services/topic_service.py` | RoBERTa-wwm-ext 17 类多标签，ONNX 推理，后台分布页 |
| 可观测性 | `app/observability/` | Langfuse trace / span / 意图元数据，全链路旁路降级 |

数据层：MySQL 存业务表、会话消息、知识块原文；Milvus 只存向量与检索所需的标量字段，原文一律回 MySQL 取。

---

## 快速开始

```bash
# 1. 依赖
pip install -r requirements.txt

# 2. 配置
cp .env.example .env          # 填 LLM_API_KEY

# 3. 模型权重（约 4.5G，不入库）
pip install -U huggingface_hub
huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3
huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir models/bge-reranker-v2-m3

# 4. 数据库（Docker MySQL 8.4，主机端口 3307）
docker compose up -d
python scripts/apply_schema.py
python scripts/seed_faq.py

# 5. 知识库（切片 + 嵌入，需要上一步的 bge-m3）
python scripts/seed_knowledge.py
python scripts/build_knowledge.py

# 6. 起服务
python -m uvicorn app.main:app --port 8000
# 打开 http://127.0.0.1:8000/
```

向量库默认用 Milvus Lite 本地文件（`./data/ch07_milvus.db`，与 `ch07.db` 里 986 块语料配套；`data/milvus_lite.db` 是早期遗留的坏文件，别指过去），零 Docker 依赖；切服务端只改 `VECTOR_STORE_URI`。

主题分类器（409MB，17 类多标签）同样不入库：用 `scripts/train_topic_model.py` 训练、`scripts/export_topic_onnx.py` 导出到 `models/topic_classifier/`，或从 `scripts/fetch_topic_model.py` 指定的实验目录搬运（会校验哈希）。

### 零依赖演示（SQLite + Milvus Lite，不起 Docker）

```bash
# 演示主库：986 块知识库 + 演示用池子数据（.env 里写的是 MySQL，要显式覆盖）
DATABASE_URL=sqlite+aiosqlite:///./data/ch07.db \
VECTOR_STORE_URI=./data/ch07_milvus.db \
python -m uvicorn app.main:app --port 8000
```

三种验证各管一段，别混用：

| 命令 | 验什么 | 会不会改库 |
|---|---|---|
| `python scripts/demo_smoke.py --rebuild` | 八个出口还在不在（重建 `demo.db` + 8 条探针） | 只动 `demo.db`/`demo_milvus.db` |
| `python scripts/demo_rehearsal.py` | 演示脚本那 12 幕顺不顺、各花多久（打 `ch07.db`） | 会（含飞轮入池 → 通过 → 写库） |
| `python scripts/demo_rehearsal.py --only C2,C3` | 只验某一幕 | 视幕而定 |

逐条演示清单（输入什么、屏幕上看到什么、讲什么、失败了换哪句）见
[`docs/演示脚本.md`](docs/演示脚本.md)，耗时来自 `reports/demo_rehearsal.md`。

---

## 目录结构

```
app/
  api/            路由与接口
  agent/          LangGraph 图、意图、闸、ReAct、退款子流程
  core/           配置之外的公共件：工具、上下文、prompt、mock 数据
  rag/            切分、嵌入、索引、检索、重排、引用
  services/       业务服务（对话编排、飞轮、评估、主题、用量）
  repositories/   数据访问层
  eval/           指标实现
  observability/  Langfuse 接入
  static/         聊天页与后台页
scripts/          建库、评测、演示、运维脚本
tests/            单测 + eval 用例与执行器
docs/             设计与方案文档
dev-notes/        逐章开发记录
reports/          评估报告产物
```

---

## 测试与评测

```bash
python -m pytest        # 全量单测
```

各章另有 eval 集（抽取、工具选择、召回、检索四策略、意图、多轮指代、对话、主题分类），阈值 90%，运行方式见 `tests/eval/` 与 `dev-notes/`。检索与生成类指标是分桶统计 + 人工抽检，不是单一分数。

---

## 文档

- `docs/superpowers/specs/` — 各章设计文档
- `docs/superpowers/plans/` — 实施计划
- `dev-notes/ch01.md` ~ `ch10.md` — 逐章开发记录（踩坑、取舍、为什么这么写）
- `reports/` — 评估报告与对比数据

---

## 边界

本地/内网演示定位：无鉴权、无用户体系；订单、商品、物流数据为合成 mock；转人工是前端模拟；向量库单机嵌入式。真实生产化还需要鉴权、支付与工单系统对接、检索侧纠错、多标签派单等一整套工作——这些不在本项目范围内，逐条说明见 `dev-notes/`。
