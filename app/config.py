"""Application settings loaded from environment / .env file."""

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-flash"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 512
    llm_timeout: int = 60
    llm_max_retries: int = 1
    # with_structured_output method: "json_mode"(DeepSeek/GLM) 或 "function_calling"。
    # DeepSeek 的 JSON 模式要求提示词里出现 "json" 字样,本项目提示词均满足。
    llm_structured_method: str = "json_mode"
    # DeepSeek 的思考模式默认开启(默认 effort=high),关掉才快。
    # 空字符串 = 不发这个字段(换到不认识的供应商时用)。
    # 注意:思考模式 + tools 时,上一轮的 reasoning_content 必须原样回传,否则 400;
    # 本项目 CH02 的工具循环没做这件事,所以默认 disabled。
    llm_thinking: str = "disabled"
    history_token_budget: int = 2048
    # CH02: database (Docker MySQL on host port 3307, see docker-compose.yml).
    database_url: str = "mysql+aiomysql://root:root@127.0.0.1:3307/kefu_agent"
    db_echo: bool = False
    # CH02: tool execution timeout (registry executor).
    tool_timeout_seconds: float = 10.0
    # CH03: RAG — 向量库连接与嵌入模型。
    # 注意:不能叫 milvus_uri(env MILVUS_URI)——pymilvus 自己会读 .env 里这个名字,
    # import 时就按 http(s):// 形式解析,相对路径直接抛 ConnectionConfigException。
    vector_store_uri: str = "./data/milvus_lite.db"
    embedding_model: str = "BAAI/bge-m3"
    # CH04: 检索增强 — 重排模型 + 生成前的证据自评开关。
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    # 在线检索策略(dense / bm25 / hybrid / hybrid_rerank)。四策略都能切,
    # 但线上只该跑一种:评估口径跟着它走,否则趋势图的数字和用户看到的不是同一条链路。
    # 默认 hybrid_rerank:两路召回 RRF 融合后再精排,是检索指标最好的一档
    # (见 reports/ch04_eval_v2.md:hybrid R@5 0.84 高于 dense 0.81)。
    # 代价是重排模型常驻约 2.2G 内存、单次检索多几百毫秒,内存紧的开发机可
    # 用 RAG_STRATEGY=dense 切回纯向量。
    rag_strategy: str = "hybrid_rerank"
    # 检索证据自评(生成前闸门):关掉就只按「零命中」拒答
    evidence_selfcheck_enabled: bool = True
    # 混合检索两路各召回多少条。语料涨到几百块后 50 等于扫 10%,
    # 会把评估重新变成「大范围扫描」——按库规模调到 20(约 4%)。
    recall_route_limit: int = 20
    # CH04 评测:eval 的裁判模型。留空回落生成模型,报告会显式标注「裁判=生成」。
    # 想让绝对分数可信,必须填成与 llm_model 不同的模型(最好不同厂商),
    # 需要时同时给 judge_base_url / judge_api_key。
    judge_model: str = ""
    judge_base_url: str = ""
    judge_api_key: str = ""
    # CH05: 工作流编排 — 知识类问题的置信度闸阈值与主力 Agent 的停止条件。
    # kb_min_score 是「算不算一条有效证据」的门槛(ch09 起) — 实测 357 块库上库内问题
    # 0.49~0.78、库外 0.38~0.62 且有重叠,单值分不开「沾边」和「答得上」,
    # 所以它只用来数有效证据条数,真正的判定看下面的 evidence_min_confidence。
    kb_min_score: float = 0.45
    # CH09: 三信号合成的置信度阈值(0.60*top1 + 0.25*有效证据数 + 0.15*分差)。
    # 0.45 由 scripts/calibrate_gate.py 扫出,报告见 reports/ch09_gate_calibration.md。
    # 986 块库上的复标结果:0.45 处正例误杀 4.2%(4/96)、负例拦下 80.0%(12/15);
    # 0.36~0.45 是同分平台,选 0.45 是为了给 top1 留一点余量,不是拟合出来的点。
    # 换检索策略等于换分数分布,必须重扫:同一份题在 dense 上最优是 0.63,
    # 且负例只能拦下 46.7%——闸在 dense 档失去分辨力,别只改阈值了事。
    # 改这个值等于改「多容易被拒答」,别凭手感调。
    evidence_min_confidence: float = 0.45
    # CH05/CH09: 退款子流程单独一根门槛。退款问题里必然混着订单号和商品名
    # (「订单1003的大容量双肩通勤包能退吗」),同一块退货条款在知识路上精排 0.99、
    # 在退款路上只 0.14,复用 0.45 会把所有退款请求判成拒答。
    # 0.10 的依据:986 块库上该路 top1 实测 0.13~0.7,库外问题多在 0.05 以下。
    refund_gate_min_score: float = 0.10
    agent_max_steps: int = 4
    agent_token_budget: int = 6000
    # CH07: 上下文管理 — 预算从窗口倒推,不写死历史长度。
    # 这五项是倒推公式的输入,改任何一个都会改变层 1/层 2 的预算(验收 2 用它演示级联)。
    model_context_window: int = 65536
    max_output_tokens: int = 2000
    max_user_input_tokens: int = 2000
    tool_result_max_tokens: int = 1200
    rerank_top_k: int = 5
    # 层 2 里客服答复只留开头这么多字(用户原话不动,工具结果换成一行标识)
    layer2_assistant_head: int = 40
    # 后台摘要开关与超时(摘要失败不影响当轮回复)
    context_summary_enabled: bool = True
    context_summary_timeout: int = 60
    # CH03: 对话挖知识定时任务(默认关;开了 lifespan 起后台循环,按 interval 跑一轮)
    mine_qa_enabled: bool = False
    mine_qa_interval_hours: int = 24
    mine_qa_batch_size: int = 20
    # CH08: MCP 工具接入。mcp_servers 用「名字=URL」逗号分隔,例如
    # logistics=http://127.0.0.1:8121/mcp,aftersale=http://127.0.0.1:8122/mcp
    # 开了之后内置 query_logistics 下线,由物流 Server 接管(注册表不许重名)。
    mcp_enabled: bool = False
    mcp_servers: str = ""
    # 清单缓存 TTL:Server 重启加了工具,最多等这么久就能被用上,不用重启客服服务
    mcp_tool_ttl_seconds: int = 30
    # 外部 MCP 工具的调用超时通常比内置长(多一跳网络)
    mcp_timeout_seconds: float = 20.0
    # CH10: 主题分类器(旁路,主链路不调)。模型资产 400MB+ 不入 git,
    # 由 scripts/fetch_topic_model.py 拉到这个目录并校验哈希。
    topic_enabled: bool = False
    topic_model_dir: str = "./models/topic_classifier"
    # 阈值是验证集上扫出来的,不是 0.5;换模型必须重新扫
    topic_threshold: float = 0.20
    # 攒够一批归一次:脚本每次最多取这么多条未归类问题
    topic_batch_size: int = 50
    # CH09: 评估流水线(默认关)。跑一轮要加载嵌入 + 重排模型,开发台不该常驻 4G。
    eval_pipeline_enabled: bool = False
    eval_pipeline_interval_hours: int = 24
    # 用哪份用例跑:ch04 那份是唯一有分桶与权重的完整检索集
    eval_cases_path: str = "tests/eval/ch04_cases.jsonl"
    # CH09: Langfuse 可观测性。默认全关——没配好时聊天链路零感知
    # (key 缺失 / host 不通都静默降级,见 app/observability/tracing.py)。
    # host 默认官方 Cloud;自托管六件套见 docker-compose.langfuse.yml。
    # 账号在美区的(us.cloud.langfuse.com)常按 SDK 文档写成
    # LANGFUSE_BASE_URL——两个名字都认:只认一个的话,另一个会被
    # 静默忽略, trace 全发去错的大区,key 还会鉴权失败。
    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        validation_alias=AliasChoices("LANGFUSE_HOST", "LANGFUSE_BASE_URL"),
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
