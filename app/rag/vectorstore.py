"""Milvus 封装(ch03 建库 / ch04 混合检索)。

集合 knowledge 四个字段:
- chunk_id   Int64 主键,与 MySQL knowledge_chunks.id 严格对齐(双写幂等键);
- embedding  FloatVector 1024 维(BGE-M3),COSINE,FLAT 暴力索引(见 INDEX_TYPE 注释);
- text       VarChar,开启分词(内置 jieba 中文分词),挂 BM25 函数自动产出 sparse 向量;
- sparse     SparseFloatVector,BM25 函数的输出字段,SPARSE_INVERTED_INDEX + BM25 度量;
- category   VarChar 标量字段,元数据过滤用(先过滤再检索)。

原文仍只存 MySQL(权威源),Milvus 里存 text 是为了让 BM25 有词可索引——这是
ch04「关键词召回这条路也落在 Milvus 上」的前提,不是把原文搬进向量库。

upsert 语义:同主键重复写不报错、不产生重复行(建库重跑的前提)。
三条检索路径:dense(COSINE)、sparse(BM25)、hybrid(两路各 Top-N 后 RRF 融合)。
异常统一包成 VectorStoreError,上层(retriever / query_faq)据此降级。

环境变量名注意事项:pymilvus 自己会读 `.env` 里的 `MILVUS_URI`,故本项目配置项叫
`VECTOR_STORE_URI`,避开与 pymilvus 内部配置撞名。

测试替身 FakeMilvus:内存 dict + 余弦/词重叠暴力扫 + 本地 RRF,不依赖 Docker/文件库。
"""

from __future__ import annotations

import logging
import math
import re
import time

from app.rag.embeddings import DIMENSION

logger = logging.getLogger(__name__)

COLLECTION_NAME = "knowledge"
# 索引类型定死 FLAT(暴力精确检索),不用默认的 AUTOINDEX(HNSW 近似):
# 1. 演示量级(几百块)暴力全扫比近似检索更准,构建开销可忽略;
# 2. milvus-lite 的 FLAT 走 BruteForceIndex,索引持久化用 Python open()——
#    AUTOINDEX 走 faiss 的 write_index(const char*),Windows 上 faiss 用窄字符
#    ANSI API,UTF-8 编码的中文路径(D:\客服agent\...)直接 ENOENT,第二进程
#    load_collection 必崩(实踩:跨进程重开 Lite 文件库复现)。
INDEX_TYPE = "FLAT"

# 集合处于 released 态时的重试次数(含首次共 4 次)。Lite 上 load 是异步生效的,
# 只重试一次经常仍拿到 code=101,见 _run 的说明。
_RELEASED_RETRIES = 3

# 中文分词器:Milvus 服务端可配 chinese(lindera),但 milvus-lite 3.2.2 只编进了
# standard / jieba(实测 chinese 报 unknown tokenizer type)。jieba 同为 Milvus
# 内置中文分词器,词典更贴电商口语,这里统一用 jieba。
ANALYZER = {"type": "jieba"}

# text 字段长度上限(Milvus VARCHAR 上限 65535;块正文远小于此,取 8192 留余量)
TEXT_MAX_LENGTH = 8192
CATEGORY_MAX_LENGTH = 512

DEFAULT_RRF_K = 60
DEFAULT_ROUTE_LIMIT = 50  # 每一路粗排召回条数(ch04:各召回 Top-50)


class VectorStoreError(RuntimeError):
    """Milvus 不可用(连接失败 / 集合不存在 / 查询异常)。"""


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+")


def _tokens(text: str) -> list[str]:
    """Fake 用极简切词:中文按 2 字滑窗 + 英文数字整词,够验排序语义。"""
    out: list[str] = []
    for piece in _TOKEN_RE.findall(text or ""):
        if piece.isascii():
            out.append(piece.lower())
        else:
            out.extend(piece[i : i + 2] for i in range(max(1, len(piece) - 1)))
    return out


class FakeMilvus:
    """内存替身,镜像 MilvusClient 高层接口(has_collection/create_collection/
    upsert/search/hybrid_search),不依赖 Docker/文件库。

    与真客户端唯一差别:无索引/无持久化,BM25 用词重叠打分而非真实 IDF——
    单测只验调用契约、融合与排序语义,数值不参与断言。
    """

    def __init__(self) -> None:
        self._rows: dict[int, dict] = {}
        self._collections: set[str] = set()
        self.create_collection_calls = 0

    def has_collection(self, name: str) -> bool:
        return name in self._collections

    def load_collection(self, name: str, **kwargs) -> None:
        pass  # Fake 无加载态概念

    def create_collection(self, name, schema=None, index_params=None, **kwargs) -> None:
        self.create_collection_calls += 1
        self._collections.add(name)

    def upsert(self, collection_name: str, data: list[dict], **kwargs) -> dict:
        for r in data:
            self._rows[r["chunk_id"]] = dict(r)
        return {"upsert_count": len(data)}

    # ------------------------------------------------------------ 单路检索

    def _match_filter(self, row: dict, expr: str | None) -> bool:
        if not expr:
            return True
        m = re.search(r'category\s*==\s*"([^"]*)"', expr)
        return True if not m else row.get("category") == m.group(1)

    def search(
        self,
        collection_name: str,
        data,
        anns_field: str = "embedding",
        limit: int = 10,
        filter: str | None = None,
        output_fields: list[str] | None = None,
        **kwargs,
    ) -> list[list[dict]]:
        if anns_field == "sparse":
            query = _tokens(data[0] if isinstance(data, list) else data)
            scored = []
            for cid, row in self._rows.items():
                if not self._match_filter(row, filter):
                    continue
                doc = _tokens(row.get("text", ""))
                overlap = sum(1 for t in set(query) if t in doc)
                if overlap:
                    scored.append((cid, float(overlap)))
        else:
            query = data[0]
            scored = [
                (cid, _cosine(query, row["embedding"]))
                for cid, row in self._rows.items()
                if self._match_filter(row, filter)
            ]
        scored.sort(key=lambda t: (-t[1], t[0]))
        return [[{"chunk_id": cid, "distance": score} for cid, score in scored[:limit]]]

    # ------------------------------------------------------------ 混合检索

    def hybrid_search(
        self,
        collection_name: str,
        reqs: list,
        ranker=None,
        limit: int = 10,
        output_fields: list[str] | None = None,
        **kwargs,
    ) -> list[list[dict]]:
        """本地 RRF 融合,语义与 Milvus hybrid_search 一致:按各路排名求和
        1/(k+rank),同分按 chunk_id 升序稳定。"""
        k = DEFAULT_RRF_K
        if ranker is not None and hasattr(ranker, "dict"):
            params = (ranker.dict() or {}).get("params") or {}
            k = int(params.get("k", DEFAULT_RRF_K))
        scores: dict[int, float] = {}
        for req in reqs:
            res = self.search(
                collection_name,
                req.data,
                anns_field=req.anns_field,
                limit=req.limit,
                filter=getattr(req, "expr", None),
            )
            for rank, hit in enumerate(res[0] if res else []):
                cid = hit["chunk_id"]
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        ranked = sorted(scores.items(), key=lambda t: (-t[1], t[0]))[:limit]
        return [[{"chunk_id": cid, "distance": score} for cid, score in ranked]]


class MilvusStore:
    """MilvusClient 薄封装;client 可注入(FakeMilvus 单测用),否则按 uri 真连。"""

    def __init__(self, uri: str | None = None, client=None) -> None:
        self._client = client
        if self._client is None:
            from app.config import get_settings

            from pymilvus import MilvusClient

            try:
                self._client = MilvusClient(uri or get_settings().vector_store_uri)
            except Exception as exc:  # noqa: BLE001 - 统一包成 VectorStoreError
                raise VectorStoreError(f"Milvus 连接失败:{exc}") from exc

    # ------------------------------------------------------------ 集合管理

    def ensure_collection(self) -> None:
        """幂等建集合(Lite 文件库 / 服务端同行为);已存在则跳过并确保已加载。

        显式 schema:chunk_id 主键 + embedding 向量(FLAT/COSINE)+ text(分词后产
        BM25 稀疏向量)+ sparse(BM25 输出)+ category(元数据过滤)。

        load_collection 不可省:Milvus Lite 文件库跨进程重开时集合是 released 态,
        不 load 直接 search 报 code=101。
        """
        try:
            if self._client.has_collection(COLLECTION_NAME):
                self._client.load_collection(COLLECTION_NAME)
                return
            from pymilvus import (
                CollectionSchema,
                DataType,
                FieldSchema,
                Function,
                FunctionType,
            )
            from pymilvus.milvus_client.index import IndexParams

            schema = CollectionSchema(
                fields=[
                    FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=True),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
                    FieldSchema(
                        name="text",
                        dtype=DataType.VARCHAR,
                        max_length=TEXT_MAX_LENGTH,
                        enable_analyzer=True,
                        analyzer_params=ANALYZER,
                    ),
                    FieldSchema(
                        name="category",
                        dtype=DataType.VARCHAR,
                        max_length=CATEGORY_MAX_LENGTH,
                    ),
                    FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
                ],
                functions=[
                    Function(
                        name="bm25",
                        input_field_names=["text"],
                        output_field_names=["sparse"],
                        function_type=FunctionType.BM25,
                    )
                ],
            )
            index_params = IndexParams()
            index_params.add_index(
                field_name="embedding", index_type=INDEX_TYPE, metric_type="COSINE"
            )
            index_params.add_index(
                field_name="sparse",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
            )
            self._client.create_collection(
                COLLECTION_NAME, schema=schema, index_params=index_params
            )
        except VectorStoreError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Milvus 建集合失败:{exc}") from exc

    def drop_collection(self) -> None:
        """删集合重建用(schema 升级必须重建:新增字段无法回填旧行)。"""
        try:
            if self._client.has_collection(COLLECTION_NAME):
                self._client.drop_collection(COLLECTION_NAME)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Milvus 删集合失败:{exc}") from exc

    # ------------------------------------------------------------ 写入

    def upsert(self, rows: list[dict]) -> None:
        """rows: [{"chunk_id": int, "embedding": [...], "text": str, "category": str}]。"""
        if not rows:
            return
        payload = [
            {
                "chunk_id": r["chunk_id"],
                "embedding": r["embedding"],
                "text": (r.get("text") or "")[:TEXT_MAX_LENGTH],
                "category": (r.get("category") or "")[:CATEGORY_MAX_LENGTH],
            }
            for r in rows
        ]
        try:
            self._client.upsert(COLLECTION_NAME, data=payload)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Milvus upsert 失败:{exc}") from exc

    # ------------------------------------------------------------ 检索

    def search(
        self, vector: list[float], top_k: int = 5, expr: str | None = None
    ) -> list[tuple[int, float]]:
        """dense 单路检索。返回 [(chunk_id, score)] 按分降序;无命中返回 []。"""
        return self._run(
            lambda: self._client.search(
                COLLECTION_NAME,
                data=[vector],
                anns_field="embedding",
                limit=top_k,
                filter=expr,
            )
        )

    def search_sparse(
        self, query: str, top_k: int = 5, expr: str | None = None
    ) -> list[tuple[int, float]]:
        """BM25 单路检索;Milvus 对 BM25 返回负分(越大越相关),原样透出。"""
        return self._run(
            lambda: self._client.search(
                COLLECTION_NAME,
                data=[query],
                anns_field="sparse",
                limit=top_k,
                filter=expr,
            )
        )

    def hybrid_search(
        self,
        vector: list[float],
        query: str,
        top_k: int = 10,
        expr: str | None = None,
        route_limit: int = DEFAULT_ROUTE_LIMIT,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> list[tuple[int, float]]:
        """混合检索:两路各召回 route_limit 条,hybrid_search 配 RRF 融合取 top_k。"""
        from pymilvus import AnnSearchRequest, RRFRanker

        dense_req = AnnSearchRequest(
            data=[vector],
            anns_field="embedding",
            param={"metric_type": "COSINE"},
            limit=route_limit,
            expr=expr,
        )
        sparse_req = AnnSearchRequest(
            data=[query],
            anns_field="sparse",
            param={"metric_type": "BM25"},
            limit=route_limit,
            expr=expr,
        )
        return self._run(
            lambda: self._client.hybrid_search(
                COLLECTION_NAME,
                [dense_req, sparse_req],
                ranker=RRFRanker(rrf_k),
                limit=top_k,
            )
        )

    def _run(self, call) -> list[tuple[int, float]]:
        """统一异常口径 + released 态自动 load 重试(最多 3 次,退避)。

        跨进程场景:建库进程 upsert 后,应用进程里的 store 再 search 会撞
        「collection released」。`load_collection` 返回不等于加载完成——Lite 上实测
        紧接着重试仍会拿到同一个错,所以退避几次再试。

        为什么值得退避:检索抛错会被上层当成「这一路零命中」,多路检索时表现为
        证据变少、置信度被压低,最后莫名拒答——查起来像模型问题,其实是库的状态。
        """
        attempt = 0
        while True:
            try:
                res = call()
                break
            except Exception as exc:  # noqa: BLE001
                if not _is_released_error(exc) or attempt >= _RELEASED_RETRIES:
                    raise VectorStoreError(f"Milvus 检索失败:{exc}") from exc
                attempt += 1
                logger.warning("集合未加载,load 后重试(%d/%d)", attempt, _RELEASED_RETRIES)
                try:
                    self._client.load_collection(COLLECTION_NAME)
                except Exception as load_exc:  # noqa: BLE001
                    logger.warning("load_collection 失败:%s", load_exc)
                time.sleep(0.2 * attempt)
        hits = res[0] if res else []
        return [(_extract_pk(hit), float(hit["distance"])) for hit in hits]


def _is_released_error(exc: Exception) -> bool:
    """识别 Milvus「集合未加载」错(code=101 / released)。"""
    text = str(exc).lower()
    return "released" in text or "code=101" in text or "call load()" in text


def _extract_pk(hit) -> int:
    """从检索命中取主键。

    pymilvus 3.0 的命中按主键字段名暴露(fast-path 建集合时叫 chunk_id;
    默认 id);三种名字都兼容,取不到就抛错而不是静默错位。
    """
    for key in ("chunk_id", "id", "pk"):
        try:
            if key in hit:
                return int(hit[key])
        except (KeyError, TypeError, ValueError):
            continue
    raise VectorStoreError(f"检索命中缺少主键字段,keys={list(hit.keys())}")
