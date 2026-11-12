"""回滚一条「飞轮补充」知识：把演示用的探针题恢复成「还没补过」。

    python scripts/flywheel_reset.py --keyword 安全座椅
    python scripts/flywheel_reset.py --keyword 安全座椅 --dry-run   # 只看会删什么
    python scripts/flywheel_reset.py --list                        # 现有飞轮补充块

为什么需要它:彩排跑一遍就把那条题补进知识库了,再演「拒答」就演不出来。要么换题
(题是耗材),要么把这条回滚掉再来一遍。录制前用这个脚本回滚,比重新建库快得多。

只动三类东西,范围按关键字精确匹配:
1. `knowledge_chunks` 里 `section_path` 以「飞轮补充」开头、且问题含关键字的块;
2. 上面这些块的 Milvus 实体(必须先 load_collection,集合 released 态删不动);
3. `review_queue` 里同问题的行:置回待审、清 `approved_answer`、次数归 1。

池内原话(`low_confidence_questions`)不删:它是「这条问题被问过」的事实,
删了反而让下次拒答再也入不了池。
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DB = "sqlite+aiosqlite:///./data/ch07.db"
DEFAULT_VECTOR = "./data/ch07_milvus.db"
COLLECTION = "knowledge"
FLYWHEEL_PATH_PREFIX = "飞轮补充"


def _sqlite_file(url: str) -> Path:
    tail = url.split("///", 1)[-1]
    return Path(tail) if Path(tail).is_absolute() else REPO_ROOT / tail.lstrip("./")


def _conn(url: str) -> sqlite3.Connection:
    con = sqlite3.connect(_sqlite_file(url))
    con.row_factory = sqlite3.Row
    return con


def list_flywheel_chunks(con: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(
        con.execute(
            "select id, questions, answer from knowledge_chunks "
            "where section_path like ? order by id",
            (FLYWHEEL_PATH_PREFIX + "%",),
        )
    )


def _delete_vectors(uri: str, chunk_ids: list[int], dry_run: bool) -> tuple[int, int]:
    """返回 (删前实体数, 删后实体数)。集合要先 load 才查得到、删得动。"""
    import time

    from pymilvus import MilvusClient

    client = MilvusClient(uri)
    for _ in range(8):
        try:
            client.load_collection(COLLECTION)
            time.sleep(0.4)
            before = client.query(COLLECTION, filter="chunk_id >= 0", output_fields=["count(*)"])
            break
        except Exception:  # noqa: BLE001 - released 态要等 load 生效
            time.sleep(1.0)
    else:
        raise SystemExit("向量库加载失败:检查 VECTOR_STORE_URI 是否被别的进程占用")

    before_n = int(str(before[0]).strip("{}'\"").split(":")[-1].strip(" '\""))
    if dry_run:
        return before_n, before_n
    client.delete(COLLECTION, ids=[str(i) for i in chunk_ids])
    time.sleep(0.3)
    for _ in range(8):
        try:
            after = client.query(COLLECTION, filter="chunk_id >= 0", output_fields=["count(*)"])
            after_n = int(str(after[0]).strip("{}'\"").split(":")[-1].strip(" '\""))
            break
        except Exception:  # noqa: BLE001
            time.sleep(1.0)
    else:
        after_n = before_n - len(chunk_ids)
    return before_n, after_n


def reset(db_url: str, vector_uri: str, keyword: str, *, dry_run: bool = False) -> dict:
    con = _conn(db_url)
    try:
        chunks = [
            r for r in list_flywheel_chunks(con) if keyword in (r["questions"] or "")
        ]
        if not chunks:
            return {"chunks": [], "reviews": [], "vectors": (0, 0)}

        ids = [int(r["id"]) for r in chunks]
        placeholders = ",".join("?" * len(ids))
        reviews = list(
            con.execute(
                "select id, normalized_question from review_queue where normalized_question like ?",
                (f"%{keyword}%",),
            )
        )
        review_ids = [int(r["id"]) for r in reviews]

        vectors = _delete_vectors(vector_uri, ids, dry_run)
        if dry_run:
            return {"chunks": ids, "reviews": review_ids, "vectors": vectors}

        con.execute(f"delete from knowledge_chunks where id in ({placeholders})", ids)
        for rid in review_ids:
            con.execute(
                "update review_queue set review_status='待审', approved_answer=NULL, "
                "occurrence_count=1 where id=?",
                (rid,),
            )
        con.commit()
        left = con.execute("select count(*) from knowledge_chunks").fetchone()[0]
        return {"chunks": ids, "reviews": review_ids, "vectors": vectors, "left": left}
    finally:
        con.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="回滚一条飞轮补充知识(演示探针复位)")
    parser.add_argument("--keyword", default="", help="问题里的关键字,如 安全座椅")
    parser.add_argument("--database-url", default=DEFAULT_DB)
    parser.add_argument("--vector-store-uri", default=DEFAULT_VECTOR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true", help="列出所有飞轮补充块")
    args = parser.parse_args()

    con = _conn(args.database_url)
    try:
        rows = list_flywheel_chunks(con)
        if args.list or not args.keyword:
            if not rows:
                print("没有飞轮补充块")
            for r in rows:
                print(f"  #{r['id']:>4}  {r['questions']}")
            return 0 if rows or args.list else 2
    finally:
        con.close()

    stats = reset(
        args.database_url, args.vector_store_uri, args.keyword, dry_run=args.dry_run
    )
    if not stats["chunks"]:
        print(f"没有含「{args.keyword}」的飞轮补充块,不用回滚")
        return 2
    head = "预览(未落库)" if args.dry_run else "已回滚"
    print(f"{head}:块 {stats['chunks']},审核行 {stats['reviews']}")
    print(f"向量实体 {stats['vectors'][0]} → {stats['vectors'][1]}")
    if not args.dry_run:
        print(f"知识库剩 {stats['left']} 块")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
