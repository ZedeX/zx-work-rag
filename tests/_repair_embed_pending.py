"""Repair embedding for files that still have text_extracted=1 and embedded=0.

This script writes directly to the main ChromaDB collection `work_corpus` with upsert,
then marks each successfully processed file as embedded in SQLite.
"""
import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import chromadb
from loguru import logger
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import config  # noqa: E402

LOG_FILE = Path(config.LOG_DIR) / "repair_embed_pending.log"
PROGRESS_FILE = Path(config.DATA_DIR) / "repair_embed_pending_progress.json"


def setup_logging():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    logger.add(str(LOG_FILE), rotation="20 MB", encoding="utf-8", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


def load_progress():
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    return {"last_file_id": 0, "files_done": 0, "files_failed": 0, "chunks_done": 0, "start_time": time.time()}


def save_progress(progress):
    PROGRESS_FILE.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def get_embed_client():
    if config.EMBEDDING_MODE == "local":
        return OpenAI(api_key="lm-studio", base_url=config.LM_STUDIO_BASE_URL), config.LM_STUDIO_EMBEDDING_MODEL
    return OpenAI(api_key=config.ARK_API_KEY, base_url=config.ARK_BASE_URL), config.ARK_EMBEDDING_MODEL


def chunk_text(text, chunk_size=None, overlap=None):
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def embed_texts(client, model, texts):
    for attempt in range(3):
        try:
            response = client.embeddings.create(model=model, input=texts)
            return [item.embedding for item in response.data]
        except Exception as exc:
            if attempt == 2:
                raise
            wait = 2 ** attempt
            logger.warning(f"Embedding failed, retrying in {wait}s: {exc}")
            time.sleep(wait)


def count_pending(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*)
        FROM files
        WHERE is_duplicate = 0
          AND text_extracted = 1
          AND embedded = 0
          AND status = 'text_extracted'
          AND extracted_text IS NOT NULL
          AND extracted_text != ''
    """)
    return cur.fetchone()[0]


def iter_pending_files(conn, after_file_id, limit=None):
    cur = conn.cursor()
    sql = """
        SELECT id, file_path, file_name, extension, file_category,
               folder_path, folder_tags, extracted_text, modified_time, size, status
        FROM files
        WHERE is_duplicate = 0
          AND text_extracted = 1
          AND embedded = 0
          AND status = 'text_extracted'
          AND extracted_text IS NOT NULL
          AND extracted_text != ''
          AND id > ?
        ORDER BY id
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    cur.execute(sql, (after_file_id,))
    while True:
        row = cur.fetchone()
        if row is None:
            break
        yield row


def main():
    parser = argparse.ArgumentParser(description="Repair pending embeddings")
    parser.add_argument("--limit", type=int, default=None, help="Optional max files to process")
    parser.add_argument("--reset-progress", action="store_true", help="Reset repair progress checkpoint")
    args = parser.parse_args()

    setup_logging()
    config.ensure_dirs()

    if args.reset_progress and PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        logger.info("Reset repair progress checkpoint")

    client, model = get_embed_client()
    logger.info(f"Using embedding model: {model} (mode={config.EMBEDDING_MODE})")

    chroma_client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    collection = chroma_client.get_or_create_collection(name="work_corpus", metadata={"hnsw:space": "cosine"})
    logger.info(f"Target collection work_corpus initial count: {collection.count()}")

    conn = sqlite3.connect(config.DB_PATH, timeout=60)
    pending = count_pending(conn)
    logger.info(f"Pending files before repair: {pending}")

    progress = load_progress()
    logger.info(f"Loaded progress: {progress}")

    batch_limit = 10 if config.EMBEDDING_MODE == "local" else config.BATCH_SIZE
    processed_this_run = 0

    for row in iter_pending_files(conn, progress["last_file_id"], args.limit):
        (file_id, file_path, file_name, ext, category, folder_path, folder_tags,
         text, modified_time, size, status) = row

        chunks = chunk_text(text)
        if not chunks:
            conn.execute("UPDATE files SET embedded = 1 WHERE id = ?", (file_id,))
            conn.commit()
            progress["last_file_id"] = file_id
            progress["files_done"] += 1
            save_progress(progress)
            continue

        ids = [f"f{file_id}_c{i}" for i in range(len(chunks))]
        metas = [{
            "file_id": file_id,
            "file_name": file_name or "",
            "extension": ext or "",
            "category": category or "",
            "folder_path": folder_path or "",
            "folder_tags": folder_tags or "",
            "modified_time": modified_time or "",
            "chunk_index": i,
            "total_chunks": len(chunks),
            "repair_status": status or "",
        } for i in range(len(chunks))]

        try:
            for start in range(0, len(chunks), batch_limit):
                sub_texts = chunks[start:start + batch_limit]
                sub_ids = ids[start:start + batch_limit]
                sub_metas = metas[start:start + batch_limit]
                vectors = embed_texts(client, model, sub_texts)
                collection.upsert(ids=sub_ids, embeddings=vectors, documents=sub_texts, metadatas=sub_metas)
                progress["chunks_done"] += len(sub_texts)
                time.sleep(0.05 if config.EMBEDDING_MODE == "local" else 0.3)

            conn.execute("UPDATE files SET embedded = 1 WHERE id = ?", (file_id,))
            conn.commit()
            progress["last_file_id"] = file_id
            progress["files_done"] += 1
            processed_this_run += 1

            if progress["files_done"] % 100 == 0:
                elapsed = max(time.time() - progress.get("start_time", time.time()), 1)
                logger.info(
                    f"Progress: files_done={progress['files_done']}, chunks_done={progress['chunks_done']}, "
                    f"failed={progress['files_failed']}, last_file_id={progress['last_file_id']}, "
                    f"rate={progress['chunks_done'] / elapsed:.2f} chunks/s"
                )
            save_progress(progress)

        except Exception as exc:
            logger.error(f"File failed id={file_id}, name={file_name}: {exc}")
            progress["files_failed"] += 1
            progress["last_file_id"] = file_id
            save_progress(progress)
            conn.commit()

    final_pending = count_pending(conn)
    final_count = collection.count()
    conn.close()

    logger.info("=== Repair complete ===")
    logger.info(f"Processed this run: {processed_this_run}")
    logger.info(f"Progress: {progress}")
    logger.info(f"Pending files after repair: {final_pending}")
    logger.info(f"Target collection work_corpus final count: {final_count}")


if __name__ == "__main__":
    main()
