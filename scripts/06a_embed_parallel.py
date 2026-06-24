"""Step 6 Parallel: Generate embeddings with multiple workers.

Architecture:
- N workers (default 4), each handles 1/N of the files
- Each worker writes to its OWN ChromaDB collection (work_corpus_p0..pN-1)
- Each worker tracks progress in its OWN JSON progress file
- No SQLite writes during embedding (avoids lock contention)
- After all workers finish, run 06b_merge_embeddings.py to merge collections

Usage:
  python 06a_embed_parallel.py [--workers 4] [--worker-id 0]  # run single worker
  python 06a_embed_parallel.py --all --workers 4               # run all workers
"""
import sys
import json
import time
import argparse
import sqlite3
from pathlib import Path
from multiprocessing import Process

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import config
from loguru import logger
from openai import OpenAI


def setup_logging(worker_id: int):
    log_file = Path(config.LOG_DIR) / f"06a_embed_w{worker_id}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_file), rotation="10 MB", encoding="utf-8",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


def get_embed_client() -> tuple[OpenAI, str]:
    if config.EMBEDDING_MODE == "local":
        client = OpenAI(api_key="lm-studio", base_url=config.LM_STUDIO_BASE_URL)
        model = config.LM_STUDIO_EMBEDDING_MODEL
    else:
        client = OpenAI(api_key=config.ARK_API_KEY, base_url=config.ARK_BASE_URL)
        model = config.ARK_EMBEDDING_MODEL
    return client, model


def embed_texts(client: OpenAI, texts: list[str], model: str) -> list[list[float]]:
    batch_limit = 10 if config.EMBEDDING_MODE == "local" else 100
    if len(texts) > batch_limit:
        all_vectors = []
        for i in range(0, len(texts), batch_limit):
            sub = texts[i:i + batch_limit]
            all_vectors.extend(embed_texts(client, sub, model))
        return all_vectors

    for attempt in range(3):
        try:
            response = client.embeddings.create(model=model, input=texts)
            return [item.embedding for item in response.data]
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                time.sleep(2 ** attempt)
            elif "connect" in str(e).lower():
                time.sleep(5 * (attempt + 1))
            else:
                raise
    raise RuntimeError("Failed after 3 retries")


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def get_progress_path(worker_id: int) -> Path:
    return Path(config.DATA_DIR) / f"embed_progress_w{worker_id}.json"


def load_progress(worker_id: int) -> dict:
    p = get_progress_path(worker_id)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"last_file_id": 0, "embedded_count": 0, "chunk_count": 0, "failed_count": 0}


def save_progress(worker_id: int, progress: dict):
    p = get_progress_path(worker_id)
    p.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def run_worker(worker_id: int, total_workers: int):
    setup_logging(worker_id)
    logger.info(f"Worker {worker_id}/{total_workers} starting")

    import chromadb
    client, model = get_embed_client()
    logger.info(f"Using model: {model} (mode={config.EMBEDDING_MODE})")

    chroma_client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    collection_name = f"work_corpus_p{worker_id}"
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    logger.info(f"Collection: {collection_name} (existing: {collection.count()} vectors)")

    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    c = conn.cursor()

    # Count total pending files
    c.execute("""
        SELECT COUNT(*) FROM files
        WHERE is_duplicate = 0 AND text_extracted = 1 AND embedded = 0
        AND extracted_text IS NOT NULL AND extracted_text != ''
    """)
    total_pending = c.fetchone()[0]

    # First get just the file IDs assigned to this worker (lightweight query)
    c.execute("""
        SELECT id FROM (
            SELECT id, ROW_NUMBER() OVER (ORDER BY id) AS rn
            FROM files
            WHERE is_duplicate = 0 AND text_extracted = 1 AND embedded = 0
            AND extracted_text IS NOT NULL AND extracted_text != ''
        )
        WHERE rn %% %d = %d
        ORDER BY id
    """ % (total_workers, worker_id))
    file_ids = [r[0] for r in c.fetchall()]
    logger.info(f"Total pending: {total_pending}, this worker: {len(file_ids)} file IDs")

    progress = load_progress(worker_id)
    batch_texts = []
    batch_metas = []
    batch_ids = []
    effective_batch = 10 if config.EMBEDDING_MODE == "local" else config.BATCH_SIZE

    for file_idx, file_id in enumerate(file_ids):
        # Skip already processed (resume support)
        if file_id <= progress["last_file_id"]:
            continue

        # Fetch one file at a time to save memory
        c.execute("""
            SELECT id, file_path, file_name, extension, file_category,
                   folder_path, folder_tags, extracted_text, modified_time, size
            FROM files WHERE id = ?
        """, (file_id,))
        row = c.fetchone()
        if not row:
            continue

        (file_id, filepath, file_name, ext, category,
         folder_path, folder_tags, text, modified_time, size) = row

        chunks = chunk_text(text)

        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"f{file_id}_c{chunk_idx}"
            meta = {
                "file_id": file_id,
                "file_name": file_name,
                "extension": ext,
                "category": category,
                "folder_path": folder_path,
                "folder_tags": folder_tags or "",
                "modified_time": modified_time,
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks),
            }
            batch_texts.append(chunk)
            batch_metas.append(meta)
            batch_ids.append(chunk_id)

            if len(batch_texts) >= effective_batch:
                try:
                    vectors = embed_texts(client, batch_texts, model)
                    collection.add(
                        ids=batch_ids, embeddings=vectors,
                        documents=batch_texts, metadatas=batch_metas,
                    )
                    progress["embedded_count"] += len(batch_texts)
                    progress["chunk_count"] += len(batch_texts)
                    progress["last_file_id"] = file_id
                    save_progress(worker_id, progress)

                    if progress["chunk_count"] % 500 == 0 or progress["chunk_count"] % 1000 == 0:
                        logger.info(f"W{worker_id}: {progress['chunk_count']} chunks, {progress['embedded_count']} embedded, {progress['failed_count']} failed")
                except Exception as e:
                    logger.error(f"W{worker_id} batch failed: {e}")
                    # Try one by one
                    for bt, bm, bi in zip(batch_texts, batch_metas, batch_ids):
                        try:
                            vecs = embed_texts(client, [bt], model)
                            collection.add(ids=[bi], embeddings=vecs, documents=[bt], metadatas=[bm])
                            progress["embedded_count"] += 1
                            progress["chunk_count"] += 1
                        except Exception as e2:
                            progress["failed_count"] += 1
                            if progress["failed_count"] <= 20:
                                logger.warning(f"W{worker_id} single failed {bi}: {e2}")

                    progress["last_file_id"] = file_id
                    save_progress(worker_id, progress)

                batch_texts = []
                batch_metas = []
                batch_ids = []
                time.sleep(0.05 if config.EMBEDDING_MODE == "local" else 0.3)

    # Remaining batch
    if batch_texts:
        try:
            vectors = embed_texts(client, batch_texts, model)
            collection.add(ids=batch_ids, embeddings=vectors, documents=batch_texts, metadatas=batch_metas)
            progress["embedded_count"] += len(batch_texts)
            progress["chunk_count"] += len(batch_texts)
        except Exception as e:
            logger.error(f"W{worker_id} final batch failed: {e}")
            progress["failed_count"] += len(batch_texts)

    save_progress(worker_id, progress)
    logger.info(f"W{worker_id} DONE. Chunks: {progress['chunk_count']}, Embedded: {progress['embedded_count']}, Failed: {progress['failed_count']}")
    logger.info(f"W{worker_id} Collection {collection_name} count: {collection.count()}")
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Parallel embedding")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--worker-id", type=int, default=None, help="Run single worker by ID")
    parser.add_argument("--all", action="store_true", help="Run all workers in parallel")
    args = parser.parse_args()

    config.ensure_dirs()

    if args.worker_id is not None:
        # Run single worker
        run_worker(args.worker_id, args.workers)
    elif args.all:
        # Run all workers in parallel
        logger.info(f"Starting {args.workers} workers in parallel")
        processes = []
        for wid in range(args.workers):
            p = Process(target=run_worker, args=(wid, args.workers))
            p.start()
            processes.append(p)
            logger.info(f"Started worker {wid} (PID {p.pid})")

        for p in processes:
            p.join()

        logger.info(f"All {args.workers} workers finished")
        # Print summary
        total_chunks = 0
        total_embedded = 0
        total_failed = 0
        for wid in range(args.workers):
            prog = load_progress(wid)
            total_chunks += prog["chunk_count"]
            total_embedded += prog["embedded_count"]
            total_failed += prog["failed_count"]
            logger.info(f"W{wid}: {prog}")
        logger.info(f"TOTAL: chunks={total_chunks}, embedded={total_embedded}, failed={total_failed}")
        logger.info("Next step: python 06b_merge_embeddings.py")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
