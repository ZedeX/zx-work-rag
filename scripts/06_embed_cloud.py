"""Step 6: Generate embeddings and store in ChromaDB.

Supports two modes:
- "local": Use LM Studio local embedding model (qwen3-embedding-0.6b)
- "cloud": Use Volcengine Ark API (doubao-embedding)

Set EMBEDDING_MODE in .env to switch between modes.
"""
import sys
import sqlite3
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import config
from loguru import logger

from openai import OpenAI


def setup_logging():
    log_file = Path(config.LOG_DIR) / "06_embed.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_file), rotation="10 MB", encoding="utf-8",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


def get_embed_client() -> tuple[OpenAI, str]:
    """Get embedding client and model based on EMBEDDING_MODE.

    Returns (client, model_name) tuple.
    """
    if config.EMBEDDING_MODE == "local":
        client = OpenAI(
            api_key="lm-studio",
            base_url=config.LM_STUDIO_BASE_URL,
        )
        model = config.LM_STUDIO_EMBEDDING_MODEL
        logger.info(f"Using LOCAL embedding: {model} via LM Studio")
    else:
        client = OpenAI(
            api_key=config.ARK_API_KEY,
            base_url=config.ARK_BASE_URL,
        )
        model = config.ARK_EMBEDDING_MODEL
        logger.info(f"Using CLOUD embedding: {model} via Volcengine Ark")
    return client, model


def embed_texts(client: OpenAI, texts: list[str],
                model: str = None) -> list[list[float]]:
    """Embed a batch of texts. Returns list of embedding vectors.

    Handles rate limiting with retries.
    For local mode, uses smaller batches (10) to avoid timeout.
    """
    max_retries = 3
    # Local LM Studio handles smaller batches better
    batch_limit = 10 if config.EMBEDDING_MODE == "local" else 100

    if len(texts) > batch_limit:
        # Split into sub-batches
        all_vectors = []
        for i in range(0, len(texts), batch_limit):
            sub_batch = texts[i:i + batch_limit]
            vectors = embed_texts(client, sub_batch, model)
            all_vectors.extend(vectors)
        return all_vectors

    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model=model,
                input=texts,
            )
            return [item.embedding for item in response.data]

        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                wait = 2 ** attempt
                logger.warning(f"Rate limited, waiting {wait}s... ({e})")
                time.sleep(wait)
            elif "connect" in str(e).lower() or "connection" in str(e).lower():
                wait = 5 * (attempt + 1)
                logger.warning(f"Connection error, waiting {wait}s... ({e})")
                time.sleep(wait)
            else:
                logger.error(f"Embedding API error: {e}")
                raise

    raise RuntimeError(f"Failed after {max_retries} retries")


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    """Split text into overlapping chunks."""
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap

    return chunks


def process_all(conn: sqlite3.Connection):
    """Process all extracted text: chunk -> embed -> store in ChromaDB."""
    import chromadb

    client, model = get_embed_client()
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    collection = chroma_client.get_or_create_collection(
        name="work_corpus",
        metadata={"hnsw:space": "cosine"}
    )

    c = conn.cursor()
    c.execute("""
        SELECT id, file_path, file_name, extension, file_category,
               folder_path, folder_tags, extracted_text, modified_time, size
        FROM files
        WHERE is_duplicate = 0
        AND text_extracted = 1
        AND embedded = 0
        AND extracted_text IS NOT NULL
        AND extracted_text != ''
        ORDER BY id
    """)
    rows = c.fetchall()
    logger.info(f"Files to embed: {len(rows)}")

    total_chunks = 0
    total_embedded = 0
    total_failed = 0
    batch_texts = []
    batch_metas = []
    batch_ids = []

    # For local mode, use smaller batch sizes
    effective_batch = 10 if config.EMBEDDING_MODE == "local" else config.BATCH_SIZE

    for row in rows:
        (file_id, filepath, file_name, ext, category,
         folder_path, folder_tags, text, modified_time, size) = row

        # Chunk the text
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
            total_chunks += 1

            # Send batch when full
            if len(batch_texts) >= effective_batch:
                try:
                    vectors = embed_texts(client, batch_texts, model)
                    collection.add(
                        ids=batch_ids,
                        embeddings=vectors,
                        documents=batch_texts,
                        metadatas=batch_metas,
                    )
                    total_embedded += len(batch_texts)
                    for bid in batch_ids:
                        fid = int(bid.split("_")[0][1:])
                        c.execute("UPDATE files SET embedded = 1 WHERE id = ?", (fid,))

                    conn.commit()
                    logger.info(f"Embedded batch: {total_embedded} chunks total ({total_failed} failed)")

                except Exception as e:
                    logger.error(f"Batch embedding failed: {e}")
                    # Try one by one
                    for bt, bm, bi in zip(batch_texts, batch_metas, batch_ids):
                        try:
                            vecs = embed_texts(client, [bt], model)
                            collection.add(ids=[bi], embeddings=vecs, documents=[bt], metadatas=[bm])
                            total_embedded += 1
                            fid = int(bi.split("_")[0][1:])
                            c.execute("UPDATE files SET embedded = 1 WHERE id = ?", (fid,))
                        except Exception as e2:
                            logger.error(f"Single embed failed for {bi}: {e2}")
                            total_failed += 1

                # Reset batch
                batch_texts = []
                batch_metas = []
                batch_ids = []

                # Rate limiting
                if config.EMBEDDING_MODE == "cloud":
                    time.sleep(0.5)
                else:
                    time.sleep(0.1)  # Local is faster

    # Process remaining batch
    if batch_texts:
        try:
            vectors = embed_texts(client, batch_texts, model)
            collection.add(
                ids=batch_ids,
                embeddings=vectors,
                documents=batch_texts,
                metadatas=batch_metas,
            )
            total_embedded += len(batch_texts)
            for bid in batch_ids:
                fid = int(bid.split("_")[0][1:])
                c.execute("UPDATE files SET embedded = 1 WHERE id = ?", (fid,))
            conn.commit()
        except Exception as e:
            logger.error(f"Final batch embedding failed: {e}")

    logger.info(f"Embedding complete. Total chunks: {total_chunks}, Embedded: {total_embedded}, Failed: {total_failed}")

    # Summary
    c.execute("SELECT COUNT(*) FROM files WHERE embedded = 1")
    logger.info(f"Files with embeddings: {c.fetchone()[0]}")
    logger.info(f"ChromaDB collection count: {collection.count()}")


def main():
    setup_logging()
    logger.info("=" * 60)
    logger.info(f"Step 6: Embedding (mode={config.EMBEDDING_MODE})")
    logger.info("=" * 60)

    config.ensure_dirs()
    conn = sqlite3.connect(config.DB_PATH)
    process_all(conn)
    conn.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
