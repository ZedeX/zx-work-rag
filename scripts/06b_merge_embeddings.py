"""Step 6b: Merge parallel embedding collections into one.

After 06a_embed_parallel.py finishes, this script:
1. Reads all work_corpus_p0..pN-1 collections
2. Merges their vectors into work_corpus
3. Updates SQLite: sets embedded=1 for all successfully embedded files
4. Reports final statistics
"""
import sys
import sqlite3
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import config
from loguru import logger
import chromadb


def setup_logging():
    log_file = Path(config.LOG_DIR) / "06b_merge.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_file), rotation="10 MB", encoding="utf-8",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


def load_all_progress(num_workers: int) -> dict:
    total = {"chunk_count": 0, "embedded_count": 0, "failed_count": 0}
    for wid in range(num_workers):
        p = Path(config.DATA_DIR) / f"embed_progress_w{wid}.json"
        if p.exists():
            prog = json.loads(p.read_text(encoding="utf-8"))
            total["chunk_count"] += prog["chunk_count"]
            total["embedded_count"] += prog["embedded_count"]
            total["failed_count"] += prog["failed_count"]
    return total


def merge_collections(num_workers: int):
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_DIR)

    # Create or get target collection
    target = chroma_client.get_or_create_collection(
        name="work_corpus",
        metadata={"hnsw:space": "cosine"}
    )
    logger.info(f"Target collection 'work_corpus' has {target.count()} vectors")

    total_merged = 0
    total_ids = set()

    # Get existing IDs to avoid duplicates
    try:
        existing = target.get(include=[])
        existing_ids = set(existing["ids"])
        logger.info(f"Existing IDs in target: {len(existing_ids)}")
    except Exception:
        existing_ids = set()

    for wid in range(num_workers):
        coll_name = f"work_corpus_p{wid}"
        try:
            source = chroma_client.get_collection(name=coll_name)
            count = source.count()
            logger.info(f"Merging {coll_name}: {count} vectors")

            if count == 0:
                continue

            # ChromaDB max get limit is typically ~5000
            batch_size = 5000
            offset = 0
            while offset < count:
                batch = source.get(
                    include=["embeddings", "documents", "metadatas"],
                    limit=batch_size,
                    offset=offset,
                )
                if not batch["ids"]:
                    break

                # Filter out already existing IDs
                new_ids = []
                new_embs = []
                new_docs = []
                new_metas = []
                for i, bid in enumerate(batch["ids"]):
                    if bid not in existing_ids and bid not in total_ids:
                        new_ids.append(bid)
                        new_embs.append(batch["embeddings"][i])
                        new_docs.append(batch["documents"][i])
                        new_metas.append(batch["metadatas"][i])
                        total_ids.add(bid)

                if new_ids:
                    target.add(
                        ids=new_ids,
                        embeddings=new_embs,
                        documents=new_docs,
                        metadatas=new_metas,
                    )
                    total_merged += len(new_ids)

                offset += batch_size
                logger.info(f"  Merged {offset}/{count} from {coll_name}, new: {len(new_ids)}")

        except Exception as e:
            logger.warning(f"Collection {coll_name} not found or error: {e}")

    logger.info(f"Total merged: {total_merged} new vectors")
    logger.info(f"Target collection now has: {target.count()} vectors")

    # Update SQLite - extract file_ids from ALL vectors in target collection
    # (both pre-existing and newly merged)
    conn = sqlite3.connect(config.DB_PATH)
    c = conn.cursor()

    # Get all file_ids from both existing and newly merged vectors
    embedded_file_ids = set()

    # From newly merged vectors (total_ids)
    for bid in total_ids:
        fid = int(bid.split("_")[0][1:])
        embedded_file_ids.add(fid)

    # From pre-existing vectors (existing_ids) - THIS WAS THE BUG
    # Previously only total_ids were used, missing ~62k files from earlier runs
    for bid in existing_ids:
        try:
            fid = int(bid.split("_")[0][1:])
            embedded_file_ids.add(fid)
        except (ValueError, IndexError):
            pass

    # Also include previously embedded files in SQLite
    prev = c.execute("SELECT id FROM files WHERE embedded = 1").fetchall()
    for (fid,) in prev:
        embedded_file_ids.add(fid)

    # Batch update
    updated = 0
    for fid in embedded_file_ids:
        c.execute("UPDATE files SET embedded = 1 WHERE id = ? AND embedded = 0", (fid,))
        if c.rowcount > 0:
            updated += 1

    conn.commit()
    logger.info(f"SQLite updated: {updated} files marked as embedded")

    # Final stats
    total_files = c.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    unique_files = c.execute("SELECT COUNT(*) FROM files WHERE is_duplicate = 0").fetchone()[0]
    text_extracted = c.execute("SELECT COUNT(*) FROM files WHERE text_extracted = 1").fetchone()[0]
    embedded = c.execute("SELECT COUNT(*) FROM files WHERE embedded = 1").fetchone()[0]
    conn.close()

    logger.info("=== Final Statistics ===")
    logger.info(f"Total files: {total_files}")
    logger.info(f"Unique files: {unique_files}")
    logger.info(f"Text extracted: {text_extracted}")
    logger.info(f"Embedded: {embedded}")
    logger.info(f"ChromaDB vectors: {target.count()}")

    progress = load_all_progress(num_workers)
    logger.info(f"Embedding progress: {progress}")


def main():
    setup_logging()
    logger.info("=" * 60)
    logger.info("Step 6b: Merging parallel embedding collections")
    logger.info("=" * 60)

    # Detect number of workers from progress files
    num_workers = 0
    for f in Path(config.DATA_DIR).glob("embed_progress_w*.json"):
        try:
            wid = int(f.stem.split("_w")[1])
            num_workers = max(num_workers, wid + 1)
        except (ValueError, IndexError):
            pass

    if num_workers == 0:
        logger.error("No progress files found. Did you run 06a_embed_parallel.py?")
        return

    logger.info(f"Detected {num_workers} workers")
    merge_collections(num_workers)
    logger.info("Merge complete.")


if __name__ == "__main__":
    main()
