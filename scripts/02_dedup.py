"""Step 2: Deduplicate files by MD5 hash.

Marks duplicates with is_duplicate=1 and duplicate_of=original_id.
Keeps the file with the shortest path (most likely the original).
"""
import sys
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import config
from loguru import logger


def setup_logging():
    log_file = Path(config.LOG_DIR) / "02_dedup.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_file), rotation="10 MB", encoding="utf-8",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


def dedup(conn: sqlite3.Connection):
    """Find and mark duplicate files by MD5 hash."""
    c = conn.cursor()

    # Find all duplicate hashes
    c.execute("""
        SELECT md5_hash, COUNT(*) as cnt
        FROM files
        WHERE md5_hash != '' AND is_duplicate = 0
        GROUP BY md5_hash
        HAVING cnt > 1
    """)
    dup_hashes = c.fetchall()
    logger.info(f"Found {len(dup_hashes)} hash groups with duplicates")

    total_dupes = 0
    for md5_hash, count in dup_hashes:
        # Get all files with this hash, ordered by path length (shortest first = likely original)
        c.execute("""
            SELECT id, file_path
            FROM files
            WHERE md5_hash = ? AND is_duplicate = 0
            ORDER BY LENGTH(file_path) ASC, file_path ASC
        """, (md5_hash,))
        rows = c.fetchall()

        if len(rows) <= 1:
            continue

        # Keep the first one (shortest path), mark rest as duplicates
        original_id = rows[0][0]
        for dup_id, dup_path in rows[1:]:
            c.execute("""
                UPDATE files
                SET is_duplicate = 1, duplicate_of = ?, status = 'duplicate'
                WHERE id = ?
            """, (original_id, dup_id))
            total_dupes += 1

        if total_dupes % 1000 == 0:
            conn.commit()
            logger.info(f"Marked {total_dupes} duplicates so far...")

    conn.commit()
    logger.info(f"Dedup complete. Total duplicates marked: {total_dupes}")

    # Summary
    c.execute("SELECT COUNT(*) FROM files WHERE is_duplicate = 1")
    logger.info(f"Total duplicate files: {c.fetchone()[0]}")
    c.execute("SELECT COUNT(*) FROM files WHERE is_duplicate = 0")
    logger.info(f"Unique files: {c.fetchone()[0]}")


def main():
    setup_logging()
    logger.info("=" * 60)
    logger.info("Step 2: Deduplication")
    logger.info("=" * 60)

    conn = sqlite3.connect(config.DB_PATH)
    dedup(conn)
    conn.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
