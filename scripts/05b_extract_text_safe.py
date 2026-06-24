"""Safe resumable text extraction with per-file timeout.

This script avoids getting stuck on complex/corrupt Office/PDF files by
running extraction in a child process per file. Timed-out files are marked as
extraction_failed and skipped in future runs.
"""
import sys
import sqlite3
import multiprocessing as mp
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import config
from loguru import logger


MODULE_PATH = Path(__file__).parent / "05_extract_text.py"
spec = importlib.util.spec_from_file_location("extract_text_module", MODULE_PATH)
extmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extmod)

TIMEOUT_SECONDS = 90


def setup_logging():
    log_file = Path(config.LOG_DIR) / "05b_extract_text_safe.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_file), rotation="10 MB", encoding="utf-8",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


def extract_worker(path: str, ext: str, queue: mp.Queue):
    try:
        text, ok, err = extmod.extract_text(path, ext)
        queue.put((text, ok, err))
    except Exception as e:
        queue.put(("", False, str(e)[:500]))


def extract_with_timeout(path: str, ext: str, timeout: int = TIMEOUT_SECONDS):
    queue = mp.Queue()
    proc = mp.Process(target=extract_worker, args=(path, ext, queue))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        return "", False, f"Timeout after {timeout}s"
    if queue.empty():
        return "", False, "Extractor returned no result"
    return queue.get()


def main():
    setup_logging()
    logger.info("=" * 60)
    logger.info("Step 5b: Safe Text Extraction")
    logger.info("=" * 60)

    conn = sqlite3.connect(config.DB_PATH)
    extmod.add_text_columns(conn)
    c = conn.cursor()

    c.execute("""
        SELECT id, file_path, extension, file_category, folder_path,
               folder_tags, file_name, size, converted_path, status
        FROM files
        WHERE is_duplicate = 0
        AND text_extracted = 0
        AND status IN ('scanned', 'identified', 'converted')
        ORDER BY file_category, extension, size ASC
    """)
    rows = c.fetchall()
    logger.info(f"Files to process: {len(rows)}")

    text_extracted = 0
    metadata_only = 0
    failed = 0

    for idx, row in enumerate(rows, 1):
        (file_id, filepath, ext, category, folder_path,
         folder_tags, file_name, size, converted_path, status) = row

        if category in ("image", "video", "audio", "archive", "code", "other"):
            meta_text = extmod.create_metadata_record(
                filepath, folder_path, folder_tags, file_name, ext, size
            )
            c.execute("""
                UPDATE files
                SET extracted_text = ?, text_length = ?, text_extracted = 1, status = 'metadata_only'
                WHERE id = ?
            """, (meta_text, len(meta_text), file_id))
            metadata_only += 1
        else:
            actual_path = converted_path if converted_path and Path(converted_path).exists() else filepath
            actual_ext = Path(converted_path).suffix.lower() if converted_path and Path(converted_path).exists() else ext

            if not Path(actual_path).exists():
                c.execute("UPDATE files SET status = 'missing' WHERE id = ?", (file_id,))
                failed += 1
            else:
                text, ok, err = extract_with_timeout(actual_path, actual_ext)
                if ok:
                    if len(text) > 100000:
                        text = text[:100000] + "\n... (truncated)"
                    c.execute("""
                        UPDATE files
                        SET extracted_text = ?, text_length = ?, text_extracted = 1, status = 'text_extracted'
                        WHERE id = ?
                    """, (text, len(text), file_id))
                    text_extracted += 1
                else:
                    c.execute("""
                        UPDATE files
                        SET text_extracted = 0, status = 'extraction_failed', error_message = ?
                        WHERE id = ?
                    """, (err[:500], file_id))
                    failed += 1
                    if failed <= 50 or failed % 500 == 0:
                        logger.warning(f"Failed: {filepath} - {err[:200]}")

        if idx % 100 == 0:
            conn.commit()
        if idx % 500 == 0:
            logger.info(f"Progress {idx}/{len(rows)} | Text: {text_extracted}, Meta: {metadata_only}, Failed: {failed}")

    conn.commit()
    logger.info(f"Safe extraction complete. Text: {text_extracted}, Meta: {metadata_only}, Failed: {failed}")

    c.execute("SELECT status, COUNT(*) FROM files WHERE is_duplicate = 0 GROUP BY status ORDER BY COUNT(*) DESC")
    logger.info("=== Status Summary ===")
    for row in c.fetchall():
        logger.info(f"  {row[0]}: {row[1]}")
    conn.close()


if __name__ == "__main__":
    mp.freeze_support()
    main()
