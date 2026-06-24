"""Step 5c: Fast single-process extraction for remaining files.

Skips PDFs in this pass to avoid hangs. Old Office files are extracted only when
LibreOffice conversion produced a converted_path.
"""
import sys
import sqlite3
import importlib.util
import signal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import config
from loguru import logger


MODULE_PATH = Path(__file__).parent / "05_extract_text.py"
spec = importlib.util.spec_from_file_location("extract_text_module", MODULE_PATH)
extmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extmod)

OLD_OFFICE_EXTS = {".doc", ".xls", ".ppt", ".dot", ".pot", ".xlt"}
PDF_EXTS = {".pdf"}
FAST_TEXT_EXTS = {
    ".docx", ".xlsx", ".pptx", ".txt", ".csv", ".html", ".htm",
    ".xml", ".json", ".md", ".log", ".rtf",
}
SKIP_FAST_EXTS = {".xlsx", ".pptx"}
METADATA_CATEGORIES = {"image", "video", "audio", "archive", "code", "other"}
STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    logger.warning("Stop requested; finishing current file and committing progress.")


signal.signal(signal.SIGINT, request_stop)
if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, request_stop)


def setup_logging():
    log_file = Path(config.LOG_DIR) / "05c_extract_remaining_fast.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_file), rotation="10 MB", encoding="utf-8",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


def mark_failed(cursor, file_id: int, error: str):
    cursor.execute("""
        UPDATE files
        SET text_extracted = 0, status = 'extraction_failed', error_message = ?
        WHERE id = ?
    """, (error[:500], file_id))


def mark_text(cursor, file_id: int, text: str, status: str = "text_extracted"):
    if len(text) > 100000:
        text = text[:100000] + "\n... (truncated)"
    cursor.execute("""
        UPDATE files
        SET extracted_text = ?, text_length = ?, text_extracted = 1, status = ?
        WHERE id = ?
    """, (text, len(text), status, file_id))


def main():
    setup_logging()
    logger.info("=" * 60)
    logger.info("Step 5c: Fast Remaining Extraction")
    logger.info("=" * 60)

    conn = sqlite3.connect(config.DB_PATH)
    extmod.add_text_columns(conn)
    c = conn.cursor()

    c.execute("""
        SELECT id, file_path, lower(extension), file_category, folder_path,
               folder_tags, file_name, size, converted_path, status
        FROM files
        WHERE is_duplicate = 0
        AND text_extracted = 0
        AND status NOT IN ('extraction_failed', 'missing')
        ORDER BY
            CASE
                WHEN file_category IN ('image', 'video', 'audio', 'archive', 'code', 'other') THEN 1
                WHEN lower(extension) = '.pdf' THEN 2
                WHEN lower(extension) IN ('.doc', '.xls', '.ppt', '.dot', '.pot', '.xlt') THEN 3
                ELSE 4
            END,
            file_category, lower(extension), size ASC
    """)
    rows = c.fetchall()
    logger.info(f"Files to process: {len(rows)}")

    text_extracted = 0
    metadata_only = 0
    skipped_pdf = 0
    old_office_not_converted = 0
    failed = 0
    unsupported = 0

    for idx, row in enumerate(rows, 1):
        (file_id, filepath, ext, category, folder_path,
         folder_tags, file_name, size, converted_path, status) = row
        ext = ext or ""

        try:
            if category in METADATA_CATEGORIES:
                meta_text = extmod.create_metadata_record(
                    filepath, folder_path, folder_tags, file_name, ext, size
                )
                mark_text(c, file_id, meta_text, "metadata_only")
                metadata_only += 1
            elif ext in PDF_EXTS:
                mark_failed(c, file_id, "Skipped PDF in fast pass")
                skipped_pdf += 1
            else:
                has_converted = converted_path and Path(converted_path).exists()
                actual_path = converted_path if has_converted else filepath
                actual_ext = Path(converted_path).suffix.lower() if has_converted else ext

                if ext in OLD_OFFICE_EXTS and not has_converted:
                    mark_failed(c, file_id, "Old Office not converted")
                    old_office_not_converted += 1
                elif actual_ext in SKIP_FAST_EXTS:
                    mark_failed(c, file_id, f"Skipped {actual_ext} in fast pass")
                    unsupported += 1
                elif actual_ext not in FAST_TEXT_EXTS:
                    mark_failed(c, file_id, f"No fast extractor for {actual_ext}")
                    unsupported += 1
                elif not Path(actual_path).exists():
                    c.execute("UPDATE files SET status = 'missing' WHERE id = ?", (file_id,))
                    failed += 1
                else:
                    text, ok, err = extmod.extract_text(actual_path, actual_ext)
                    if ok:
                        mark_text(c, file_id, text, "text_extracted")
                        text_extracted += 1
                    else:
                        mark_failed(c, file_id, err)
                        failed += 1
                        if failed <= 50 or failed % 500 == 0:
                            logger.warning(f"Failed: {filepath} - {err[:200]}")
        except Exception as e:
            mark_failed(c, file_id, str(e))
            failed += 1
            if failed <= 50 or failed % 500 == 0:
                logger.warning(f"Exception: {filepath} - {str(e)[:200]}")

        if idx % 100 == 0:
            conn.commit()
        if idx % 500 == 0:
            logger.info(
                f"Progress {idx}/{len(rows)} | Text: {text_extracted}, Meta: {metadata_only}, "
                f"PDF skipped: {skipped_pdf}, OldOffice no convert: {old_office_not_converted}, "
                f"Unsupported: {unsupported}, Failed: {failed}"
            )
        if STOP_REQUESTED:
            conn.commit()
            logger.warning(f"Stopped at {idx}/{len(rows)} after commit.")
            break

    conn.commit()
    logger.info(
        f"Fast extraction complete. Text: {text_extracted}, Meta-only: {metadata_only}, "
        f"PDF skipped: {skipped_pdf}, OldOffice no convert: {old_office_not_converted}, "
        f"Unsupported: {unsupported}, Failed: {failed}"
    )

    c.execute("SELECT status, COUNT(*) FROM files WHERE is_duplicate = 0 GROUP BY status ORDER BY COUNT(*) DESC")
    logger.info("=== Status Summary ===")
    for row in c.fetchall():
        logger.info(f"  {row[0]}: {row[1]}")
    conn.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
