"""Step 4: Convert old Office formats (.doc, .xls, .ppt) to modern formats.

Uses LibreOffice headless mode for batch conversion.
Falls back to marking as 'conversion_failed' if LibreOffice not available.
"""
import sys
import sqlite3
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import config, Config
from loguru import logger


def setup_logging():
    log_file = Path(config.LOG_DIR) / "04_convert_formats.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_file), rotation="10 MB", encoding="utf-8",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


# Conversion mapping: old_ext -> (new_ext, LibreOffice export filter)
CONVERT_MAP = {
    ".doc": (".docx", "docx:Office Open XML Text"),
    ".dot": (".docx", "docx:Office Open XML Text"),
    ".xls": (".xlsx", "xlsx:Calc MS Excel 2007 XML"),
    ".xlt": (".xlsx", "xlsx:Calc MS Excel 2007 XML"),
    ".ppt": (".pptx", "pptx:Impress MS PowerPoint 2007 XML"),
    ".pot": (".pptx", "pptx:Impress MS PowerPoint 2007 XML"),
}


def find_soffice() -> str:
    """Find LibreOffice soffice executable (cross-platform)."""
    import platform

    candidates = []
    system = platform.system()

    if system == "Windows":
        candidates = [
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
        ]
    elif system == "Darwin":  # macOS
        candidates = [
            "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        ]
    else:  # Linux
        candidates = [
            "/usr/bin/soffice",
            "/usr/local/bin/soffice",
            "/snap/bin/soffice",
        ]

    for c in candidates:
        if Path(c).exists():
            return c

    # Try PATH
    try:
        result = subprocess.run(
            ["where" if system == "Windows" else "which", "soffice"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return result.stdout.strip().split("\n")[0]
    except Exception:
        pass

    return ""


def convert_file(soffice_path: str, input_path: str, output_dir: str) -> tuple:
    """Convert a single file using LibreOffice headless mode.

    Returns (input_path, output_path, success, error_message)
    """
    ext = Path(input_path).suffix.lower()
    mapping = CONVERT_MAP.get(ext)
    if not mapping:
        return input_path, "", False, f"No conversion mapping for {ext}"
    target_ext, export_filter = mapping

    try:
        result = subprocess.run(
            [
                soffice_path,
                "--headless",
                "--convert-to", export_filter,
                "--outdir", output_dir,
                input_path,
            ],
            capture_output=True, text=True, timeout=120,
            encoding="utf-8", errors="replace"
        )

        if result.returncode == 0:
            # Check if output file was created
            expected_name = Path(input_path).stem + target_ext
            output_path = Path(output_dir) / expected_name
            if output_path.exists():
                return input_path, str(output_path), True, ""
            else:
                return input_path, "", False, f"Output not found: {output_path}"
        else:
            return input_path, "", False, result.stderr[:200]

    except subprocess.TimeoutExpired:
        return input_path, "", False, "Timeout (120s)"
    except Exception as e:
        return input_path, "", False, str(e)[:200]


def convert_all(conn: sqlite3.Connection):
    """Convert all old-format Office files."""
    soffice = find_soffice()
    if not soffice:
        logger.error("LibreOffice not found! Cannot convert old formats.")
        logger.error("Install LibreOffice from https://www.libreoffice.org/download/")
        logger.error("Or skip this step - old format files will be marked as 'conversion_needed'")
        # Mark all as needing conversion
        c = conn.cursor()
        old_office_exts = tuple(CONVERT_MAP.keys())
        placeholders = ",".join("?" for _ in old_office_exts)
        c.execute(f"""
            UPDATE files SET status = 'conversion_needed'
            WHERE lower(extension) IN ({placeholders})
            AND is_duplicate = 0
            AND status IN ('scanned', 'identified', 'conversion_needed', 'extraction_failed', 'conversion_failed')
        """, old_office_exts)
        conn.commit()
        return

    logger.info(f"Found LibreOffice at: {soffice}")

    # Get files to convert by extension instead of file_category because old Office
    # files may have been classified as regular text by settings.TEXT_EXTENSIONS.
    c = conn.cursor()
    old_office_exts = tuple(CONVERT_MAP.keys())
    placeholders = ",".join("?" for _ in old_office_exts)
    c.execute(f"""
        SELECT id, file_path, extension FROM files
        WHERE lower(extension) IN ({placeholders})
        AND is_duplicate = 0
        AND status IN ('scanned', 'identified', 'conversion_needed', 'extraction_failed', 'conversion_failed')
    """, old_office_exts)
    rows = c.fetchall()
    logger.info(f"Files to convert: {len(rows)}")

    converted_dir = Path(config.CONVERTED_DIR)
    converted_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0

    for file_id, filepath, ext in rows:
        if not Path(filepath).exists():
            c.execute("UPDATE files SET status = 'missing' WHERE id = ?", (file_id,))
            continue

        input_path, output_path, ok, err = convert_file(soffice, filepath, str(converted_dir))

        if ok:
            c.execute("""
                UPDATE files
                SET status = 'converted', converted_path = ?
                WHERE id = ?
            """, (output_path, file_id))
            success += 1
        else:
            c.execute("""
                UPDATE files
                SET status = 'conversion_failed', error_message = ?
                WHERE id = ?
            """, (err, file_id))
            failed += 1
            if failed <= 20:
                logger.warning(f"Conversion failed: {filepath} - {err}")

        total = success + failed
        if total % 100 == 0:
            conn.commit()
            logger.info(f"Converted {success}, Failed {failed} / {len(rows)}")

    conn.commit()
    logger.info(f"Conversion complete. Success: {success}, Failed: {failed}")


def add_converted_path_column(conn: sqlite3.Connection):
    """Add converted_path column if not exists."""
    c = conn.cursor()
    try:
        c.execute("ALTER TABLE files ADD COLUMN converted_path TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass  # Column already exists
    conn.commit()


def main():
    setup_logging()
    logger.info("=" * 60)
    logger.info("Step 4: Format Conversion")
    logger.info("=" * 60)

    conn = sqlite3.connect(config.DB_PATH)
    add_converted_path_column(conn)
    convert_all(conn)
    conn.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
