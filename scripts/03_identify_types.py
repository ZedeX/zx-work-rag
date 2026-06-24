"""Step 3: Identify file types for extensionless files.

Uses file magic (via subprocess `file` command) to identify file types
for the ~27k files without extensions.
"""
import sys
import sqlite3
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import config, Config
from loguru import logger


def setup_logging():
    log_file = Path(config.LOG_DIR) / "03_identify_types.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_file), rotation="10 MB", encoding="utf-8",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


# Map file command output to extensions and categories
MIME_MAP = {
    "Microsoft Word": (".doc", "text_old"),
    "Microsoft Excel": (".xls", "text_old"),
    "Microsoft PowerPoint": (".ppt", "text_old"),
    "PDF document": (".pdf", "text"),
    "HTML document": (".html", "text"),
    "XML document": (".xml", "text"),
    "ASCII text": (".txt", "text"),
    "UTF-8 Unicode text": (".txt", "text"),
    "text": (".txt", "text"),
    "PNG image": (".png", "image"),
    "JPEG image": (".jpg", "image"),
    "GIF image": (".gif", "image"),
    "Bitmap image": (".bmp", "image"),
    "TIFF image": (".tif", "image"),
    "SVG Scalable Vector": (".svg", "image"),
    "MPEG sequence": (".mp4", "video"),
    "Matroska data": (".mkv", "video"),
    "RIFF (little-endian) data, AVI": (".avi", "video"),
    "MP4": (".mp4", "video"),
    "Zip archive": (".zip", "archive"),
    "RAR archive": (".rar", "archive"),
    "7-zip archive": (".7z", "archive"),
    "gzip compressed": (".gz", "archive"),
    "JSON data": (".json", "text"),
    "Python script": (".py", "code"),
    "JavaScript": (".js", "code"),
    "Java source": (".java", "code"),
    "executable": (".exe", "other"),
    "DLL": (".dll", "other"),
    "SQLite": (".db", "other"),
}


# Magic byte signatures for cross-platform file identification
MAGIC_BYTES = {
    b"\xd0\xcf\x11\xe0": (".doc", "text_old"),  # OLE2 (doc/xls/ppt)
    b"PK\x03\x04": (".docx", "text"),            # ZIP-based (docx/xlsx/pptx)
    b"%PDF": (".pdf", "text"),
    b"\x89PNG": (".png", "image"),
    b"\xff\xd8\xff": (".jpg", "image"),
    b"GIF8": (".gif", "image"),
    b"BM": (".bmp", "image"),
    b"II*\x00": (".tif", "image"),
    b"MM\x00*": (".tif", "image"),
    b"\x1a\x45\xdf\xa3": (".mkv", "video"),      # MKV/WebM
    b"RIFF": (".avi", "video"),                    # AVI/WAV
    b"\x00\x00\x00\x18ftypmp4": (".mp4", "video"),
    b"\x00\x00\x00\x1cftypisom": (".mp4", "video"),
    b"ID3": (".mp3", "audio"),
    b"\xff\xfb": (".mp3", "audio"),
    b"fLaC": (".flac", "audio"),
    b"Rar!": (".rar", "archive"),
    b"7z\xbc\xaf\x27\x1c": (".7z", "archive"),
    b"\x1f\x8b": (".gz", "archive"),
    b"SQLite format 3": (".db", "other"),
}


def _identify_by_magic_bytes(filepath: str) -> tuple:
    """Fallback: identify file type by reading magic bytes."""
    try:
        with open(filepath, "rb") as f:
            header = f.read(32)

        for magic, (ext, cat) in MAGIC_BYTES.items():
            if header.startswith(magic):
                # Distinguish OLE2 subtypes
                if magic == b"\xd0\xcf\x11\xe0":
                    # Read more to distinguish doc/xls/ppt
                    with open(filepath, "rb") as f:
                        f.seek(512)
                        clsid = f.read(16)
                    # This is a rough heuristic; mark as generic old office
                    return ".doc", "text_old"
                return ext, cat

        # Check if it's plain text
        try:
            with open(filepath, "r", encoding="utf-8", errors="strict") as f:
                f.read(1024)
            return ".txt", "text"
        except (UnicodeDecodeError, ValueError):
            pass

        return "", "other"
    except (OSError, IOError):
        return "", "other"


def identify_file(filepath: str) -> tuple:
    """Use `file` command to identify file type. Returns (detected_ext, category)."""
    try:
        # Try 'file' command (Linux/Mac native, Windows via Git Bash or MSYS2)
        result = subprocess.run(
            ["file", "--brief", filepath],
            capture_output=True, text=True, timeout=10,
            encoding="utf-8", errors="replace"
        )
        output = result.stdout.strip()

        # On Windows without 'file' command, fall back to magic-byte detection
        if not output or result.returncode != 0:
            return _identify_by_magic_bytes(filepath)

        for keyword, (ext, cat) in MIME_MAP.items():
            if keyword.lower() in output.lower():
                return ext, cat

        # Fallback: check if it's text-like
        if "text" in output.lower():
            return ".txt", "text"
        if "image" in output.lower():
            return ".img", "image"
        if "video" in output.lower():
            return ".vid", "video"
        if "audio" in output.lower():
            return ".aud", "audio"

        return "", "other"

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.warning(f"file command failed for {filepath}: {e}")
        return "", "other"


def identify_all(conn: sqlite3.Connection):
    """Identify types for all extensionless files."""
    c = conn.cursor()
    c.execute("""
        SELECT id, file_path FROM files
        WHERE (extension = '' OR extension IS NULL)
        AND is_duplicate = 0
        AND status != 'identified'
    """)
    rows = c.fetchall()
    logger.info(f"Files to identify: {len(rows)}")

    identified = 0
    for file_id, filepath in rows:
        if not Path(filepath).exists():
            c.execute("UPDATE files SET status = 'missing' WHERE id = ?", (file_id,))
            continue

        ext, category = identify_file(filepath)
        if ext:
            c.execute("""
                UPDATE files
                SET extension = ?, file_category = ?, status = 'identified'
                WHERE id = ?
            """, (ext, category, file_id))
        else:
            c.execute("""
                UPDATE files SET status = 'unidentified' WHERE id = ?
            """, (file_id,))

        identified += 1
        if identified % 500 == 0:
            conn.commit()
            logger.info(f"Identified {identified}/{len(rows)} files...")

    conn.commit()
    logger.info(f"Identification complete. Processed: {identified}")

    # Summary
    c.execute("""
        SELECT extension, COUNT(*) FROM files
        WHERE status = 'identified'
        GROUP BY extension ORDER BY COUNT(*) DESC LIMIT 20
    """)
    logger.info("=== Identified Extensions ===")
    for row in c.fetchall():
        logger.info(f"  {row[0]}: {row[1]}")

    c.execute("SELECT COUNT(*) FROM files WHERE status = 'unidentified'")
    logger.info(f"  Unidentified: {c.fetchone()[0]}")


def main():
    setup_logging()
    logger.info("=" * 60)
    logger.info("Step 3: Type Identification")
    logger.info("=" * 60)

    conn = sqlite3.connect(config.DB_PATH)
    identify_all(conn)
    conn.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
