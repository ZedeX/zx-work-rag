"""Step 1: Scan all files in source directory and save to SQLite database.

Records: file_path, file_name, extension, size, modified_time, md5_hash,
         file_category, folder_path, status
"""
import sys
import hashlib
import sqlite3
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import config, Config
from loguru import logger


def setup_logging():
    log_file = Path(config.LOG_DIR) / "01_scan_files.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_file), rotation="10 MB", encoding="utf-8",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


def init_db():
    """Create SQLite database with files table."""
    conn = sqlite3.connect(config.DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE NOT NULL,
            file_name TEXT NOT NULL,
            extension TEXT DEFAULT '',
            size INTEGER DEFAULT 0,
            modified_time TEXT DEFAULT '',
            md5_hash TEXT DEFAULT '',
            file_category TEXT DEFAULT 'unknown',
            folder_path TEXT DEFAULT '',
            folder_tags TEXT DEFAULT '',
            status TEXT DEFAULT 'scanned',
            text_extracted INTEGER DEFAULT 0,
            embedded INTEGER DEFAULT 0,
            is_duplicate INTEGER DEFAULT 0,
            duplicate_of INTEGER DEFAULT NULL,
            error_message TEXT DEFAULT '',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_extension ON files(extension)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_category ON files(file_category)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_status ON files(status)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_hash ON files(md5_hash)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_duplicate ON files(is_duplicate)")
    conn.commit()
    return conn


def categorize_file(ext: str) -> str:
    """Categorize file by extension."""
    ext = ext.lower()
    if ext in Config.TEXT_EXTENSIONS:
        return "text"
    elif ext in Config.IMAGE_EXTENSIONS:
        return "image"
    elif ext in Config.VIDEO_EXTENSIONS:
        return "video"
    elif ext in Config.AUDIO_EXTENSIONS:
        return "audio"
    elif ext in Config.ARCHIVE_EXTENSIONS:
        return "archive"
    elif ext in Config.CODE_EXTENSIONS:
        return "code"
    elif ext in Config.OLD_OFFICE_EXTENSIONS:
        return "text_old"
    else:
        return "other"


def extract_folder_tags(folder_path: str, source_dir: str) -> str:
    """Extract semantic tags from folder path.

    e.g. '18 TX/_培训/数据建模' -> '腾讯,培训,数据建模'
    """
    try:
        rel = Path(folder_path).relative_to(source_dir)
    except (ValueError, TypeError):
        return ""
    parts = []
    for p in rel.parts:
        # Clean up folder names: remove leading numbers, underscores
        clean = p.strip()
        if clean.startswith("_"):
            clean = clean[1:]
        # Skip pure numbers or very short names
        if len(clean) > 1 and not clean.isdigit():
            parts.append(clean)
    return ",".join(parts)


def compute_md5(file_path: str) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except (PermissionError, OSError) as e:
        logger.warning(f"Cannot read file for hash: {file_path} - {e}")
        return ""


def scan_files(source_dir: str, conn: sqlite3.Connection, compute_hashes: bool = True):
    """Scan all files and insert into database."""
    source = Path(source_dir)
    if not source.exists():
        logger.error(f"Source directory does not exist: {source_dir}")
        return

    c = conn.cursor()
    scanned = 0
    errors = 0

    logger.info(f"Starting scan of: {source_dir}")

    # Get existing paths to avoid re-scanning
    c.execute("SELECT file_path FROM files")
    existing = {row[0] for row in c.fetchall()}

    for filepath in source.rglob("*"):
        if not filepath.is_file():
            continue

        filepath_str = str(filepath)

        # Skip if already scanned
        if filepath_str in existing:
            continue

        try:
            ext = filepath.suffix.lower()
            stat = filepath.stat()
            category = categorize_file(ext)
            folder_tags = extract_folder_tags(str(filepath.parent), source_dir)

            # Compute hash (can be slow for large files, skip if >100MB)
            md5 = ""
            if compute_hashes and stat.st_size < 100 * 1024 * 1024:
                md5 = compute_md5(filepath_str)

            c.execute("""
                INSERT OR IGNORE INTO files
                (file_path, file_name, extension, size, modified_time,
                 md5_hash, file_category, folder_path, folder_tags, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                filepath_str,
                filepath.name,
                ext,
                stat.st_size,
                datetime.fromtimestamp(stat.st_mtime).isoformat(),
                md5,
                category,
                str(filepath.parent),
                folder_tags,
                "scanned",
            ))

            scanned += 1
            if scanned % 1000 == 0:
                conn.commit()
                logger.info(f"Scanned {scanned} files...")

        except (PermissionError, OSError) as e:
            errors += 1
            if errors <= 50:
                logger.warning(f"Error scanning {filepath_str}: {e}")
            continue

    conn.commit()
    logger.info(f"Scan complete. Scanned: {scanned}, Errors: {errors}")

    # Print summary
    c.execute("SELECT file_category, COUNT(*) FROM files GROUP BY file_category ORDER BY COUNT(*) DESC")
    logger.info("=== File Category Summary ===")
    for row in c.fetchall():
        logger.info(f"  {row[0]}: {row[1]}")

    c.execute("SELECT COUNT(*) FROM files WHERE extension = ''")
    no_ext = c.fetchone()[0]
    if no_ext > 0:
        logger.info(f"  Files without extension: {no_ext}")


def main():
    setup_logging()
    logger.info("=" * 60)
    logger.info("Step 1: File Scanner")
    logger.info("=" * 60)

    config.ensure_dirs()
    conn = init_db()
    scan_files(config.SOURCE_DIR, conn, compute_hashes=True)
    conn.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
