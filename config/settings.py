"""Project configuration - loads from .env file."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_project_dir = Path(__file__).parent.parent
load_dotenv(_project_dir / ".env")


class Config:
    # Volcengine Ark API
    ARK_API_KEY: str = os.getenv("ARK_API_KEY", "")
    ARK_BASE_URL: str = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    ARK_CODING_BASE_URL: str = os.getenv("ARK_CODING_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding/v3")
    ARK_EMBEDDING_MODEL: str = os.getenv("ARK_EMBEDDING_MODEL", "doubao-embedding-text-240915")
    ARK_LLM_MODEL: str = os.getenv("ARK_LLM_MODEL", "doubao-seed-2.0-pro")

    # LM Studio (local LLM + local Embedding)
    LM_STUDIO_BASE_URL: str = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    LM_STUDIO_CHAT_MODEL: str = os.getenv("LM_STUDIO_CHAT_MODEL", "google/gemma-4-e2b")
    LM_STUDIO_EMBEDDING_MODEL: str = os.getenv("LM_STUDIO_EMBEDDING_MODEL", "text-embedding-embeddinggemma-300m")

    # Embedding mode: "local" (LM Studio) or "cloud" (Volcengine Ark)
    EMBEDDING_MODE: str = os.getenv("EMBEDDING_MODE", "local")

    # Source data
    SOURCE_DIR: str = os.getenv("SOURCE_DIR", "")

    # Project data dirs
    PROJECT_DIR: str = os.getenv("PROJECT_DIR", str(_project_dir))
    DATA_DIR: str = os.getenv("DATA_DIR", str(_project_dir / "data"))
    CONVERTED_DIR: str = os.getenv("CONVERTED_DIR", str(_project_dir / "data" / "converted"))
    CHROMA_DIR: str = os.getenv("CHROMA_DIR", str(_project_dir / "data" / "chroma_db"))
    LOG_DIR: str = os.getenv("LOG_DIR", str(_project_dir / "data" / "logs"))
    DB_PATH: str = os.getenv("DB_PATH", str(_project_dir / "data" / "file_manifest.db"))

    # Processing params
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "100"))

    # File type categories
    TEXT_EXTENSIONS = {
        ".doc", ".docx", ".pdf", ".ppt", ".pptx", ".xls", ".xlsx",
        ".txt", ".csv", ".md", ".rtf", ".odt", ".ods", ".odp",
        ".html", ".htm", ".xml", ".json", ".log",
    }
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".svg", ".psd", ".ai"}
    VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".ts", ".webm"}
    AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma"}
    ARCHIVE_EXTENSIONS = {".zip", ".rar", ".7z", ".tar", ".gz"}
    CODE_EXTENSIONS = {".js", ".css", ".less", ".h", ".py", ".java", ".ts", ".map", ".strings"}

    # Old Office formats that need LibreOffice conversion
    OLD_OFFICE_EXTENSIONS = {".doc", ".ppt", ".xls", ".dot", ".pot", ".xlt"}

    @classmethod
    def ensure_dirs(cls):
        """Create all required directories."""
        for d in [cls.DATA_DIR, cls.CONVERTED_DIR, cls.CHROMA_DIR, cls.LOG_DIR]:
            Path(d).mkdir(parents=True, exist_ok=True)


config = Config()
