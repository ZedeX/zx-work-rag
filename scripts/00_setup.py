#!/usr/bin/env python3
"""Interactive setup wizard for zx-work-rag.

Guides users through:
1. Setting SOURCE_DIR (work files directory)
2. Choosing embedding provider (LM Studio / Ollama / OpenAI-compatible / Volcengine Ark)
3. Choosing chat LLM provider
4. Creating .env configuration file
5. Creating virtual environment and installing dependencies

Usage:
    python scripts/00_setup.py
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"


def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def ask(prompt, default=""):
    if default:
        answer = input(f"{prompt} [{default}]: ").strip()
        return answer if answer else default
    return input(f"{prompt}: ").strip()


def ask_yes_no(prompt, default=True):
    default_str = "Y/n" if default else "y/N"
    answer = input(f"{prompt} [{default_str}]: ").strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")


def check_command(cmd):
    """Check if a command is available on PATH."""
    return shutil.which(cmd) is not None


def test_api_connection(base_url, api_key="test", timeout=5):
    """Test if an OpenAI-compatible API is reachable."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        client.models.list()
        return True
    except Exception:
        return False


def detect_embedding_provider():
    """Auto-detect available embedding providers."""
    print("Detecting available embedding providers...\n")

    providers = []

    # Check LM Studio
    if test_api_connection("http://localhost:1234/v1", "lm-studio"):
        print("  [FOUND] LM Studio running at http://localhost:1234/v1")
        providers.append("lm_studio")
    else:
        print("  [----] LM Studio not detected (start it if you want local embedding)")

    # Check Ollama
    if test_api_connection("http://localhost:11434/v1", "ollama"):
        print("  [FOUND] Ollama running at http://localhost:11434/v1")
        providers.append("ollama")
    else:
        print("  [----] Ollama not detected (start it if you want local embedding)")

    return providers


def run_setup():
    print_header("zx-work-rag Setup Wizard")

    # Step 0: Check Python version
    py_version = sys.version_info
    if py_version < (3, 10):
        print(f"ERROR: Python 3.10+ required, got {py_version.major}.{py_version.minor}")
        sys.exit(1)
    print(f"Python {py_version.major}.{py_version.minor}.{py_version.micro} OK")

    # Step 1: Source directory
    print_header("Step 1: Work Files Directory")
    print("Point this to the directory containing your work files.")
    print("The scanner will recursively index all files under this path.\n")

    current_source = os.getenv("SOURCE_DIR", "")
    if ENV_FILE.exists():
        # Read existing .env
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("SOURCE_DIR="):
                    current_source = line.strip().split("=", 1)[1]
                    break

    if current_source:
        print(f"  Current: {current_source}")

    source_dir = ask("Source directory path", current_source if current_source else "")
    while source_dir and not Path(source_dir).exists():
        print(f"  WARNING: Path does not exist: {source_dir}")
        if not ask_yes_no("Continue anyway?", False):
            source_dir = ask("Source directory path", "")
        else:
            break

    # Step 2: Embedding provider
    print_header("Step 2: Embedding Provider")
    print("Choose how to generate vector embeddings for your documents.\n")
    print("  1. LM Studio (local, free, recommended)")
    print("     - Download from https://lmstudio.ai/")
    print("     - Load an embedding model (e.g. nomic-ai/nomic-embed-text-v1.5)")
    print("     - Best for: privacy, no API costs")
    print()
    print("  2. Ollama (local, free)")
    print("     - Install from https://ollama.com/")
    print("     - Run: ollama pull nomic-embed-text")
    print("     - Best for: Linux/Mac users, easy setup")
    print()
    print("  3. OpenAI-compatible API (self-hosted or cloud)")
    print("     - Any server that implements the /v1/embeddings endpoint")
    print("     - Examples: vLLM, LocalAI, text-embeddings-inference")
    print()
    print("  4. Volcengine Ark (cloud, paid)")
    print("     - Fast at scale, good Chinese support")
    print("     - Requires API key from https://www.volcengine.com/")
    print()

    # Auto-detect
    detected = detect_embedding_provider()
    default_choice = "1"
    if detected:
        if "lm_studio" in detected:
            default_choice = "1"
            print(f"\n  Auto-detected: LM Studio")
        elif "ollama" in detected:
            default_choice = "2"
            print(f"\n  Auto-detected: Ollama")

    choice = ask("Choose embedding provider (1-4)", default_choice)

    embed_mode = "local"
    embed_base_url = ""
    embed_model = ""
    ark_api_key = ""

    if choice == "1":
        embed_mode = "local"
        embed_base_url = ask("LM Studio API URL", "http://localhost:1234/v1")
        embed_model = ask("Embedding model name", "nomic-ai/nomic-embed-text-v1.5")
    elif choice == "2":
        embed_mode = "local"
        embed_base_url = ask("Ollama API URL", "http://localhost:11434/v1")
        embed_model = ask("Embedding model name", "nomic-embed-text")
    elif choice == "3":
        embed_mode = "local"
        embed_base_url = ask("API base URL", "http://localhost:8080/v1")
        embed_model = ask("Embedding model name", "")
        if not embed_model:
            print("  WARNING: Model name is required for custom API")
    elif choice == "4":
        embed_mode = "cloud"
        ark_api_key = ask("Volcengine Ark API Key", "")
        embed_base_url = "https://ark.cn-beijing.volces.com/api/v3"
        embed_model = "doubao-embedding-text-240915"

    # Step 3: Chat LLM provider
    print_header("Step 3: Chat LLM Provider (for AI Q&A)")
    print("Choose a chat model for generating answers from retrieved context.")
    print("This is optional — you can always use search-only mode.\n")
    print("  1. LM Studio (local)")
    print("  2. Ollama (local)")
    print("  3. OpenAI-compatible API")
    print("  4. Volcengine Ark (cloud)")
    print("  5. Skip (search-only mode)")

    chat_choice = ask("Choose chat LLM provider (1-5)", "5")
    chat_base_url = ""
    chat_model = ""
    chat_api_key = ""

    if chat_choice == "1":
        chat_base_url = ask("LM Studio API URL", embed_base_url or "http://localhost:1234/v1")
        chat_model = ask("Chat model name", "google/gemma-4-e2b")
        chat_api_key = "lm-studio"
    elif chat_choice == "2":
        chat_base_url = ask("Ollama API URL", "http://localhost:11434/v1")
        chat_model = ask("Chat model name", "qwen2.5:7b")
        chat_api_key = "ollama"
    elif chat_choice == "3":
        chat_base_url = ask("API base URL", "http://localhost:8080/v1")
        chat_model = ask("Chat model name", "")
        chat_api_key = ask("API key (or 'none')", "none")
    elif chat_choice == "4":
        chat_base_url = "https://ark.cn-beijing.volces.com/api/coding/v3"
        chat_model = "doubao-seed-2.0-pro"
        if not ark_api_key:
            ark_api_key = ask("Volcengine Ark API Key", "")
        chat_api_key = ark_api_key

    # Step 4: Processing parameters
    print_header("Step 4: Processing Parameters")
    chunk_size = ask("Chunk size (tokens)", "512")
    chunk_overlap = ask("Chunk overlap (tokens)", "64")
    batch_size = ask("Batch size", "100")

    # Step 5: Write .env
    print_header("Step 5: Writing Configuration")

    env_lines = [
        f"# Auto-generated by setup wizard",
        f"# Edit manually if needed",
        f"",
    ]

    # Ark API
    if ark_api_key:
        env_lines.extend([
            f"# Volcengine Ark API (cloud mode)",
            f"ARK_API_KEY={ark_api_key}",
            f"ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3",
            f"ARK_CODING_BASE_URL=https://ark.cn-beijing.volces.com/api/coding/v3",
            f"ARK_EMBEDDING_MODEL=doubao-embedding-text-240915",
            f"ARK_LLM_MODEL=doubao-seed-2.0-pro",
            f"",
        ])
    else:
        env_lines.extend([
            f"# Volcengine Ark API (cloud mode)",
            f"ARK_API_KEY=",
            f"ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3",
            f"ARK_CODING_BASE_URL=https://ark.cn-beijing.volces.com/api/coding/v3",
            f"ARK_EMBEDDING_MODEL=doubao-embedding-text-240915",
            f"ARK_LLM_MODEL=doubao-seed-2.0-pro",
            f"",
        ])

    # LM Studio / Ollama / Custom
    env_lines.extend([
        f"# Local embedding & chat (LM Studio / Ollama / OpenAI-compatible)",
        f"LM_STUDIO_BASE_URL={embed_base_url}",
        f"LM_STUDIO_CHAT_MODEL={chat_model}",
        f"LM_STUDIO_EMBEDDING_MODEL={embed_model}",
        f"",
    ])

    # Mode & paths
    env_lines.extend([
        f"# Embedding mode: 'local' or 'cloud'",
        f"EMBEDDING_MODE={embed_mode}",
        f"",
        f"# Source data directory",
        f"SOURCE_DIR={source_dir}",
        f"",
        f"# Project data directories (defaults are usually fine)",
        f"PROJECT_DIR=.",
        f"DATA_DIR=./data",
        f"CONVERTED_DIR=./data/converted",
        f"CHROMA_DIR=./data/chroma_db",
        f"LOG_DIR=./data/logs",
        f"DB_PATH=./data/file_manifest.db",
        f"",
        f"# Processing parameters",
        f"CHUNK_SIZE={chunk_size}",
        f"CHUNK_OVERLAP={chunk_overlap}",
        f"BATCH_SIZE={batch_size}",
    ])

    env_content = "\n".join(env_lines) + "\n"

    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.write(env_content)
    print(f"  Written: {ENV_FILE}")

    # Step 6: Virtual environment
    print_header("Step 6: Virtual Environment")

    venv_dir = PROJECT_ROOT / ".venv"
    if venv_dir.exists():
        print(f"  Virtual environment already exists at {venv_dir}")
    else:
        if ask_yes_no("Create virtual environment?", True):
            print("  Creating virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
            print(f"  Created: {venv_dir}")

    # Install dependencies
    pip_path = str(venv_dir / "Scripts" / "pip.exe" if sys.platform == "win32" else venv_dir / "bin" / "pip")
    if Path(pip_path).exists():
        if ask_yes_no("Install dependencies from requirements.txt?", True):
            print("  Installing dependencies...")
            subprocess.run([pip_path, "install", "-r", str(PROJECT_ROOT / "requirements.txt"), "-q"], check=True)
            print("  Dependencies installed.")
    else:
        print(f"  WARNING: pip not found at {pip_path}")
        print(f"  Run manually: pip install -r requirements.txt")

    # Done
    print_header("Setup Complete!")
    print(f"  Configuration saved to: {ENV_FILE}")
    print()
    print("  Next steps:")
    print()
    print("    1. Start your embedding provider (LM Studio / Ollama)")
    print("    2. Run the pipeline:")
    if sys.platform == "win32":
        print("       run_pipeline.bat all")
    else:
        print("       bash run_pipeline.sh all")
    print("    3. Start the web UI:")
    if sys.platform == "win32":
        print("       run_pipeline.bat web")
    else:
        print("       bash run_pipeline.sh web")
    print()
    print("  Or use the interactive setup to reconfigure anytime:")
    print("    python scripts/00_setup.py")


if __name__ == "__main__":
    run_setup()
