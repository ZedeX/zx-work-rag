# zx-work-rag

Personal work document RAG (Retrieval-Augmented Generation) system. Converts 200K+ work files (doc, docx, ppt, pptx, xls, xlsx, pdf, etc.) into a searchable vector knowledge base, enabling semantic search and AI-powered Q\&A over your entire work archive.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Query Interfaces                       │
│       IDE (MCP)  │  Streamlit Web UI  │  CLI            │
├──────────────────────────────────────────────────────────┤
│                    RAG Query Engine                       │
│  Semantic Search  │  Metadata Filtering  │  LLM Answer   │
├──────────────────────────────────────────────────────────┤
│                    Storage Layer                          │
│  ChromaDB (Vectors)  │  SQLite (Metadata & Full Text)    │
├──────────────────────────────────────────────────────────┤
│                    Embedding Layer                        │
│  Local: LM Studio / Ollama / vLLM / LocalAI             │
│  Cloud: Volcengine Ark / OpenAI                          │
├──────────────────────────────────────────────────────────┤
│                    Chat LLM Layer                         │
│  Local: LM Studio / Ollama / vLLM                       │
│  Cloud: Volcengine Ark / OpenAI                          │
└──────────────────────────────────────────────────────────┘
```

## Prerequisites

| Requirement        | Required? | Notes                              |
| ------------------ | --------- | ---------------------------------- |
| Python 3.10+       | Yes       | Core runtime                       |
| Embedding provider | Yes       | One of the options below           |
| Chat LLM provider  | Optional  | Only needed for AI Q\&A mode       |
| LibreOffice        | Optional  | Only for .doc/.xls/.ppt conversion |

### Embedding Provider Options

Any **OpenAI-compatible API server** works. Choose one:

| Provider           | Cost | Platform      | Setup                          | Best For                     |
| ------------------ | ---- | ------------- | ------------------------------ | ---------------------------- |
| **LM Studio**      | Free | Win/Mac       | Desktop app, load model        | Windows users, easiest setup |
| **Ollama**         | Free | Win/Mac/Linux | `ollama pull nomic-embed-text` | Linux/Mac, CLI lovers        |
| **vLLM**           | Free | Linux         | Self-hosted GPU server         | Production, high throughput  |
| **LocalAI**        | Free | Win/Mac/Linux | Self-hosted                    | CPU-only machines            |
| **Volcengine Ark** | Paid | Any           | API key                        | Large scale, Chinese docs    |
| **OpenAI**         | Paid | Any           | API key                        | Quick start, high quality    |

**Recommended local models for embedding:**

- `nomic-ai/nomic-embed-text-v1.5` — Good multilingual, fast
- `Alibaba-NLP/gte-Qwen2-0.5B` — Best Chinese support
- `text-embedding-embeddinggemma-300m` — Fast, good quality

**Recommended local models for chat:**

- `google/gemma-4-e2b` — Zero hallucination, fast
- `qwen3.5:9b` — Best Chinese, but slower + hallucination risk
- `llama3.1:8b` — Good English, balanced

### LM Studio Alternatives

If you don't have LM Studio, you can use **Ollama** as a drop-in replacement:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh   # Linux/macOS
# Or download from https://ollama.com/            # Windows

# Pull models
ollama pull nomic-embed-text    # Embedding model
ollama pull qwen2.5:7b          # Chat model

# Ollama auto-starts an OpenAI-compatible API at http://localhost:11434/v1
```

Then set in `.env`:

```env
LM_STUDIO_BASE_URL=http://localhost:11434/v1
LM_STUDIO_EMBEDDING_MODEL=nomic-embed-text
LM_STUDIO_CHAT_MODEL=qwen2.5:7b
```

## Pipeline Overview

The data processing pipeline consists of 6 sequential steps that transform raw files into a vector-indexed knowledge base:

| Step | Script                                        | Description                                                                      |
| ---- | --------------------------------------------- | -------------------------------------------------------------------------------- |
| 1    | `01_scan_files.py`                            | Scan source directory, record file metadata + MD5 hash into SQLite               |
| 2    | `02_dedup.py`                                 | Deduplicate files by MD5 hash (keeps shortest path as original)                  |
| 3    | `03_identify_types.py`                        | Identify file types for extensionless files via magic bytes or `file` command    |
| 4    | `04_convert_formats.py`                       | Convert legacy Office formats (.doc/.xls/.ppt) to modern formats via LibreOffice |
| 5    | `05_extract_text.py`                          | Extract text from all document types (docx, pptx, xlsx, pdf, txt, html, etc.)    |
| 6    | `06_embed_cloud.py` / `06a_embed_parallel.py` | Generate vector embeddings and store in ChromaDB                                 |

After parallel embedding, run `06b_merge_embeddings.py` to merge partition collections into the main `work_corpus` collection.

## Features

- **Multi-format text extraction**: docx, pptx, xlsx, pdf, txt, csv, html, xml, json, md, and more
- **Legacy format conversion**: Automatic .doc/.xls/.ppt → .docx/.xlsx/.pptx via LibreOffice headless mode
- **MD5 deduplication**: Identifies and marks duplicate files, keeping the original
- **Extensionless file identification**: Uses magic bytes (cross-platform) or `file` command to detect types
- **Any OpenAI-compatible provider**: LM Studio, Ollama, vLLM, LocalAI, Volcengine Ark, OpenAI
- **Parallel embedding**: 8-worker parallel processing with progress tracking and resume support
- **Semantic search**: Vector similarity search with metadata filtering (category, extension, year, folder)
- **AI-powered Q\&A**: Generate answers from retrieved context using local or cloud LLMs
- **MCP Server integration**: Search your knowledge base directly from Trae IDE
- **Streamlit Web UI**: Browser-based search interface with filters and export
- **Context export**: Export search results as Markdown/JSONL for use with other LLMs
- **Cross-platform**: Works on Windows, Linux, and macOS

## Project Structure

```
zx-work-rag/
├── config/
│   ├── __init__.py
│   └── settings.py            # Configuration (API keys, paths, params)
├── scripts/
│   ├── 00_setup.py            # Interactive setup wizard
│   ├── 01_scan_files.py       # Step 1: File scanner
│   ├── 02_dedup.py            # Step 2: MD5 deduplication
│   ├── 03_identify_types.py   # Step 3: Extensionless file identification
│   ├── 04_convert_formats.py  # Step 4: Legacy format conversion
│   ├── 05_extract_text.py     # Step 5: Text extraction
│   ├── 05b_extract_text_safe.py  # Step 5b: Safe extraction with timeouts
│   ├── 05c_extract_remaining_fast.py  # Step 5c: Fast remaining extraction
│   ├── 06_embed_cloud.py      # Step 6: Single-worker embedding
│   ├── 06a_embed_parallel.py  # Step 6a: Multi-worker parallel embedding
│   └── 06b_merge_embeddings.py # Step 6b: Merge partition collections
├── server/
│   ├── __init__.py
│   ├── rag_query.py           # RAG query service (search + LLM answer)
│   ├── mcp_server.py          # MCP Server for Trae IDE integration
│   └── web_app.py             # Streamlit web UI
├── tests/                     # Benchmark & test scripts
├── data/                      # Runtime data (gitignored)
│   ├── file_manifest.db       # SQLite metadata database
│   ├── chroma_db/             # ChromaDB vector store
│   ├── converted/             # Converted Office files
│   └── logs/                  # Pipeline logs
├── .env                       # Environment variables (gitignored)
├── .env.example               # Configuration template
├── requirements.txt           # Python dependencies
├── run_pipeline.bat           # Pipeline runner (Windows)
└── run_pipeline.sh            # Pipeline runner (Linux/macOS)
```

## Quick Start

### Option A: Interactive Setup Wizard (Recommended)

```bash
# Clone the repository
git clone https://github.com/<your-username>/zx-work-rag.git
cd zx-work-rag

# Create virtual environment
python -m venv .venv

# Activate
.venv\Scripts\activate         # Windows
source .venv/bin/activate      # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Run setup wizard
python scripts/00_setup.py
```

The wizard will guide you through:

1. Setting your work files directory (`SOURCE_DIR`)
2. Choosing an embedding provider (auto-detects LM Studio / Ollama)
3. Choosing a chat LLM provider
4. Writing `.env` configuration
5. Installing dependencies

### Option B: Manual Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/zx-work-rag.git
cd zx-work-rag

# Create virtual environment
python -m venv .venv

# Activate
.venv\Scripts\activate         # Windows
source .venv/bin/activate      # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env and set SOURCE_DIR to your work files directory
```

### Configuration

Edit `.env` with your settings:

```env
# Source data directory (REQUIRED)
SOURCE_DIR=/path/to/your/work/files

# Embedding provider URL (any OpenAI-compatible API)
# LM Studio:
LM_STUDIO_BASE_URL=http://localhost:1234/v1
# Ollama:
# LM_STUDIO_BASE_URL=http://localhost:11434/v1
# vLLM:
# LM_STUDIO_BASE_URL=http://localhost:8000/v1

# Model names (adjust to your loaded models)
LM_STUDIO_EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
LM_STUDIO_CHAT_MODEL=qwen2.5:7b

# Embedding mode: "local" or "cloud"
EMBEDDING_MODE=local
```

### Run the Pipeline

```bash
# Windows
run_pipeline.bat scan       # Step 1: Scan files
run_pipeline.bat dedup      # Step 2: Deduplicate
run_pipeline.bat identify   # Step 3: Identify types
run_pipeline.bat convert    # Step 4: Convert formats
run_pipeline.bat extract    # Step 5: Extract text
run_pipeline.bat embed      # Step 6: Generate embeddings
run_pipeline.bat all        # Run all steps
run_pipeline.bat stats      # View statistics
run_pipeline.bat web        # Start web UI

# Linux/macOS
bash run_pipeline.sh scan
bash run_pipeline.sh all
bash run_pipeline.sh web
```

For parallel embedding (recommended for large corpora):

```bash
# Run 8 workers in parallel
python scripts/06a_embed_parallel.py --all --workers 8

# After all workers finish, merge into main collection
python scripts/06b_merge_embeddings.py
```

### Query the Knowledge Base

**CLI:**

```bash
# Semantic search only (no LLM needed)
python server/rag_query.py "digital transformation" --mode none --top-k 5

# Search with AI-generated answer (local LLM)
python server/rag_query.py "digital transformation" --mode local --top-k 5

# Search with cloud LLM answer
python server/rag_query.py "digital transformation" --mode cloud --top-k 5

# Export results as Markdown
python server/rag_query.py "digital transformation" --export markdown --top-k 10

# Filter by metadata
python server/rag_query.py "financial report" --category text --year 2023
```

**Streamlit Web UI:**

```bash
streamlit run server/web_app.py --server.port 8501
```

The web UI automatically shows a setup guide if the system is not yet configured.

**MCP Server (Trae IDE):**

Add to your Trae MCP configuration:

```json
{
  "mcpServers": {
    "zx-work-rag": {
      "command": "/path/to/.venv/bin/python",
      "args": ["/path/to/zx-work-rag/server/mcp_server.py"]
    }
  }
}
```

MCP tools available:

- `search_knowledge_base` — Semantic search with optional filters
- `get_document_detail` — Retrieve full text of a specific document
- `get_knowledge_base_stats` — View database statistics

## Benchmark Results

Comprehensive benchmarks were conducted across 3 rounds with local models via LM Studio:

### Chat Models

| Model           | Quality | Chinese | Speed  | Hallucination | Verdict         |
| --------------- | ------- | ------- | ------ | ------------- | --------------- |
| gemma-4-e2b     | 4.8/5   | 4.0/5   | 73 tps | **Zero**      | **Recommended** |
| qwen3.5-9b      | 4.6/5   | 4.4/5   | 6 tps  | Risky         | Backup only     |
| nemotron-3-nano | 3.6/5   | 3.6/5   | 34 tps | Present       | Not recommended |

### Embedding Models

| Model       | Dims | Discrimination | Throughput  | 80K Est. | Verdict          |
| ----------- | ---- | -------------- | ----------- | -------- | ---------------- |
| qwen3-0.6b  | 1024 | **0.5536**     | 28.5 t/s    | \~12h    | Best quality     |
| gemma-300m  | 768  | 0.4538         | **103 t/s** | \~2.4h   | Best speed/value |
| nomic-embed | 768  | 0.0756         | 165 t/s     | \~1.4h   | Chinese-unusable |

### Recommended Combinations

| Scenario     | Embedding  | Chat        | Rationale                                                         |
| ------------ | ---------- | ----------- | ----------------------------------------------------------------- |
| Production   | gemma-300m | gemma-4-e2b | Fastest pipeline + zero hallucination                             |
| Quality      | qwen3-0.6b | gemma-4-e2b | Best retrieval precision + zero hallucination                     |
| Deep Chinese | qwen3-0.6b | qwen3.5-9b  | Best retrieval + most natural Chinese (slow + hallucination risk) |

## Processing Statistics (Sample Run)

| Metric                              | Count   |
| ----------------------------------- | ------- |
| Total files scanned                 | 201,434 |
| Unique files (after dedup)          | 157,825 |
| Text extracted                      | 124,625 |
| Files embedded                      | 99,519  |
| ChromaDB vectors                    | 816,562 |
| Parallel embedding time (8 workers) | \~2.5h  |

## Supported File Types

| Category      | Extensions                                                                       |
| ------------- | -------------------------------------------------------------------------------- |
| Text          | .docx, .pptx, .xlsx, .pdf, .txt, .csv, .md, .html, .htm, .xml, .json, .log, .rtf |
| Legacy Office | .doc, .xls, .ppt, .dot, .pot, .xlt (requires LibreOffice)                        |
| Image         | .png, .jpg, .jpeg, .gif, .bmp, .tif, .tiff, .svg, .psd, .ai                      |
| Video         | .mp4, .mkv, .avi, .mov, .wmv, .flv, .ts, .webm                                   |
| Audio         | .mp3, .wav, .flac, .aac, .ogg, .m4a, .wma                                        |
| Archive       | .zip, .rar, .7z, .tar, .gz                                                       |
| Code          | .js, .css, .py, .java, .ts, .h                                                   |

Non-text files (images, videos, audio, archives) are indexed with metadata-only records, enabling folder/tag-based discovery.

## Tech Stack

- **Python 3.10+** — Core language
- **ChromaDB** — Vector storage with HNSW cosine similarity
- **SQLite** — File metadata and full-text storage
- **OpenAI-compatible API** — Unified interface for any LLM/embedding provider
- **pdfplumber / PyMuPDF** — PDF text extraction
- **python-docx / python-pptx / openpyxl** — Office document parsing
- **BeautifulSoup4 + lxml** — HTML extraction
- **LangChain text-splitters** — Text chunking
- **Streamlit** — Web UI
- **MCP (Model Context Protocol)** — IDE integration
- **Loguru** — Structured logging

## Known Issues

1. **PDF extraction timeouts**: Some malformed PDFs can hang the extraction process; use `05b_extract_text_safe.py` for timeout-protected extraction
2. **LibreOffice dependency**: Step 4 (format conversion) requires LibreOffice to be installed; without it, old-format files are only marked as `conversion_needed`
3. **Parallel worker discrepancy**: Workers W0-W3 may show inflated progress counts from previous interrupted runs; actual ChromaDB collection counts are authoritative

## License

This project is for personal use. The source code is provided as-is.
