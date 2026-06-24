#!/usr/bin/env bash
# zx-work-rag Pipeline Runner
# Usage: ./run_pipeline.sh [step]
# Steps: scan, dedup, identify, convert, extract, embed, all, web, stats

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"
SCRIPTS="$SCRIPT_DIR/scripts"

# Fallback to system python if venv not found
if [ ! -f "$PYTHON" ]; then
    PYTHON="python3"
fi

case "${1:-}" in
    scan)
        echo "[Step 1/6] Scanning files..."
        $PYTHON "$SCRIPTS/01_scan_files.py"
        ;;
    dedup)
        echo "[Step 2/6] Deduplicating..."
        $PYTHON "$SCRIPTS/02_dedup.py"
        ;;
    identify)
        echo "[Step 3/6] Identifying file types..."
        $PYTHON "$SCRIPTS/03_identify_types.py"
        ;;
    convert)
        echo "[Step 4/6] Converting old formats..."
        $PYTHON "$SCRIPTS/04_convert_formats.py"
        ;;
    extract)
        echo "[Step 5/6] Extracting text..."
        $PYTHON "$SCRIPTS/05_extract_text.py"
        ;;
    embed)
        echo "[Step 6/6] Generating embeddings..."
        $PYTHON "$SCRIPTS/06_embed_cloud.py"
        ;;
    all)
        echo "Running full pipeline..."
        echo "[Step 1/6] Scanning files..."
        $PYTHON "$SCRIPTS/01_scan_files.py"
        echo "[Step 2/6] Deduplicating..."
        $PYTHON "$SCRIPTS/02_dedup.py"
        echo "[Step 3/6] Identifying file types..."
        $PYTHON "$SCRIPTS/03_identify_types.py"
        echo "[Step 4/6] Converting old formats..."
        $PYTHON "$SCRIPTS/04_convert_formats.py"
        echo "[Step 5/6] Extracting text..."
        $PYTHON "$SCRIPTS/05_extract_text.py"
        echo "[Step 6/6] Generating embeddings..."
        $PYTHON "$SCRIPTS/06_embed_cloud.py"
        echo "Pipeline complete!"
        ;;
    web)
        echo "Starting Streamlit web app..."
        STREAMLIT="$SCRIPT_DIR/.venv/bin/streamlit"
        if [ ! -f "$STREAMLIT" ]; then
            STREAMLIT="streamlit"
        fi
        $STREAMLIT run "$SCRIPT_DIR/server/web_app.py" --server.port 8501
        ;;
    stats)
        $PYTHON -c "import sys; sys.path.insert(0, '$SCRIPT_DIR'); from server.rag_query import RAGQueryService; svc = RAGQueryService('none'); import json; print(json.dumps(svc.get_stats(), indent=2)); svc.close()"
        ;;
    setup)
        $PYTHON "$SCRIPTS/00_setup.py"
        ;;
    *)
        echo "Usage: ./run_pipeline.sh [step]"
        echo "Steps:"
        echo "  scan      - Scan all files in source directory"
        echo "  dedup     - Deduplicate by MD5 hash"
        echo "  identify  - Identify types for extensionless files"
        echo "  convert   - Convert old Office formats (.doc, .xls, .ppt)"
        echo "  extract   - Extract text from all files"
        echo "  embed     - Generate embeddings"
        echo "  all       - Run all steps in sequence"
        echo "  web       - Start Streamlit web interface"
        echo "  stats     - Show database statistics"
        echo "  setup     - Run interactive setup wizard"
        ;;
esac
