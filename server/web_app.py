"""Streamlit Web UI for the RAG Query Service.

Run: streamlit run server/web_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from openai import OpenAI
from server.rag_query import RAGQueryService
from config.settings import config


def is_configured():
    """Check if the system has been set up (database exists and has data)."""
    db_path = Path(config.DB_PATH)
    if not db_path.exists():
        return False
    import sqlite3
    try:
        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM files")
        count = c.fetchone()[0]
        conn.close()
        return count > 0
    except Exception:
        return False


def show_setup_guide():
    """Show setup instructions when the system is not configured."""
    st.header("Welcome to Work Knowledge Base!")
    st.markdown("""
    It looks like this is your first time running the system. Follow these steps to get started:

    ### Quick Setup

    **Option 1: Interactive Setup Wizard (Recommended)**
    ```bash
    python scripts/00_setup.py
    ```

    **Option 2: Manual Setup**

    1. Copy `.env.example` to `.env` and configure:
       ```bash
       cp .env.example .env
       ```

    2. Edit `.env` and set `SOURCE_DIR` to your work files directory

    3. Start your embedding provider:
       - **LM Studio**: Download from https://lmstudio.ai/, load an embedding model
       - **Ollama**: Install from https://ollama.com/, run `ollama pull nomic-embed-text`

    4. Run the pipeline:
       ```bash
       # Windows
       run_pipeline.bat all

       # Linux/macOS
       bash run_pipeline.sh all
       ```

    5. Refresh this page when done!

    ### Embedding Provider Options

    | Provider | Cost | Setup | Best For |
    |----------|------|-------|----------|
    | LM Studio | Free | Desktop app | Windows users, privacy |
    | Ollama | Free | CLI | Linux/Mac, easy setup |
    | vLLM / LocalAI | Free | Self-hosted | GPU servers |
    | Volcengine Ark | Paid | API key | Large scale, Chinese docs |
    | OpenAI | Paid | API key | Quick start, high quality |
    """)


def init_service():
    if "service" not in st.session_state:
        st.session_state.service = RAGQueryService(llm_mode="none")
    return st.session_state.service


def ensure_llm_client(svc, mode):
    """Ensure LLM client is initialized for the given mode."""
    if mode == "local":
        svc.llm_client = OpenAI(api_key="lm-studio", base_url=config.LM_STUDIO_BASE_URL)
        svc.llm_model = config.LM_STUDIO_CHAT_MODEL
    elif mode == "cloud":
        svc.llm_client = OpenAI(api_key=config.ARK_API_KEY, base_url=config.ARK_CODING_BASE_URL)
        svc.llm_model = config.ARK_LLM_MODEL
    svc.llm_mode = mode


def main():
    st.set_page_config(
        page_title="Work Knowledge Base",
        page_icon="📚",
        layout="wide",
    )

    st.title("📚 Work Knowledge Base")
    st.caption("Search and query across all your work documents")

    # Check if configured
    if not is_configured():
        show_setup_guide()
        return

    svc = init_service()

    # Sidebar - Stats & Settings
    with st.sidebar:
        st.header("Stats")
        stats = svc.get_stats()
        st.metric("Unique Files", f"{stats['total_unique_files']:,}")
        st.metric("Text Extracted", f"{stats['text_extracted']:,}")
        st.metric("Embedded", f"{stats['embedded']:,}")
        st.metric("ChromaDB Chunks", f"{stats['chroma_chunks']:,}")

        st.divider()
        st.header("LLM Mode")
        llm_mode_label = st.radio("Answer generation:", ["Search Only", "Local (LM Studio / Ollama)", "Cloud (Ark)"],
                                   index=0)
        mode_map = {"Search Only": "none", "Local (LM Studio / Ollama)": "local", "Cloud (Ark)": "cloud"}
        ensure_llm_client(svc, mode_map[llm_mode_label])

        st.divider()
        st.header("Filters")
        category = st.selectbox("Category", [None, "text", "image", "video", "audio", "code", "other"])
        year = st.text_input("Year", placeholder="e.g. 2023")
        folder = st.text_input("Folder contains", placeholder="e.g. data modeling")

    # Main area - Search
    query = st.text_input("Ask a question or search:",
                          placeholder="e.g. What digital transformation projects have I done?")

    col1, col2, col3 = st.columns(3)
    with col1:
        search_btn = st.button("🔍 Search", type="primary", use_container_width=True)
    with col2:
        ask_btn = st.button("💬 Ask AI", use_container_width=True)
    with col3:
        export_btn = st.button("📋 Export", use_container_width=True)

    if query:
        filter_kwargs = {}
        if category:
            filter_kwargs["category"] = category
        if year:
            filter_kwargs["year"] = year
        if folder:
            filter_kwargs["folder_contains"] = folder

        if ask_btn:
            # Auto-switch to local mode if still "none"
            if svc.llm_mode == "none":
                ensure_llm_client(svc, "local")
            with st.spinner("Searching and generating answer..."):
                results = svc.search(query, top_k=5, **filter_kwargs)
                if not results:
                    st.info("No relevant documents found.")
                else:
                    answer = svc.ask(query, top_k=5, **filter_kwargs)
                    st.markdown("### AI Answer")
                    st.markdown(answer)
                    with st.expander("View sources"):
                        for i, r in enumerate(results, 1):
                            meta = r["metadata"]
                            st.markdown(f"**#{i} {meta.get('file_name', 'unknown')}** "
                                        f"(relevance: {r['relevance']:.3f})")
                            st.caption(f"Path: {meta.get('folder_path', '')} | "
                                       f"Extension: {meta.get('extension', '')}")
                            st.text(r["text"][:1000])

        elif search_btn:
            with st.spinner("Searching..."):
                results = svc.search(query, top_k=10, **filter_kwargs)

            if not results:
                st.info("No results found.")
            else:
                st.success(f"Found {len(results)} results")
                for i, r in enumerate(results, 1):
                    meta = r["metadata"]
                    with st.expander(
                        f"#{i} {meta.get('file_name', 'unknown')} "
                        f"(relevance: {r['relevance']:.3f})",
                        expanded=(i <= 3)
                    ):
                        st.caption(f"Path: {meta.get('folder_path', '')} | "
                                   f"Category: {meta.get('category', '')} | "
                                   f"Extension: {meta.get('extension', '')}")
                        st.text(r["text"][:2000])

        elif export_btn:
            with st.spinner("Exporting..."):
                exported = svc.export_context(query, top_k=20, format="markdown", **filter_kwargs)
            st.code(exported, language="markdown")
            st.download_button("Download as Markdown", exported,
                               file_name="rag_export.md", mime="text/markdown")


if __name__ == "__main__":
    main()
