"""RAG Query Service - Local query over ChromaDB + SQLite.

Supports:
1. Semantic search (vector similarity)
2. Metadata filtering (by year, folder, category, extension)
3. Answer generation via LM Studio (local) or Volcengine Ark (cloud)
4. Export results for other LLMs (Claude Code, Doubao, etc.)
"""
import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import config
from loguru import logger

import chromadb
from openai import OpenAI


class RAGQueryService:
    """Local RAG query service."""

    def __init__(self, llm_mode: str = "local"):
        """
        Args:
            llm_mode: "local" (LM Studio), "cloud" (Volcengine Ark), or "none" (search only)
        """
        self.llm_mode = llm_mode
        self.chroma_client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        self.collection = self.chroma_client.get_or_create_collection(
            name="work_corpus",
            metadata={"hnsw:space": "cosine"}
        )
        self.sqlite_conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
        self.sqlite_conn.row_factory = sqlite3.Row

        # Initialize LLM client
        if llm_mode == "local":
            self.llm_client = OpenAI(
                api_key="lm-studio",
                base_url=config.LM_STUDIO_BASE_URL,
            )
            self.llm_model = config.LM_STUDIO_CHAT_MODEL
        elif llm_mode == "cloud":
            self.llm_client = OpenAI(
                api_key=config.ARK_API_KEY,
                base_url=config.ARK_CODING_BASE_URL,
            )
            self.llm_model = config.ARK_LLM_MODEL
        else:
            self.llm_client = None
            self.llm_model = None

    def search(self, query: str, top_k: int = 10,
               category: Optional[str] = None,
               extension: Optional[str] = None,
               year: Optional[str] = None,
               folder_contains: Optional[str] = None,
               ) -> list[dict]:
        """Semantic search with optional metadata filtering.

        Args:
            query: Search query text
            top_k: Number of results
            category: Filter by file category (text, image, video, audio, etc.)
            extension: Filter by file extension
            year: Filter by year (e.g. "2023")
            folder_contains: Filter by folder path containing this string

        Returns:
            List of search results with metadata
        """
        # Build ChromaDB where filter
        where_filter = {}
        conditions = []

        if category:
            conditions.append({"category": category})
        if extension:
            conditions.append({"extension": extension})
        if folder_contains:
            conditions.append({"folder_tags": {"$contains": folder_contains}})

        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}

        # Query ChromaDB - embed the query using same mode as indexing
        if config.EMBEDDING_MODE == "local":
            embed_client = OpenAI(
                api_key="lm-studio",
                base_url=config.LM_STUDIO_BASE_URL,
            )
            embed_model = config.LM_STUDIO_EMBEDDING_MODEL
        else:
            embed_client = OpenAI(
                api_key=config.ARK_API_KEY,
                base_url=config.ARK_BASE_URL,
            )
            embed_model = config.ARK_EMBEDDING_MODEL

        query_embedding = embed_client.embeddings.create(
            model=embed_model,
            input=[query],
        ).data[0].embedding

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter if where_filter else None,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        formatted = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                doc = results["documents"][0][i] if results["documents"] else ""
                dist = results["distances"][0][i] if results["distances"] else 0

                # Year filtering (post-filter since ChromaDB may not support date range)
                if year and meta.get("modified_time", ""):
                    if not meta["modified_time"].startswith(year):
                        continue

                formatted.append({
                    "id": doc_id,
                    "text": doc,
                    "metadata": meta,
                    "distance": dist,
                    "relevance": 1 - dist,  # cosine distance -> similarity
                })

        return formatted

    def ask(self, question: str, top_k: int = 5, **filter_kwargs) -> str:
        """Ask a question and get an AI-generated answer.

        1. Search for relevant documents
        2. Build a prompt with context
        3. Generate answer using LLM
        """
        if not self.llm_client:
            return "Error: No LLM configured. Set llm_mode to 'local' or 'cloud'."

        # Search for relevant context
        results = self.search(question, top_k=top_k, **filter_kwargs)

        if not results:
            return "No relevant documents found for your question."

        # Build context
        context_parts = []
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            source = f"{meta.get('file_name', 'unknown')} ({meta.get('folder_path', '')})"
            context_parts.append(f"[Source {i}: {source}]\n{r['text'][:2000]}")

        context = "\n\n".join(context_parts)

        # Build prompt
        system_prompt = """You are a helpful assistant that answers questions based on the user's personal work documents.
Answer in the same language as the question (Chinese by default).
When citing information, mention the source file name.
If the context doesn't contain enough information to answer, say so honestly.
Be concise but thorough."""

        user_prompt = f"""Based on the following documents from my work files:

{context}

---

Question: {question}

Please answer based on the above documents. Cite source file names when possible."""

        # Generate answer
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=4000,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"LLM error: {e}"

    def export_context(self, question: str, top_k: int = 20,
                       format: str = "markdown", **filter_kwargs) -> str:
        """Export search results as context for other LLMs (Claude Code, Doubao, etc.).

        Args:
            format: "markdown", "jsonl", or "text"
        """
        results = self.search(question, top_k=top_k, **filter_kwargs)

        if format == "jsonl":
            lines = []
            for r in results:
                lines.append(json.dumps({
                    "text": r["text"],
                    "source": r["metadata"].get("file_name", ""),
                    "path": r["metadata"].get("folder_path", ""),
                    "relevance": round(r["relevance"], 3),
                }, ensure_ascii=False))
            return "\n".join(lines)

        elif format == "markdown":
            parts = [f"# Search Results: {question}\n"]
            for i, r in enumerate(results, 1):
                meta = r["metadata"]
                parts.append(f"## Result {i}: {meta.get('file_name', 'unknown')}")
                parts.append(f"- Path: {meta.get('folder_path', '')}")
                parts.append(f"- Category: {meta.get('category', '')}")
                parts.append(f"- Relevance: {r['relevance']:.3f}")
                parts.append(f"\n{r['text'][:3000]}\n")
            return "\n".join(parts)

        else:  # plain text
            parts = []
            for r in results:
                parts.append(r["text"])
            return "\n\n---\n\n".join(parts)

    def get_stats(self) -> dict:
        """Get database statistics."""
        c = self.sqlite_conn.cursor()
        c.execute("SELECT COUNT(*) FROM files WHERE is_duplicate = 0")
        total_unique = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM files WHERE embedded = 1")
        embedded = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM files WHERE text_extracted = 1")
        extracted = c.fetchone()[0]

        return {
            "total_unique_files": total_unique,
            "text_extracted": extracted,
            "embedded": embedded,
            "chroma_chunks": self.collection.count(),
        }

    def close(self):
        self.sqlite_conn.close()


# CLI interface for quick testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG Query Service")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--mode", default="none", choices=["local", "cloud", "none"],
                        help="LLM mode")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--category", help="Filter by category")
    parser.add_argument("--export", choices=["markdown", "jsonl", "text"],
                        help="Export format for other LLMs")
    args = parser.parse_args()

    svc = RAGQueryService(llm_mode=args.mode)

    if args.export:
        print(svc.export_context(args.query, top_k=args.top_k, format=args.export,
                                 category=args.category))
    elif args.mode != "none":
        print(svc.ask(args.query, top_k=args.top_k, category=args.category))
    else:
        results = svc.search(args.query, top_k=args.top_k, category=args.category)
        for r in results:
            meta = r["metadata"]
            print(f"\n[{r['relevance']:.3f}] {meta.get('file_name', '')} ({meta.get('category', '')})")
            print(f"  Path: {meta.get('folder_path', '')}")
            print(f"  {r['text'][:200]}...")

    svc.close()
