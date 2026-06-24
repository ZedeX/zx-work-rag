"""MCP Server for Trae IDE integration.

Provides tools for searching the local knowledge base.
Trae can call these tools to answer questions about your work files.

Setup in Trae MCP config:
{
  "mcpServers": {
    "zx-work-rag": {
      "command": "/path/to/your/.venv/Scripts/python.exe",
      "args": ["/path/to/zx-work-rag/server/mcp_server.py"]
    }
  }
}
"""
import sys
import json
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import config

from mcp.server.fastmcp import FastMCP

# Import RAG service
from server.rag_query import RAGQueryService

# Create MCP server
mcp = FastMCP(
    "zx-work-rag",
    instructions="Personal work knowledge base. Search and query across all work documents (doc, ppt, xlsx, pdf, etc.)."
)

# Lazy-init service
_service: Optional[RAGQueryService] = None


def get_service() -> RAGQueryService:
    global _service
    if _service is None:
        _service = RAGQueryService(llm_mode="none")
    return _service


@mcp.tool()
def search_knowledge_base(query: str, top_k: int = 5,
                          category: str = None,
                          extension: str = None,
                          year: str = None,
                          folder_contains: str = None) -> str:
    """Search personal work knowledge base for relevant documents.

    Use this tool when the user asks questions about their past work,
    projects, documents, reports, or any content from their work files.

    Args:
        query: Natural language search query (in Chinese or English)
        top_k: Number of results to return (default 5, max 20)
        extension: Filter by file extension (e.g. ".pdf", ".docx")
        category: Filter by category: text, image, video, audio, code, other
        year: Filter by year (e.g. "2023")
        folder_contains: Filter by folder name or tag
    """
    top_k = min(top_k, 20)
    svc = get_service()
    results = svc.search(
        query=query,
        top_k=top_k,
        category=category,
        extension=extension,
        year=year,
        folder_contains=folder_contains,
    )

    if not results:
        return json.dumps({"found": False, "message": "No relevant documents found"}, ensure_ascii=False)

    output = []
    for r in results:
        meta = r["metadata"]
        output.append({
            "file_name": meta.get("file_name", ""),
            "folder_path": meta.get("folder_path", ""),
            "category": meta.get("category", ""),
            "extension": meta.get("extension", ""),
            "relevance": round(r["relevance"], 3),
            "text_preview": r["text"][:500],
        })

    return json.dumps({"found": True, "count": len(output), "results": output}, ensure_ascii=False, indent=2)


@mcp.tool()
def get_document_detail(file_name: str = None, folder_contains: str = None) -> str:
    """Get full text content of a specific document from the knowledge base.

    Use this when you need more detail about a specific document found
    in a previous search.

    Args:
        file_name: Exact or partial file name to look for
        folder_contains: Folder path filter to narrow results
    """
    import sqlite3

    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    conditions = ["is_duplicate = 0", "text_extracted = 1"]
    params = []

    if file_name:
        conditions.append("file_name LIKE ?")
        params.append(f"%{file_name}%")
    if folder_contains:
        conditions.append("folder_path LIKE ?")
        params.append(f"%{folder_contains}%")

    c.execute(f"""
        SELECT file_name, file_path, extension, category, folder_path,
               folder_tags, extracted_text, text_length
        FROM files
        WHERE {' AND '.join(conditions)}
        ORDER BY text_length DESC
        LIMIT 3
    """, params)

    rows = c.fetchall()
    conn.close()

    if not rows:
        return json.dumps({"found": False, "message": "Document not found"}, ensure_ascii=False)

    results = []
    for row in rows:
        results.append({
            "file_name": row["file_name"],
            "folder_path": row["folder_path"],
            "extension": row["extension"],
            "text_length": row["text_length"],
            "text": row["extracted_text"][:3000] if row["extracted_text"] else "",
        })

    return json.dumps({"found": True, "results": results}, ensure_ascii=False, indent=2)


@mcp.tool()
def get_knowledge_base_stats() -> str:
    """Get statistics about the personal work knowledge base.

    Returns counts of files, categories, and indexing progress.
    """
    svc = get_service()
    stats = svc.get_stats()
    return json.dumps(stats, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
