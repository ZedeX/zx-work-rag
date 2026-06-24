"""Full RAG validation after repair embedding."""
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server.rag_query import RAGQueryService
from config.settings import config

svc = RAGQueryService("none")

# 1. Stats
print("=== Database Stats ===")
conn = sqlite3.connect(config.DB_PATH)
c = conn.cursor()
c.execute("SELECT COUNT(*) FROM files WHERE is_duplicate=0")
unique = c.fetchone()[0]
c.execute("SELECT COUNT(*) FROM files WHERE is_duplicate=0 AND text_extracted=1")
extracted = c.fetchone()[0]
c.execute("SELECT COUNT(*) FROM files WHERE is_duplicate=0 AND embedded=1")
embedded = c.fetchone()[0]
c.execute("SELECT COUNT(*) FROM files WHERE is_duplicate=0 AND text_extracted=1 AND embedded=0 AND status='text_extracted'")
pending_text = c.fetchone()[0]
c.execute("SELECT COUNT(*) FROM files WHERE is_duplicate=0 AND text_extracted=1 AND embedded=0 AND status='metadata_only'")
pending_meta = c.fetchone()[0]
conn.close()

print(f"  Unique files:     {unique:,}")
print(f"  Text extracted:   {extracted:,}")
print(f"  Embedded:         {embedded:,}")
print(f"  Pending (text):   {pending_text:,}")
print(f"  Pending (meta):   {pending_meta:,}")
print(f"  Embed coverage:   {embedded/extracted*100:.1f}%")

# 2. ChromaDB count
import chromadb
chroma = chromadb.PersistentClient(path=config.CHROMA_DIR)
coll = chroma.get_collection("work_corpus")
print(f"  ChromaDB vectors: {coll.count():,}")

# 3. Search tests
queries = [
    "数字化转型",
    "年度总结报告",
    "项目计划书",
    "Photoshop tutorial",
    "budget allocation",
]

print("\n=== Search Validation ===")
for q in queries:
    results = svc.search(q, top_k=3)
    if results:
        top = results[0]
        rel = top.get("relevance", 0)
        fname = top.get("metadata", {}).get("file_name", "?")
        ext = top.get("metadata", {}).get("extension", "?")
        print(f"  [{rel:.4f}] '{q}' -> {fname}{ext}")
    else:
        print(f"  [NO RESULTS] '{q}'")

svc.close()
print("\n=== Validation Complete ===")
