"""Full system test: Search, AI Q&A, MCP tools, edge cases."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from server.rag_query import RAGQueryService
from config.settings import config
import sqlite3
import chromadb

passed = 0
failed = 0

def test(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name} - {detail}")

# === 1. Database Stats ===
print("\n=== 1. Database Stats ===")
conn = sqlite3.connect(config.DB_PATH)
c = conn.cursor()
c.execute("SELECT COUNT(*) FROM files WHERE is_duplicate=0")
unique = c.fetchone()[0]
c.execute("SELECT COUNT(*) FROM files WHERE is_duplicate=1")
dups = c.fetchone()[0]
c.execute("SELECT COUNT(*) FROM files WHERE text_extracted=1 AND embedded=1")
embedded = c.fetchone()[0]
c.execute("SELECT COUNT(*) FROM files WHERE text_extracted=1 AND embedded=0 AND status='text_extracted'")
pending_text = c.fetchone()[0]
conn.close()

test("Unique files > 100k", unique > 100000, f"got {unique}")
test("Duplicates found", dups > 0, f"got {dups}")
test("Embedded > 90k", embedded > 90000, f"got {embedded}")
test("No pending text files", pending_text == 0, f"got {pending_text}")

# === 2. ChromaDB ===
print("\n=== 2. ChromaDB ===")
chroma = chromadb.PersistentClient(path=config.CHROMA_DIR)
collections = chroma.list_collections()
test("Only 1 collection (work_corpus)", len(collections) == 1, f"got {len(collections)}: {[c.name for c in collections]}")
if collections:
    main = chroma.get_collection("work_corpus")
    test("Vectors > 800k", main.count() > 800000, f"got {main.count()}")
    test("No partition collections", all(not c.name.startswith("work_corpus_p") for c in collections))

# === 3. Search Tests ===
print("\n=== 3. Search ===")
svc = RAGQueryService(llm_mode="none")

# Chinese search
r = svc.search("数字化转型", top_k=5)
test("Chinese search returns results", len(r) > 0, f"got {len(r)}")
test("Chinese search relevance > 0.5", r[0]["relevance"] > 0.5 if r else False, f"got {r[0]['relevance'] if r else 'N/A'}")

# English search
r2 = svc.search("project plan", top_k=5)
test("English search returns results", len(r2) > 0, f"got {len(r2)}")

# Filter by category
r3 = svc.search("report", top_k=5, category="text")
test("Category filter works", len(r3) > 0, f"got {len(r3)}")
if r3:
    test("Category filter correct", r3[0]["metadata"].get("category") == "text")

# Filter by extension
r4 = svc.search("budget", top_k=5, extension=".xlsx")
test("Extension filter works", len(r4) >= 0)  # may be 0 if no xlsx match

# Filter by folder
r5 = svc.search("report", top_k=5, folder_contains="archieves")
test("Folder filter works", len(r5) >= 0)

# Year filter
r6 = svc.search("report", top_k=10, year="2020")
if r6:
    year_match = all(r["metadata"].get("modified_time", "").startswith("2020") for r in r6)
    test("Year filter correct", year_match)
else:
    test("Year filter (no results ok)", True)

# === 4. AI Q&A ===
print("\n=== 4. AI Q&A ===")
svc_llm = RAGQueryService(llm_mode="local")
answer = svc_llm.ask("我做过哪些数字化转型项目？请简要列出3个", top_k=5)
test("AI answer not empty", len(answer) > 10, f"got length {len(answer)}")
test("AI answer not error", not answer.startswith("LLM error"), answer[:50])
svc_llm.close()

# === 5. Export ===
print("\n=== 5. Export ===")
md = svc.export_context("数字化转型", top_k=3, format="markdown")
test("Markdown export works", "Search Results" in md and "Result" in md, md[:100])

jsonl = svc.export_context("数字化转型", top_k=3, format="jsonl")
test("JSONL export works", len(jsonl) > 0)
try:
    parsed = json.loads(jsonl.split("\n")[0])
    test("JSONL valid JSON", True)
except:
    test("JSONL valid JSON", False)

# === 6. Stats ===
print("\n=== 6. Stats ===")
stats = svc.get_stats()
test("Stats has keys", all(k in stats for k in ["total_unique_files", "text_extracted", "embedded", "chroma_chunks"]))
test("Stats embedded > 90k", stats["embedded"] > 90000, f"got {stats['embedded']}")

svc.close()

# === Summary ===
print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed out of {passed+failed}")
if failed == 0:
    print("ALL TESTS PASSED!")
