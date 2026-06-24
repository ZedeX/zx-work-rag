"""Temporary verification script for embedding pipeline."""
import sqlite3
import chromadb
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "data" / "file_manifest.db"
CHROMA_DIR = ROOT / "data" / "chroma_db"

print("=" * 60)
print("SQLite Statistics")
print("=" * 60)
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

total = c.execute("SELECT COUNT(*) FROM files").fetchone()[0]
unique = c.execute("SELECT COUNT(*) FROM files WHERE is_duplicate = 0").fetchone()[0]
text_extracted = c.execute("SELECT COUNT(*) FROM files WHERE text_extracted = 1").fetchone()[0]
embedded = c.execute("SELECT COUNT(*) FROM files WHERE embedded = 1").fetchone()[0]
not_embedded = c.execute(
    "SELECT COUNT(*) FROM files WHERE text_extracted = 1 AND embedded = 0 AND is_duplicate = 0"
).fetchone()[0]

print(f"Total files:          {total}")
print(f"Unique files:         {unique}")
print(f"Text extracted:       {text_extracted}")
print(f"Embedded:             {embedded}")
print(f"Text extracted but NOT embedded (unique): {not_embedded}")
conn.close()

print()
print("=" * 60)
print("ChromaDB Statistics")
print("=" * 60)
client = chromadb.PersistentClient(path=CHROMA_DIR)

main_coll = client.get_collection("work_corpus")
print(f"Main collection 'work_corpus': {main_coll.count()} vectors")

print()
print("All collections:")
for coll in client.list_collections():
    print(f"  {coll.name}: {coll.count()} vectors")

print()
print("=" * 60)
print("Verification complete.")
