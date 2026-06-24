"""Clean up partition collections (p0-p7) from ChromaDB.

These were used during parallel embedding and are now redundant
since all vectors have been merged into work_corpus.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import config
import chromadb

chroma_client = chromadb.PersistentClient(path=config.CHROMA_DIR)

# List all collections
collections = chroma_client.list_collections()
print(f"Total collections: {len(collections)}")
for coll in collections:
    print(f"  {coll.name}: {coll.count()} vectors")

# Delete partition collections
deleted = 0
freed_vectors = 0
for coll in collections:
    if coll.name.startswith("work_corpus_p"):
        count = coll.count()
        print(f"Deleting {coll.name} ({count} vectors)...", end=" ")
        chroma_client.delete_collection(coll.name)
        print("OK")
        deleted += 1
        freed_vectors += count

print(f"\nDeleted {deleted} partition collections, freed {freed_vectors:,} redundant vectors")

# Verify main collection
main = chroma_client.get_collection("work_corpus")
print(f"Main collection 'work_corpus': {main.count():,} vectors (intact)")
