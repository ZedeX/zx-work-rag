"""Analyze files that still need embedding after merge fix."""
import sqlite3
import json
from pathlib import Path

DB_PATH = r"E:\git\zx-work-rag\data\file_manifest.db"
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# 1. Overall stats
c.execute("SELECT COUNT(*) FROM files WHERE is_duplicate=0")
unique = c.fetchone()[0]
c.execute("SELECT COUNT(*) FROM files WHERE is_duplicate=0 AND text_extracted=1")
extracted = c.fetchone()[0]
c.execute("SELECT COUNT(*) FROM files WHERE is_duplicate=0 AND embedded=1")
embedded = c.fetchone()[0]
c.execute("SELECT COUNT(*) FROM files WHERE is_duplicate=0 AND text_extracted=1 AND embedded=0")
pending = c.fetchone()[0]

print("=== Overall Stats ===")
print(f"Unique files:        {unique:,}")
print(f"Text extracted:      {extracted:,}")
print(f"Embedded:            {embedded:,}")
print(f"Pending embed:       {pending:,}")

# 2. Pending embed by category
print("\n=== Pending Embed by Category ===")
c.execute("""
    SELECT file_category, COUNT(*) as cnt
    FROM files
    WHERE is_duplicate=0 AND text_extracted=1 AND embedded=0
    GROUP BY file_category ORDER BY cnt DESC
""")
for row in c.fetchall():
    print(f"  {row[0]}: {row[1]:,}")

# 3. Pending embed by extension (top 20)
print("\n=== Pending Embed by Extension (top 20) ===")
c.execute("""
    SELECT extension, COUNT(*) as cnt
    FROM files
    WHERE is_duplicate=0 AND text_extracted=1 AND embedded=0
    GROUP BY extension ORDER BY cnt DESC LIMIT 20
""")
for row in c.fetchall():
    print(f"  {row[0]}: {row[1]:,}")

# 4. Pending embed by status
print("\n=== Pending Embed by Status ===")
c.execute("""
    SELECT status, COUNT(*) as cnt
    FROM files
    WHERE is_duplicate=0 AND text_extracted=1 AND embedded=0
    GROUP BY status ORDER BY cnt DESC
""")
for row in c.fetchall():
    print(f"  {row[0]}: {row[1]:,}")

# 5. Check how many pending files have actual text content (not just metadata)
c.execute("""
    SELECT COUNT(*) FROM files
    WHERE is_duplicate=0 AND text_extracted=1 AND embedded=0
    AND status = 'text_extracted'
    AND extracted_text IS NOT NULL AND extracted_text != ''
""")
real_text = c.fetchone()[0]
print(f"\n=== Pending with real text content (status=text_extracted): {real_text:,} ===")

c.execute("""
    SELECT COUNT(*) FROM files
    WHERE is_duplicate=0 AND text_extracted=1 AND embedded=0
    AND status = 'metadata_only'
""")
meta_only = c.fetchone()[0]
print(f"Pending with metadata_only: {meta_only:,}")

# 6. Estimate chunks for real text files
c.execute("""
    SELECT SUM(text_length) FROM files
    WHERE is_duplicate=0 AND text_extracted=1 AND embedded=0
    AND status = 'text_extracted'
    AND extracted_text IS NOT NULL AND extracted_text != ''
""")
total_chars = c.fetchone()[0] or 0
chunk_size = 512
overlap = 64
est_chunks = total_chars // (chunk_size - overlap) if total_chars > 0 else 0
print(f"Total chars in pending text: {total_chars:,}")
print(f"Estimated chunks (512/64): {est_chunks:,}")
print(f"Estimated embed time at 100 t/s: {est_chunks/100/3600:.1f} hours")

conn.close()
