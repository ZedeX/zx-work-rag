import sqlite3
from pathlib import Path

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "file_manifest.db"
conn = sqlite3.connect(str(DB_PATH))
c = conn.cursor()

c.execute("SELECT status, COUNT(*) FROM files WHERE is_duplicate=0 GROUP BY status ORDER BY COUNT(*) DESC")
rows = c.fetchall()
print("=== Unique files by status ===")
for r in rows:
    print(f"  {r[0]}: {r[1]:,}")

c.execute("SELECT file_category, COUNT(*) FROM files WHERE is_duplicate=0 GROUP BY file_category ORDER BY COUNT(*) DESC")
rows = c.fetchall()
print("\n=== By category ===")
for r in rows:
    print(f"  {r[0]}: {r[1]:,}")

c.execute("""SELECT COUNT(*) FROM files WHERE is_duplicate=0 AND text_extracted=1 AND embedded=0 AND extracted_text IS NOT NULL AND extracted_text != ''""")
pending = c.fetchone()[0]
print(f"\nPending embed (extracted but not embedded): {pending:,}")

c.execute("SELECT COUNT(*) FROM files WHERE is_duplicate=0 AND text_extracted=0 AND status NOT IN ('duplicate')")
not_extracted = c.fetchone()[0]
print(f"Not extracted yet: {not_extracted:,}")

conn.close()
