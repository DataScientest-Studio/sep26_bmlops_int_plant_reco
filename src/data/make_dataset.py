"""One-time script: store a subset of the PlantVillage images in a SQLite database."""
import random
import sqlite3
from pathlib import Path

RAW_DIR = Path("data/raw/color")
DB_PATH = Path("data/plants.db")
N_PER_CLASS = 100
SEED = 42


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_path TEXT UNIQUE,
            label TEXT NOT NULL,
            image BLOB NOT NULL,
            added_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    random.seed(SEED)
    for class_dir in sorted(RAW_DIR.iterdir()):
        files = sorted(class_dir.glob("*"))
        chosen = random.sample(files, min(N_PER_CLASS, len(files)))
        for f in chosen:
            conn.execute(
                "INSERT OR IGNORE INTO images (source_path, label, image) VALUES (?, ?, ?)",
                (f"{class_dir.name}/{f.name}", class_dir.name, f.read_bytes()),
            )
    conn.commit()

    total = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    n_classes = conn.execute("SELECT COUNT(DISTINCT label) FROM images").fetchone()[0]
    print(f"{total} images from {n_classes} classes stored in {DB_PATH}")
    conn.close()


if __name__ == "__main__":
    main()
