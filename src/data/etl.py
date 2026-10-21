"""ETL step: add the next batch of not yet used images to PostgreSQL (simulated new data)."""
import sys
from pathlib import Path

from src.data.db import get_connection

RAW_DIR = Path("data/raw/color")
PER_CLASS = 20


def run_etl(per_class=PER_CLASS):
    """Extract new image files, turn them into rows and load them into the images table."""
    added = 0
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT source_path FROM images")
            known = {row["source_path"] for row in cur.fetchall()}

            for class_dir in sorted(RAW_DIR.iterdir()):
                if not class_dir.is_dir():
                    continue
                new_files = [
                    f for f in sorted(class_dir.iterdir())
                    if f.is_file() and f"{class_dir.name}/{f.name}" not in known
                ]
                for image_file in new_files[:per_class]:
                    cur.execute(
                        """
                        INSERT INTO images (source_path, label, image)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (source_path) DO NOTHING
                        """,
                        (
                            f"{class_dir.name}/{image_file.name}",
                            class_dir.name,
                            image_file.read_bytes(),
                        ),
                    )
                    added += cur.rowcount

            cur.execute("SELECT COUNT(*) AS total FROM images")
            total = cur.fetchone()["total"]

    return {"added": added, "total": total}


if __name__ == "__main__":
    per_class = int(sys.argv[1]) if len(sys.argv) > 1 else PER_CLASS
    print(run_etl(per_class))
