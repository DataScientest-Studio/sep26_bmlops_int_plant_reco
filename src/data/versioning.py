"""Version the training data with DVC (without Git) and return its hash for MLflow."""
import csv
import subprocess
from pathlib import Path

import yaml

from src.data.db import get_connection

DVC_DIR = Path("dvc_data")
SNAPSHOT = "images.csv"


def version_dataset():
    """Write the list of images in the database to a CSV file, add it to DVC, return its md5."""
    DVC_DIR.mkdir(exist_ok=True)
    if not (DVC_DIR / ".dvc").exists():
        subprocess.run(["dvc", "init", "--no-scm"], cwd=DVC_DIR, check=True)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, source_path, label FROM images ORDER BY id")
            rows = cur.fetchall()

    with open(DVC_DIR / SNAPSHOT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "source_path", "label"])
        for row in rows:
            writer.writerow([row["id"], row["source_path"], row["label"]])

    subprocess.run(["dvc", "add", SNAPSHOT], cwd=DVC_DIR, check=True)
    meta = yaml.safe_load((DVC_DIR / f"{SNAPSHOT}.dvc").read_text())
    return meta["outs"][0]["md5"], len(rows)
