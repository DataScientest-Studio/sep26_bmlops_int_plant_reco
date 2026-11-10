
"""Send modified PlantVillage images to the API to simulate drift."""

import argparse
import io
import random
from pathlib import Path

import requests
from PIL import Image, ImageEnhance, ImageFilter

from src.data.db import get_connection

RAW_DIR = Path("data/raw/color")
API_URL = "http://localhost:8000/predict"
N_IMAGES = 30
SEED = 42


def get_images():
    """Return 30 deterministic image paths from the dataset."""
    files = sorted(
        path
        for path in RAW_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if len(files) < N_IMAGES:
        raise ValueError(f"Need at least {N_IMAGES} images, found {len(files)}")

    return random.Random(SEED).sample(files, N_IMAGES)


def transform_image(image, mode):
    """Apply the selected image transformation."""
    if mode == "dark":
        image = ImageEnhance.Brightness(image).enhance(0.4)
    elif mode == "bright":
        image = ImageEnhance.Brightness(image).enhance(1.6)
    elif mode == "blur":
        image = image.filter(ImageFilter.GaussianBlur(4))

    return image


def send_image(path, mode):
    """Transform one image and send it to the prediction API."""
    with Image.open(path) as original:
        image = transform_image(original.convert("RGB"), mode)

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)

    response = requests.post(
        API_URL,
        files={"file": ("leaf.jpg", buffer, "image/jpeg")},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def print_statistics():
    """Print average confidence, brightness and sharpness of last 30 predictions."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    AVG(confidence) AS avg_confidence,
                    AVG(brightness) AS avg_brightness,
                    AVG(sharpness) AS avg_sharpness
                FROM (
                    SELECT confidence, brightness, sharpness
                    FROM predictions
                    ORDER BY created_at DESC, id DESC
                    LIMIT 30
                ) AS recent
                """
            )
            stats = cur.fetchone()

    print("\nStatistics for the last 30 predictions:")
    print(f"Average confidence: {stats['avg_confidence']:.4f}")
    print(f"Average brightness: {stats['avg_brightness']:.2f}")
    print(f"Average sharpness: {stats['avg_sharpness']:.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=["normal", "dark", "bright", "blur"],
        help="Image condition to simulate",
    )
    args = parser.parse_args()

    if not RAW_DIR.is_dir():
        raise FileNotFoundError(f"Image directory not found: {RAW_DIR.resolve()}")

    images = get_images()
    print(f"Sending {len(images)} {args.mode} images to {API_URL}")

    for index, path in enumerate(images, start=1):
        result = send_image(path, args.mode)
        print(f"{index}/{N_IMAGES}: {result}")

    print_statistics()


if __name__ == "__main__":
    main()
