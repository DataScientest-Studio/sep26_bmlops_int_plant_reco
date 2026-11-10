
"""Store API predictions in PostgreSQL."""

from src.data.db import get_connection


def save_prediction(result, features):
    """Save one prediction and its image features in PostgreSQL."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO predictions (
                    label,
                    confidence,
                    model_version,
                    brightness,
                    contrast,
                    mean_r,
                    mean_g,
                    mean_b,
                    sharpness
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    result["label"],
                    float(result["confidence"]),
                    int(result["model_version"]),
                    features["brightness"],
                    features["contrast"],
                    features["mean_r"],
                    features["mean_g"],
                    features["mean_b"],
                    features["sharpness"],
                ),
            )
