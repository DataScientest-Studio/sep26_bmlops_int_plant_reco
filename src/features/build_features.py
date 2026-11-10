
import io

import numpy as np
from PIL import Image


def image_features(image_bytes):
    """
    Extract numerical features from an image.

    Parameters
    ----------
    image_bytes : bytes
        Original image content.

    Returns
    -------
    dict
        Image brightness, contrast, RGB means and sharpness.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Convert image to NumPy arrays
    rgb = np.asarray(image, dtype=np.float64)
    gray = np.asarray(image.convert("L"), dtype=np.float64)

    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "mean_r": float(rgb[:, :, 0].mean()),
        "mean_g": float(rgb[:, :, 1].mean()),
        "mean_b": float(rgb[:, :, 2].mean()),
        "sharpness": float(np.abs(np.diff(gray, axis=1)).mean()),
    }
