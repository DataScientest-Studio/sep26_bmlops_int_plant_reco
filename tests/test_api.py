"""Simple tests that need no database and no MLflow server."""
import io

from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app
from src.features.build_features import image_features
from src.models.training import IMG_SIZE, preprocess

client = TestClient(app)


def make_image(color):
    """Return a small one-colour JPEG image as bytes."""
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_root_says_ok():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_rejects_other_file_types():
    files = {"file": ("notes.txt", b"not an image")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400


def test_metrics_show_our_histograms():
    response = client.get("/metrics")
    assert response.status_code == 200
    for name in ["prediction_confidence", "image_brightness",
                 "image_sharpness"]:
        assert name in response.text


def test_preprocess_gives_flat_vector():
    vector = preprocess(make_image("green"))
    assert vector.shape == (IMG_SIZE * IMG_SIZE * 3,)
    assert vector.min() >= 0 and vector.max() <= 1


def test_image_features_of_white_and_black_images():
    white = image_features(make_image("white"))
    black = image_features(make_image("black"))
    assert white["brightness"] > 250
    assert black["brightness"] < 5
    assert white["sharpness"] == 0  # one colour: no edges
