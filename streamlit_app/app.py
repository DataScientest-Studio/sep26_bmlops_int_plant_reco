"""Streamlit app: get a prediction for a leaf photo."""
import os

import requests
import streamlit as st
from PIL import UnidentifiedImageError

API_URL = os.getenv("API_URL", "http://api:8000")


def api_is_up():
    """Return True if the API answers its health check."""
    try:
        return requests.get(API_URL, timeout=5).ok
    except requests.RequestException:
        return False


def predict(image):
    """Send the image to the API and return its answer."""
    files = {"file": (image.name, image.getvalue())}
    response = requests.post(
        f"{API_URL}/predict", files=files, timeout=60
    )
    response.raise_for_status()
    return response.json()


st.title("Plant Reco")
st.write(
    "Upload a photo of a leaf. The app sends it to "
    "our API, which answers with the production model."
)

if api_is_up():
    st.sidebar.success("API is up")
else:
    st.sidebar.error("API is not reachable")

image = st.file_uploader(
    "Leaf image", type=["jpg", "jpeg", "png"]
)

if image is not None:
    try:
        st.image(image, width=256)
    except UnidentifiedImageError:
        st.error("This file is not a valid image.")
        st.stop()
    if st.button("Predict"):
        try:
            result = predict(image)
        except requests.RequestException as error:
            st.error(f"No prediction: {error}")
        else:
            plant, disease = result["label"].split("___")
            answer = f"{plant}: {disease}".replace("_", " ")
            st.success(answer)
            left, right = st.columns(2)
            left.metric(
                "Confidence", f"{result['confidence']:.0%}"
            )
            right.metric(
                "Model version", result["model_version"]
            )
