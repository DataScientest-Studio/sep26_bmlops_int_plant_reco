# Plant Reco — MLOps project

Recognise the plant species and its disease from a photo of a leaf, and build everything
around the model that makes it production-ready.

Team: Christoph, Adrian · Mentor: Nicolas · Liora MLOps bootcamp (Sep 2026)

## Project objectives

Recognise the plant species and its disease from a photo of a leaf
(PlantVillage dataset, 38 classes including healthy leaves),
and make the model production-ready: a local database, training
and prediction scripts, an API, and later tracking, versioning,
automation and monitoring.

## Key metrics

**Model**
- Macro-F1 on a held-out test set (main metric: the classes are unevenly sized)
- Accuracy (secondary)

**Service**
- The API answers /predict correctly (status 200, a class name returned)
- Prediction time per image (latency)

## Phase 1 — Foundations

| Component | File | What it does |
|---|---|---|
| Database | `src/data/make_dataset.py` | One-time script: stores 100 images per class (3,800) in SQLite `data/plants.db` |
| Training | `src/models/training.py` | Reads the images from the database, trains a Random Forest on 32×32 pixels, saves `models/model.joblib` |
| Prediction | `src/models/predict.py` | Predicts the class of one image |
| API | `src/api/main.py` | FastAPI with `POST /training` and `POST /predict` |

Baseline result (20 % test split, 760 images): **accuracy 0.537, macro-F1 0.516**.
The model is deliberately simple; the focus of this project is the MLOps around it.

## How to run

```bash
#1.Environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

#2.Data (needs a Kaggle API token in KAGGLE_API_TOKEN)
kaggle datasets download -d abdallahalidev/plantvillage-dataset -p data/raw
unzip -q data/raw/plantvillage-dataset.zip "plantvillage dataset/color/*" -d data/raw
mv "data/raw/plantvillage dataset/color" data/raw/color

#3.Database (run once)
python src/data/make_dataset.py

#4.Train and predict from the command line
python -m src.models.training
python -m src.models.predict path/to/leaf.jpg

# 5. API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
curl -X POST http://localhost:8000/training
curl -X POST -F "file=@path/to/leaf.jpg" http://localhost:8000/predict
```
