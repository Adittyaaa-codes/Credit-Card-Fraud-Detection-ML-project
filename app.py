import sys
import os

import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()
mongo_db_url = os.getenv("DB_URL")

import pymongo
from src.exception.exception import CustomException
from src.logging.logger import logging
from src.pipeline.train_pipeline import TrainingPipeline

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Request
from uvicorn import run as app_run
from fastapi.responses import Response, JSONResponse
from starlette.responses import RedirectResponse
import pandas as pd

from src.utils.utils import load_object
from src.utils.ml_utils import DetectionModel

from src.constants.app_constants import DATABASE_NAME, COLLECTION_NAME

client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
database = client[DATABASE_NAME]
collection = database[COLLECTION_NAME]

app = FastAPI(title="Credit Card Fraud Detection API", version="1.0.0")
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Health"])
async def health():
    return JSONResponse({"status": "ok"})


@app.get("/train", tags=["Training"])
async def train_route():
    try:
        logging.info("Training pipeline triggered via API")
        train_pipeline = TrainingPipeline()
        model_artifact = train_pipeline.run_pipeline()
        return JSONResponse({
            "message": "Training successful",
            "trained_model_path": model_artifact.trained_model_file_path,
            "train_f1": round(model_artifact.train_metric_artifact.f1_score, 4),
            "test_f1":  round(model_artifact.test_metric_artifact.f1_score, 4),
            "train_precision": round(model_artifact.train_metric_artifact.precision_score, 4),
            "test_precision":  round(model_artifact.test_metric_artifact.precision_score, 4),
            "train_recall": round(model_artifact.train_metric_artifact.recall_score, 4),
            "test_recall":  round(model_artifact.test_metric_artifact.recall_score, 4),
        })
    except Exception as e:
        raise CustomException(e, sys)


@app.post("/predict", tags=["Prediction"])
async def predict_route(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        logging.info(f"Received prediction request: {df.shape[0]} rows")

        model = load_object("final_model/model.pkl")
        input_df = df.drop(columns=["Class"], errors="ignore")
        y_pred = model.predict(input_df)

        df["predicted_column"] = y_pred
        df["fraud_label"] = df["predicted_column"].map({0: "Legitimate", 1: "Fraudulent"})

        os.makedirs("prediction_output", exist_ok=True)
        df.to_csv("prediction_output/output.csv", index=False)

        n_fraud = int((y_pred == 1).sum())
        n_legit = int((y_pred == 0).sum())

        logging.info(f"Prediction done: {n_fraud} fraud, {n_legit} legit")
        return JSONResponse({
            "total": len(y_pred),
            "fraudulent": n_fraud,
            "legitimate": n_legit,
            "fraud_rate_pct": round(n_fraud / len(y_pred) * 100, 2),
            "predictions": df[["predicted_column", "fraud_label"]].to_dict(orient="records"),
        })
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    app_run(app, host="localhost", port=8000)
