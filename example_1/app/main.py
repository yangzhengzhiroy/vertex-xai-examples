"""
Custom server entry point
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import tensorflow as tf
import uvicorn

from app.entities import PredictionRequest


# Logging setup
formatter = logging.Formatter(
    "[%(asctime)s] - [%(levelname)s] - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %Z00",
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger = logging.getLogger("tf-custom-xai")
logger.addHandler(handler)
logger.setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    global model_serving
    _default_model_path = "<local-model-path>"
    model_path = os.getenv('MODEL_PATH', _default_model_path)
    if os.getenv("ENV") != "local":
        model_path = f"gs://{model_path}"
    model = tf.saved_model.load(model_path)
    model_serving = model.signatures['serving_default']
    logger.info("Model loaded successfully.")
    yield


app = FastAPI(lifespan=lifespan)


@app.get(os.getenv("AIP_HEALTH_ROUTE", "/health"), status_code=200)
def health():
    return JSONResponse({"status": "OK"}, 200)


@app.post(os.getenv("AIP_PREDICT_ROUTE", "/predict"))
def predict(request: PredictionRequest):
    req = request.model_dump()
    pred_ls = []
    for instance in req["instances"]:
        pred = model_serving(**instance)
        pred = pred["output"].numpy().tolist()[0][0]
        pred_ls.append({"output": pred})
    return JSONResponse({"predictions": pred_ls}, 200)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
