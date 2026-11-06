"""FastAPI prediction service."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import load_model, predict

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["model"] = load_model()
        logger.info("Model loaded")
    except Exception:
        logger.exception("Model failed to load")
        state["model"] = None
    yield


app = FastAPI(title="House Price Prediction API", version="1.0.0", lifespan=lifespan)


class PredictionInput(BaseModel):
    MedInc: float = Field(..., gt=0, description="Median income (10k USD)")
    HouseAge: float = Field(..., ge=0, le=100)
    AveRooms: float = Field(..., gt=0)
    AveBedrms: float = Field(..., gt=0)
    Population: float = Field(..., gt=0)
    AveOccup: float = Field(..., gt=0)
    Latitude: float = Field(..., ge=32, le=43)
    Longitude: float = Field(..., ge=-125, le=-114)

    model_config = {"json_schema_extra": {"example": {
        "MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 6.98, "AveBedrms": 1.02,
        "Population": 322.0, "AveOccup": 2.55, "Latitude": 37.88, "Longitude": -122.23}}}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state.get("model") is not None}


@app.post("/predict")
def predict_endpoint(payload: PredictionInput):
    if state.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not available")
    try:
        result = predict(state["model"], [payload.model_dump()])
    except Exception:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail="Prediction failed")
    return {"prediction": result, "unit": "100k USD"}