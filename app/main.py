import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import redis
import sklearn
from fastapi import FastAPI
from pydantic import Field, create_model

# ---------- logging: stdout only ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
logging.basicConfig(level=LOG_LEVEL, handlers=[_h], force=True)
for _n in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    _l = logging.getLogger(_n)
    _l.handlers = [_h]
    _l.setLevel(LOG_LEVEL)
    _l.propagate = False
log = logging.getLogger("triage")

# ---------- schema ----------
_BASES = ["radius", "texture", "perimeter", "area", "smoothness", "compactness",
          "concavity", "concave_points", "symmetry", "fractal_dimension"]
FEATURES = ([f"mean_{b}" for b in _BASES] + [f"{b}_error" for b in _BASES]
            + [f"worst_{b}" for b in _BASES])

PredictRequest = create_model(
    "PredictRequest",
    **{f: (float, Field(ge=0, allow_inf_nan=False)) for f in FEATURES},
)

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
    card_path = path.with_name("model_card.json")
    if not path.is_file() or not card_path.is_file():
        log.critical("Model artifact not found at %s (or model_card.json). Exiting.", path)
        raise RuntimeError(f"model artifact missing: {path}")
    state["model"] = joblib.load(path)
    state["card"] = json.loads(card_path.read_text())
    if state["card"]["feature_names"] != FEATURES:
        log.critical("Model card feature names do not match API schema")
        raise RuntimeError("feature mismatch")
    state["threshold"] = float(os.getenv("DECISION_THRESHOLD", "0.5"))
    state["version"] = os.getenv("MODEL_VERSION") or state["card"]["model_version"]

    host = os.getenv("REDIS_HOST")
    state["redis"] = None
    if host:
        state["redis"] = redis.Redis(
            host=host, port=int(os.getenv("REDIS_PORT", "6379")),
            socket_connect_timeout=1, socket_timeout=1)
        log.info("Redis counting enabled (%s)", host)
    else:
        log.info("REDIS_HOST not set: request counting disabled")
    log.info("Model loaded: %s v%s threshold=%s",
             state["card"]["model_family"], state["version"], state["threshold"])
    yield


app = FastAPI(title="Breast Cancer Triage API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    x = np.array([[getattr(req, f) for f in FEATURES]], dtype=float)
    p = float(state["model"].predict_proba(x)[0, 1])   # P(malignant)
    t = state["threshold"]
    r = state["redis"]
    if r is not None:
        try:
            r.incr("predictions_total")
        except redis.RedisError as e:
            log.warning("Redis unavailable, counter not incremented: %s", e)
    return {
        "label": "malignant" if p >= t else "benign",
        "malignant_probability": round(p, 4),
        "threshold": t,
        "model_version": state["version"],
    }


@app.get("/model/info")
def model_info():
    c = state["card"]
    return {
        "model_version": state["version"],
        "model_family": c["model_family"],
        "scikit_learn_version": sklearn.__version__,
        "git_commit": os.getenv("GIT_SHA", "unknown"),
        "decision_threshold": state["threshold"],
        "feature_count": len(c["feature_names"]),
        "metrics": c["metrics"]["test"],
    }


@app.get("/stats")
def stats():
    r = state["redis"]
    if r is None:
        return {"redis_enabled": False, "predictions_served": None}
    try:
        return {"redis_enabled": True,
                "predictions_served": int(r.get("predictions_total") or 0)}
    except redis.RedisError:
        return {"redis_enabled": True, "predictions_served": None,
                "error": "redis unreachable"}