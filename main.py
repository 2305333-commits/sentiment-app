from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import re
import os

app = FastAPI()

# ── CORS (allows browser fetch() to talk to this server) ─────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model & vectorizer ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model      = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))

# ── Request schema ────────────────────────────────────────────────────────────
class TextInput(BaseModel):
    text: str

# ── Text cleaning (must match training) ──────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    return text.strip()

# ── Serve front-end ───────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

@app.get("/")
def root():
    return FileResponse(os.path.join(BASE_DIR, "templates", "index.html"))

# ── Prediction endpoint ───────────────────────────────────────────────────────
@app.post("/predict")
def predict(payload: TextInput):
    cleaned = clean_text(payload.text)
    vec     = vectorizer.transform([cleaned])
    pred    = model.predict(vec)[0]
    proba   = model.predict_proba(vec)[0]
    label   = "Positive" if pred == 1 else "Negative"
    confidence = round(float(max(proba)) * 100, 1)
    return {
        "label":      label,
        "confidence": confidence,
        "emoji":      "😊" if pred == 1 else "😞",
    }
