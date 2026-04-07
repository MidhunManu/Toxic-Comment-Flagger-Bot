from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

from predictor import ContextAwarePredictor

app = FastAPI()

# --- Load model and tokenizer once at startup ---
model = tf.keras.models.load_model("base_toxic_model.h5")

with open("base_tokenizer.json", "r") as f:
    tokenizer = tokenizer_from_json(f.read())

predictor = ContextAwarePredictor(model, tokenizer, similarity_threshold=0.92)

# --- Request schemas ---
class CommentRequest(BaseModel):
    text: str

class FeedbackRequest(BaseModel):
    text: str
    labels_to_forgive: Optional[List[str]] = None  # None = forgive all flags

# --- Endpoints ---
@app.post("/predict")
def predict(req: CommentRequest):
    return predictor.predict(req.text)

@app.post("/feedback/safe")
def mark_safe(req: FeedbackRequest):
    predictor.flag_as_safe(req.text, req.labels_to_forgive)
    return {"status": "ok", "message": f"Stored correction for: '{req.text}'"}

@app.get("/feedback/list")
def list_feedback():
    return [
            {"text": item["original_text"], "forgive_labels": item["forgive_labels"]}
            for item in predictor.forgiven_embeddings
            ]
