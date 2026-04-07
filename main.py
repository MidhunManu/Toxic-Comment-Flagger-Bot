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

predictor = ContextAwarePredictor(model, tokenizer, similarity_threshold=0.3)

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
            for item in predictor.forgiven_texts
            ]

@app.post("/debug/similarity")
def debug_similarity(req: CommentRequest):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    if not predictor.forgiven_texts:
        return {"error": "no feedback stored"}
    
    forgiven_texts = [item["original_text"] for item in predictor.forgiven_texts]
    all_texts = forgiven_texts + [req.text]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    current_vec = tfidf_matrix[-1]
    
    results = []
    for i, item in enumerate(predictor.forgiven_texts):
        sim = cosine_similarity(current_vec, tfidf_matrix[i])[0][0]
        results.append({
            "forgiven_text": item["original_text"],
            "input_text": req.text,
            "similarity": float(round(sim, 4)),
            "threshold": predictor.similarity_threshold,
            "would_suppress": bool(sim >= predictor.similarity_threshold)
        })
    
    return results
