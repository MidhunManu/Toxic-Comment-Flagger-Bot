import numpy as np
import json
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.sequence import pad_sequences

FEEDBACK_FILE = "feedback_store.json"

class ContextAwarePredictor:
    def __init__(self, model, tokenizer, max_len=150, threshold=0.5, similarity_threshold=0.6):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.threshold = threshold
        self.similarity_threshold = similarity_threshold
        self.label_names = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
        self.forgiven_texts = []  # just store raw texts now
        self._load_feedback()

    def _is_similar_to_forgiven(self, text):
        if not self.forgiven_texts:
            return set()

        all_texts = [item["original_text"] for item in self.forgiven_texts] + [text]
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        current_vec = tfidf_matrix[-1]

        labels_to_suppress = set()
        for i, item in enumerate(self.forgiven_texts):
            sim = cosine_similarity(current_vec, tfidf_matrix[i])[0][0]
            if sim >= self.similarity_threshold:
                labels_to_suppress.update(item["forgive_labels"])

        return labels_to_suppress

    def predict(self, text):
        seq = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=self.max_len, padding='post', truncating='post')
        pred = self.model.predict(padded, verbose=0)[0]

        suppressed = self._is_similar_to_forgiven(text)

        scores = {}
        flags = []
        for label, p in zip(self.label_names, pred):
            score = float(round(p, 3))
            scores[label] = score
            if score >= self.threshold and label not in suppressed:
                flags.append(label)

        return {
            "scores": scores,
            "flags": flags,
            "suppressed_labels": list(suppressed)
        }

    def flag_as_safe(self, text, labels_to_forgive=None):
        entry = {
            "original_text": text,
            "forgive_labels": labels_to_forgive or self.label_names
        }
        self.forgiven_texts.append(entry)
        self._save_feedback()

    def _save_feedback(self):
        with open(FEEDBACK_FILE, "w") as f:
            json.dump(self.forgiven_texts, f)

    def _load_feedback(self):
        try:
            with open(FEEDBACK_FILE, "r") as f:
                self.forgiven_texts = json.load(f)
            print(f"✅ Loaded {len(self.forgiven_texts)} feedback entries")
        except FileNotFoundError:
            pass
