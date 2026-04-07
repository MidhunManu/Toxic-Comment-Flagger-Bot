import numpy as np
import json
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

FEEDBACK_FILE = "feedback_store.json"

class ContextAwarePredictor:
    def __init__(self, model, tokenizer, max_len=150, threshold=0.5, similarity_threshold=0.85):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.threshold = threshold
        self.similarity_threshold = similarity_threshold
        self.label_names = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

        # REPLACE with this:
        self.embedding_model = tf.keras.Sequential([model.layers[0]])
        self.embedding_model.build(input_shape=(None, self.max_len))

        """
        self.embedding_model = tf.keras.Model(
                inputs=model.input,
                outputs=model.layers[0].output
                )
        """

        # Load saved feedback from disk if it exists
        self.forgiven_embeddings = []
        self._load_feedback()

    def _get_embedding(self, text):
        seq = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=self.max_len, padding='post', truncating='post')
        emb = self.embedding_model.predict(padded, verbose=0)
        return emb[0].mean(axis=0)

    def _is_similar_to_forgiven(self, embedding):
        labels_to_suppress = set()
        for item in self.forgiven_embeddings:
            sim = cosine_similarity([embedding], [item["embedding"]])[0][0]
            if sim >= self.similarity_threshold:
                labels_to_suppress.update(item["forgive_labels"])
        return labels_to_suppress

    def predict(self, text):
        seq = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=self.max_len, padding='post', truncating='post')
        pred = self.model.predict(padded, verbose=0)[0]

        emb = self._get_embedding(text)
        suppressed = self._is_similar_to_forgiven(emb)

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
        emb = self._get_embedding(text)
        entry = {
                "embedding": emb,
                "forgive_labels": labels_to_forgive or self.label_names,
                "original_text": text
                }
        self.forgiven_embeddings.append(entry)
        self._save_feedback()

    # --- Persistence ---
    def _save_feedback(self):
        data = [
                {
                    "embedding": item["embedding"].tolist(),
                    "forgive_labels": item["forgive_labels"],
                    "original_text": item["original_text"]
                    }
                for item in self.forgiven_embeddings
                ]
        with open(FEEDBACK_FILE, "w") as f:
            json.dump(data, f)

    def _load_feedback(self):
        try:
            with open(FEEDBACK_FILE, "r") as f:
                data = json.load(f)
            self.forgiven_embeddings = [
                    {
                        "embedding": np.array(item["embedding"]),
                        "forgive_labels": item["forgive_labels"],
                        "original_text": item["original_text"]
                        }
                    for item in data
                    ]
            print(f"✅ Loaded {len(self.forgiven_embeddings)} feedback entries")
        except FileNotFoundError:
            pass  # first run, no feedback yet
