import os
import time

import httpx
import numpy as np

from classify.lexicon import LEXICON_WEIGHT, lexicon_distribution
from classify.questions import (
    INTENT_LABELS,
    INTENT_PRIOR,
    INTENT_PROTOTYPES,
    RESOLUTION_LABELS,
    RESOLUTION_PROTOTYPES,
    SENTIMENT_FACTOR,
    SENTIMENT_LABELS,
    SENTIMENT_PROTOTYPES,
    URGENCY_FACTOR,
    URGENCY_LABELS,
    URGENCY_PROTOTYPES,
    build_jev_questions,
)
from classify.schema import Classification

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
PROTOTYPE_TEMPERATURE = 0.05


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


class PrototypeBackend:
    name = "prototype-embedding"

    def __init__(self, temperature: float = PROTOTYPE_TEMPERATURE):
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self.temperature = temperature
        self._embed_fn = DefaultEmbeddingFunction()
        self._axes = {
            "intent": (INTENT_PROTOTYPES, INTENT_LABELS),
            "sentiment": (SENTIMENT_PROTOTYPES, SENTIMENT_LABELS),
            "urgency": (URGENCY_PROTOTYPES, URGENCY_LABELS),
            "resolution": (RESOLUTION_PROTOTYPES, RESOLUTION_LABELS),
        }
        self._matrices: dict[str, np.ndarray] = {}
        for axis, (protos, labels) in self._axes.items():
            all_texts = [text for label in labels for text in protos[label]]
            vectors = np.array(
                [_normalize(np.array(v)) for v in self._embed_fn(all_texts)]
            )
            matrix = []
            offset = 0
            for label in labels:
                count = len(protos[label])
                centroid = _normalize(vectors[offset : offset + count].mean(axis=0))
                matrix.append(centroid)
                offset += count
            self._matrices[axis] = np.array(matrix)

    def classify(self, subject: str, body: str) -> Classification:
        text = f"{subject}\n{body}"
        query = _normalize(np.array(self._embed_fn([text])[0]))

        intent = self._axis("intent", INTENT_LABELS, query)
        sentiment = self._blend(
            text, "sentiment", SENTIMENT_LABELS, self._axis("sentiment", SENTIMENT_LABELS, query)
        )
        urgency = self._blend(
            text, "urgency", URGENCY_LABELS, self._axis("urgency", URGENCY_LABELS, query)
        )
        resolution = self._blend(
            text, "resolution", RESOLUTION_LABELS, self._axis("resolution", RESOLUTION_LABELS, query)
        )

        top_intent = max(intent, key=intent.get)
        top_sentiment = max(sentiment, key=sentiment.get)
        top_urgency = max(urgency, key=urgency.get)
        action_probability = resolution["action"]
        auto = (
            intent[top_intent]
            * INTENT_PRIOR[top_intent]
            * SENTIMENT_FACTOR[top_sentiment]
            * URGENCY_FACTOR[top_urgency]
            * (1.0 - 0.6 * action_probability)
        )
        return Classification(
            intent=intent,
            sentiment=sentiment,
            urgency=urgency,
            auto_resolvable_confidence=float(auto),
        )

    def _blend(
        self, text: str, axis: str, labels: list[str], embed_probs: dict[str, float]
    ) -> dict[str, float]:
        lex = lexicon_distribution(text, axis)
        return {
            label: (1 - LEXICON_WEIGHT) * embed_probs[label] + LEXICON_WEIGHT * lex[label]
            for label in labels
        }

    def _axis(self, axis: str, labels: list[str], query: np.ndarray) -> dict[str, float]:
        sims = self._matrices[axis] @ query
        probs = _softmax(sims / self.temperature)
        return {label: float(p) for label, p in zip(labels, probs)}


class JevBackend:
    name = "jev-system-one"

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str = JEV_ENDPOINT,
        model: str = JEV_MODEL,
        timeout: float = 30.0,
    ):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY") or os.environ.get("JEV_API_KEY")
        if not self.api_key:
            raise RuntimeError("JevBackend requires TYPESAFE_API_KEY or JEV_API_KEY")
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout

    @staticmethod
    def available() -> bool:
        return bool(os.environ.get("TYPESAFE_API_KEY") or os.environ.get("JEV_API_KEY"))

    def classify(self, subject: str, body: str) -> Classification:
        payload = {
            "model": self.model,
            "state": {"subject": subject, "body": body},
            "questions": build_jev_questions(),
        }
        response = httpx.post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        answers = response.json()["answers"]

        intent = self._renorm(
            {label: float(answers["intent"]["probabilities"].get(label, 0.0)) for label in INTENT_LABELS}
        )
        sentiment = self._renorm(
            {label: float(answers["sentiment"]["probabilities"].get(label, 0.0)) for label in SENTIMENT_LABELS}
        )
        raw_urgency = answers["urgency"]["probabilities"]
        urgency = self._renorm(
            {
                label: float(raw_urgency.get(str(i), raw_urgency.get(i, 0.0)))
                for i, label in enumerate(URGENCY_LABELS)
            }
        )
        auto = float(answers["auto_resolvable"]["noul"])
        return Classification(
            intent=intent,
            sentiment=sentiment,
            urgency=urgency,
            auto_resolvable_confidence=auto,
        )

    @staticmethod
    def _renorm(probs: dict[str, float]) -> dict[str, float]:
        total = sum(probs.values())
        if total <= 0:
            n = len(probs)
            return {k: 1.0 / n for k in probs}
        if abs(total - 1.0) > 1e-6:
            return {k: v / total for k, v in probs.items()}
        return probs


_default_backend = None


def get_default_backend():
    global _default_backend
    if _default_backend is None:
        _default_backend = JevBackend() if JevBackend.available() else PrototypeBackend()
    return _default_backend
