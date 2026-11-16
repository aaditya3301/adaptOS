"""The switchboard: decide which adapter (if any) should answer a request.

A small sentence-embedding model turns the request into a vector, and a linear classifier
picks one of the adapter tasks or ``other``. ``other`` and low-confidence requests go to the
plain base model.

The classifier is stored as plain JSON (labels, weights, bias), not as a pickle, so loading a
router version from the registry never executes code.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

OTHER = "other"  # label for general requests: answered by the base model
ROUTER_FILE = "router.json"

EmbedFn = Callable[[Sequence[str]], np.ndarray]


@dataclass(frozen=True)
class RouteDecision:
    adapter: str | None  # adapter task to use, or None for the plain base model
    label: str  # the classifier's top label
    confidence: float
    probabilities: dict[str, float]
    reason: str  # "adapter" | "general_request" | "low_confidence"


def sentence_embedder(model_id: str, revision: str) -> EmbedFn:
    """Lazy-loading embedding function (normalized vectors)."""
    model: Any = None

    def embed(texts: Sequence[str]) -> np.ndarray:
        nonlocal model
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_id, revision=revision, device="cpu")
        return model.encode(
            list(texts), batch_size=64, normalize_embeddings=True, convert_to_numpy=True
        )

    return embed


class RouterModel:
    def __init__(
        self,
        labels: Sequence[str],
        coef: np.ndarray,
        intercept: np.ndarray,
        embedding_model: str,
        embedding_revision: str,
        threshold: float,
        embed_fn: EmbedFn | None = None,
    ):
        if len(labels) < 3:
            raise ValueError("router needs at least 3 labels (softmax over a multinomial model)")
        if coef.shape[0] != len(labels) or intercept.shape != (len(labels),):
            raise ValueError("coef/intercept shapes do not match the labels")
        self.labels = list(labels)
        self.coef = np.asarray(coef, dtype=np.float32)
        self.intercept = np.asarray(intercept, dtype=np.float32)
        self.embedding_model = embedding_model
        self.embedding_revision = embedding_revision
        self.threshold = threshold
        self.embed = embed_fn or sentence_embedder(embedding_model, embedding_revision)

    # ------------------------------------------------------------- persistence
    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / ROUTER_FILE
        path.write_text(
            json.dumps(
                {
                    "labels": self.labels,
                    "coef": self.coef.tolist(),
                    "intercept": self.intercept.tolist(),
                    "embedding_model": self.embedding_model,
                    "embedding_revision": self.embedding_revision,
                    "threshold": self.threshold,
                }
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(
        cls, directory: Path, threshold: float | None = None, embed_fn: EmbedFn | None = None
    ) -> RouterModel:
        data = json.loads((Path(directory) / ROUTER_FILE).read_text(encoding="utf-8"))
        return cls(
            labels=data["labels"],
            coef=np.array(data["coef"]),
            intercept=np.array(data["intercept"]),
            embedding_model=data["embedding_model"],
            embedding_revision=data["embedding_revision"],
            threshold=data["threshold"] if threshold is None else threshold,
            embed_fn=embed_fn,
        )

    # --------------------------------------------------------------- inference
    def probabilities_from_embeddings(self, x: np.ndarray) -> np.ndarray:
        logits = x @ self.coef.T + self.intercept
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return self.probabilities_from_embeddings(self.embed(texts))

    def decide(self, probs: np.ndarray) -> RouteDecision:
        i = int(probs.argmax())
        label, confidence = self.labels[i], float(probs[i])
        distribution = {lab: round(float(p), 4) for lab, p in zip(self.labels, probs, strict=True)}
        if label == OTHER:
            return RouteDecision(None, label, confidence, distribution, "general_request")
        if confidence < self.threshold:
            return RouteDecision(None, label, confidence, distribution, "low_confidence")
        return RouteDecision(label, label, confidence, distribution, "adapter")

    def route(self, text: str) -> RouteDecision:
        return self.decide(self.predict_proba([text])[0])

    def route_batch(self, texts: Sequence[str]) -> list[RouteDecision]:
        return [self.decide(p) for p in self.predict_proba(texts)]
