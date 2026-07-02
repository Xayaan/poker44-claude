"""Production inference for the Poker44 bot-detection ensemble.

Loads the trained artifact once and scores whole requests (all chunks of a
DetectionSynapse together, so batch-relative features mirror training).
Falls back to a deterministic heuristic if the artifact is unavailable so the
miner never drops a response.
"""

from __future__ import annotations

import pickle
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from poker44.detection.features import FEATURE_NAMES, extract_chunk_features

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
_NEUTRAL_SCORE = 0.5


class DetectionModel:
    """Thread-safe scorer: one risk score per chunk, higher = more bot-like."""

    def __init__(self, model_path: Optional[Path] = None):
        self._path = Path(model_path) if model_path is not None else MODEL_PATH
        self._lock = threading.Lock()
        self._artifact: Optional[Dict[str, Any]] = None
        self._load_error: Optional[str] = None
        self._load()

    def _load(self) -> None:
        try:
            with open(self._path, "rb") as fh:
                artifact = pickle.load(fh)
            if artifact.get("format") != "poker44-detection-v1":
                raise ValueError(f"unexpected artifact format: {artifact.get('format')}")
            if int(artifact.get("n_features", -1)) != len(FEATURE_NAMES):
                raise ValueError(
                    f"feature count mismatch: artifact={artifact.get('n_features')} "
                    f"code={len(FEATURE_NAMES)}"
                )
            self._artifact = artifact
            self._load_error = None
        except Exception as exc:  # noqa: BLE001
            self._artifact = None
            self._load_error = f"{type(exc).__name__}: {exc}"

    @property
    def ready(self) -> bool:
        return self._artifact is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def score_chunks(self, chunks: List[List[dict]]) -> List[float]:
        """Score every chunk of one request. Always returns len(chunks) floats
        in [0, 1]; never raises."""
        if not chunks:
            return []
        try:
            return self._score_chunks_model(chunks)
        except Exception:  # noqa: BLE001
            return [self._heuristic_chunk(chunk) for chunk in chunks]

    def _score_chunks_model(self, chunks: List[List[dict]]) -> List[float]:
        if self._artifact is None:
            return [self._heuristic_chunk(chunk) for chunk in chunks]

        absolute = np.vstack([self._safe_features(chunk) for chunk in chunks])
        if absolute.shape[0] >= 3:
            relative = absolute - np.median(absolute, axis=0)
        else:
            relative = np.zeros_like(absolute)
        X = np.hstack([absolute, relative])

        with self._lock:
            gbms = self._artifact["gbms"]
            lr = self._artifact["lr"]
            lr_weight = float(self._artifact.get("lr_weight", 0.15))
            proba = np.mean([m.predict_proba(X)[:, 1] for m in gbms], axis=0)
            proba = (1.0 - lr_weight) * proba + lr_weight * lr.predict_proba(X)[:, 1]

        scores: List[float] = []
        for chunk, p in zip(chunks, proba):
            if not isinstance(chunk, list) or not chunk:
                scores.append(_NEUTRAL_SCORE)
                continue
            value = float(p)
            if not np.isfinite(value):
                value = _NEUTRAL_SCORE
            scores.append(min(1.0, max(0.0, round(value, 6))))
        return scores

    @staticmethod
    def _safe_features(chunk: Any) -> np.ndarray:
        try:
            if isinstance(chunk, list):
                return extract_chunk_features(chunk)
        except Exception:  # noqa: BLE001
            pass
        return np.zeros(len(FEATURE_NAMES), dtype=float)

    @staticmethod
    def _heuristic_chunk(chunk: Any) -> float:
        """Dependency-free fallback using the two most stable signals:
        action-sequence repetition and bet-size skew."""
        if not isinstance(chunk, list) or not chunk:
            return _NEUTRAL_SCORE
        try:
            seqs: Dict[str, int] = {}
            amounts: List[float] = []
            n_hands = 0
            for hand in chunk:
                if not isinstance(hand, dict):
                    continue
                actions = hand.get("actions") or []
                if not isinstance(actions, list):
                    continue
                n_hands += 1
                parts = []
                for action in actions:
                    if not isinstance(action, dict):
                        continue
                    parts.append(str(action.get("action_type") or "")[:2])
                    try:
                        amount = float(action.get("normalized_amount_bb") or 0.0)
                    except (TypeError, ValueError):
                        amount = 0.0
                    if amount > 0:
                        amounts.append(amount)
                seq = "|".join(parts)
                seqs[seq] = seqs.get(seq, 0) + 1
            if n_hands < 2:
                return _NEUTRAL_SCORE
            collision = sum(c * (c - 1) for c in seqs.values()) / float(
                n_hands * (n_hands - 1)
            )
            # Training-population anchors: human collision ~0.025, bot ~0.037.
            score = 0.5 + 6.0 * (collision - 0.031)
            if amounts:
                mean_amount = sum(amounts) / len(amounts)
                score += 0.004 * (mean_amount - 51.0)
            return min(1.0, max(0.0, round(score, 6)))
        except Exception:  # noqa: BLE001
            return _NEUTRAL_SCORE


_default_model: Optional[DetectionModel] = None
_default_model_lock = threading.Lock()


def get_default_model() -> DetectionModel:
    global _default_model
    with _default_model_lock:
        if _default_model is None:
            _default_model = DetectionModel()
        return _default_model
