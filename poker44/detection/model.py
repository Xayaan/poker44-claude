"""Production inference for the Poker44 bot-detection ensemble.

Loads the trained artifact once and scores whole requests (all chunks of a
DetectionSynapse together, so batch-relative features mirror training).

Artifact preference order:
  1. model_v2.npz  — numpy-native export of the ensemble (flat arrays, no
     pickle, no scikit-learn at inference; immune to library version drift)
  2. model.pkl     — original scikit-learn pickle (requires a compatible
     scikit-learn/numpy at runtime)
  3. deterministic heuristic — so the miner never drops a response.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from poker44.detection.features import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    RequestContext,
    compute_request_context,
    extract_chunk_features,
)

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
MODEL_V2_PATH = Path(__file__).resolve().parent / "model_v2.npz"
_NEUTRAL_SCORE = 0.5
# Test-time augmentation: average scores over deterministic sub-chunk views.
# The reward is a pure ranking metric; averaging views reduces per-chunk score
# variance and lifted the per-window reward minimum on temporal holdout
# (research/ENGINEERING_LOG.md §9). Deterministic seed -> reproducible scores.
_TTA_VIEWS = 3
_TTA_MIN_HANDS = 19
_TTA_SEED = 1789


class _NumpyEnsemble:
    """Dependency-free predictor for the exported HistGB + LR blend.

    Replicates scikit-learn HistGradientBoostingClassifier inference exactly:
    per tree, walk nodes comparing x[feature_idx] <= threshold (NaN follows
    missing_go_to_left); sum leaf values onto the baseline; sigmoid. The six
    GBM probabilities are averaged, then blended with a standardized logistic
    regression.
    """

    def __init__(self, npz: Any):
        self.n_gbms = int(npz["n_gbms"][0])
        self.n_features = int(npz["n_features"][0])
        self.lr_weight = float(npz["lr_weight"][0])
        self.gbms = []
        for g in range(self.n_gbms):
            self.gbms.append(
                {
                    "tree_starts": npz[f"g{g}_tree_starts"].astype(np.int64),
                    "feature_idx": npz[f"g{g}_feature_idx"].astype(np.int64),
                    "threshold": npz[f"g{g}_threshold"].astype(np.float64),
                    "left": npz[f"g{g}_left"].astype(np.int64),
                    "right": npz[f"g{g}_right"].astype(np.int64),
                    "is_leaf": npz[f"g{g}_is_leaf"].astype(bool),
                    "missing_left": npz[f"g{g}_missing_left"].astype(bool),
                    "value": npz[f"g{g}_value"].astype(np.float64),
                    "baseline": float(npz[f"g{g}_baseline"][0]),
                }
            )
        self.lr_mean = npz["lr_mean"].astype(np.float64)
        self.lr_scale = npz["lr_scale"].astype(np.float64)
        self.lr_coef = npz["lr_coef"].astype(np.float64).ravel()
        self.lr_intercept = float(npz["lr_intercept"][0])

    @staticmethod
    def _sigmoid(raw: np.ndarray) -> np.ndarray:
        out = np.empty_like(raw)
        pos = raw >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-raw[pos]))
        e = np.exp(raw[~pos])
        out[~pos] = e / (1.0 + e)
        return out

    def _gbm_raw(self, gbm: Dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        raw = np.full(n, gbm["baseline"], dtype=np.float64)
        starts = gbm["tree_starts"]
        is_leaf = gbm["is_leaf"]
        feature_idx = gbm["feature_idx"]
        threshold = gbm["threshold"]
        left = gbm["left"]
        right = gbm["right"]
        missing_left = gbm["missing_left"]
        value = gbm["value"]
        rows = np.arange(n)
        for t in range(len(starts) - 1):
            base = starts[t]
            cur = np.full(n, base, dtype=np.int64)
            active = ~is_leaf[cur]
            # bounded walk: HistGB trees are shallow; 64 covers any depth
            for _ in range(64):
                if not active.any():
                    break
                idx = cur[active]
                x = X[rows[active], feature_idx[idx]]
                thr = threshold[idx]
                go_left = np.where(np.isnan(x), missing_left[idx], x <= thr)
                cur[active] = base + np.where(go_left, left[idx], right[idx])
                active = ~is_leaf[cur]
            raw += value[cur]
        return raw

    def predict_proba_1(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        proba = np.zeros(X.shape[0], dtype=np.float64)
        for gbm in self.gbms:
            proba += self._sigmoid(self._gbm_raw(gbm, X))
        proba /= float(self.n_gbms)
        z = (X - self.lr_mean) / self.lr_scale
        lr_p = self._sigmoid(z @ self.lr_coef + self.lr_intercept)
        return (1.0 - self.lr_weight) * proba + self.lr_weight * lr_p


class DetectionModel:
    """Thread-safe scorer: one risk score per chunk, higher = more bot-like."""

    def __init__(self, model_path: Optional[Path] = None):
        self._explicit_path = model_path is not None
        self._path = Path(model_path) if model_path is not None else MODEL_PATH
        self._lock = threading.Lock()
        self._numpy_model: Optional[_NumpyEnsemble] = None
        self._artifact: Optional[Dict[str, Any]] = None
        self._active_full_idx: np.ndarray = np.arange(
            2 * len(FEATURE_NAMES), dtype=np.int64
        )
        self._load_error: Optional[str] = None
        self._engine = "heuristic"
        self._load()

    def _load(self) -> None:
        errors: List[str] = []
        if not self._explicit_path:
            try:
                self._load_npz(MODEL_V2_PATH)
                self._engine = "numpy-v2"
                self._load_error = None
                return
            except Exception as exc:  # noqa: BLE001
                errors.append(f"npz: {type(exc).__name__}: {exc}")
        try:
            self._load_pickle(self._path)
            self._engine = "sklearn-pickle-v1"
            self._load_error = None
            return
        except Exception as exc:  # noqa: BLE001
            errors.append(f"pkl: {type(exc).__name__}: {exc}")
        self._engine = "heuristic"
        self._load_error = " | ".join(errors)

    def _load_npz(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as npz:
            if str(npz["format"][0]) != "poker44-detection-v2":
                raise ValueError(f"unexpected artifact format: {npz['format'][0]}")
            artifact_fv = int(npz["feature_version"][0]) if "feature_version" in npz else 1
            if artifact_fv != FEATURE_VERSION:
                raise ValueError(
                    f"feature version mismatch: artifact={artifact_fv} code={FEATURE_VERSION}"
                )
            active_idx = (
                npz["active_full_idx"].astype(np.int64)
                if "active_full_idx" in npz
                else np.arange(2 * len(FEATURE_NAMES), dtype=np.int64)
            )
            model = _NumpyEnsemble(npz)
        if active_idx.size and int(active_idx.max()) >= 2 * len(FEATURE_NAMES):
            raise ValueError("active_full_idx out of range for extracted features")
        if model.n_features != int(active_idx.size):
            raise ValueError(
                f"feature count mismatch: artifact={model.n_features} "
                f"active={int(active_idx.size)}"
            )
        self._active_full_idx = active_idx
        self._numpy_model = model

    def _load_pickle(self, path: Path) -> None:
        import pickle  # local: only the legacy path needs it

        with open(path, "rb") as fh:
            artifact = pickle.load(fh)
        if artifact.get("format") != "poker44-detection-v1":
            raise ValueError(f"unexpected artifact format: {artifact.get('format')}")
        if int(artifact.get("feature_version", 1)) != FEATURE_VERSION:
            raise ValueError(
                f"feature version mismatch: artifact={artifact.get('feature_version', 1)} "
                f"code={FEATURE_VERSION}"
            )
        if int(artifact.get("n_features", -1)) != len(FEATURE_NAMES):
            raise ValueError(
                f"feature count mismatch: artifact={artifact.get('n_features')} "
                f"code={len(FEATURE_NAMES)}"
            )
        idx = artifact.get("active_full_idx")
        self._active_full_idx = (
            np.asarray(idx, dtype=np.int64)
            if idx is not None
            else np.arange(2 * len(FEATURE_NAMES), dtype=np.int64)
        )
        self._artifact = artifact

    @property
    def ready(self) -> bool:
        return self._numpy_model is not None or self._artifact is not None

    @property
    def engine(self) -> str:
        return self._engine

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def score_chunks(self, chunks: List[List[dict]]) -> List[float]:
        """Score every chunk of one request. Always returns len(chunks) floats
        in [0, 1]; never raises."""
        if not chunks:
            return []
        try:
            scores = self._score_chunks_model(chunks)
        except Exception:  # noqa: BLE001
            scores = [self._heuristic_chunk(chunk) for chunk in chunks]
        if len(scores) != len(chunks):  # defensive; should be unreachable
            scores = [self._heuristic_chunk(chunk) for chunk in chunks]
        return scores

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        with self._lock:
            if self._numpy_model is not None:
                return self._numpy_model.predict_proba_1(X)
            if self._artifact is not None:
                gbms = self._artifact["gbms"]
                lr = self._artifact["lr"]
                lr_weight = float(self._artifact.get("lr_weight", 0.15))
                proba = np.mean([m.predict_proba(X)[:, 1] for m in gbms], axis=0)
                return (1.0 - lr_weight) * proba + lr_weight * lr.predict_proba(X)[:, 1]
        raise RuntimeError("no artifact loaded")

    def _view_proba(self, rows: np.ndarray) -> np.ndarray:
        if rows.shape[0] >= 3:
            relative = rows - np.median(rows, axis=0)
        else:
            relative = np.zeros_like(rows)
        X = np.hstack([rows, relative])[:, self._active_full_idx]
        return self._predict_proba(X)

    def _score_chunks_model(self, chunks: List[List[dict]]) -> List[float]:
        if not self.ready:
            return [self._heuristic_chunk(chunk) for chunk in chunks]

        ctx = compute_request_context(chunks)
        absolute = np.vstack([self._safe_features(chunk, ctx) for chunk in chunks])
        probas = [self._view_proba(absolute)]

        # TTA sub-chunk views share the request context (the calibration
        # anchor is the request, not the subsample).
        rng = np.random.default_rng(_TTA_SEED)
        for _ in range(_TTA_VIEWS):
            rows = []
            for chunk in chunks:
                if isinstance(chunk, list) and len(chunk) >= _TTA_MIN_HANDS:
                    n = len(chunk)
                    m = int(rng.integers(max(18, int(0.7 * n)), n + 1))
                    idx = rng.choice(n, size=m, replace=False)
                    sub = [chunk[j] for j in idx]
                else:
                    sub = chunk if isinstance(chunk, list) else []
                rows.append(self._safe_features(sub, ctx))
            probas.append(self._view_proba(np.vstack(rows)))
        proba = np.mean(probas, axis=0)

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
    def _safe_features(chunk: Any, ctx: Optional[RequestContext] = None) -> np.ndarray:
        try:
            if isinstance(chunk, list):
                return extract_chunk_features(chunk, ctx)
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

    def self_check(self) -> Dict[str, Any]:
        """Score a tiny synthetic request end-to-end; used at miner startup to
        prove which engine is live. Never raises."""
        try:
            hand = {
                "actions": [
                    {"action_type": "bet", "normalized_amount_bb": 4.0},
                    {"action_type": "call", "normalized_amount_bb": 4.0},
                ],
                "players": [{}, {}],
                "streets": [{}, {}],
                "outcome": {},
            }
            chunks = [[dict(hand) for _ in range(4)], [dict(hand) for _ in range(3)]]
            scores = self.score_chunks(chunks)
            return {
                "engine": self.engine,
                "ready": self.ready,
                "scores": scores,
                "load_error": self.load_error,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "engine": self.engine,
                "ready": self.ready,
                "scores": None,
                "load_error": f"self_check: {type(exc).__name__}: {exc}",
            }


_default_model: Optional[DetectionModel] = None
_default_model_lock = threading.Lock()


def get_default_model() -> DetectionModel:
    global _default_model
    with _default_model_lock:
        if _default_model is None:
            _default_model = DetectionModel()
        return _default_model
