"""Export the sklearn ensemble (model.pkl) to the numpy-native v2 artifact
(model_v2.npz) and verify bit-level parity on the full benchmark dataset.

The v2 artifact contains only flat float64/int64/bool arrays (allow_pickle
False end to end), so production inference needs numpy alone — no
scikit-learn, no pickle, no version coupling.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from poker44.detection.features import (  # noqa: E402
    FEATURE_NAMES,
    FEATURE_VERSION,
    extract_features_matrix,
)

PKL_PATH = REPO_ROOT / "poker44" / "detection" / "model.pkl"
NPZ_PATH = REPO_ROOT / "poker44" / "detection" / "model_v2.npz"
DATASET = Path(__file__).resolve().parent / "dataset.pkl"


def export(pkl_path: Path = PKL_PATH, npz_path: Path = NPZ_PATH) -> None:
    with open(pkl_path, "rb") as fh:
        artifact = pickle.load(fh)
    assert artifact["format"] == "poker44-detection-v1", artifact["format"]
    gbms = artifact["gbms"]
    lr_pipeline = artifact["lr"]
    lr_weight = float(artifact.get("lr_weight", 0.15))

    artifact_fv = int(artifact.get("feature_version", 1))
    if artifact_fv != FEATURE_VERSION:
        raise ValueError(
            f"model.pkl feature_version={artifact_fv} does not match "
            f"features.FEATURE_VERSION={FEATURE_VERSION}; retrain first"
        )
    active_idx = np.asarray(
        artifact.get("active_full_idx", list(range(2 * len(FEATURE_NAMES)))),
        dtype=np.int64,
    )
    payload: dict[str, np.ndarray] = {
        "format": np.array(["poker44-detection-v2"]),
        "feature_version": np.array([FEATURE_VERSION], dtype=np.int64),
        "active_full_idx": active_idx,
        "n_gbms": np.array([len(gbms)], dtype=np.int64),
        "n_features": np.array([int(active_idx.size)], dtype=np.int64),
        "lr_weight": np.array([lr_weight], dtype=np.float64),
    }

    for g, clf in enumerate(gbms):
        feature_idx, threshold, left, right = [], [], [], []
        is_leaf, missing_left, value = [], [], []
        tree_starts = [0]
        for iteration in clf._predictors:
            assert len(iteration) == 1, "binary classifier expected"
            nodes = iteration[0].nodes
            if nodes["is_categorical"].any():
                raise ValueError("categorical splits are not supported")
            feature_idx.append(nodes["feature_idx"].astype(np.int64))
            threshold.append(nodes["num_threshold"].astype(np.float64))
            left.append(nodes["left"].astype(np.int64))
            right.append(nodes["right"].astype(np.int64))
            is_leaf.append(nodes["is_leaf"].astype(bool))
            missing_left.append(nodes["missing_go_to_left"].astype(bool))
            value.append(nodes["value"].astype(np.float64))
            tree_starts.append(tree_starts[-1] + len(nodes))
        payload[f"g{g}_tree_starts"] = np.array(tree_starts, dtype=np.int64)
        payload[f"g{g}_feature_idx"] = np.concatenate(feature_idx)
        payload[f"g{g}_threshold"] = np.concatenate(threshold)
        payload[f"g{g}_left"] = np.concatenate(left)
        payload[f"g{g}_right"] = np.concatenate(right)
        payload[f"g{g}_is_leaf"] = np.concatenate(is_leaf)
        payload[f"g{g}_missing_left"] = np.concatenate(missing_left)
        payload[f"g{g}_value"] = np.concatenate(value)
        payload[f"g{g}_baseline"] = np.array(
            [float(np.ravel(clf._baseline_prediction)[0])], dtype=np.float64
        )

    scaler = lr_pipeline.named_steps["standardscaler"]
    logreg = lr_pipeline.named_steps["logisticregression"]
    payload["lr_mean"] = scaler.mean_.astype(np.float64)
    payload["lr_scale"] = scaler.scale_.astype(np.float64)
    payload["lr_coef"] = logreg.coef_.astype(np.float64)
    payload["lr_intercept"] = np.array(
        [float(np.ravel(logreg.intercept_)[0])], dtype=np.float64
    )

    np.savez_compressed(npz_path, **payload)
    size_mb = npz_path.stat().st_size / 1e6
    n_nodes = sum(int(payload[f"g{g}_tree_starts"][-1]) for g in range(len(gbms)))
    print(f"exported {npz_path.name} ({size_mb:.1f} MB, {len(gbms)} gbms, {n_nodes} nodes)")


def verify_parity() -> None:
    """sklearn proba vs numpy-engine proba on every benchmark chunk group,
    plus a round-trip through the production DetectionModel path."""
    from poker44.detection.model import DetectionModel, _NumpyEnsemble

    with open(PKL_PATH, "rb") as fh:
        artifact = pickle.load(fh)
    gbms, lr = artifact["gbms"], artifact["lr"]
    lr_weight = float(artifact.get("lr_weight", 0.15))

    with np.load(NPZ_PATH, allow_pickle=False) as npz:
        numpy_model = _NumpyEnsemble(npz)

    active_idx = np.asarray(
        artifact.get("active_full_idx", list(range(2 * len(FEATURE_NAMES)))),
        dtype=np.int64,
    )
    records = pickle.load(open(DATASET, "rb"))
    dates = sorted({r["date"] for r in records})
    max_diff = 0.0
    for date in dates:
        recs = [r for r in records if r["date"] == date]
        X = extract_features_matrix([r["hands"] for r in recs])[:, active_idx]
        p_sk = np.mean([m.predict_proba(X)[:, 1] for m in gbms], axis=0)
        p_sk = (1.0 - lr_weight) * p_sk + lr_weight * lr.predict_proba(X)[:, 1]
        p_np = numpy_model.predict_proba_1(X)
        max_diff = max(max_diff, float(np.max(np.abs(p_sk - p_np))))
    print(f"proba parity across {len(dates)} dates: max|sklearn - numpy| = {max_diff:.3e}")
    assert max_diff < 1e-9, "parity failure"

    # production-path round trip on the newest date
    newest = dates[-1]
    recs = [r for r in records if r["date"] == newest]
    chunks = [r["hands"] for r in recs]
    model = DetectionModel()
    assert model.engine == "numpy-v2", model.engine
    s_np = model.score_chunks(chunks)
    pkl_model = DetectionModel(model_path=PKL_PATH)
    assert pkl_model.engine == "sklearn-pickle-v1", pkl_model.engine
    s_sk = pkl_model.score_chunks(chunks)
    assert s_np == s_sk, "score_chunks mismatch between engines"
    print(f"score_chunks round-trip identical on {newest} ({len(chunks)} chunks) ✓")


if __name__ == "__main__":
    export()
    verify_parity()
