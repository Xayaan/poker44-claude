"""Train the final Poker44 detection ensemble on ALL release dates and export
the production artifact to poker44/detection/model.pkl (+ model_v2.npz).

Recipe (v4, validated on temporal holdout 2026-07-08 — see
research/ENGINEERING_LOG.md §9):
  features : poker44.detection.features v4 (abs + batch-relative, 141 active)
  views    : domain randomization — every group trains twice: original view
             and live-mimic view (research/live_mimic.py), each with its own
             per-(date,view) request context and batch median
  aug      : 8 random sub-chunks per group per view (18..n hands)
  ensemble : HistGB {(300,d3,lr.06,15), (300,d4,lr.06,31), (500,d3,lr.04,15)}
             x seeds {0,1,2,3}  ->  mean proba, blended 0.85/0.15 with
             StandardScaler+LogisticRegression(C=0.2)
Temporal holdout (train <=06-26, test 11 later dates, per-date reward):
  original view mean 0.858, live-mimic view mean 0.841 (with serving TTA);
  baseline v3.1 recipe was 0.847 / 0.812.
"""

from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from poker44.detection.features import (  # noqa: E402
    ACTIVE_FULL_IDX,
    FEATURE_NAMES,
    FEATURE_VERSION,
    LIVE_DEGENERATE_FEATURES,
    compute_request_context,
    extract_chunk_features,
)
from live_mimic import build_mimic_records  # noqa: E402

RESEARCH = Path(__file__).resolve().parent
MODEL_PATH = REPO_ROOT / "poker44" / "detection" / "model.pkl"
META_PATH = REPO_ROOT / "poker44" / "detection" / "model_meta.json"

GBM_CONFIGS = [
    dict(max_iter=300, max_depth=3, learning_rate=0.06, l2_regularization=1.0, max_leaf_nodes=15),
    dict(max_iter=300, max_depth=4, learning_rate=0.06, l2_regularization=1.0, max_leaf_nodes=31),
    dict(max_iter=500, max_depth=3, learning_rate=0.04, l2_regularization=1.0, max_leaf_nodes=15),
]
SEEDS = (0, 1, 2, 3)
LR_WEIGHT = 0.15
N_AUG = 8
AUG_MIN = 18


def _view_rows(recs: list[dict], view: str, rng: np.random.Generator):
    """Feature rows (abs + batch-relative) for one view of the dataset:
    per-date request context + per-date batch median, originals + N_AUG
    sub-chunk augmentations per group."""
    dates = sorted({r["date"] for r in recs})
    ctx_by_date = {
        d: compute_request_context([r["hands"] for r in recs if r["date"] == d])
        for d in dates
    }
    feats = [extract_chunk_features(r["hands"], ctx_by_date[r["date"]]) for r in recs]
    A = np.vstack(feats)
    rec_dates = np.array([r["date"] for r in recs])
    date_median = {d: np.median(A[rec_dates == d], axis=0) for d in dates}

    X_rows, y_rows = [], []
    for i, r in enumerate(recs):
        X_rows.append(np.hstack([A[i], A[i] - date_median[r["date"]]]))
        y_rows.append(r["label"])
    for r in recs:
        hands = r["hands"]
        n = len(hands)
        ctx = ctx_by_date[r["date"]]
        for _ in range(N_AUG):
            m = int(rng.integers(min(AUG_MIN, n), n + 1))
            idx = rng.choice(n, size=m, replace=False)
            fa = extract_chunk_features([hands[j] for j in idx], ctx)
            X_rows.append(np.hstack([fa, fa - date_median[r["date"]]]))
            y_rows.append(r["label"])
    print(f"  view={view}: {len(X_rows)} rows")
    return X_rows, y_rows


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    with open(RESEARCH / "dataset.pkl", "rb") as fh:
        recs = pickle.load(fh)
    dates = sorted({r["date"] for r in recs})
    rng = np.random.default_rng(7)

    print(f"extracting features for {len(recs)} groups across {len(dates)} dates (2 views)...")
    X_rows, y_rows = _view_rows(recs, "original", rng)
    mimic = build_mimic_records(recs)
    xr_m, yr_m = _view_rows(mimic, "live-mimic", rng)
    X_rows += xr_m
    y_rows += yr_m

    X = np.vstack(X_rows)[:, ACTIVE_FULL_IDX]  # drop live-degenerate columns
    y = np.array(y_rows)
    print(f"training matrix: {X.shape}")

    gbms = []
    for seed in SEEDS:
        for k, cfg in enumerate(GBM_CONFIGS):
            model = HistGradientBoostingClassifier(random_state=seed * 1000 + k * 100, **cfg)
            model.fit(X, y)
            gbms.append(model)
            print(f"  fitted GBM cfg{k} seed{seed}")
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.2, max_iter=4000))
    lr.fit(X, y)
    print("  fitted LR")

    artifact = {
        "format": "poker44-detection-v1",
        "feature_version": FEATURE_VERSION,
        "n_features": len(FEATURE_NAMES),
        "active_full_idx": [int(i) for i in ACTIVE_FULL_IDX],
        "gbms": gbms,
        "lr": lr,
        "lr_weight": LR_WEIGHT,
    }
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(artifact, fh)

    meta = {
        "format": "poker44-detection-v1",
        "feature_version": FEATURE_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_dates": [str(d) for d in dates],
        "n_chunk_groups": len(recs),
        "n_training_rows": int(X.shape[0]),
        "n_features_active": int(X.shape[1]),
        "live_degenerate_masked": len(LIVE_DEGENERATE_FEATURES),
        "domain_randomization": (
            "each group trains in two views: original benchmark and the "
            "live-regime transform (research/live_mimic.py; transform targets "
            "are aggregate unlabeled capture statistics — no validator payload "
            "enters training data)"
        ),
        "validation_note": (
            "Validated by temporal holdout in research/ (not computed inline). "
            "v4 model: per-date mean ~0.86 (original view) / ~0.84 (live-mimic "
            "view) on 11 unseen benchmark dates with serving-side TTA. Live "
            "reward is lower (benchmark->live population gap); see "
            "research/ENGINEERING_LOG.md."
        ),
        "data_source": "https://api.poker44.net/api/v1/benchmark (public training benchmark)",
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    size_mb = MODEL_PATH.stat().st_size / 1e6
    print(f"saved {MODEL_PATH.name} ({size_mb:.1f} MB) + {META_PATH.name}")

    # numpy-native production artifact + parity proof (see export_v2.py)
    import export_v2

    export_v2.export()
    export_v2.verify_parity()


if __name__ == "__main__":
    main()
