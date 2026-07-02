"""Train and evaluate the Poker44 detection model under validator parity.

- Features: poker44.detection.features (absolute + batch-relative blocks)
- Labels/metric: exact validator reward() = 0.75*AP + 0.25*recall@FPR<=5%
- CV: leave-one-date-out (LODO) across all release dates
- Augmentation: random sub-chunks per group (size-invariant features)
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from poker44.detection.features import extract_chunk_features, FEATURE_NAMES  # noqa: E402
from poker44.score.scoring import reward  # noqa: E402

RESEARCH = Path(__file__).resolve().parent
RNG = np.random.default_rng(7)

N_AUG = 8
AUG_MIN = 18


def load_features(rebuild: bool = False):
    cache = RESEARCH / "features_cache.pkl"
    if cache.exists() and not rebuild:
        with open(cache, "rb") as fh:
            return pickle.load(fh)

    with open(RESEARCH / "dataset.pkl", "rb") as fh:
        recs = pickle.load(fh)

    dates = sorted({r["date"] for r in recs})
    X_abs, y, rec_dates = [], [], []
    for r in recs:
        X_abs.append(extract_chunk_features(r["hands"]))
        y.append(r["label"])
        rec_dates.append(r["date"])
    X_abs = np.vstack(X_abs)
    y = np.array(y)
    rec_dates = np.array(rec_dates)

    # Batch-relative block: anchor = median over all chunks of the same date
    # (mirrors one live request holding the whole eval window).
    date_median = {}
    X_rel = np.zeros_like(X_abs)
    for d in dates:
        m = rec_dates == d
        date_median[d] = np.median(X_abs[m], axis=0)
        X_rel[m] = X_abs[m] - date_median[d]

    # Augmented sub-chunks (train-time only).
    Xa_abs, ya, aug_dates = [], [], []
    for r in recs:
        hands = r["hands"]
        n = len(hands)
        for _ in range(N_AUG):
            m = int(RNG.integers(AUG_MIN, n + 1))
            idx = RNG.choice(n, size=m, replace=False)
            sub = [hands[i] for i in idx]
            Xa_abs.append(extract_chunk_features(sub))
            ya.append(r["label"])
            aug_dates.append(r["date"])
    Xa_abs = np.vstack(Xa_abs)
    ya = np.array(ya)
    aug_dates = np.array(aug_dates)
    Xa_rel = np.vstack([Xa_abs[i] - date_median[d] for i, d in enumerate(aug_dates)])

    out = {
        "X_abs": X_abs, "X_rel": X_rel, "y": y, "dates": rec_dates,
        "Xa_abs": Xa_abs, "Xa_rel": Xa_rel, "ya": ya, "aug_dates": aug_dates,
        "all_dates": dates,
    }
    with open(cache, "wb") as fh:
        pickle.dump(out, fh)
    return out


def evaluate_lodo(data, model_fn, use_rel=True, use_aug=True, seeds=(0,)):
    """Leave-one-date-out CV. Returns per-date rewards and pooled predictions."""
    X = np.hstack([data["X_abs"], data["X_rel"]]) if use_rel else data["X_abs"]
    Xa = np.hstack([data["Xa_abs"], data["Xa_rel"]]) if use_rel else data["Xa_abs"]
    y, dates = data["y"], data["dates"]
    ya, aug_dates = data["ya"], data["aug_dates"]

    per_date_rewards, per_date_ap = [], []
    pooled_pred = np.zeros(len(y))
    for d in data["all_dates"]:
        te = dates == d
        tr = ~te
        X_tr, y_tr = X[tr], y[tr]
        if use_aug:
            m_aug = aug_dates != d
            X_tr = np.vstack([X_tr, Xa[m_aug]])
            y_tr = np.concatenate([y_tr, ya[m_aug]])
        preds = np.zeros(te.sum())
        for seed in seeds:
            model = model_fn(seed)
            model.fit(X_tr, y_tr)
            preds += model.predict_proba(X[te])[:, 1]
        preds /= len(seeds)
        pooled_pred[te] = preds
        rew, res = reward(preds, y[te])
        per_date_rewards.append(rew)
        per_date_ap.append(res["ap_score"])

    pooled_rew, pooled_res = reward(pooled_pred, y)
    return {
        "per_date_rewards": np.array(per_date_rewards),
        "per_date_ap": np.array(per_date_ap),
        "pooled_reward": pooled_rew,
        "pooled_ap": pooled_res["ap_score"],
        "pooled_recall": pooled_res["bot_recall"],
        "pooled_pred": pooled_pred,
    }


def summarize(name: str, res: dict) -> None:
    r = res["per_date_rewards"]
    print(
        f"{name:34s} | date-reward min={r.min():.3f} p25={np.percentile(r, 25):.3f} "
        f"med={np.median(r):.3f} mean={r.mean():.3f} | >=0.65: {(r >= 0.65).sum()}/38 "
        f">=0.70: {(r >= 0.70).sum()}/38 | pooled={res['pooled_reward']:.3f} "
        f"(AP={res['pooled_ap']:.3f} R@5={res['pooled_recall']:.3f})"
    )


if __name__ == "__main__":
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rebuild = "--rebuild" in sys.argv
    data = load_features(rebuild=rebuild)
    print(f"features: {data['X_abs'].shape} | augmented: {data['Xa_abs'].shape}")

    def gbm(seed):
        return HistGradientBoostingClassifier(
            max_iter=300, max_depth=3, learning_rate=0.06,
            l2_regularization=1.0, max_leaf_nodes=15, random_state=seed,
        )

    def lr(seed):
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.2, max_iter=4000, random_state=seed),
        )

    summarize("GBM abs (no aug)", evaluate_lodo(data, gbm, use_rel=False, use_aug=False))
    summarize("GBM abs+rel (no aug)", evaluate_lodo(data, gbm, use_rel=True, use_aug=False))
    summarize("GBM abs+rel + aug", evaluate_lodo(data, gbm, use_rel=True, use_aug=True))
    summarize("GBM abs+rel + aug x3 seeds", evaluate_lodo(data, gbm, seeds=(0, 1, 2)))
    summarize("LR  abs+rel + aug", evaluate_lodo(data, lr, use_rel=True, use_aug=True))
