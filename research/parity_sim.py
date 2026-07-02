"""Full validator-parity simulation of the forward cycle.

Replays poker44/validator/forward.py exactly, per release date:
  1. provider batches  = benchmark chunk groups (raw hands as served)
  2. miner payloads    = prepare_hand_for_miner(hand) per hand  (forward.py:121)
  3. wire round-trip   = json.dumps/loads of chunks (DetectionSynapse body)
  4. miner scoring     = DetectionModel.score_chunks (neurons/miner.py forward)
  5. reward            = poker44.score.scoring.reward over the request window
                         (window == chunk count, matching _compute_windowed_rewards)
  6. winner selection  = argmax reward (winner-take-all _select_weight_targets)

Modes:
  --production : score with the shipped artifact (in-sample on train dates)
  --holdout    : retrain on dates <= 2026-06-25 and score the last 7 dates
                 (honest forward-looking estimate; default)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from poker44.detection.features import extract_chunk_features  # noqa: E402
from poker44.detection.model import DetectionModel, MODEL_PATH  # noqa: E402
from poker44.validator.payload_view import prepare_hand_for_miner  # noqa: E402
from poker44.score.scoring import reward  # noqa: E402

RESEARCH = Path(__file__).resolve().parent
DATA_DIR = RESEARCH / "data"
HOLDOUT_SPLIT_DATE = "2026-06-25"
HOLDOUT_MODEL_PATH = RESEARCH / "model_holdout.pkl"


def load_provider_batches(date: str):
    """Raw benchmark groups for one date, as the eval provider would serve."""
    payload = json.loads((DATA_DIR / f"{date}.json").read_text())
    batches, labels = [], []
    for chunk_payload in payload["chunks"]:
        for group, label in zip(chunk_payload["chunks"], chunk_payload["groundTruth"]):
            batches.append(group)
            labels.append(int(label))
    return batches, np.array(labels)


def validator_prepares(batches):
    """forward.py: every hand through the canonicalizer, then the synapse body."""
    chunks = [[prepare_hand_for_miner(hand) for hand in batch] for batch in batches]
    # DetectionSynapse serializes chunks as JSON in the request body.
    return json.loads(json.dumps(chunks))


def train_holdout_model() -> DetectionModel:
    """Train the exact production recipe on dates <= HOLDOUT_SPLIT_DATE."""
    import pickle

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if HOLDOUT_MODEL_PATH.exists():
        return DetectionModel(model_path=HOLDOUT_MODEL_PATH)

    with open(RESEARCH / "dataset.pkl", "rb") as fh:
        recs = pickle.load(fh)
    recs_train = [r for r in recs if r["date"] <= HOLDOUT_SPLIT_DATE]
    dates = sorted({r["date"] for r in recs_train})
    rng = np.random.default_rng(7)

    feats = [extract_chunk_features(r["hands"]) for r in recs_train]
    A = np.vstack(feats)
    rec_dates = np.array([r["date"] for r in recs_train])
    med = {d: np.median(A[rec_dates == d], axis=0) for d in dates}

    X_rows, y_rows = [], []
    for i, r in enumerate(recs_train):
        X_rows.append(np.hstack([A[i], A[i] - med[r["date"]]]))
        y_rows.append(r["label"])
    for r in recs_train:
        hands = r["hands"]
        n = len(hands)
        for _ in range(8):
            m = int(rng.integers(18, n + 1))
            idx = rng.choice(n, size=m, replace=False)
            fa = extract_chunk_features([hands[j] for j in idx])
            X_rows.append(np.hstack([fa, fa - med[r["date"]]]))
            y_rows.append(r["label"])
    X = np.vstack(X_rows)
    y = np.array(y_rows)

    cfgs = [
        dict(max_iter=300, max_depth=3, learning_rate=0.06, l2_regularization=1.0, max_leaf_nodes=15),
        dict(max_iter=300, max_depth=4, learning_rate=0.06, l2_regularization=1.0, max_leaf_nodes=31),
        dict(max_iter=500, max_depth=3, learning_rate=0.04, l2_regularization=1.0, max_leaf_nodes=15),
    ]
    gbms = []
    for seed in (0, 1):
        for k, cfg in enumerate(cfgs):
            model = HistGradientBoostingClassifier(random_state=seed * 1000 + k * 100, **cfg)
            model.fit(X, y)
            gbms.append(model)
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.2, max_iter=4000))
    lr.fit(X, y)
    artifact = {
        "format": "poker44-detection-v1",
        "feature_version": 1,
        "n_features": A.shape[1],
        "gbms": gbms,
        "lr": lr,
        "lr_weight": 0.15,
    }
    with open(HOLDOUT_MODEL_PATH, "wb") as fh:
        pickle.dump(artifact, fh)
    return DetectionModel(model_path=HOLDOUT_MODEL_PATH)


def reference_heuristic_scores(chunks):
    """The previously-shipped reference miner, for the competitor baseline."""
    from collections import Counter

    def clamp(v):
        return max(0.0, min(1.0, v))

    def score_hand(hand):
        actions = hand.get("actions") or []
        players = hand.get("players") or []
        streets = hand.get("streets") or []
        outcome = hand.get("outcome") or {}
        counts = Counter(a.get("action_type") for a in actions)
        meaningful = max(1, sum(counts.get(k, 0) for k in ("call", "check", "bet", "raise", "fold")))
        s = 0.32 * (len(streets) / 3.0)
        s += 0.22 * (1.0 if outcome.get("showdown") else 0.0)
        s += 0.18 * clamp(counts.get("call", 0) / meaningful / 0.35)
        s += 0.12 * clamp(counts.get("check", 0) / meaningful / 0.30)
        s += 0.08 * clamp((6 - min(len(players), 6)) / 4.0 if players else 0.0)
        s -= 0.18 * clamp(counts.get("fold", 0) / meaningful / 0.55)
        s -= 0.10 * clamp(counts.get("raise", 0) / meaningful / 0.20)
        return clamp(s)

    return [
        sum(score_hand(h) for h in c) / len(c) if c else 0.5
        for c in chunks
    ]


def main() -> None:
    mode = "production" if "--production" in sys.argv else "holdout"
    all_dates = sorted(p.stem for p in DATA_DIR.glob("*.json"))
    if mode == "holdout":
        model = train_holdout_model()
        eval_dates = [d for d in all_dates if d > HOLDOUT_SPLIT_DATE]
        print(f"holdout model (trained <= {HOLDOUT_SPLIT_DATE}), evaluating {len(eval_dates)} unseen dates")
    else:
        model = DetectionModel(model_path=MODEL_PATH)
        eval_dates = all_dates
        print(f"production artifact, evaluating all {len(eval_dates)} dates (in-sample)")
    assert model.ready, model.load_error

    print(f"\n{'date':12s} {'chunks':>6s} {'OURS':>7s} {'AP':>7s} {'R@5':>7s} {'FPR':>6s} | {'REF':>6s} {'win?':>5s}")
    our_wins = 0
    pooled_scores, pooled_ref, pooled_labels = [], [], []
    per_date_rewards = []
    for date in eval_dates:
        batches, labels = load_provider_batches(date)
        chunks = validator_prepares(batches)
        scores = model.score_chunks(chunks)
        assert len(scores) == len(chunks), "contract violation"
        rew, res = reward(np.array(scores), labels)
        ref = reference_heuristic_scores(chunks)
        ref_rew, _ = reward(np.array(ref), labels)
        win = rew > ref_rew
        our_wins += int(win)
        per_date_rewards.append(rew)
        pooled_scores.extend(scores)
        pooled_ref.extend(ref)
        pooled_labels.extend(labels.tolist())
        print(
            f"{date:12s} {len(chunks):6d} {rew:7.3f} {res['ap_score']:7.3f} "
            f"{res['bot_recall']:7.3f} {res['fpr']:6.3f} | {ref_rew:6.3f} {'YES' if win else 'no':>5s}"
        )

    pooled_rew, pooled_res = reward(np.array(pooled_scores), np.array(pooled_labels))
    ref_rew, _ = reward(np.array(pooled_ref), np.array(pooled_labels))
    r = np.array(per_date_rewards)
    print(
        f"\nPOOLED: ours={pooled_rew:.3f} (AP={pooled_res['ap_score']:.3f} "
        f"R@5={pooled_res['bot_recall']:.3f}) vs reference={ref_rew:.3f}"
    )
    print(
        f"per-date reward: min={r.min():.3f} med={np.median(r):.3f} mean={r.mean():.3f} "
        f"| beat reference on {our_wins}/{len(eval_dates)} dates"
    )
    print("dashboard context: live leader=0.599, best-ever close=0.643, last close=0.553")


if __name__ == "__main__":
    main()
