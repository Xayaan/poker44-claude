#!/usr/bin/env bash
# Daily retrain with a safety gate. Intended for cron on the deploy host,
# shortly after the benchmark release drops (00:05 UTC):
#   5 1 * * * cd /root/poker44-claude && bash scripts/miner/retrain_daily.sh >> retrain.log 2>&1
#
# Flow: download new releases -> rebuild dataset -> GATE (train without the
# newest date, require reward >= GATE_MIN on it) -> full retrain + numpy
# export (parity-checked) -> verify artifact loads -> restart miner.
# On any failure the previous artifact keeps serving.
set -euo pipefail

cd "$(dirname "$0")/../.."
PY="${PY:-./miner_env/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
PM2_NAME="${PM2_NAME:-poker44_miner}"
GATE_MIN="${GATE_MIN:-0.60}"

echo "=== $(date -u +%FT%TZ) retrain_daily start (py=$PY) ==="

"$PY" research/download_benchmark.py
"$PY" research/build_dataset.py

# --- gate: newest-date out-of-sample check --------------------------------
"$PY" - <<PYEOF
import pickle, sys
import numpy as np
sys.path.insert(0, ".")
from poker44.detection.features import ACTIVE_FULL_IDX, compute_request_context, extract_chunk_features
from poker44.score.scoring import reward
from sklearn.ensemble import HistGradientBoostingClassifier

recs = pickle.load(open("research/dataset.pkl", "rb"))
dates = sorted({r["date"] for r in recs})
newest = dates[-1]
ctx = {d: compute_request_context([r["hands"] for r in recs if r["date"] == d]) for d in dates}
feats = {id(r): extract_chunk_features(r["hands"], ctx[r["date"]]) for r in recs}
A = {d: np.vstack([feats[id(r)] for r in recs if r["date"] == d]) for d in dates}
med = {d: np.median(A[d], axis=0) for d in dates}
Xtr, ytr = [], []
for r in recs:
    if r["date"] == newest:
        continue
    f = feats[id(r)]
    Xtr.append(np.hstack([f, f - med[r["date"]]]))
    ytr.append(r["label"])
X = np.vstack(Xtr)[:, ACTIVE_FULL_IDX]
m = HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.06,
                                   l2_regularization=1.0, max_leaf_nodes=31, random_state=0)
m.fit(X, np.array(ytr))
trecs = [r for r in recs if r["date"] == newest]
F = np.vstack([feats[id(r)] for r in trecs])
Xt = np.hstack([F, F - np.median(F, axis=0)])[:, ACTIVE_FULL_IDX]
rew, _ = reward(m.predict_proba(Xt)[:, 1], np.array([r["label"] for r in trecs]))
print(f"GATE newest={newest} oos_reward={rew:.3f} min=${GATE_MIN}")
sys.exit(0 if rew >= float("${GATE_MIN}") else 1)
PYEOF
echo "gate passed"

# --- full retrain + numpy export (includes sklearn/numpy parity assert) ---
"$PY" research/train_final.py

# --- verify the artifact the miner will load ------------------------------
"$PY" - <<'PYEOF'
import sys
sys.path.insert(0, ".")
from poker44.detection.model import DetectionModel
m = DetectionModel()
assert m.engine == "numpy-v2" and m.ready, (m.engine, m.load_error)
check = m.self_check()
assert check["scores"] is not None
print("artifact ok:", check)
PYEOF

pm2 restart "$PM2_NAME" && pm2 save
sleep 8
grep -hE "Detector self-check" ~/.pm2/logs/*out*.log 2>/dev/null | tail -1 || true
echo "=== $(date -u +%FT%TZ) retrain_daily done ==="
