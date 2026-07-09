#!/usr/bin/env bash
# Nightly retrain with a safety gate and manifest consistency guarantee.
# Cron (deploy host), shortly after the benchmark release drops (00:05 UTC):
#   5 1 * * * cd /root/poker44-claude && bash scripts/miner/retrain_daily.sh >> retrain.log 2>&1
#
# Consistency invariant: the miner only ever serves artifacts that exist at a
# PUBLISHED commit of the manifest repo. Flow:
#   sync to origin/main (auto-deploys code updates)
#   -> download new releases -> rebuild dataset
#   -> GATE: train without the newest date, require reward >= GATE_MIN on it
#   -> full retrain + numpy export (sklearn/numpy parity asserted)
#   -> refresh MODEL_MANIFEST.json artifact hash
#   -> commit artifacts and PUSH; only then restart the miner
# If the gate, training, or push fails: artifacts revert to HEAD and the
# previous (consistent) model keeps serving — BUT a code update pulled by the
# origin sync still restarts the miner (2026-07-09 incident: the publish step
# failed on missing git credentials night after night, the failure path never
# restarted PM2, and a fully-verified new model sat undeployed on disk while
# the old in-memory process kept serving four rounds). After revert_artifacts
# the tree equals a PUBLISHED commit, so restarting is always
# consistency-safe.
set -euo pipefail

cd "$(dirname "$0")/../.."
PY="${PY:-./miner_env/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
PM2_NAME="${PM2_NAME:-poker44_miner}"
GATE_MIN="${GATE_MIN:-0.60}"

echo "=== $(date -u +%FT%TZ) retrain_daily start (py=$PY) ==="

git fetch origin
PREV_HEAD="$(git rev-parse HEAD)"
git reset --hard origin/main
BASE_HEAD="$(git rev-parse HEAD)"
echo "base commit: $(git rev-parse --short HEAD)"

restart_miner() {
    pm2 restart "$PM2_NAME" && pm2 save
    sleep 8
    grep -hE "Detector self-check|Manifest summary" ~/.pm2/logs/*out*.log 2>/dev/null | tail -2 || true
}

# Failure exit that still lands a pending code deploy (tree is at a published
# commit whenever this is called).
bail() {
    echo "$1; previous model keeps serving"
    if [ "$PREV_HEAD" != "$BASE_HEAD" ]; then
        echo "code updated this run (${PREV_HEAD:0:7} -> ${BASE_HEAD:0:7}); restarting miner onto published state"
        restart_miner
    fi
    exit 1
}

"$PY" research/download_benchmark.py
"$PY" research/build_dataset.py

# --- gate: newest-date out-of-sample check --------------------------------
if ! "$PY" - <<PYEOF
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
then
    bail "GATE FAILED"
fi
echo "gate passed"

revert_artifacts() {
    echo "reverting artifacts to committed state ($(git rev-parse --short HEAD))"
    git checkout -- poker44/detection/ MODEL_MANIFEST.json 2>/dev/null || true
}

# --- full retrain + numpy export (includes sklearn/numpy parity assert) ---
if ! "$PY" research/train_final.py; then
    revert_artifacts
    bail "TRAIN FAILED"
fi

# --- refresh manifest artifact hash + verify the artifact loads -----------
if ! "$PY" - <<'PYEOF'
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, ".")
mf = json.loads(Path("MODEL_MANIFEST.json").read_text())
mf["artifact_sha256"] = hashlib.sha256(Path("poker44/detection/model_v2.npz").read_bytes()).hexdigest()
Path("MODEL_MANIFEST.json").write_text(json.dumps(mf, indent=2, sort_keys=True) + "\n")
from poker44.detection.model import DetectionModel
m = DetectionModel()
assert m.engine == "numpy-v2" and m.ready, (m.engine, m.load_error)
check = m.self_check()
assert check["scores"] is not None
print("artifact ok:", check)
PYEOF
then
    revert_artifacts
    bail "MANIFEST REFRESH FAILED"
fi

# --- publish: served model must equal a public commit ----------------------
git add poker44/detection/model.pkl poker44/detection/model_v2.npz \
        poker44/detection/model_meta.json MODEL_MANIFEST.json
if git diff --cached --quiet; then
    echo "no artifact changes; nothing to publish"
else
    git -c user.name="poker44-miner" -c user.email="miner@leadpoet" \
        commit -q -m "Nightly retrain: $(date -u +%F) benchmark releases" \
        || { revert_artifacts; bail "COMMIT FAILED"; }
    if ! git push --no-verify origin main; then
        echo "push rejected; re-syncing and retrying once"
        NEW_HEAD="$(git rev-parse HEAD)"
        git fetch origin && git reset --hard origin/main
        BASE_HEAD="$(git rev-parse HEAD)"
        git -c user.name="poker44-miner" -c user.email="miner@leadpoet" \
            cherry-pick "$NEW_HEAD" \
            || { git cherry-pick --abort || true; revert_artifacts; bail "PUSH FAILED"; }
        git push --no-verify origin main \
            || { git reset --hard origin/main; bail "PUSH FAILED"; }
    fi
    echo "published commit: $(git rev-parse --short HEAD)"
fi

restart_miner
echo "=== $(date -u +%FT%TZ) retrain_daily done ==="
