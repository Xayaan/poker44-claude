# Poker44 Miner — Engineering Log & Runbook

Continuation handoff for the subnet-126 (Poker44 bot-detection) miner in this
repo. Read this end-to-end before changing anything: it records what the code
does, **why each piece exists**, how to operate it, and how every change was
verified. A companion private operator playbook (VPS access, wallet mapping,
forward strategy) lives in the maintainer's session memory, not here.

Status as of commit `08958e6` (2026-07-06). **We are not winning yet** — see
§1. This log is honest about that on purpose; a future maintainer who believes
the code is "winning" will make bad calls.

---

## 1. Current state (the honest scoreboard)

- Deployed: UID 48 on netuid 126, axon `157.180.87.74:8091`, model
  `poker44-seqcollision-ensemble 3.1.0`, engine `numpy-v2`, feature version 3.
- Live round scores (per-chunk reward = 0.75·AP + 0.25·recall@FPR≤5%):
  - Cycle 1: R2 **0.463**, R3 **0.478**, R4 **0.525** (R1 0.000 — nothing
    was serving yet).
  - Cycle 2: R1 **0.475**.
  - Leader band across these rounds: **0.58–0.63**.
- Benchmark temporal holdout (v3.1, trained ≤06-26, tested on later unseen
  dates): pooled **0.798**, per-date min ~0.70.
- **The gap is real and unsolved.** It closed from −0.17 (R2) to ~−0.06 (R4)
  across five fixes, then cycle-2 R1 came in at 0.475 — one round can't tell
  "regression" from "hard window" given the round-to-round live variance we
  see. The remaining gap is **benchmark→live domain transfer** (§4, cause 5;
  forward plan in the private playbook), not any code defect we know of.

The reward is **winner-take-all on the 5-round cycle composite** (mean of
evaluated rounds); only rank #1 at cycle close is paid. So the operational
goal is: best transferring model × zero missed rounds × correct compliance,
sustained into a clean cycle.

---

## 2. How the competition works (extracted from validator code)

Source: `poker44/validator/forward.py`, `poker44/score/scoring.py`,
`poker44/base/miner.py`, `poker44/validator/payload_view.py`.

- **Reward** (`scoring.reward`): `0.75 * average_precision + 0.25 *
  recall_at_fpr<=0.05`, computed over per-chunk bot scores (1=bot). It is a
  **ranking** metric — absolute score calibration is irrelevant, only the
  ordering and the low-FPR head matter. `AP` dominates; the recall term
  rewards a clean high-precision top slice.
- **One request = whole eval window.** The validator sends every chunk of the
  current window in a single `DetectionSynapse` (`synapse.chunks`). We exploit
  this: the request itself is our calibration context and unsupervised drift
  anchor (§3).
- **Reward buffering** (`_compute_windowed_rewards`, forward.py:176–259): only
  *returned* predictions enter the buffer; a missed/timed-out/blacklisted
  query records coverage 0 but does **not** poison the score with zeros. So a
  low round score means our served predictions ranked poorly — not that we
  missed queries. (Missed queries hurt only by reducing sample count/coverage.)
- **Winner selection** (`_select_weight_targets`): pure `max(reward)` over
  UIDs, winner-take-all, optional burn to UID 0. No tie-breaking on
  compliance. Compliance/manifest does **not** enter the reward math (verified)
  — but the platform rules warn that an unverifiable manifest on a high-scoring
  model can be penalized/zeroed out of band, so treat it as a hard gate.
- **Candidate selection** (`_get_candidate_miners`): excludes UID 0, excludes
  high-stake validator-permit holders, requires a served axon ip/port, rotates
  `POKER44_MINERS_PER_CYCLE` (=16) miners per cycle. `active`/`last_update`
  metagraph flags do **not** gate queries (they look stale for all miners
  because miners never set weights — ignore the dashboard "Active: No").
- **Blacklist** (`common_blacklist`): `force_validator_permit` defaults **True**
  → only permitted validators can query us. Consequence: **we cannot self-probe
  our own axon** with our miner hotkey (no permit) — validation is done locally
  and via the diagnostic capture, never by hitting the live axon ourselves.
- **Payload projection**: the validator canonicalizes every hand through
  `prepare_hand_for_miner` (payload_view.py) before sending. Training data is
  projected through the **exact same** transform (`research/build_dataset.py`).
- **Benchmark vs live are different populations.** The public benchmark
  (`api.poker44.net/api/v1/benchmark`) is for training/regression; the docs
  state explicitly it is "not the live validator evaluation stream and should
  not be treated as a fixed production pattern." The captures in §4 proved this
  literally — it is the central difficulty of the whole project.

---

## 3. Architecture of the served model

**Inference chain** (`poker44/detection/model.py`, `DetectionModel`):
`model_v2.npz` (numpy-native) → `model.pkl` (sklearn) → deterministic
heuristic. The miner never drops a response. Startup logs the live engine
(`Detector self-check | engine=numpy-v2 ...`) and every request logs
`engine=... scores[min/mean/max/mid]` where `mid` = fraction in [0.4,0.6]
(the "dead zone" that flags a domain-saturated model).

- **`model_v2.npz`** — the production artifact. A flat-array export of the
  ensemble (`_NumpyEnsemble`): per-tree node arrays + a standardized logistic
  head, evaluated with pure numpy. **No scikit-learn, no pickle at inference.**
  This exists because the deploy host runs sklearn 1.7.2 while training uses
  1.9.0, and unpickling across versions silently produces garbage or fails
  (§4, cause 1). `np.load(..., allow_pickle=False)`.
- **Guards**: the loader hard-checks `feature_version` and `active_full_idx`
  size against the code. A stale artifact can **never** silently score
  mismatched features — it errors and falls back rather than mis-scoring.
- **Parity**: `research/export_v2.py` asserts sklearn-proba vs numpy-proba
  agree to <1e-9 on every benchmark date (observed 1.1e-16) and that a
  scikit-learn-free venv produces byte-identical scores. This runs inside
  `train_final.py` every retrain.

**Features** (`poker44/detection/features.py`, `FEATURE_VERSION = 3`, 139
names, 17 masked → 122 active absolute, 244 active with the relative block):

- **Scale-free by construction (v2).** Every monetary feature is normalized by
  a `RequestContext` computed from the whole request: bet sizes in units of the
  request's median bet, histograms at request quantiles reported as *ECDF
  anomaly vs the request's own pooled distribution*, pots/stacks unit-normed.
  This is why the extractor is invariant to the data source's money scale
  (benchmark ~37bb median vs live ~1bb). Proven: rescale-and-grid-snap
  transform moves features by median 0.000.
- **Pool-anchored regularity block (v3).** 33 features that measure how
  *deterministic / self-consistent* a player is, since real and synthetic bots
  share regularity even when they don't share exact-sequence repetition:
  action-type×street rate anomalies, conditional-action entropy offsets,
  √n-scaled within-chunk behavioral drift (bots don't tilt), bet-sizing modal
  concentration. **Every one is an offset from the request-pool baseline** —
  raw levels are domain-dependent and would recreate the saturation trap.
- **Live-degeneracy mask (v3.1).** 17 features are variance-collapsed on live
  traffic (measured on unlabeled captures only — no label leak): unit-pinned
  amount quantiles, the uniform-stack block, near-absent aggressive bigrams,
  the single-action artifact. They stay in the extractor for audit but are
  excluded from model input via `LIVE_DEGENERATE_FEATURES` →
  `ACTIVE_FULL_IDX`, enforced through the artifact like the version guard. This
  stops the model spending capacity on signals that are constant in production.
- **Batch-relative block.** `extract_features_matrix` appends each chunk's
  offset from the per-request median (the whole window arrives at once, so this
  is a label-free drift anchor).

**Ensemble**: 3 HistGradientBoosting configs × 2 seeds → mean proba, blended
0.85/0.15 with StandardScaler+LogisticRegression(C=0.2). Recipe in
`train_final.py`; selected originally by leave-one-date-out CV.

---

## 4. The journey: five root causes, in order

Each was found by evidence, fixed at the root, verified, then shipped. This is
the most important section — it stops a future maintainer re-treading it.

1. **Fallback was serving, not the model** (commit `c563496`). Live R2 0.463 ≈
   the miner's heuristic on that day's release, while the ensemble scored 0.703
   OOS on the same data. Cause: the v1 `model.pkl` was a scikit-learn pickle;
   the VPS had sklearn 1.7.2 vs 1.9.0 at train, so unpickle threw
   `InconsistentVersionWarning` and degraded silently. Fix: the numpy-native
   `model_v2.npz` (§3) — inference no longer depends on any library version.
   Added startup self-check + per-request engine logging so this failure class
   is impossible to miss again.

2. **Manifest was unverifiable** (commit `8072cce`). `implementation_sha256`
   hashed VPS-absolute paths, so no one could reproduce it from a checkout. Fix:
   hash repo-relative POSIX paths; add `artifact_url` + `artifact_sha256` for
   `model_v2.npz`; add `MODEL_MANIFEST.json` at repo root with exact
   verification recipes.

3. **Scale saturation** (commit `fec3449`). Captured live payloads showed bets
   at ~1.02bb median (62 distinct values) vs benchmark 36.8bb (254 values);
   v1's fixed bet-size grids pinned to a corner on live, blinding half the
   model and squashing **49%** of live scores into [0.4,0.6]. Fix: v2
   request-calibrated scale-free features (§3). Invariance proven; benchmark
   holdout 0.741.

4. **The collision edge is dead on live** (commit `7011392`). Deep analysis of
   captures: live chunks collide at 0.008 vs benchmark *humans* 0.023 — real
   live hands (7-action, balanced mix) have too rich a sequence space for the
   synthetic-bot repetition trick; the model was ranking on street depth alone.
   Fix: v3 pool-anchored regularity block (§3). Holdout 0.785; live score
   distribution gained a rankable top tail (mid 49%→11%).

5. **Model leaning on live-dead features + manifest identity drift** (commits
   `2a17f82`, `08958e6`). (a) 17 features informative on benchmark are constant
   on live → masked out (§3), holdout 0.798. (b) The dashboard showed
   `2.0.0·c563496` while serving 3.1.0 — **PM2 persists the original start
   command's `POKER44_MODEL_*` env vars across every restart, including cron's**,
   overriding code defaults. Fix: never pass those env vars (code self-derives
   name/version/commit/hashes); the nightly loop now commits+pushes its
   artifacts so the served model always equals a published commit.

Two ideas were **tested and rejected** (don't re-ship them): an unsupervised
within-window cluster-split head (0.555 on windows the supervised model
separates cleanly — the dominant PCA axis isn't the bot axis) and recency-×2
training weight (0.765, hurt). Also from the original build: merged-pair
augmentation hurts (dilutes per-player signal).

---

## 5. Verification methodology (how a change earns a ship)

No change ships without passing this battery. A future maintainer should hold
the same bar.

- **Temporal holdout** — train on dates ≤ T, test on later unseen dates, pooled
  reward under the exact validator metric. This is the closest labeled proxy
  for "will it generalize forward." Reference scaffold:
  `research/` temporal-eval pattern (built ad hoc in scratch; keep the recipe
  identical to `train_final.py`).
- **Invariance test** — rescale/grid-snap benchmark data toward the live regime
  and confirm the feature vector barely moves (median abs deviation ~0). Guards
  against re-introducing a scale trap.
- **sklearn↔numpy parity** — `export_v2.verify_parity()` asserts <1e-9 on all
  dates; runs every train.
- **sklearn-free venv** — score in a numpy-only environment; must be
  byte-identical to the sklearn env (proves no hidden dependency).
- **Wire test** — `research/wire_test.py` runs a real `bt.Axon`↔`bt.Dendrite`
  loopback; wire scores must equal local, 100-chunk window well under the 180s
  budget.
- **Unit suite** — `.venv-bt/bin/python -m unittest discover -s tests` (50
  tests) including feature shape/finiteness, size-invariance, malformed-input
  safety, manifest compliance.
- **Live rescore** — after deploy, `analyze_live_capture.py --rescore` scores
  captured live windows with the new model and compares `mid` share vs what was
  served at capture time. The only signal short of a round report.

Discipline used throughout: **diagnose before fixing** (each of the five causes
was proven with data, not guessed), and **captures are diagnostics only** —
never training input (the manifest attests this; feature masks are derived from
capture *variance*, which carries no labels).

---

## 6. Operational runbook

**Environments.** Research (this machine): `.venv` (sklearn 1.9), `.venv-bt`
(bittensor 10.5, py3.12), a numpy-only venv for parity. VPS: `miner_env`
(py3.10, sklearn 1.7.2). Git pushes from research need `--no-verify` (vestigial
LFS hooks, no LFS files).

**Full local retrain + publish:**
```bash
.venv/bin/python research/download_benchmark.py    # fetch new releases
.venv/bin/python research/build_dataset.py         # project through payload_view
.venv/bin/python research/train_final.py           # trains + exports npz + parity
.venv-bt/bin/python -m unittest discover -s tests  # 50 tests
# regenerate MODEL_MANIFEST.json artifact hash, commit, push --no-verify
```

**Deploy on VPS (MUST be delete+fresh, not restart — see §4 cause 5):**
```bash
cd ~/poker44-claude && git fetch origin && git reset --hard origin/main
pm2 delete poker44_miner
WALLET_NAME=ready HOTKEY=readyh NETUID=126 NETWORK=finney AXON_PORT=8091 \
  PM2_NAME=poker44_miner bash ./scripts/miner/run/run_miner.sh
pm2 save
# verify — no POKER44_MODEL_* env this time:
grep -hE "Manifest summary|Detector self-check" ~/.pm2/logs/poker44-miner*out.log | tail -2
# success: version=3.1.0 ... commit=<HEAD> ... engine=numpy-v2
```

**Nightly self-retrain** (`scripts/miner/retrain_daily.sh`, cron `5 1 * * *`):
resets to origin/main (also auto-deploys code pushes), downloads, gates
(newest-date OOS ≥ `GATE_MIN`=0.60), retrains, refreshes the manifest hash,
**commits and pushes** the artifacts, then restarts. Any failure reverts
artifacts to HEAD and keeps the last published model serving. Needs a git push
credential on the VPS (PAT in the remote URL); without it, the loop still
retrains-and-verifies but holds the last published artifact rather than
swapping to an unpublished one.

**Health checks:**
```bash
timedatectl                                  # clock synced → no nonce rejections
grep -ac "NotVerified" ~/.pm2/logs/poker44-miner-error.log   # rejected queries
grep -a "Scored" ~/.pm2/logs/poker44-miner-out.log | tail    # engine + mid share
```

**Diagnostic loop for the transfer gap** (the actual work each round):
```bash
# on VPS, after queries land:
python research/analyze_live_capture.py 'captures/req_*.json.gz' --rescore
# scp captures to research machine for full per-feature analysis:
scp 'root@157.180.87.74:~/poker44-claude/captures/req_*.json.gz' captures/
python research/analyze_live_capture.py 'captures/req_*.json.gz' --benchmark
```
`captures/` is gitignored (validator data never enters the public repo).

---

## 7. Code map

- `poker44/detection/features.py` — feature extraction, `RequestContext`,
  `LIVE_DEGENERATE_FEATURES`, `ACTIVE_FULL_IDX`, `FEATURE_VERSION`.
- `poker44/detection/model.py` — `DetectionModel` (npz→pkl→heuristic),
  `_NumpyEnsemble`, guards, `self_check`.
- `neurons/miner.py` — production miner: scoring, manifest build, self-check,
  fingerprint log, rotating capture. **Do not add `POKER44_MODEL_*` env pins.**
- `poker44/utils/model_manifest.py` — manifest build + compliance;
  repo-relative implementation hash.
- `MODEL_MANIFEST.json` — canonical public manifest with verification recipes.
- `research/train_final.py` — train + export + parity (the retrain entrypoint).
- `research/export_v2.py` — sklearn→numpy artifact export + parity proof.
- `research/analyze_live_capture.py` — live-vs-benchmark diagnostics, `--rescore`.
- `research/build_dataset.py` / `download_benchmark.py` — data pipeline.
- `research/wire_test.py` — real Axon/Dendrite loopback.
- `scripts/miner/retrain_daily.sh` — gated, consistency-first nightly loop.
- `research/README.md` — pipeline usage.

---

## 8. Known warts / gotchas

- **PM2 env persistence** — the single most costly trap (§4.5). Restart reuses
  the original env; only `pm2 delete` + fresh start clears it. Never pin
  `POKER44_MODEL_*`.
- **`model_meta.json`** — regenerated each train; its metrics are descriptive
  only. (Older commits carried stale v1 `lodo_*` constants; removed as of this
  log.)
- **sklearn version drift** — the reason inference is numpy-native. If you ever
  add an inference path that imports sklearn, you reintroduce cause 1.
- **Benchmark ≠ live** — the entire difficulty. Any feature validated only on
  benchmark can still be dead or inverted live; always cross-check on captures.
- **Release format changed** — 2026-07-06 release jumped to ~142 groups/5000
  hands (~10× prior). The pipeline handled it, but watch for schema/scale
  shifts in new releases.
- **Self-probe impossible** — `force_validator_permit` blocks our own hotkey;
  don't build tooling that expects to query the live axon.
