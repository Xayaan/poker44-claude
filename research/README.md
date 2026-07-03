# Poker44 Detection Research Pipeline

Training and validation pipeline for the shipped detection model. Production
inference uses `poker44/detection/model_v2.npz` (format `poker44-detection-v2`)
— a numpy-native export of the ensemble: flat arrays only, loaded with
`allow_pickle=False`, **no scikit-learn and no pickle at inference**, so it is
immune to library version drift on the deploy host. `model.pkl` (v1, sklearn
pickle) is kept as a research/secondary artifact.

## Pipeline

```bash
.venv/bin/python research/download_benchmark.py   # fetch all benchmark releases (cache: research/data/)
.venv/bin/python research/build_dataset.py        # project hands through the exact validator canonicalizer
.venv/bin/python research/train.py                # leave-one-date-out CV experiments
.venv/bin/python research/train_final.py          # train final ensemble -> model.pkl + model_v2.npz (parity-checked)
.venv/bin/python research/export_v2.py            # re-export npz from an existing model.pkl + parity proof
.venv/bin/python research/parity_sim.py           # validator-parity forward-cycle simulation (temporal holdout)
.venv-bt/bin/python research/wire_test.py         # real bt.Axon <-> bt.Dendrite loopback test
```

Verified environment: Python 3.12/3.14, numpy 2.5.0, scikit-learn 1.9.0,
bittensor 10.5.0 (training). Inference needs numpy only; a scikit-learn-free
venv produces byte-identical scores (verified). Artifact preference at
runtime: `model_v2.npz` → `model.pkl` → deterministic heuristic; the miner
logs the active engine at startup (`Detector self-check | engine=...`) and on
every scored request.

## Incident note (2026-07-03, R1/R2 of the live cycle)

Live R2 scored 0.463 ≈ the miner-level heuristic on that day's release
(0.460), while the trained ensemble scored 0.703 out-of-sample on the same
release — i.e. production served a fallback, not the ensemble (env drift or a
dirty tree on the deploy host; v1 was a sklearn/numpy pickle and any
unpickle/predict failure degrades silently). The v2 numpy artifact plus
startup self-check + per-request engine logging make that failure class
impossible / immediately visible.

## Model

- **Features** (`poker44/detection/features.py`, 106 per chunk + 106
  batch-relative): every feature is chunk-size invariant (rates, quantiles,
  pairwise-collision U-statistics). Key stable signals discovered:
  - action-sequence collision (bots repeat exact action sequences more than
    humans; bot/human direction stable on 38/38 release dates),
  - bet-size distribution shape (bucket histogram, quantiles; stable 34-37/38),
  - pot dynamics, stack distribution, action bigrams, hero behavior.
  - batch-relative block: chunk features minus the per-request median —
    the validator sends the whole eval window in one request, so this
    anchors away day-level drift without labels.
- **Ensemble**: 3 HistGradientBoosting configs x 2 seeds, mean probability,
  blended 0.85/0.15 with StandardScaler+LogisticRegression(C=0.2).
- **Training**: all 38 benchmark release dates (2026-05-26..2026-07-02),
  540 labeled chunk groups + 8 random sub-chunk augmentations each
  (4,860 rows). Labels are never read from hand payloads.

## Validation results (exact validator reward = 0.75*AP + 0.25*recall@FPR<=5%)

| evaluation | pooled | per-date min | per-date median |
|---|---|---|---|
| leave-one-date-out CV (38 dates) | 0.859 (AP 0.916) | 0.657 | 0.895 |
| temporal holdout (trained <= 06-25, tested 06-26..07-02) | 0.826 (AP 0.893) | 0.778 | 0.889 |
| reference heuristic miner (same holdout) | 0.429 | — | — |

Dashboard context at build time: live provisional leader composite 0.599,
best historical close 0.643, last close 0.553.

Robustness (LODO-trained models): merged double-size chunks 0.831 mean,
80/20-unbalanced batch anchor 0.881 mean (no degradation), missing relative
block 0.817 mean, live-scale latency 91 chunks x 100 hands in ~4s over the
wire (180s budget).
