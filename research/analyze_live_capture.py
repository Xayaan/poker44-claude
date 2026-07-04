"""Diagnose live-vs-benchmark distribution shift from captured payloads.

Usage:
  python research/analyze_live_capture.py captures/req_*.json.gz
      -> live-side structural + score stats (compact, paste-able; runs on VPS)
  python research/analyze_live_capture.py captures/req_*.json.gz --benchmark
      -> adds per-feature shift vs research/dataset.pkl and rescoring sanity
         (requires the local research environment)

Captured payloads are DIAGNOSTIC ONLY and are never used for training
(see the miner manifest's private_data_attestation).
"""

from __future__ import annotations

import glob
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from poker44.detection.features import (  # noqa: E402
    FEATURE_NAMES,
    compute_request_context,
    extract_chunk_features,
    extract_features_matrix,
)


def load_captures(patterns: list[str]) -> list[dict]:
    files: list[str] = []
    for p in patterns:
        files.extend(glob.glob(p))
    reqs = []
    for f in sorted(set(files)):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            reqs.append(json.load(fh))
    if not reqs:
        raise SystemExit("no capture files matched")
    return reqs


def live_stats(reqs: list[dict]) -> np.ndarray:
    all_rows = []
    for req in reqs:
        chunks = req.get("chunks") or []
        scores = req.get("scores") or []
        hand_counts = [len(c) for c in chunks if isinstance(c, list)]
        action_counts, amounts, atypes = [], [], Counter()
        keysets = Counter()
        for c in chunks:
            if not isinstance(c, list):
                continue
            for h in c:
                if not isinstance(h, dict):
                    continue
                keysets[tuple(sorted(h.keys()))] += 1
                acts = h.get("actions") or []
                if isinstance(acts, list):
                    action_counts.append(len(acts))
                    for a in acts:
                        if isinstance(a, dict):
                            atypes[str(a.get("action_type"))] += 1
                            amt = a.get("normalized_amount_bb")
                            if isinstance(amt, (int, float)) and amt > 0:
                                amounts.append(float(amt))
        s = np.array(scores, dtype=float) if scores else np.array([0.0])
        print(f"== req ts={req.get('ts')} chunks={len(chunks)}")
        print(f"   hands/chunk: min={min(hand_counts)} med={int(np.median(hand_counts))} max={max(hand_counts)}")
        print(f"   actions/hand: med={np.median(action_counts):.1f} p90={np.percentile(action_counts, 90):.1f}")
        ta = ", ".join(f"{k}:{v}" for k, v in atypes.most_common(8))
        print(f"   action_types: {ta}")
        if amounts:
            am = np.array(amounts)
            print(f"   amounts_bb: n={len(am)} med={np.median(am):.2f} p90={np.percentile(am, 90):.2f} "
                  f"distinct={len(np.unique(np.round(am, 4)))}")
        print(f"   hand key-sets: {[(list(k)[:8], v) for k, v in keysets.most_common(2)]}")
        print(f"   logged scores: min={s.min():.3f} med={np.median(s):.3f} mean={s.mean():.3f} max={s.max():.3f} "
              f"| frac in [0.4,0.6]: {float(np.mean((s >= 0.4) & (s <= 0.6))):.2f}")
        good = [c for c in chunks if isinstance(c, list) and c]
        ctx = compute_request_context(good)
        all_rows.extend(_safe_feats(c, ctx) for c in good)
    X = np.vstack(all_rows)
    print(f"\nlive feature matrix: {X.shape[0]} chunks x {X.shape[1]} features "
          f"| zero-rows={int(np.sum(~X.any(axis=1)))}")
    return X


def _safe_feats(chunk: list, ctx=None) -> np.ndarray:
    try:
        return extract_chunk_features(chunk, ctx)
    except Exception:  # noqa: BLE001
        return np.zeros(len(FEATURE_NAMES), dtype=float)


def benchmark_compare(X_live: np.ndarray) -> None:
    import pickle

    ds = Path(__file__).resolve().parent / "dataset.pkl"
    records = pickle.load(open(ds, "rb"))
    dates = sorted({r["date"] for r in records})
    recent = set(dates[-7:])
    recs = [r for r in records if r["date"] in recent]
    ctx_by_date = {
        d: compute_request_context([r["hands"] for r in recs if r["date"] == d])
        for d in sorted(recent)
    }
    X_b = np.vstack([_safe_feats(r["hands"], ctx_by_date[r["date"]]) for r in recs])
    y_b = np.array([r["label"] for r in recs])

    med_b = np.median(X_b, axis=0)
    mad_b = np.median(np.abs(X_b - med_b), axis=0) + 1e-9
    med_l = np.median(X_live, axis=0)
    shift = np.abs(med_l - med_b) / mad_b

    order = np.argsort(-shift)
    print(f"\nfeature shift vs benchmark (last {len(recent)} dates, {len(recs)} chunks):")
    print(f"  features with |shift| > 3 MADs: {int(np.sum(shift > 3))}/{len(shift)}")
    for j in order[:15]:
        # where does the live median sit relative to the class medians?
        mb_h = np.median(X_b[y_b == 0, j])
        mb_b = np.median(X_b[y_b == 1, j])
        print(f"  {FEATURE_NAMES[j]:<38} shift={shift[j]:7.2f}  "
              f"bench_med={med_b[j]:9.4f} (hum {mb_h:9.4f} / bot {mb_b:9.4f})  live_med={med_l[j]:9.4f}")

    # score-distribution comparison through the deployed model
    from poker44.detection.model import DetectionModel

    m = DetectionModel()
    rel_b = X_b - np.median(X_b, axis=0)
    rel_l = X_live - np.median(X_live, axis=0)
    p_b = m._predict_proba(np.hstack([X_b, rel_b]))
    p_l = m._predict_proba(np.hstack([X_live, rel_l]))
    print(f"\nmodel scores | benchmark(recent): frac[0.4,0.6]={float(np.mean((p_b >= .4) & (p_b <= .6))):.2f} "
          f"| live: frac[0.4,0.6]={float(np.mean((p_l >= .4) & (p_l <= .6))):.2f}")
    print(f"benchmark score split: hum_med={float(np.median(p_b[y_b == 0])):.3f} bot_med={float(np.median(p_b[y_b == 1])):.3f}")


def rescore(reqs: list[dict]) -> None:
    """Rescore captured requests with the CURRENT local model (works on the
    VPS; no dataset needed). Compares against the scores served at capture
    time."""
    from poker44.detection.model import DetectionModel

    m = DetectionModel()
    print(f"\nrescore engine={m.engine} ready={m.ready}")
    for req in reqs:
        chunks = req.get("chunks") or []
        new = np.array(m.score_chunks(chunks), dtype=float)
        old = np.array(req.get("scores") or [], dtype=float)
        mid = float(np.mean((new >= 0.4) & (new <= 0.6))) if new.size else 0.0
        line = (
            f"req ts={req.get('ts')}: new scores min={new.min():.3f} "
            f"med={float(np.median(new)):.3f} max={new.max():.3f} mid={mid:.0%}"
        )
        if old.size == new.size and old.size:
            line += f" | served-at-capture mid={float(np.mean((old >= 0.4) & (old <= 0.6))):.0%}"
        print(line)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    reqs = load_captures(args)
    X_live = live_stats(reqs)
    if "--benchmark" in sys.argv:
        benchmark_compare(X_live)
    if "--rescore" in sys.argv:
        rescore(reqs)
