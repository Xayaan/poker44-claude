"""Live-regime domain randomization for training (features v4+).

Projects labeled benchmark hands into the live platform regime so the model
trains on BOTH views of every labeled example:

  - bet amounts collapse to the live lattice {1.0, 1.5, 2.0} bb
  - pots collapse to {3, 4, 6, 8} bb
  - table sizes mix 6/7/8/9-max
  - streets arrays cap at 3 entries

The transform is an order-preserving cumulative-mass bucket map applied to the
RAW benchmark hands, followed by re-projection through the real validator
transform (`prepare_hand_for_miner`), so the platform's deterministic
bucket+noise obfuscation produces genuinely live-shaped payloads.

Provenance note (mirrors the manifest's private_data_attestation): the target
marginals below are aggregate, UNLABELED statistics measured on captured live
payloads (2026-07-04 window) — bucket shares, pot shares, table-size mix.
No validator payload enters training data; training inputs remain transformed
copies of the public benchmark with benchmark labels.

Validated 2026-07-08: domain-randomized training improved temporal holdout on
both views (orig 0.847->0.858, live-mimic view 0.812->0.841 with the v4
model levers) with per-date minima up on both.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np

from poker44.validator.payload_view import prepare_hand_for_miner

_BUCKETS = np.array(
    [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 36.0, 56.0, 84.0, 126.0]
)
_VISIBLE_BB = 0.02

# Live marginal targets (unlabeled capture statistics, 2026-07-04).
AMT_TARGETS = [(0.79, 1.0), (0.99, 1.5), (2.0, 2.0)]
POT_TARGETS = [(0.38, 3.0), (0.80, 4.0), (0.996, 6.0), (2.0, 8.0)]
SEATS_DIST = [(6, 0.56), (7, 0.08), (8, 0.18), (9, 0.18)]


def _snap(v: float) -> float:
    return float(_BUCKETS[np.argmin(np.abs(_BUCKETS - v))])


def _build_map(pool_buckets: Counter, targets) -> Dict[float, float]:
    """Order-preserving (by bucket value) cumulative-mass map to the live
    lattice. A bucket goes to the live bucket whose cumulative band contains
    the bucket's cumulative midpoint — robust to chunky bucket masses."""
    total = max(1, sum(pool_buckets.values()))
    out: Dict[float, float] = {}
    cum = 0.0
    for b in sorted(pool_buckets):
        share = pool_buckets[b] / total
        mid = cum + share / 2.0
        cum += share
        for thr, live_b in targets:
            if mid < thr + 1e-12:
                out[b] = live_b
                break
    return out


def build_date_maps(projected_hands: List[dict]) -> tuple[dict, dict]:
    """Bucket maps derived from one date's PROJECTED hands (the same pool a
    serving request would show)."""
    amt_pool: Counter = Counter()
    pot_pool: Counter = Counter()
    for h in projected_hands:
        for a in h.get("actions") or []:
            v = a.get("normalized_amount_bb")
            if isinstance(v, (int, float)) and v > 0:
                amt_pool[_snap(float(v))] += 1
            for f in ("pot_before", "pot_after"):
                p = a.get(f)
                if isinstance(p, (int, float)) and p > 0:
                    pot_pool[_snap(float(p) / _VISIBLE_BB)] += 1
    return _build_map(amt_pool, AMT_TARGETS), _build_map(pot_pool, POT_TARGETS)


def mimic_raw_hand(
    raw: dict, amt_map: dict, pot_map: dict, rng: np.random.Generator
) -> dict:
    """Live-regime copy of one RAW benchmark hand (pre-projection)."""
    h = {k: v for k, v in raw.items()}
    md = dict(h.get("metadata") or {})
    bb = float(md.get("bb", _VISIBLE_BB) or _VISIBLE_BB)
    r = rng.random()
    cum = 0.0
    for seats, p in SEATS_DIST:
        cum += p
        if r <= cum:
            md["max_seats"] = seats
            break
    h["metadata"] = md
    h["streets"] = list(h.get("streets") or [])[:3]
    new_actions = []
    for a in h.get("actions") or []:
        if not isinstance(a, dict):
            continue
        a = dict(a)
        amt = float(a.get("amount") or 0.0)
        if amt > 0:
            lb = amt_map.get(_snap(amt / bb))
            if lb is not None:
                a["amount"] = lb * bb
        for f in ("raise_to", "call_to", "pot_before", "pot_after"):
            v = a.get(f)
            if isinstance(v, (int, float)) and v > 0:
                lb = pot_map.get(_snap(float(v) / bb))
                if lb is not None:
                    a[f] = lb * bb
        new_actions.append(a)
    h["actions"] = new_actions
    return h


def build_mimic_records(recs: List[dict], seed: int = 11) -> List[dict]:
    """Transformed+reprojected copy of every dataset record (labels ride
    along). Records keep their date so training batches stay per (date, view)."""
    dates = sorted({r["date"] for r in recs})
    maps = {}
    for d in dates:
        proj = [h for r in recs if r["date"] == d for h in r["hands"]]
        maps[d] = build_date_maps(proj)
    rng = np.random.default_rng(seed)
    out = []
    for r in recs:
        amt_map, pot_map = maps[r["date"]]
        hands = [
            prepare_hand_for_miner(mimic_raw_hand(raw, amt_map, pot_map, rng))
            for raw in r["raw_hands"]
        ]
        out.append(
            {
                "date": r["date"],
                "label": r["label"],
                "hands": hands,
            }
        )
    return out
