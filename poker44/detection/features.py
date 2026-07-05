"""Chunk-level feature extraction for Poker44 bot detection.

Operates on miner-visible hand payloads (the output of
``poker44.validator.payload_view.prepare_hand_for_miner``). Every feature is
chunk-size invariant: rates, quantiles, and pairwise-collision U-statistics
only, so scores stay stable whether a chunk holds 30 or 100+ hands.

Feature version 2: every monetary feature is normalized by request-level
context (median bet size, request quantile grids), so the extractor is
invariant to the absolute bet/stack/pot scale of the data source. The public
benchmark quotes amounts around tens of bb while live platform hands quote
around 1 bb; v1's fixed grids saturated on live data, v2 self-calibrates per
request. Training builds the same context per date batch, mirroring serving
where the validator sends the whole eval window in one request.

Used by both the training pipeline and the production miner — keep pure
stdlib + numpy and tolerant of missing/malformed fields.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

FEATURE_VERSION = 3

# Fallback bet-size grid (v1 canonicalizer buckets), used only when a request
# carries too few positive amounts to build a quantile grid.
_BUCKETS = np.array(
    [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 36.0, 56.0, 84.0, 126.0]
)
_VISIBLE_BB = 0.02
_STREET_ORDER = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
_ACTION_TYPES = ("fold", "check", "call", "bet", "raise")
_ACTION_IDX = {t: i for i, t in enumerate(_ACTION_TYPES)}
_N_HIST_BINS = 15
_N_SEQ_BUCKETS = 8
_MIN_CTX_AMOUNTS = 8


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(out):
        return 0.0
    return out


def _collision(counts: Sequence[int], n: int) -> float:
    """Unbiased pairwise collision rate: P(two random items identical)."""
    if n < 2:
        return 0.0
    return float(sum(c * (c - 1) for c in counts)) / float(n * (n - 1))


def _quantiles(values: List[float], qs: Sequence[float]) -> List[float]:
    if not values:
        return [0.0] * len(qs)
    arr = np.asarray(values, dtype=float)
    return [float(np.percentile(arr, q)) for q in qs]


def _entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


class _HandView:
    """Pre-parsed single hand."""

    __slots__ = (
        "actions",
        "n_players",
        "hero_seat",
        "stacks",
        "max_street",
        "pot0",
        "pot_last",
    )

    def __init__(self, hand: Dict[str, Any]):
        metadata = hand.get("metadata") if isinstance(hand.get("metadata"), dict) else {}
        players = hand.get("players") if isinstance(hand.get("players"), list) else []
        actions_raw = hand.get("actions") if isinstance(hand.get("actions"), list) else []

        self.hero_seat = int(_safe_float(metadata.get("hero_seat")))
        self.n_players = len(players)
        self.stacks = [
            round(_safe_float(p.get("starting_stack")) / _VISIBLE_BB, 2)
            for p in players
            if isinstance(p, dict)
        ]

        self.actions = []
        self.max_street = 0
        for action in actions_raw:
            if not isinstance(action, dict):
                continue
            a_type = str(action.get("action_type") or "")
            if a_type not in _ACTION_IDX:
                continue
            street = str(action.get("street") or "preflop").lower()
            street_i = _STREET_ORDER.get(street, 0)
            amount_bb = _safe_float(action.get("normalized_amount_bb"))
            raise_to = action.get("raise_to")
            call_to = action.get("call_to")
            pot_before = _safe_float(action.get("pot_before")) / _VISIBLE_BB
            pot_after = _safe_float(action.get("pot_after")) / _VISIBLE_BB
            actor = int(_safe_float(action.get("actor_seat")))
            self.actions.append(
                {
                    "type": a_type,
                    "street_i": street_i,
                    "amount_bb": amount_bb,
                    "raise_to": _safe_float(raise_to) if raise_to is not None else None,
                    "call_to": _safe_float(call_to) if call_to is not None else None,
                    "pot_before": pot_before,
                    "pot_after": pot_after,
                    "is_hero": actor == self.hero_seat and self.hero_seat > 0,
                }
            )
            self.max_street = max(self.max_street, street_i)

        self.pot0 = self.actions[0]["pot_before"] if self.actions else 0.0
        self.pot_last = self.actions[-1]["pot_after"] if self.actions else 0.0


class RequestContext:
    """Scale calibration shared by every chunk of one request (or one
    training date batch): bet-size unit, quantile grids, stack unit, and the
    pooled stack multiset for duplication features."""

    __slots__ = (
        "amt_scale",
        "hist_edges",
        "hist_pool_shares",
        "hist_pool_entropy",
        "big_thresholds",
        "big_pool_shares",
        "pool_modal_share",
        "pool_distinct_ratio",
        "pool_top3_share",
        "pool_ts_joint",
        "pool_h_type",
        "pool_h_type_street",
        "pool_h_type_prev",
        "pool_acts_cv",
        "pool_perhand_check_std",
        "pool_perhand_fold_std",
        "seq_edges",
        "stack_scale",
        "stack_counts",
    )

    def __init__(self, hands: List[_HandView]):
        amounts = [
            a["amount_bb"] for h in hands for a in h.actions if a["amount_bb"] > 0
        ]
        stacks = [s for h in hands for s in h.stacks]
        if len(amounts) >= _MIN_CTX_AMOUNTS:
            arr = np.asarray(amounts, dtype=float)
            self.amt_scale = float(np.median(arr)) or 1.0
            self.hist_edges = np.percentile(
                arr, np.linspace(0.0, 100.0, _N_HIST_BINS + 1)
            )[1:-1]
            self.big_thresholds = [float(np.percentile(arr, q)) for q in (60, 80, 90)]
            self.seq_edges = np.percentile(
                arr, np.linspace(0.0, 100.0, _N_SEQ_BUCKETS + 1)
            )[1:-1]
        else:
            arr = np.asarray(amounts, dtype=float) if amounts else np.array([])
            self.amt_scale = 1.0
            self.hist_edges = _BUCKETS[1:]
            self.big_thresholds = [24.0, 56.0, 84.0]
            self.seq_edges = _BUCKETS[::2]
        # Pool baselines: with heavy ties (live feeds quote few distinct
        # sizes) equal-mass bins are spiky; features report each chunk's
        # occupancy anomaly vs these baselines, which is bounded and
        # 0-centered in every domain.
        if arr.size:
            pool_bins = np.searchsorted(self.hist_edges, arr, side="right")
            counts = np.bincount(pool_bins, minlength=_N_HIST_BINS).astype(float)
            self.hist_pool_shares = counts / max(1.0, counts.sum())
            self.hist_pool_entropy = _entropy(self.hist_pool_shares)
            self.big_pool_shares = [
                float(np.mean(arr >= t)) for t in self.big_thresholds
            ]
            amt_counts = Counter(round(v, 6) for v in amounts)
            top = sorted(amt_counts.values(), reverse=True)
            self.pool_modal_share = top[0] / len(amounts)
            self.pool_top3_share = sum(top[:3]) / len(amounts)
            self.pool_distinct_ratio = len(amt_counts) / len(amounts)
        else:
            self.hist_pool_shares = np.zeros(_N_HIST_BINS)
            self.hist_pool_entropy = 0.0
            self.big_pool_shares = [0.0, 0.0, 0.0]
            self.pool_modal_share = 0.0
            self.pool_top3_share = 0.0
            self.pool_distinct_ratio = 0.0
        med_stack = float(np.median(stacks)) if stacks else 0.0
        self.stack_scale = med_stack if med_stack > 0 else 1.0
        self.stack_counts = Counter(stacks)

        # Pooled regularity baselines for the v3 block: chunk features report
        # their offset from these, so levels stay comparable across domains
        # with very different action richness.
        log5 = float(np.log(5.0))
        pool_actions = [a for h in hands for a in h.actions]
        n_pa = max(1, len(pool_actions))
        ts = np.zeros((5, 4), dtype=float)
        bigram = np.zeros((5, 5), dtype=float)
        n_bg = 0
        acts_ph: List[int] = []
        ph_check: List[float] = []
        ph_fold: List[float] = []
        for h in hands:
            if h.actions:
                acts_ph.append(len(h.actions))
                n_a = len(h.actions)
                ph_check.append(sum(1 for a in h.actions if a["type"] == "check") / n_a)
                ph_fold.append(sum(1 for a in h.actions if a["type"] == "fold") / n_a)
            for a in h.actions:
                ts[_ACTION_IDX[a["type"]], a["street_i"]] += 1.0
            for prev, cur in zip(h.actions, h.actions[1:]):
                bigram[_ACTION_IDX[prev["type"]], _ACTION_IDX[cur["type"]]] += 1.0
                n_bg += 1
        self.pool_ts_joint = ts / n_pa
        self.pool_h_type = _entropy(self.pool_ts_joint.sum(axis=1)) / log5
        self.pool_h_type_street = (
            _entropy(self.pool_ts_joint.ravel())
            - _entropy(self.pool_ts_joint.sum(axis=0))
        ) / log5
        if n_bg:
            bigram /= n_bg
            self.pool_h_type_prev = (
                _entropy(bigram.ravel()) - _entropy(bigram.sum(axis=1))
            ) / log5
        else:
            self.pool_h_type_prev = 0.0
        acts_mean = float(np.mean(acts_ph)) if acts_ph else 0.0
        self.pool_acts_cv = (
            float(np.std(acts_ph)) / max(1.0, acts_mean) if acts_ph else 0.0
        )
        self.pool_perhand_check_std = float(np.std(ph_check)) if ph_check else 0.0
        self.pool_perhand_fold_std = float(np.std(ph_fold)) if ph_fold else 0.0

    def amt_bucket(self, amount: float) -> int:
        if amount <= 0:
            return -1
        return int(np.searchsorted(self.seq_edges, amount, side="right"))

    def hist_bin(self, amount: float) -> int:
        return int(np.searchsorted(self.hist_edges, amount, side="right"))


def compute_request_context(chunks: List[List[Dict[str, Any]]]) -> RequestContext:
    hands = [
        _HandView(h)
        for chunk in chunks
        if isinstance(chunk, list)
        for h in chunk
        if isinstance(h, dict)
    ]
    return RequestContext(hands)


def _hand_sequences(hand: _HandView, ctx: RequestContext) -> tuple[str, str, bool]:
    seq_types = "".join(a["type"][0] for a in hand.actions)
    seq_full = "|".join(
        f"{a['street_i']}{a['type'][0]}{ctx.amt_bucket(a['amount_bb'])}"
        for a in hand.actions
    )
    # Validator artifact: single-action source hands are emitted as ~12
    # copies of one action.
    single_action_marker = (
        len(hand.actions) >= 10
        and len(
            {
                (a["type"], a["street_i"], round(a["amount_bb"], 6))
                for a in hand.actions
            }
        )
        == 1
    )
    return seq_types, seq_full, single_action_marker


def extract_chunk_features(
    chunk: List[Dict[str, Any]], ctx: Optional[RequestContext] = None
) -> np.ndarray:
    """Extract one feature vector for a chunk (list of miner-visible hands).

    ctx carries the request-level scale calibration; when omitted the chunk
    self-calibrates (used only by legacy callers/tests)."""
    hands = [_HandView(h) for h in chunk if isinstance(h, dict)]
    hands = [h for h in hands if h.actions or h.n_players]
    features: List[float] = []
    if not hands:
        return np.zeros(len(FEATURE_NAMES), dtype=float)
    if ctx is None:
        ctx = RequestContext(hands)

    n_hands = len(hands)
    all_actions = [a for h in hands for a in h.actions]
    n_actions = max(1, len(all_actions))
    unit = ctx.amt_scale

    # --- A. action-type rates and volume shape ---------------------------
    type_counts = Counter(a["type"] for a in all_actions)
    for t in _ACTION_TYPES:
        features.append(type_counts.get(t, 0) / n_actions)
    amounts = [a["amount_bb"] for a in all_actions if a["amount_bb"] > 0]
    features.append(1.0 - len(amounts) / n_actions)  # zero-amount share
    features.append(sum(1 for a in all_actions if a["raise_to"] is not None) / n_actions)
    features.append(sum(1 for a in all_actions if a["call_to"] is not None) / n_actions)

    acts_per_hand = [len(h.actions) for h in hands]
    features.extend(_quantiles(acts_per_hand, (10, 50, 90)))
    features.append(float(np.std(acts_per_hand)))

    players_per_hand = [h.n_players for h in hands]
    features.append(float(np.mean(players_per_hand)))
    for k in (2, 3, 4, 5, 6):
        features.append(sum(1 for p in players_per_hand if p == k) / n_hands)

    # street shares and reach
    street_counts = Counter(a["street_i"] for a in all_actions)
    for s in range(4):
        features.append(street_counts.get(s, 0) / n_actions)
    max_streets = [h.max_street for h in hands]
    features.append(float(np.mean(max_streets)))
    for s in (1, 2, 3):
        features.append(sum(1 for m in max_streets if m >= s) / n_hands)

    # --- B. sequence repetition (size-unbiased collision statistics) -----
    seqs = [_hand_sequences(h, ctx) for h in hands]
    seq_counter = Counter(s[0] for s in seqs)
    features.append(_collision(list(seq_counter.values()), n_hands))
    seq_full_counter = Counter(s[1] for s in seqs)
    features.append(_collision(list(seq_full_counter.values()), n_hands))
    long_seqs = [s[0] for h, s in zip(hands, seqs) if len(h.actions) >= 6]
    features.append(_collision(list(Counter(long_seqs).values()), len(long_seqs)))
    combo = [f"{h.n_players}|{s[0]}" for h, s in zip(hands, seqs)]
    features.append(_collision(list(Counter(combo).values()), n_hands))
    features.append(sum(1 for s in seqs if s[2]) / n_hands)

    # --- C. bet-size distribution (request-normalized) --------------------
    features.extend(v / unit for v in _quantiles(amounts, (10, 25, 50, 75, 90)))
    mean_amt = float(np.mean(amounts)) if amounts else 0.0
    features.append(mean_amt / unit)
    features.append(float(np.std(amounts)) / mean_amt if mean_amt > 0 else 0.0)
    bin_counts = Counter(ctx.hist_bin(v) for v in amounts)
    n_binned = max(1, sum(bin_counts.values()))
    hist = np.array([bin_counts.get(i, 0) / n_binned for i in range(_N_HIST_BINS)])
    features.extend((hist - ctx.hist_pool_shares).tolist())
    features.append(_entropy(hist) - ctx.hist_pool_entropy)
    for threshold, pool_share in zip(ctx.big_thresholds, ctx.big_pool_shares):
        share = sum(1 for v in amounts if v >= threshold) / max(1, len(amounts))
        features.append(share - pool_share)
    # per-street mean amounts (request-normalized)
    for s in range(4):
        street_amts = [
            a["amount_bb"] for a in all_actions if a["street_i"] == s and a["amount_bb"] > 0
        ]
        features.append(float(np.mean(street_amts)) / unit if street_amts else 0.0)

    # --- D. pot dynamics (request-normalized) -----------------------------
    pot0s = [h.pot0 for h in hands]
    features.append(float(np.mean(pot0s)) / unit)
    features.append(float(np.std(pot0s)) / unit)
    features.append(
        _collision(list(Counter(round(p / unit, 2) for p in pot0s).values()), n_hands)
    )
    pot_growth = [
        (h.pot_last - h.pot0) / max(1, len(h.actions)) for h in hands if h.actions
    ]
    features.extend(v / unit for v in _quantiles(pot_growth, (50, 90)))
    violations = sum(1 for a in all_actions if a["pot_after"] < a["pot_before"] - 1e-9)
    features.append(violations / n_actions)

    # --- E. stacks (request-normalized; duplication is scale-free) --------
    stacks = [s for h in hands for s in h.stacks]
    s_unit = ctx.stack_scale
    features.extend(v / s_unit for v in _quantiles(stacks, (10, 50, 90)))
    features.append(float(np.std(stacks)) / s_unit if stacks else 0.0)
    features.append(_collision(list(Counter(stacks).values()), len(stacks)))
    dup_stacks = sum(1 for s in stacks if ctx.stack_counts.get(s, 0) >= 2)
    features.append(dup_stacks / max(1, len(stacks)))
    hand_mean_stacks = [float(np.mean(h.stacks)) for h in hands if h.stacks]
    features.append(
        float(np.std(hand_mean_stacks)) / s_unit if hand_mean_stacks else 0.0
    )

    # --- F. action-type bigrams (within hand) -------------------------------
    bigram = np.zeros((5, 5), dtype=float)
    n_bigrams = 0
    for h in hands:
        for prev, cur in zip(h.actions, h.actions[1:]):
            bigram[_ACTION_IDX[prev["type"]], _ACTION_IDX[cur["type"]]] += 1
            n_bigrams += 1
    if n_bigrams:
        bigram /= n_bigrams
    features.extend(bigram.ravel().tolist())

    # --- G. hero behavior ----------------------------------------------------
    hero_actions = [a for a in all_actions if a["is_hero"]]
    n_hero = max(1, len(hero_actions))
    features.append(len(hero_actions) / n_actions)
    hero_counts = Counter(a["type"] for a in hero_actions)
    for t in _ACTION_TYPES:
        features.append(hero_counts.get(t, 0) / n_hero)
    hero_amts = [a["amount_bb"] for a in hero_actions if a["amount_bb"] > 0]
    features.append(float(np.mean(hero_amts)) / unit if hero_amts else 0.0)

    # --- H. regularity block (v3): signals that generalize to real bots ------
    # Bots — synthetic or live — are more deterministic and more
    # self-consistent than humans. All of this is scale-free and defined in
    # every domain (unlike exact-sequence collisions, which die when the
    # action space is rich).

    # H1. action-type x street joint rates, as anomaly vs the request pool
    ts_counts = np.zeros((5, 4), dtype=float)
    for a in all_actions:
        ts_counts[_ACTION_IDX[a["type"]], a["street_i"]] += 1.0
    ts_joint = ts_counts / n_actions
    features.extend((ts_joint - ctx.pool_ts_joint).ravel().tolist())

    # H2. determinism: entropies as offsets from the request pool (levels are
    # domain-dependent; offsets are comparable everywhere)
    log5 = float(np.log(5.0))
    p_type = ts_joint.sum(axis=1)
    features.append(_entropy(p_type) / log5 - ctx.pool_h_type)
    p_street = ts_joint.sum(axis=0)
    h_cond_street = (_entropy(ts_joint.ravel()) - _entropy(p_street)) / log5
    features.append(h_cond_street - ctx.pool_h_type_street)
    if n_bigrams:
        p_prev = bigram.sum(axis=1)
        h_cond_prev = (_entropy(bigram.ravel()) - _entropy(p_prev)) / log5
    else:
        h_cond_prev = 0.0
    features.append(h_cond_prev - ctx.pool_h_type_prev)

    # H3. within-chunk behavioral drift (humans drift, bots stay put)
    def _type_rates(sub: List[_HandView]) -> np.ndarray:
        cnt = np.zeros(5, dtype=float)
        for h in sub:
            for a in h.actions:
                cnt[_ACTION_IDX[a["type"]]] += 1.0
        return cnt / max(1.0, cnt.sum())

    def _street_shares(sub: List[_HandView]) -> np.ndarray:
        cnt = np.zeros(4, dtype=float)
        for h in sub:
            for a in h.actions:
                cnt[a["street_i"]] += 1.0
        return cnt / max(1.0, cnt.sum())

    # sampling noise scales ~1/sqrt(n); rescale so 30-hand training chunks
    # and 90-hand live chunks land in the same range
    half = n_hands // 2
    root_n = float(np.sqrt(n_hands))
    first, second = hands[:half], hands[half:]
    features.append(root_n * float(np.abs(_type_rates(first) - _type_rates(second)).sum()))
    features.append(root_n * float(np.abs(_street_shares(first) - _street_shares(second)).sum()))
    features.append(root_n * float(np.abs(_type_rates(hands[::2]) - _type_rates(hands[1::2])).sum()))
    acts_mean = float(np.mean(acts_per_hand)) if acts_per_hand else 0.0
    m1 = float(np.mean([len(h.actions) for h in first])) if first else 0.0
    m2 = float(np.mean([len(h.actions) for h in second])) if second else 0.0
    features.append(root_n * abs(m1 - m2) / max(1.0, acts_mean))

    # H4. sizing-formula adherence: modal amount concentration vs the pool
    if amounts:
        amt_counts = Counter(round(v, 6) for v in amounts)
        top = sorted(amt_counts.values(), reverse=True)
        modal_share = top[0] / len(amounts)
        top3_share = sum(top[:3]) / len(amounts)
        distinct_ratio = len(amt_counts) / len(amounts)
    else:
        modal_share = top3_share = distinct_ratio = 0.0
    features.append(modal_share - ctx.pool_modal_share)
    features.append(top3_share - ctx.pool_top3_share)
    features.append(distinct_ratio - ctx.pool_distinct_ratio)

    # H5. volume regularity + per-hand rate dispersion (pool-anchored)
    features.append(
        float(np.std(acts_per_hand)) / max(1.0, acts_mean) - ctx.pool_acts_cv
    )
    ph_check, ph_fold = [], []
    for h in hands:
        if not h.actions:
            continue
        n_a = len(h.actions)
        ph_check.append(sum(1 for a in h.actions if a["type"] == "check") / n_a)
        ph_fold.append(sum(1 for a in h.actions if a["type"] == "fold") / n_a)
    features.append(
        (float(np.std(ph_check)) if ph_check else 0.0) - ctx.pool_perhand_check_std
    )
    features.append(
        (float(np.std(ph_fold)) if ph_fold else 0.0) - ctx.pool_perhand_fold_std
    )

    return np.asarray(features, dtype=float)


def _build_feature_names() -> List[str]:
    names: List[str] = []
    names += [f"rate_{t}" for t in _ACTION_TYPES]
    names += ["zero_amt_share", "raise_to_rate", "call_to_rate"]
    names += ["acts_p10", "acts_p50", "acts_p90", "acts_std"]
    names += ["players_mean"] + [f"players_{k}" for k in (2, 3, 4, 5, 6)]
    names += [f"street_share_{s}" for s in range(4)]
    names += ["street_reach_mean", "reach_flop", "reach_turn", "reach_river"]
    names += ["coll_seq", "coll_seq_full", "coll_seq_long", "coll_players_seq", "single_action_rate"]
    names += ["amt_p10", "amt_p25", "amt_p50", "amt_p75", "amt_p90", "amt_mean", "amt_cv"]
    names += [f"bucket_{i}" for i in range(15)]
    names += ["bucket_entropy", "amt_ge24", "amt_ge56", "amt_ge84"]
    names += [f"amt_street_{s}" for s in range(4)]
    names += ["pot0_mean", "pot0_std", "pot0_coll", "potgrow_p50", "potgrow_p90", "pot_violation_rate"]
    names += ["stack_p10", "stack_p50", "stack_p90", "stack_std", "stack_coll", "stack_round_share", "stack_hand_drift"]
    names += [f"bg_{a}_{b}" for a in _ACTION_TYPES for b in _ACTION_TYPES]
    names += ["hero_action_share"] + [f"hero_rate_{t}" for t in _ACTION_TYPES] + ["hero_amt_mean"]
    names += [f"ts_{t}_{s}" for t in _ACTION_TYPES for s in range(4)]
    names += ["H_type", "H_type_given_street", "H_type_given_prev"]
    names += ["drift_type_half", "drift_street_half", "drift_type_oddeven", "drift_acts_half"]
    names += ["amt_modal_anom", "amt_top3_share", "amt_distinct_anom"]
    names += ["acts_cv", "perhand_check_std", "perhand_fold_std"]
    return names


FEATURE_NAMES: List[str] = _build_feature_names()

# Features whose variance collapses on live validator traffic (measured on
# unlabeled captured payloads, 2026-07-04: live MAD < 2% of benchmark MAD).
# They stay in the extractor for auditability but are excluded from model
# input, so training cannot lean on signals that are flat in production.
# Derived purely from feature variance on unlabeled data — no label
# information can leak (see manifest notes).
LIVE_DEGENERATE_FEATURES: List[str] = [
    "acts_p10",
    "acts_p50",
    "acts_p90",
    "single_action_rate",
    "amt_p10",
    "amt_p25",
    "amt_p50",
    "amt_p75",
    "stack_std",
    "stack_coll",
    "stack_round_share",
    "stack_hand_drift",
    "bg_bet_bet",
    "bg_raise_bet",
    "bg_raise_raise",
    "ts_bet_0",
    "ts_raise_3",
]
_DEAD_SET = set(LIVE_DEGENERATE_FEATURES)
ACTIVE_IDX: np.ndarray = np.array(
    [i for i, n in enumerate(FEATURE_NAMES) if n not in _DEAD_SET], dtype=np.int64
)
# absolute block + batch-relative block share the mask
ACTIVE_FULL_IDX: np.ndarray = np.concatenate(
    [ACTIVE_IDX, ACTIVE_IDX + len(FEATURE_NAMES)]
)


def extract_features_matrix(chunks: List[List[Dict[str, Any]]]) -> np.ndarray:
    """Feature matrix for a full request: absolute + batch-relative blocks.

    The validator sends every chunk of the current eval window in a single
    request, so the request doubles as calibration context (bet-size scale,
    quantile grids) and as an unsupervised drift anchor (the second block is
    each chunk's offset from the request median).
    """
    if not chunks:
        return np.zeros((0, 2 * len(FEATURE_NAMES)), dtype=float)
    ctx = compute_request_context(chunks)
    absolute = np.vstack([extract_chunk_features(chunk, ctx) for chunk in chunks])
    if absolute.shape[0] >= 3:
        relative = absolute - np.median(absolute, axis=0)
    else:
        # Too few chunks for a meaningful batch anchor: no relative signal.
        relative = np.zeros_like(absolute)
    return np.hstack([absolute, relative])


ALL_FEATURE_NAMES: List[str] = FEATURE_NAMES + [f"rel_{n}" for n in FEATURE_NAMES]
