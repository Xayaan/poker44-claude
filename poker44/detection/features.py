"""Chunk-level feature extraction for Poker44 bot detection.

Operates on miner-visible hand payloads (the output of
``poker44.validator.payload_view.prepare_hand_for_miner``). Every feature is
chunk-size invariant: rates, quantiles, and pairwise-collision U-statistics
only, so scores stay stable whether a chunk holds 30 or 100+ hands.

Used by both the training pipeline and the production miner — keep pure
stdlib + numpy and tolerant of missing/malformed fields.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Sequence

import numpy as np

# Visible bet-size buckets used by the validator payload canonicalizer.
_BUCKETS = np.array(
    [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 36.0, 56.0, 84.0, 126.0]
)
_VISIBLE_BB = 0.02
_STREET_ORDER = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
_ACTION_TYPES = ("fold", "check", "call", "bet", "raise")
_ACTION_IDX = {t: i for i, t in enumerate(_ACTION_TYPES)}


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
        "seq_types",
        "seq_full",
        "max_street",
        "pot0",
        "pot_last",
        "single_action_marker",
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
        seq_types: List[str] = []
        seq_full: List[str] = []
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
            bucket = int(np.argmin(np.abs(_BUCKETS - amount_bb))) if amount_bb > 0 else -1
            self.actions.append(
                {
                    "type": a_type,
                    "street_i": street_i,
                    "amount_bb": amount_bb,
                    "bucket": bucket,
                    "raise_to": _safe_float(raise_to) if raise_to is not None else None,
                    "call_to": _safe_float(call_to) if call_to is not None else None,
                    "pot_before": pot_before,
                    "pot_after": pot_after,
                    "is_hero": actor == self.hero_seat and self.hero_seat > 0,
                }
            )
            self.max_street = max(self.max_street, street_i)
            seq_types.append(a_type[0])
            seq_full.append(f"{street_i}{a_type[0]}{bucket}")

        self.seq_types = "".join(seq_types)
        self.seq_full = "|".join(seq_full)
        self.pot0 = self.actions[0]["pot_before"] if self.actions else 0.0
        self.pot_last = self.actions[-1]["pot_after"] if self.actions else 0.0
        # Validator artifact: single-action source hands are emitted as 12
        # copies of one action.
        self.single_action_marker = (
            len(self.actions) >= 10
            and len({(a["type"], a["street_i"], a["bucket"]) for a in self.actions}) == 1
        )


def extract_chunk_features(chunk: List[Dict[str, Any]]) -> np.ndarray:
    """Extract one feature vector for a chunk (list of miner-visible hands)."""
    hands = [_HandView(h) for h in chunk if isinstance(h, dict)]
    hands = [h for h in hands if h.actions or h.n_players]
    features: List[float] = []
    if not hands:
        return np.zeros(len(FEATURE_NAMES), dtype=float)

    n_hands = len(hands)
    all_actions = [a for h in hands for a in h.actions]
    n_actions = max(1, len(all_actions))

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
    seq_counter = Counter(h.seq_types for h in hands)
    features.append(_collision(list(seq_counter.values()), n_hands))
    seq_full_counter = Counter(h.seq_full for h in hands)
    features.append(_collision(list(seq_full_counter.values()), n_hands))
    long_seqs = [h.seq_types for h in hands if len(h.actions) >= 6]
    features.append(_collision(list(Counter(long_seqs).values()), len(long_seqs)))
    combo = [f"{h.n_players}|{h.seq_types}" for h in hands]
    features.append(_collision(list(Counter(combo).values()), n_hands))
    features.append(sum(1 for h in hands if h.single_action_marker) / n_hands)

    # --- C. bet-size distribution ----------------------------------------
    features.extend(_quantiles(amounts, (10, 25, 50, 75, 90)))
    mean_amt = float(np.mean(amounts)) if amounts else 0.0
    features.append(mean_amt)
    features.append(float(np.std(amounts)) / mean_amt if mean_amt > 0 else 0.0)
    bucket_counts = Counter(a["bucket"] for a in all_actions if a["bucket"] >= 0)
    n_bucketed = max(1, sum(bucket_counts.values()))
    bucket_hist = np.array([bucket_counts.get(i, 0) / n_bucketed for i in range(15)])
    features.extend(bucket_hist.tolist())
    features.append(_entropy(bucket_hist))
    for threshold in (24.0, 56.0, 84.0):
        features.append(
            sum(1 for v in amounts if v >= threshold) / max(1, len(amounts))
        )
    # per-street mean amounts
    for s in range(4):
        street_amts = [
            a["amount_bb"] for a in all_actions if a["street_i"] == s and a["amount_bb"] > 0
        ]
        features.append(float(np.mean(street_amts)) if street_amts else 0.0)

    # --- D. pot dynamics ---------------------------------------------------
    pot0s = [h.pot0 for h in hands]
    features.append(float(np.mean(pot0s)))
    features.append(float(np.std(pot0s)))
    features.append(_collision(list(Counter(round(p, 2) for p in pot0s).values()), n_hands))
    pot_growth = [
        (h.pot_last - h.pot0) / max(1, len(h.actions)) for h in hands if h.actions
    ]
    features.extend(_quantiles(pot_growth, (50, 90)))
    violations = sum(1 for a in all_actions if a["pot_after"] < a["pot_before"] - 1e-9)
    features.append(violations / n_actions)

    # --- E. stacks ----------------------------------------------------------
    stacks = [s for h in hands for s in h.stacks]
    features.extend(_quantiles(stacks, (10, 50, 90)))
    features.append(float(np.std(stacks)) if stacks else 0.0)
    features.append(_collision(list(Counter(stacks).values()), len(stacks)))
    round_stacks = sum(1 for s in stacks if abs(s - round(s / 10.0) * 10.0) < 0.05)
    features.append(round_stacks / max(1, len(stacks)))
    hand_mean_stacks = [float(np.mean(h.stacks)) for h in hands if h.stacks]
    features.append(float(np.std(hand_mean_stacks)) if hand_mean_stacks else 0.0)

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
    features.append(float(np.mean(hero_amts)) if hero_amts else 0.0)

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
    return names


FEATURE_NAMES: List[str] = _build_feature_names()


def extract_features_matrix(chunks: List[List[Dict[str, Any]]]) -> np.ndarray:
    """Feature matrix for a full request: absolute + batch-relative blocks.

    The validator sends every chunk of the current eval window in a single
    request, so per-request medians act as an unsupervised drift anchor:
    the second block is each chunk's offset from the request median.
    """
    if not chunks:
        return np.zeros((0, 2 * len(FEATURE_NAMES)), dtype=float)
    absolute = np.vstack([extract_chunk_features(chunk) for chunk in chunks])
    if absolute.shape[0] >= 3:
        relative = absolute - np.median(absolute, axis=0)
    else:
        # Too few chunks for a meaningful batch anchor: no relative signal.
        relative = np.zeros_like(absolute)
    return np.hstack([absolute, relative])


ALL_FEATURE_NAMES: List[str] = FEATURE_NAMES + [f"rel_{n}" for n in FEATURE_NAMES]
