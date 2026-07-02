"""Poker44 miner: chunk-level bot detection with a trained ensemble.

Scores every chunk of a request in one pass (batch-relative features need the
whole request) with a gradient-boosted ensemble trained on the public Poker44
training benchmark. Falls back to a deterministic heuristic when the model
artifact cannot be loaded, so responses are never dropped.
"""

# from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Tuple

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.detection.model import get_default_model
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse


class Miner(BaseMinerNeuron):
    """
    Ensemble detection miner.

    Uses chunk-size-invariant behavioral features (action-sequence collision
    statistics, bet-size distribution shape, pot/stack dynamics, action
    bigrams) plus batch-relative drift anchoring, scored by a
    HistGradientBoosting ensemble blended with logistic regression.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        bt.logging.info("🤖 Poker44 ensemble detection miner starting")
        repo_root = Path(__file__).resolve().parents[1]
        self.detector = get_default_model()
        if self.detector.ready:
            bt.logging.info("Detection ensemble loaded (poker44-detection-v1).")
        else:
            bt.logging.warning(
                f"Detection artifact unavailable ({self.detector.load_error}); "
                "serving heuristic fallback. Run research/train_final.py to rebuild."
            )
        model_files = [
            Path(__file__).resolve(),
            repo_root / "poker44" / "detection" / "features.py",
            repo_root / "poker44" / "detection" / "model.py",
        ]
        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=[p for p in model_files if p.exists()],
            defaults={
                "model_name": "poker44-seqcollision-ensemble",
                "model_version": "1.0.0",
                "framework": "scikit-learn",
                "license": "MIT",
                "repo_url": "https://github.com/Xayaan/poker44-claude",
                "notes": (
                    "HistGradientBoosting ensemble + logistic blend over "
                    "chunk-size-invariant behavioral features (sequence-collision "
                    "U-statistics, bet-size histograms, pot/stack dynamics, action "
                    "bigrams) with batch-relative drift anchoring."
                ),
                "open_source": True,
                "inference_mode": "local",
                "training_data_statement": (
                    "Trained exclusively on the public Poker44 training benchmark "
                    "(api.poker44.net/api/v1/benchmark), all release dates, "
                    "projected through the validator payload canonicalizer."
                ),
                "training_data_sources": [
                    "https://api.poker44.net/api/v1/benchmark",
                ],
                "private_data_attestation": (
                    "This miner does not train on validator-only evaluation data."
                ),
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup(repo_root)
        
        # # Attach handlers after initialization
        # self.axon.attach(
        #     forward_fn = self.forward,
        #     blacklist_fn = self.blacklist,
        #     priority_fn = self.priority,
        # )
        # bt.logging.info("Attaching forward function to miner axon.")
        
        bt.logging.info(f"Axon created: {self.axon}")

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info("Open-sourced miner manifest standard active for this miner.")
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"open_source={self.model_manifest.get('open_source')}"
        )
        bt.logging.info(
            f"Manifest digest={self.manifest_digest} "
            f"inference_mode={self.model_manifest.get('inference_mode', '')}"
        )
        bt.logging.info(
            "Miner prep docs available | "
            f"miner_doc={repo_root / 'docs' / 'miner.md'}"
        )

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        """Assign one bot-risk score per chunk using the trained ensemble."""
        chunks = synapse.chunks or []
        started = time.monotonic()
        try:
            scores = self.detector.score_chunks(chunks)
        except Exception as exc:  # noqa: BLE001 - never drop a response
            bt.logging.warning(f"Ensemble scoring failed ({exc}); using heuristic.")
            scores = [self.score_chunk(chunk) for chunk in chunks]
        if len(scores) != len(chunks):
            bt.logging.warning(
                f"Score count mismatch ({len(scores)} vs {len(chunks)}); using heuristic."
            )
            scores = [self.score_chunk(chunk) for chunk in chunks]
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        elapsed = time.monotonic() - started
        total_hands = sum(len(c) for c in chunks if isinstance(c, list))
        bt.logging.info(
            f"Scored {len(chunks)} chunks ({total_hands} hands) in {elapsed:.2f}s "
            f"with {'ensemble' if self.detector.ready else 'heuristic fallback'}."
        )
        return synapse

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @classmethod
    def _score_hand(cls, hand: dict) -> float:
        actions = hand.get("actions") or []
        players = hand.get("players") or []
        streets = hand.get("streets") or []
        outcome = hand.get("outcome") or {}

        action_counts = Counter(action.get("action_type") for action in actions)
        meaningful_actions = max(
            1,
            sum(
                action_counts.get(kind, 0)
                for kind in ("call", "check", "bet", "raise", "fold")
            ),
        )

        call_ratio = action_counts.get("call", 0) / meaningful_actions
        check_ratio = action_counts.get("check", 0) / meaningful_actions
        fold_ratio = action_counts.get("fold", 0) / meaningful_actions
        raise_ratio = action_counts.get("raise", 0) / meaningful_actions
        street_depth = len(streets) / 3.0
        showdown_flag = 1.0 if outcome.get("showdown") else 0.0

        player_count_signal = 0.0
        if players:
            player_count_signal = (6 - min(len(players), 6)) / 4.0

        score = 0.0
        score += 0.32 * street_depth
        score += 0.22 * showdown_flag
        score += 0.18 * cls._clamp01(call_ratio / 0.35)
        score += 0.12 * cls._clamp01(check_ratio / 0.30)
        score += 0.08 * cls._clamp01(player_count_signal)
        score -= 0.18 * cls._clamp01(fold_ratio / 0.55)
        score -= 0.10 * cls._clamp01(raise_ratio / 0.20)

        return cls._clamp01(score)

    @classmethod
    def score_chunk(cls, chunk: list[dict]) -> float:
        if not chunk:
            return 0.5

        hand_scores = [cls._score_hand(hand) for hand in chunk]
        avg_score = sum(hand_scores) / len(hand_scores)

        return round(cls._clamp01(avg_score), 6)

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        """Determine whether to blacklist incoming requests."""
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        """Assign priority based on caller's stake."""
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Poker44 ensemble detection miner running...")
        while True:
            bt.logging.info(f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}")
            time.sleep(5 * 60)
