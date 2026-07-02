"""Tests for the detection feature extractor and production model wrapper."""

import json
import unittest

import numpy as np

from poker44.detection.features import (
    ALL_FEATURE_NAMES,
    FEATURE_NAMES,
    extract_chunk_features,
    extract_features_matrix,
)
from poker44.detection.model import DetectionModel, get_default_model


def _make_hand(seed: int = 0, n_actions: int = 5) -> dict:
    types = ["fold", "check", "call", "bet", "raise"]
    actions = []
    for i in range(n_actions):
        t = types[(seed + i) % len(types)]
        amount = 0.0 if t in ("fold", "check") else round(0.02 * (2 + (seed + i) % 30), 4)
        actions.append(
            {
                "action_id": str(i + 1),
                "street": ["preflop", "flop", "turn", "river"][min(3, i // 2)],
                "actor_seat": (seed + i) % 5 + 1,
                "action_type": t,
                "amount": amount,
                "raise_to": amount if t == "raise" else None,
                "call_to": amount if t == "call" else None,
                "normalized_amount_bb": amount / 0.02,
                "pot_before": 0.3 + 0.1 * i,
                "pot_after": 0.3 + 0.1 * (i + 1),
            }
        )
    return {
        "metadata": {
            "game_type": "Hold'em",
            "limit_type": "No Limit",
            "max_seats": 6,
            "hero_seat": seed % 5 + 1,
            "sb": 0.01,
            "bb": 0.02,
            "ante": 0.0,
        },
        "players": [
            {
                "player_uid": f"seat_{s}",
                "seat": s,
                "starting_stack": 4.0 + 0.11 * ((seed + s) % 7),
                "hole_cards": None,
                "showed_hand": False,
            }
            for s in range(1, 6)
        ],
        "streets": [],
        "actions": actions,
        "outcome": {
            "winners": [],
            "payouts": {},
            "total_pot": 0.0,
            "rake": 0.0,
            "result_reason": "",
            "showdown": False,
        },
    }


def _make_chunk(seed: int = 0, n_hands: int = 32) -> list:
    return [_make_hand(seed * 100 + i, n_actions=3 + (seed + i) % 6) for i in range(n_hands)]


class FeatureExtractionTests(unittest.TestCase):
    def test_feature_vector_shape_and_finiteness(self):
        vec = extract_chunk_features(_make_chunk(1))
        self.assertEqual(len(vec), len(FEATURE_NAMES))
        self.assertTrue(np.all(np.isfinite(vec)))

    def test_deterministic(self):
        chunk = _make_chunk(2)
        a = extract_chunk_features(chunk)
        b = extract_chunk_features(chunk)
        self.assertTrue(np.array_equal(a, b))

    def test_json_round_trip_identical(self):
        chunk = _make_chunk(3)
        round_tripped = json.loads(json.dumps(chunk))
        a = extract_chunk_features(chunk)
        b = extract_chunk_features(round_tripped)
        self.assertTrue(np.array_equal(a, b))

    def test_empty_and_malformed_inputs(self):
        for chunk in ([], [{}], [None], [{"actions": "junk", "players": 3}], ["x"]):
            vec = extract_chunk_features(chunk)  # type: ignore[arg-type]
            self.assertEqual(len(vec), len(FEATURE_NAMES))
            self.assertTrue(np.all(np.isfinite(vec)))

    def test_chunk_size_invariance_of_scale(self):
        """Doubling a chunk's hands must not systematically shift features."""
        chunk = _make_chunk(4, n_hands=30)
        doubled = chunk + chunk
        a = extract_chunk_features(chunk)
        b = extract_chunk_features(doubled)
        # Collision statistics change when duplicating hands (real dupes),
        # but rate/quantile features must be identical.
        collision_idx = {
            i for i, n in enumerate(FEATURE_NAMES) if n.startswith("coll_") or n == "stack_coll" or n == "pot0_coll"
        }
        for i, name in enumerate(FEATURE_NAMES):
            if i in collision_idx:
                continue
            self.assertAlmostEqual(a[i], b[i], places=9, msg=name)

    def test_matrix_includes_relative_block(self):
        chunks = [_make_chunk(s) for s in range(5)]
        matrix = extract_features_matrix(chunks)
        self.assertEqual(matrix.shape, (5, len(ALL_FEATURE_NAMES)))
        n = len(FEATURE_NAMES)
        median = np.median(matrix[:, :n], axis=0)
        self.assertTrue(np.allclose(matrix[:, n:], matrix[:, :n] - median))

    def test_matrix_small_batch_zero_relative(self):
        matrix = extract_features_matrix([_make_chunk(0), _make_chunk(1)])
        n = len(FEATURE_NAMES)
        self.assertTrue(np.all(matrix[:, n:] == 0.0))


class DetectionModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = get_default_model()

    def test_artifact_loads(self):
        self.assertTrue(self.model.ready, self.model.load_error)

    def test_score_contract(self):
        chunks = [_make_chunk(s) for s in range(8)]
        scores = self.model.score_chunks(chunks)
        self.assertEqual(len(scores), len(chunks))
        for s in scores:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_deterministic_scores(self):
        chunks = [_make_chunk(s) for s in range(4)]
        self.assertEqual(self.model.score_chunks(chunks), self.model.score_chunks(chunks))

    def test_empty_request(self):
        self.assertEqual(self.model.score_chunks([]), [])

    def test_single_chunk_request(self):
        scores = self.model.score_chunks([_make_chunk(0)])
        self.assertEqual(len(scores), 1)
        self.assertTrue(0.0 <= scores[0] <= 1.0)

    def test_garbage_chunks_get_neutral_scores(self):
        chunks = [_make_chunk(0), [], [None, "x"], [{}]]
        scores = self.model.score_chunks(chunks)  # type: ignore[arg-type]
        self.assertEqual(len(scores), 4)
        self.assertEqual(scores[1], 0.5)
        for s in scores:
            self.assertTrue(0.0 <= s <= 1.0)

    def test_fallback_heuristic_when_artifact_missing(self):
        broken = DetectionModel(model_path="/nonexistent/model.pkl")
        self.assertFalse(broken.ready)
        chunks = [_make_chunk(s) for s in range(3)]
        scores = broken.score_chunks(chunks)
        self.assertEqual(len(scores), 3)
        for s in scores:
            self.assertTrue(0.0 <= s <= 1.0)


if __name__ == "__main__":
    unittest.main()
