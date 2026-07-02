"""End-to-end bittensor wire test (axon <-> dendrite loopback).

Runs the production Miner.forward code behind a real bt.axon and queries it
with a real bt.dendrite using the exact DetectionSynapse contract, exactly as
the validator's forward cycle does. Verifies:
  - chunks survive HTTP serialization,
  - risk_scores come back aligned and in range,
  - wire scores match direct in-process model scores,
  - the exact validator reward on the returned scores,
  - a live-scale payload (~9k hands) within the 180s timeout.

Requires the py3.12 env with bittensor (.venv-bt).
"""

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import bittensor as bt  # noqa: E402

from poker44.detection.model import get_default_model  # noqa: E402
from poker44.score.scoring import reward  # noqa: E402
from poker44.validator.payload_view import prepare_hand_for_miner  # noqa: E402
from poker44.validator.synapse import DetectionSynapse  # noqa: E402
from neurons.miner import Miner  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
PORT = 8931


class MinerShim:
    """Carries just the state Miner.forward needs (no chain access)."""

    def __init__(self):
        self.detector = get_default_model()
        assert self.detector.ready, self.detector.load_error
        self.model_manifest = {"model_name": "poker44-seqcollision-ensemble", "model_version": "1.0.0"}

    def score_chunk(self, chunk):
        return Miner.score_chunk(chunk)


async def main() -> None:
    date = sorted(p.stem for p in DATA_DIR.glob("*.json"))[-1]
    payload = json.loads((DATA_DIR / f"{date}.json").read_text())
    batches, labels = [], []
    for chunk_payload in payload["chunks"]:
        for group, label in zip(chunk_payload["chunks"], chunk_payload["groundTruth"]):
            batches.append(group)
            labels.append(int(label))
    chunks = [[prepare_hand_for_miner(h) for h in b] for b in batches]
    labels = np.array(labels)
    print(f"date {date}: {len(chunks)} chunks, {sum(len(c) for c in chunks)} hands")

    tmp = tempfile.mkdtemp(prefix="poker44-wire-")
    wallet_miner = bt.Wallet(name="wiretest-miner", hotkey="default", path=tmp)
    wallet_miner.create_new_coldkey(use_password=False, overwrite=True)
    wallet_miner.create_new_hotkey(use_password=False, overwrite=True)
    wallet_validator = bt.Wallet(name="wiretest-vali", hotkey="default", path=tmp)
    wallet_validator.create_new_coldkey(use_password=False, overwrite=True)
    wallet_validator.create_new_hotkey(use_password=False, overwrite=True)

    shim = MinerShim()

    async def forward_fn(synapse: DetectionSynapse) -> DetectionSynapse:
        return await Miner.forward(shim, synapse)

    axon = bt.Axon(wallet=wallet_miner, ip="127.0.0.1", port=PORT, external_ip="127.0.0.1")
    axon.attach(forward_fn=forward_fn)
    axon.start()
    print(f"axon serving on 127.0.0.1:{PORT}")

    dendrite = bt.Dendrite(wallet=wallet_validator)
    try:
        # --- normal request (one full eval window) -------------------------
        synapse = DetectionSynapse(chunks=chunks)
        t0 = time.monotonic()
        responses = await dendrite(
            axons=[axon.info()], synapse=synapse, timeout=180.0, deserialize=False
        )
        elapsed = time.monotonic() - t0
        resp = responses[0]
        scores = resp.risk_scores
        status = resp.dendrite.status_code if resp.dendrite else None
        assert scores is not None, f"no risk_scores returned (status={status})"
        assert len(scores) == len(chunks), f"{len(scores)} scores vs {len(chunks)} chunks"
        assert all(0.0 <= s <= 1.0 for s in scores)

        local = shim.detector.score_chunks(json.loads(json.dumps(chunks)))
        max_dev = max(abs(a - b) for a, b in zip(scores, local))
        rew, res = reward(np.array(scores), labels)
        print(
            f"wire ok in {elapsed:.2f}s | status={status} | "
            f"max |wire-local| = {max_dev:.2e}"
        )
        print(
            f"reward on wire scores: {rew:.3f} (AP={res['ap_score']:.3f} "
            f"R@5={res['bot_recall']:.3f})"
        )
        assert max_dev < 1e-9, "wire scores diverge from local scores"

        # --- live-scale request (~91 chunks x 100 hands) --------------------
        pool = [h for c in chunks for h in c]
        big = [[pool[(k * 100 + j) % len(pool)] for j in range(100)] for k in range(91)]
        t0 = time.monotonic()
        responses = await dendrite(
            axons=[axon.info()], synapse=DetectionSynapse(chunks=big), timeout=180.0,
            deserialize=False,
        )
        elapsed = time.monotonic() - t0
        big_scores = responses[0].risk_scores
        assert big_scores is not None and len(big_scores) == 91
        print(
            f"live-scale wire ok: 91 chunks x 100 hands in {elapsed:.2f}s "
            f"(timeout budget 180s)"
        )

        # --- degenerate request ---------------------------------------------
        responses = await dendrite(
            axons=[axon.info()],
            synapse=DetectionSynapse(chunks=[[], [{"actions": None}]]),
            timeout=30.0,
            deserialize=False,
        )
        weird = responses[0].risk_scores
        assert weird is not None and len(weird) == 2
        print(f"degenerate request ok: {weird}")
        print("\nWIRE TEST PASSED")
    finally:
        axon.stop()


if __name__ == "__main__":
    asyncio.run(main())
