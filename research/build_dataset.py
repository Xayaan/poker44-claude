"""Build the validator-parity training dataset.

Loads every cached benchmark release, projects each hand through the exact
validator transform (`prepare_hand_for_miner`, deploy 0.1.33), and stores the
transformed chunk groups with labels/date/split metadata as one pickle.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from poker44.validator.payload_view import prepare_hand_for_miner  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_PATH = Path(__file__).resolve().parent / "dataset.pkl"


def main() -> None:
    records = []
    for path in sorted(DATA_DIR.glob("*.json")):
        payload = json.loads(path.read_text())
        date = payload["sourceDate"]
        for chunk_payload in payload["chunks"]:
            split = chunk_payload.get("split") or ""
            chunk_id = chunk_payload.get("chunkId") or ""
            groups = chunk_payload.get("chunks") or []
            labels = chunk_payload.get("groundTruth") or []
            if len(groups) != len(labels):
                raise RuntimeError(f"{path.name}: group/label length mismatch")
            for group_idx, (group, label) in enumerate(zip(groups, labels)):
                transformed = [prepare_hand_for_miner(hand) for hand in group]
                records.append(
                    {
                        "date": date,
                        "split": split,
                        "chunk_id": chunk_id,
                        "group_idx": group_idx,
                        "label": int(label),
                        "hands": transformed,
                        "raw_hands": group,
                    }
                )
    with open(OUT_PATH, "wb") as fh:
        pickle.dump(records, fh)
    n_bot = sum(r["label"] for r in records)
    print(
        f"{len(records)} chunk groups ({n_bot} bot / {len(records) - n_bot} human) "
        f"across {len({r['date'] for r in records})} dates -> {OUT_PATH.name}"
    )


if __name__ == "__main__":
    main()
