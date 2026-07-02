"""Download and cache all Poker44 benchmark releases locally.

Caches one JSON file per sourceDate under research/data/. Re-running only
fetches missing dates unless --refresh is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

BASE_URL = "https://api.poker44.net/api/v1/benchmark"
DATA_DIR = Path(__file__).resolve().parent / "data"


def _get(path: str, params: dict | None = None, retries: int = 4) -> dict:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(f"{BASE_URL}{path}", params=params, timeout=60)
            resp.raise_for_status()
            body = resp.json()
            if isinstance(body, dict) and body.get("success") and "data" in body:
                return body["data"]
            return body
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET {path} failed after {retries} attempts: {last_exc}")


def list_release_dates() -> list[str]:
    dates: list[str] = []
    before: str | None = None
    while True:
        params: dict = {"limit": 180}
        if before:
            params["before"] = before
        data = _get("/releases", params)
        releases = data.get("releases", [])
        if not releases:
            break
        batch = [r["sourceDate"] for r in releases]
        dates.extend(batch)
        if len(releases) < 180:
            break
        before = min(batch)
    return sorted(set(dates))


def fetch_date(source_date: str) -> dict:
    """Fetch every chunk payload for one release date, following pagination."""
    all_chunks: list[dict] = []
    cursor: str | None = None
    meta: dict = {}
    while True:
        params: dict = {"sourceDate": source_date, "limit": 48}
        if cursor:
            params["cursor"] = cursor
        data = _get("/chunks", params)
        meta = {k: v for k, v in data.items() if k != "chunks"}
        chunks = data.get("chunks", [])
        all_chunks.extend(chunks)
        cursor = data.get("nextCursor") or (data.get("pagination") or {}).get("nextCursor")
        if not cursor or not chunks:
            break
    return {"sourceDate": source_date, "meta": meta, "chunks": all_chunks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-download cached dates")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dates = list_release_dates()
    print(f"{len(dates)} release dates: {dates[0]} .. {dates[-1]}")

    for date in dates:
        out = DATA_DIR / f"{date}.json"
        if out.exists() and not args.refresh:
            continue
        payload = fetch_date(date)
        n_groups = sum(len(c.get("chunks", [])) for c in payload["chunks"])
        n_hands = sum(c.get("handCount", 0) for c in payload["chunks"])
        out.write_text(json.dumps(payload))
        print(f"{date}: {len(payload['chunks'])} payloads, {n_groups} chunk groups, {n_hands} hands -> {out.name}")

    print("done")


if __name__ == "__main__":
    main()
    sys.exit(0)
