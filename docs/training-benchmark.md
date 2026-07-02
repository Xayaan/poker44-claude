# Poker44 Training Benchmark

Public benchmark guide for Poker44 subnet `126`.

## Purpose

Poker44 provides a public training benchmark for miner development. Use it to:

- test your benchmark parser;
- build and validate feature pipelines;
- train and compare detection models;
- run regression tests across model versions;
- calibrate model outputs against labeled chunk data.

The benchmark is a development dataset for model training, validation, parser
testing, and regression testing.

## API Base

```text
https://api.poker44.net/api/v1/benchmark
```

## Endpoints

```text
GET /api/v1/benchmark
GET /api/v1/benchmark/releases
GET /api/v1/benchmark/chunks?sourceDate=YYYY-MM-DD
GET /api/v1/benchmark/chunks/:chunkId
```

## Status

`GET /api/v1/benchmark` returns aggregate availability:

- `releaseVersion`
- `schemaVersion`
- `releaseType`
- `totalChunks`
- `totalHands`
- `latestSourceDate`
- `latestReleasedAt`
- `currentUtcDate`
- `autoRelease`

Example:

```bash
curl -sS https://api.poker44.net/api/v1/benchmark
```

## Releases

`GET /api/v1/benchmark/releases` returns available benchmark dates.

Common query parameters:

- `limit`: number of releases to return.
- `before`: optional `YYYY-MM-DD` cursor for pagination.

Example:

```bash
curl -sS 'https://api.poker44.net/api/v1/benchmark/releases?limit=30'
```

Each release includes:

- `sourceDate`
- `releaseVersion`
- `schemaVersion`
- `chunkCount`
- `handCount`
- `releasedAt`
- `humanExampleCount`
- `syntheticBotExampleCount`
- `audit`
- `metadata`

## Chunks

`GET /api/v1/benchmark/chunks?sourceDate=YYYY-MM-DD` returns chunk payloads for
one release date.

Common query parameters:

- `sourceDate`: required release date in `YYYY-MM-DD` format.
- `limit`: number of chunks to return.
- `cursor`: optional pagination cursor.
- `split`: optional `train` or `validation`.

Example:

```bash
curl -sS 'https://api.poker44.net/api/v1/benchmark/chunks?sourceDate=2026-06-10&limit=24'
```

Each chunk includes:

- `chunkId`
- `chunkHash`
- `sourceDate`
- `releaseVersion`
- `split`
- `handCount`
- `batchCount`
- `chunks`
- `groundTruth`
- `groundTruthLabels`
- `metadata`

## Model Input

The `chunks` field is the miner-visible model input. It is a list of chunk
groups. Each group contains one or more poker hands.

Miners should produce one prediction per chunk group, matching the order of
`chunks`.

Current releases use at least 30 hands per chunk group.

The labels are returned separately:

- `groundTruth`: numeric labels, where `1` means bot and `0` means human.
- `groundTruthLabels`: string labels, `bot` or `human`.

Do not read labels from individual hand objects.

Minimal validation example:

```python
import requests

base_url = "https://api.poker44.net/api/v1/benchmark"
source_date = "2026-06-10"

payload = requests.get(
    f"{base_url}/chunks",
    params={"sourceDate": source_date, "limit": 100},
    timeout=30,
).json()["data"]

for chunk in payload["chunks"]:
    model_inputs = chunk["chunks"]
    labels = chunk["groundTruth"]

    predictions = model.predict_proba(model_inputs)

    assert len(predictions) == len(labels)
    assert all(0.0 <= score <= 1.0 for score in predictions)
```

## Hand Fields

Hands may include:

- `hand_id`
- `metadata`
- `players`
- `streets`
- `actions`
- `outcome`

Action records may include:

- `action_id`
- `street`
- `actor_seat`
- `action_type`
- `amount`
- `raise_to`
- `call_to`
- `normalized_amount_bb`
- `pot_before`
- `pot_after`

Code should tolerate missing optional fields and empty arrays.

## Training Guidance

Use each chunk group as one training example. The target is the matching entry
in `groundTruth`, in the same array position.

Recommended practices:

- keep release dates separate when creating train, validation, and local test
  sets;
- use the returned `split` field when it is present;
- train across multiple release dates instead of fitting one date tightly;
- save `sourceDate`, `releaseVersion`, `schemaVersion`, `chunkId`, and
  `chunkHash` with every local experiment;
- cache downloaded JSON by `chunkHash` so experiments are reproducible;
- ignore unknown response fields so clients keep working as the schema expands;
- avoid using identifiers such as `hand_id` or `chunkId` as model features.

Useful metrics:

- ROC AUC for ranking quality;
- average precision for bot-class retrieval;
- log loss or Brier score for probability calibration;
- per-release metrics to catch overfitting to one benchmark date.

## Recommended Workflow

1. Fetch release dates from `/releases`.
2. Download chunks by `sourceDate`.
3. Cache raw responses and record `chunkHash`.
4. Split by release date and by the returned `split` field when present.
5. Build features only from miner-visible hand and action data.
6. Train on `train` chunks.
7. Tune and compare on `validation` chunks.
8. Keep a held-out local set for model regression tests.
9. Track performance by release date and model version.

## Common Mistakes

- Producing one prediction per hand instead of one prediction per chunk group.
- Reordering `chunks` before pairing predictions with `groundTruth`.
- Training and validating on the same release date only.
- Treating optional fields as always present.
- Using IDs, dates, hashes, or pagination order as predictive features.
- Assuming every chunk group has the same number of hands or actions.

## Notes

- New releases may be added over time.
- Response fields may expand, so clients should ignore unknown fields.
- The chunk order and label order are significant.
- Avoid tuning a model against a single release only.
- Prefer testing across multiple release dates.
- Use the benchmark for model development, parser testing, feature validation,
  and regression testing.
