# v5 ORB NQ Backtest

Backtest the v5 ORB strategy (the same engine hooked up to Topstep/ProjectX live
accounts) on 10 years of 1-minute NQ futures data.

## Strategy changes

- `max_entries=2` per timeframe (was 4 in the live YM bot).
- NQ tick value ($20/point) and baseline index set to the dataset start.
- Everything else is copied verbatim from the v5 pure-math engine.

## Run locally (max 3 cores)

```bash
cd projects/trading/v5_orb_nq_backtest
python3 scripts/run_local.py \
    --input /path/to/NQ_1min.csv \
    --output results/final_report.json \
    --max-entries 2 \
    --max-contracts 5 \
    --workers 3
```

## Run one chunk

```bash
python3 scripts/run_chunk.py \
    --input /path/to/NQ_1min.csv \
    --start-date 2016-06-01 \
    --end-date 2016-06-30 \
    --output results/chunk_00.json \
    --max-entries 2
```

## Aggregate

```bash
python3 scripts/aggregate.py \
    --glob "results/chunk_*.json" \
    --output results/final_report.json
```

## GitHub Actions

`.github/workflows/v5_orb_nq_backtest.yml` runs 20 parallel workers in GitHub
Actions. It downloads NQ data from the configured rclone remote, splits the 10-year
range into 20 chunks, runs each chunk, and aggregates the results.

Required secrets match the `topstep-strats` project:

- `RCLONE_CONFIG_AKPANBRAIN_TYPE`
- `RCLONE_CONFIG_AKPANBRAIN_PROVIDER`
- `RCLONE_CONFIG_AKPANBRAIN_ACCESS_KEY_ID`
- `RCLONE_CONFIG_AKPANBRAIN_SECRET_ACCESS_KEY`
- `RCLONE_CONFIG_AKPANBRAIN_ENDPOINT`
- `RCLONE_CONFIG_AKPANBRAIN_REGION`
