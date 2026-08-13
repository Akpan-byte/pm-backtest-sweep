#!/usr/bin/env python3
import asyncio, os, sys, json
from datetime import datetime, timedelta, UTC
from pathlib import Path
import polars as pl

sys.path.insert(0, "/config/projects/trading/live-bots/execution_kernel/src")
from cryptography.fernet import Fernet

def creds():
    account_name = "PRAC-V2-622570-65415932"
    username = "theakpanobong@gmail.com"
    key = Path("/config/projects/trading/execution-kernel/.env/credential_key").read_bytes().strip()
    store = json.loads(Path("/config/projects/trading/execution-kernel/.env/credentials.json").read_text())
    entry = store[account_name]["projectx_api_key"]
    api_key = Fernet(key).decrypt(entry["ciphertext"].encode("ascii")).decode("utf-8")
    return account_name, api_key, username

async def main():
    account_name, api_key, username = creds()
    os.environ["PROJECT_X_USERNAME"] = username
    os.environ["PROJECT_X_API_KEY"] = api_key
    os.environ["PROJECT_X_ACCOUNT_NAME"] = account_name
    from project_x_py import ProjectX
    px = ProjectX(username=username, api_key=api_key, account_name=account_name)
    await asyncio.wait_for(px.authenticate(), timeout=30.0)

    now = datetime.now(UTC)
    frames = []
    for i in range(62, -1, -1):
        start = (now - timedelta(days=i+1)).replace(tzinfo=UTC)
        end = (now - timedelta(days=i)).replace(tzinfo=UTC)
        try:
            bars = await px.get_bars("NQ", start_time=start, end_time=end, interval=1, unit=2, limit=100000)
            if bars is None or len(bars) == 0:
                continue
            df = bars.select(["timestamp","open","high","low","close","volume"])
            df = df.with_columns(pl.col("timestamp").dt.convert_time_zone("UTC"))
            frames.append(df)
            print(f"{start.date()} bars={len(df)}", flush=True)
        except Exception as e:
            print(f"{start.date()} ERR {type(e).__name__}: {str(e)[:100]}", flush=True)
        await asyncio.sleep(0.25)

    if frames:
        full = pl.concat(frames).unique(subset="timestamp").sort("timestamp")
        out = Path("/config/projects/trading/flexing_joe_orb/market_data/NQ_topstep_api_1min.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        full.write_csv(out)
        print(f"\nTOTAL: {len(full)} rows -> {out}")
        print(f"range: {full['timestamp'].min()} -> {full['timestamp'].max()}")
        print(f"unique dates: {len(full['timestamp'].dt.date().unique())}")

asyncio.run(main())
