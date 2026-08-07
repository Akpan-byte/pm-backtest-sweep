#!/usr/bin/env python3
import pandas as pd

def load_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['timestamp'] if 'timestamp' in pd.read_csv(path, nrows=0).columns else [0])
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    else:
        df = df.set_index(df.columns[0])
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize('America/New_York')
    else:
        df.index = df.index.tz_convert('America/New_York')
    return df
