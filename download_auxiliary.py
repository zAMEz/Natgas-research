"""Download the time-safe market inputs used by the augmented NG model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from getpass import getpass
import os
from pathlib import Path

import databento as db
import pandas as pd


START = "2025-09-24"
END = "2026-09-24"
SCHEMA = "ohlcv-1m"


@dataclass(frozen=True)
class Series:
    name: str
    dataset: str
    symbol: str
    stype_in: str


SERIES = (
    Series("ng_m1", "GLBX.MDP3", "NG.c.0", "continuous"),
    Series("ng_m2", "GLBX.MDP3", "NG.c.1", "continuous"),
    Series("ng_m3", "GLBX.MDP3", "NG.c.2", "continuous"),
    Series("wti", "GLBX.MDP3", "CL.v.0", "continuous"),
    Series("lng_equity", "XNYS.PILLAR", "LNG", "raw_symbol"),
)


def get_client() -> db.Historical:
    key = os.getenv("DATABENTO_API_KEY") or getpass(
        "Databento API key (input is hidden): "
    ).strip()
    if not key.startswith("db-"):
        raise ValueError("A Databento key must start with 'db-'.")
    return db.Historical(key)


def kwargs(series: Series) -> dict[str, object]:
    return {
        "dataset": series.dataset,
        "symbols": [series.symbol],
        "stype_in": series.stype_in,
        "schema": SCHEMA,
        "start": START,
        "end": END,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-cost", type=float, default=None)
    parser.add_argument("--output-dir", default="data/auxiliary")
    args = parser.parse_args()

    client = get_client()
    estimates: list[tuple[Series, float, int]] = []
    for series in SERIES:
        request = kwargs(series)
        cost = float(client.metadata.get_cost(**request))
        records = int(client.metadata.get_record_count(**request))
        estimates.append((series, cost, records))
        print(
            f"{series.name:12s} {series.symbol:8s} "
            f"{records:>9,d} records  ${cost:,.4f}"
        )
    total_cost = sum(item[1] for item in estimates)
    print(f"Total estimated cost: ${total_cost:,.4f} USD")

    if not args.download:
        print("Cost check only. Re-run with --download to fetch the data.")
        return
    if args.max_cost is not None and total_cost > args.max_cost:
        raise SystemExit(
            f"Estimated cost ${total_cost:,.4f} exceeds "
            f"--max-cost ${args.max_cost:,.4f}."
        )

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for series, _, records in estimates:
        if records == 0:
            print(f"Skipping {series.name}: no records")
            continue
        stem = f"{series.name}_ohlcv_1m_{START}_{END}"
        dbn_path = output / f"{stem}.dbn.zst"
        parquet_path = output / f"{stem}.parquet"
        print(f"Downloading {series.symbol} ...", flush=True)
        client.timeseries.get_range(**kwargs(series), path=dbn_path)
        frame = db.DBNStore.from_file(dbn_path).to_df().reset_index()
        timestamp = "ts_event" if "ts_event" in frame else frame.columns[0]
        frame = frame.rename(columns={timestamp: "ts_event"})
        frame["ts_event"] = pd.to_datetime(frame["ts_event"], utc=True)
        frame = frame.sort_values("ts_event").drop_duplicates("ts_event")
        frame.to_parquet(parquet_path, index=False)
        print(
            f"Saved {parquet_path} ({len(frame):,} rows, "
            f"{frame['ts_event'].min()} to {frame['ts_event'].max()})"
        )


if __name__ == "__main__":
    main()
