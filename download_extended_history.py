"""Download a separate four-year Databento history without storing credentials."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from getpass import getpass
import os
from pathlib import Path

import databento as db
import pandas as pd

from download_natgas import Request, annotate_conditions, get_dataset_conditions, prepare_model_frame


@dataclass(frozen=True)
class Series:
    name: str
    dataset: str
    symbol: str
    stype_in: str


SERIES = (
    Series("ng_v0", "GLBX.MDP3", "NG.v.0", "continuous"),
    Series("ng_m1", "GLBX.MDP3", "NG.c.0", "continuous"),
    Series("ng_m2", "GLBX.MDP3", "NG.c.1", "continuous"),
    Series("ng_m3", "GLBX.MDP3", "NG.c.2", "continuous"),
    Series("wti", "GLBX.MDP3", "CL.v.0", "continuous"),
    Series("lng_equity", "XNYS.PILLAR", "LNG", "raw_symbol"),
)


def get_client() -> db.Historical:
    key = os.getenv("DATABENTO_API_KEY") or getpass("Databento API key (hidden): ").strip()
    if not key.startswith("db-"):
        raise ValueError("A Databento key must start with 'db-'.")
    return db.Historical(key)


def request_kwargs(series: Series, start: str, end: str) -> dict[str, object]:
    return {
        "dataset": series.dataset,
        "symbols": [series.symbol],
        "stype_in": series.stype_in,
        "schema": "ohlcv-1m",
        "start": start,
        "end": end,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2021-09-24")
    parser.add_argument("--end", default="2025-09-24")
    parser.add_argument("--output-dir", type=Path, default=Path("data/extended_2021_2025"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-cost", type=float, required=True)
    args = parser.parse_args()
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    if start >= end:
        parser.error("start must precede end")

    client = get_client()
    estimates = []
    for series in SERIES:
        kwargs = request_kwargs(series, args.start, args.end)
        cost = float(client.metadata.get_cost(**kwargs))
        count = int(client.metadata.get_record_count(**kwargs))
        estimates.append((series, cost, count))
        print(f"{series.name:12s} {count:>12,d} records  USD {cost:,.4f}", flush=True)
    total = sum(cost for _, cost, _ in estimates)
    print(f"Total estimated cost: USD {total:,.4f}", flush=True)
    if total > args.max_cost:
        raise SystemExit(f"Estimate exceeds hard cap of USD {args.max_cost:,.2f}; nothing downloaded")
    if not args.download:
        print("Estimate only; nothing downloaded")
        return

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    date_stem = f"{args.start}_{args.end}"
    for series, _, expected in estimates:
        stem = f"{series.name}_ohlcv_1m_{date_stem}"
        dbn_path = output / f"{stem}.dbn.zst"
        parquet_path = output / f"{stem}.parquet"
        print(f"Downloading {series.symbol} ({expected:,} records) ...", flush=True)
        client.timeseries.get_range(**request_kwargs(series, args.start, args.end), path=dbn_path)
        frame = db.DBNStore.from_file(dbn_path).to_df().reset_index()
        timestamp = "ts_event" if "ts_event" in frame else frame.columns[0]
        frame = frame.rename(columns={timestamp: "ts_event"})
        frame["ts_event"] = pd.to_datetime(frame["ts_event"], utc=True)
        frame = frame.sort_values("ts_event").drop_duplicates("ts_event").reset_index(drop=True)
        if len(frame) != expected:
            raise RuntimeError(f"{series.name}: expected {expected:,} records, got {len(frame):,}")
        frame.to_parquet(parquet_path, index=False)
        if series.name == "ng_v0":
            request = Request(start=start, end=end)
            conditions = get_dataset_conditions(client, request)
            conditions.to_csv(output / f"{stem}_conditions.csv", index=False)
            model = annotate_conditions(prepare_model_frame(frame.set_index("ts_event")), conditions)
            model.to_parquet(output / f"{stem}_model.parquet", index=False)
        print(f"Saved {parquet_path.name}: {len(frame):,} rows", flush=True)


if __name__ == "__main__":
    main()
