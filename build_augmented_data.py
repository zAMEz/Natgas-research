"""Join auxiliary markets and point-in-time EIA releases to the NG target bars."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
START = "2025-09-24"
END = "2026-09-24"
TARGET = DATA_DIR / f"ng_v0_ohlcv_1m_{START}_{END}_model.parquet"
AUXILIARY = DATA_DIR / "auxiliary"
OUTPUT = DATA_DIR / f"ng_augmented_ohlcv_1m_{START}_{END}_model.parquet"

MARKETS = {
    "ng_m1": (60, AUXILIARY / f"ng_m1_ohlcv_1m_{START}_{END}.parquet"),
    "ng_m2": (60, AUXILIARY / f"ng_m2_ohlcv_1m_{START}_{END}.parquet"),
    "ng_m3": (60, AUXILIARY / f"ng_m3_ohlcv_1m_{START}_{END}.parquet"),
    "wti": (10, AUXILIARY / f"wti_ohlcv_1m_{START}_{END}.parquet"),
    "lng_equity": (
        10,
        AUXILIARY / f"lng_equity_ohlcv_1m_{START}_{END}.parquet",
    ),
}

# The key is the storage week-ending Friday. Values are official holiday
# publication timestamps in America/New_York time during this sample.
EIA_OVERRIDES = {
    "2025-11-07": "2025-11-14 10:30",
    "2025-11-21": "2025-11-26 12:00",
    "2025-12-19": "2025-12-29 12:00",
    "2025-12-26": "2025-12-31 12:00",
}


def join_market(target: pd.DataFrame, name: str, tolerance: int, path: Path) -> pd.DataFrame:
    source = pd.read_parquet(path, columns=["ts_event", "instrument_id", "close"])
    source["ts_event"] = pd.to_datetime(source["ts_event"], utc=True)
    source = source.sort_values("ts_event").drop_duplicates("ts_event")
    source = source.rename(
        columns={
            "ts_event": f"{name}_source_time",
            "instrument_id": f"{name}_instrument_id",
            "close": f"{name}_close",
        }
    )
    joined = pd.merge_asof(
        target.sort_values("ts_event"),
        source,
        left_on="ts_event",
        right_on=f"{name}_source_time",
        direction="backward",
        tolerance=pd.Timedelta(minutes=tolerance),
    )
    age = (joined["ts_event"] - joined[f"{name}_source_time"]).dt.total_seconds() / 60
    joined[f"{name}_age_minutes"] = age.fillna(tolerance + 1).astype("float32")
    joined[f"{name}_available"] = joined[f"{name}_close"].notna().astype("int8")
    return joined.drop(columns=f"{name}_source_time")


def eia_releases(history_path: Path, revisions_path: Path) -> pd.DataFrame:
    history = pd.read_excel(history_path, sheet_name="weekly_net_changes", header=6)
    history = history.rename(
        columns={history.columns[0]: "week_ending", "Total Lower 48": "storage_change_bcf"}
    )[["week_ending", "storage_change_bcf"]]
    history["week_ending"] = pd.to_datetime(history["week_ending"], errors="coerce")
    history["storage_change_bcf"] = pd.to_numeric(
        history["storage_change_bcf"], errors="coerce"
    )
    history = history.dropna().sort_values("week_ending")

    # EIA's current history can contain later corrections. Its revisions file
    # preserves the originally published stock levels, so their weekly
    # differences are used where available instead of leaking a later revision.
    original = pd.read_excel(revisions_path, sheet_name="original_data", header=1)
    original = original.rename(
        columns={original.columns[0]: "week_ending", "Total Lower 48": "original_stock_bcf"}
    )[["week_ending", "original_stock_bcf"]]
    original["week_ending"] = pd.to_datetime(original["week_ending"], errors="coerce")
    original["original_stock_bcf"] = pd.to_numeric(
        original["original_stock_bcf"], errors="coerce"
    )
    original = original.dropna().sort_values("week_ending")
    original["original_storage_change_bcf"] = original["original_stock_bcf"].diff()
    original_change = original.set_index("week_ending")["original_storage_change_bcf"]
    replacement = history["week_ending"].map(original_change)
    history["storage_change_bcf"] = replacement.fillna(history["storage_change_bcf"])

    default_release = history["week_ending"] + pd.Timedelta(days=6, hours=10, minutes=30)
    local_naive = default_release.copy()
    for week_ending, release_time in EIA_OVERRIDES.items():
        local_naive.loc[history["week_ending"].eq(pd.Timestamp(week_ending))] = pd.Timestamp(
            release_time
        )
    history["release_time"] = (
        local_naive.dt.tz_localize("America/New_York", ambiguous="infer", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
        .astype("datetime64[ns, UTC]")
    )
    return history[["release_time", "week_ending", "storage_change_bcf"]].reset_index(
        drop=True
    )


def join_eia(target: pd.DataFrame, releases: pd.DataFrame) -> pd.DataFrame:
    joined = pd.merge_asof(
        target.sort_values("ts_event"),
        releases.sort_values("release_time"),
        left_on="ts_event",
        right_on="release_time",
        direction="backward",
    )
    minutes_since = (
        joined["ts_event"] - joined["release_time"]
    ).dt.total_seconds() / 60
    joined["eia_minutes_since_release"] = minutes_since.clip(0, 14 * 24 * 60).astype(
        "float32"
    )
    joined["eia_post_release_60m"] = minutes_since.between(0, 60).astype("int8")

    upcoming = pd.merge_asof(
        joined[["ts_event"]].sort_values("ts_event"),
        releases[["release_time"]].sort_values("release_time"),
        left_on="ts_event",
        right_on="release_time",
        direction="forward",
    )
    minutes_to = (
        upcoming["release_time"] - joined["ts_event"]
    ).dt.total_seconds() / 60
    joined["eia_release_window"] = (
        minutes_to.between(0, 30) | minutes_since.between(0, 60)
    ).astype("int8")
    joined["eia_minutes_to_release"] = minutes_to.clip(0, 14 * 24 * 60).fillna(
        14 * 24 * 60
    ).astype("float32")
    joined["storage_change_bcf"] = joined["storage_change_bcf"].astype("float32")
    return joined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=TARGET)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    frame = pd.read_parquet(args.target)
    frame["ts_event"] = pd.to_datetime(frame["ts_event"], utc=True)
    frame = frame.sort_values("ts_event").reset_index(drop=True)
    for name, (tolerance, path) in MARKETS.items():
        frame = join_market(frame, name, tolerance, path)
        coverage = 100 * frame[f"{name}_available"].mean()
        print(f"{name:12s} coverage within {tolerance:2d}m: {coverage:6.2f}%")

    releases = eia_releases(
        AUXILIARY / "eia_ngshistory.xls",
        AUXILIARY / "eia_revisions.xls",
    )
    releases.to_csv(AUXILIARY / "eia_storage_releases.csv", index=False)
    frame = join_eia(frame, releases)

    # An EIA value is available only after its official release timestamp.
    if (frame["release_time"] > frame["ts_event"]).fillna(False).any():
        raise RuntimeError("EIA look-ahead detected")
    for name in MARKETS:
        impossible_age = frame[f"{name}_age_minutes"].lt(0)
        if impossible_age.any():
            raise RuntimeError(f"Future {name} observation joined to target")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    print(f"Saved {args.output} with {len(frame):,} rows and {len(frame.columns)} columns")
    print(
        "EIA releases in target range:",
        int(releases["release_time"].between(frame.ts_event.min(), frame.ts_event.max()).sum()),
    )


if __name__ == "__main__":
    main()
