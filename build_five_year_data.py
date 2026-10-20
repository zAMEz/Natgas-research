"""Combine the preserved one-year files with four new years and rebuild joins."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from build_augmented_data import eia_releases, join_eia, join_market


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
HISTORY = DATA / "extended_2021_2025"
START, MID, END = "2021-09-24", "2025-09-24", "2026-09-24"


def combine(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in paths]
    result = pd.concat(frames, ignore_index=True)
    result["ts_event"] = pd.to_datetime(result.ts_event, utc=True)
    result = result.sort_values("ts_event").drop_duplicates("ts_event", keep="last").reset_index(drop=True)
    if not result.ts_event.is_monotonic_increasing or result.ts_event.duplicated().any():
        raise RuntimeError("Combined series is not strictly time ordered")
    return result


def main() -> None:
    target = combine([
        HISTORY / f"ng_v0_ohlcv_1m_{START}_{MID}_model.parquet",
        DATA / f"ng_v0_ohlcv_1m_{MID}_{END}_model.parquet",
    ])
    combined_dir = DATA / "five_year_sources"
    combined_dir.mkdir(exist_ok=False)
    market_paths: dict[str, tuple[int, Path]] = {}
    for name, tolerance in (("ng_m1", 60), ("ng_m2", 60), ("ng_m3", 60), ("wti", 10), ("lng_equity", 10)):
        source = combine([
            HISTORY / f"{name}_ohlcv_1m_{START}_{MID}.parquet",
            DATA / "auxiliary" / f"{name}_ohlcv_1m_{MID}_{END}.parquet",
        ])
        path = combined_dir / f"{name}_ohlcv_1m_{START}_{END}.parquet"
        source.to_parquet(path, index=False)
        market_paths[name] = (tolerance, path)
        print(f"{name:12s}: {len(source):,} rows", flush=True)

    frame = target
    for name, (tolerance, path) in market_paths.items():
        frame = join_market(frame, name, tolerance, path)
        print(f"{name:12s} coverage: {100*frame[f'{name}_available'].mean():.2f}%", flush=True)
    releases = eia_releases(
        DATA / "auxiliary" / "eia_ngshistory.xls",
        DATA / "auxiliary" / "eia_revisions.xls",
    )
    frame = join_eia(frame, releases)
    if (frame.release_time > frame.ts_event).fillna(False).any():
        raise RuntimeError("EIA look-ahead detected")
    output = DATA / f"ng_augmented_ohlcv_1m_{START}_{END}_model.parquet"
    frame.to_parquet(output, index=False)
    print(f"Saved {output}: {len(frame):,} rows, {len(frame.columns)} columns", flush=True)


if __name__ == "__main__":
    main()
