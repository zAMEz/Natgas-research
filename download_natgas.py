"""Download and prepare one year of NYMEX Henry Hub natural gas bars.

The request uses Databento's volume-ranked continuous lead contract, NG.v.0.
The raw contract instrument ID is retained and contract switches are flagged so
returns and labels never cross a futures roll.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from getpass import getpass
import os
from pathlib import Path

import databento as db
import numpy as np
import pandas as pd


DATASET = "GLBX.MDP3"
SYMBOL = "NG.v.0"
SCHEMA = "ohlcv-1m"
STYPE_IN = "continuous"


@dataclass(frozen=True)
class Request:
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def api_kwargs(self) -> dict[str, object]:
        return {
            "dataset": DATASET,
            "symbols": SYMBOL,
            "stype_in": STYPE_IN,
            "schema": SCHEMA,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
        }

    @property
    def stem(self) -> str:
        return f"ng_v0_ohlcv_1m_{self.start.date()}_{self.end.date()}"


def previous_year(end: str | pd.Timestamp | None = None) -> Request:
    """Return the previous 12 complete UTC months, with an exclusive end."""
    end_ts = (
        pd.Timestamp.now(tz="UTC").normalize()
        if end is None
        else pd.Timestamp(end)
    )
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    end_ts = end_ts.normalize()
    return Request(start=end_ts - pd.DateOffset(years=1), end=end_ts)


def get_client() -> db.Historical:
    """Create a client without writing the API key to disk or notebook output."""
    key = os.getenv("DATABENTO_API_KEY")
    if not key:
        key = getpass("Databento API key (input is hidden): ").strip()
    if not key.startswith("db-"):
        raise ValueError("A Databento key must start with 'db-'.")
    return db.Historical(key=key)


def estimate_cost(client: db.Historical, request: Request) -> float:
    """Return Databento's estimate in USD; this call does not download data."""
    return float(client.metadata.get_cost(**request.api_kwargs))


def prepare_model_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Add roll-safe one-minute returns and an exact five-minute target."""
    frame = raw.reset_index()
    timestamp_column = "ts_event" if "ts_event" in frame.columns else frame.columns[0]
    frame = frame.rename(columns={timestamp_column: "ts_event"})
    frame["ts_event"] = pd.to_datetime(frame["ts_event"], utc=True)
    frame = frame.sort_values("ts_event").drop_duplicates("ts_event").reset_index(drop=True)

    previous_id = frame["instrument_id"].shift()
    one_minute = frame["ts_event"].diff().eq(pd.Timedelta(minutes=1))
    same_previous_contract = frame["instrument_id"].eq(previous_id)

    frame["is_roll"] = frame["instrument_id"].ne(previous_id)
    if not frame.empty:
        frame.loc[0, "is_roll"] = False
    frame["is_contiguous_1m"] = one_minute & same_previous_contract
    frame["log_return_1m"] = np.log(frame["close"]).diff().where(
        frame["is_contiguous_1m"]
    )

    lookup = frame.set_index("ts_event")
    target_times = frame["ts_event"] + pd.Timedelta(minutes=5)
    future_close = lookup["close"].reindex(target_times).to_numpy()
    future_contract = lookup["instrument_id"].reindex(target_times).to_numpy()
    valid_target = future_contract == frame["instrument_id"].to_numpy()
    frame["target_log_return_5m"] = np.where(
        valid_target,
        np.log(future_close / frame["close"].to_numpy()),
        np.nan,
    )
    return frame


def get_dataset_conditions(
    client: db.Historical,
    request: Request,
) -> pd.DataFrame:
    """Fetch Databento's daily availability/quality status for the request."""
    records = client.metadata.get_dataset_condition(
        dataset=DATASET,
        start_date=str(request.start.date()),
        end_date=str(request.end.date()),
    )
    conditions = pd.DataFrame(records)
    conditions["date"] = pd.to_datetime(conditions["date"]).dt.date
    return conditions


def annotate_conditions(
    model: pd.DataFrame,
    conditions: pd.DataFrame,
) -> pd.DataFrame:
    """Attach Databento's daily data condition to every model row."""
    result = model.copy()
    condition_by_date = conditions.set_index("date")["condition"]
    utc_date = pd.to_datetime(result["ts_event"], utc=True).dt.date
    result["data_condition"] = utc_date.map(condition_by_date).fillna("unknown")
    return result


def download_and_prepare(
    client: db.Historical,
    request: Request,
    output_dir: str | Path = "data",
) -> dict[str, Path]:
    """Download DBN, and save raw and model-ready Parquet files."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    dbn_path = output / f"{request.stem}.dbn.zst"
    raw_path = output / f"{request.stem}_raw.parquet"
    model_path = output / f"{request.stem}_model.parquet"
    conditions_path = output / f"{request.stem}_conditions.csv"

    client.timeseries.get_range(**request.api_kwargs, path=dbn_path)
    store = db.DBNStore.from_file(dbn_path)
    raw = store.to_df()
    raw.reset_index().to_parquet(raw_path, index=False)
    conditions = get_dataset_conditions(client, request)
    conditions.to_csv(conditions_path, index=False)
    model = annotate_conditions(prepare_model_frame(raw), conditions)
    model.to_parquet(model_path, index=False)

    return {
        "dbn": dbn_path,
        "raw": raw_path,
        "model": model_path,
        "conditions": conditions_path,
    }


def validate_model_file(path: str | Path, request: Request) -> dict[str, object]:
    """Return compact quality checks for the saved model-ready dataset."""
    frame = pd.read_parquet(path)
    timestamps = pd.to_datetime(frame["ts_event"], utc=True)
    degraded = (
        frame["data_condition"].eq("degraded")
        if "data_condition" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    return {
        "rows": len(frame),
        "first_bar": timestamps.min(),
        "last_bar": timestamps.max(),
        "contracts": int(frame["instrument_id"].nunique()),
        "rolls": int(frame["is_roll"].sum()),
        "duplicate_timestamps": int(timestamps.duplicated().sum()),
        "bad_ohlc_rows": int(
            (
                (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
                | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
            ).sum()
        ),
        "degraded_dates": int(
            frame.loc[degraded, "ts_event"]
            .astype(str)
            .str[:10]
            .nunique()
        ),
        "degraded_bars": int(degraded.sum()),
        "valid_5m_targets": int(frame["target_log_return_5m"].notna().sum()),
        "requested_start": request.start,
        "requested_end_exclusive": request.end,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", help="Exclusive UTC end date; default is today")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download after printing the cost estimate",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help="Abort when estimated USD cost exceeds this amount",
    )
    args = parser.parse_args()

    request = previous_year(args.end)
    client = get_client()
    cost = estimate_cost(client, request)
    print(f"Request: {request.start} to {request.end} (end exclusive)")
    print(f"Estimated cost: ${cost:,.2f} USD")
    if not args.download:
        print("Cost check only. Re-run with --download to fetch the data.")
        return
    if args.max_cost is not None and cost > args.max_cost:
        raise SystemExit(
            f"Estimated cost ${cost:,.2f} exceeds --max-cost ${args.max_cost:,.2f}."
        )

    paths = download_and_prepare(client, request, args.output_dir)
    checks = validate_model_file(paths["model"], request)
    print("Saved:")
    for label, path in paths.items():
        print(f"  {label}: {path}")
    print("Validation:")
    for name, value in checks.items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
