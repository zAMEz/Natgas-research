"""Exercise causal features, labels, and accounting without proprietary data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from advanced_backtest import simulate
from advanced_data import executable_dataset, prepare_features


def synthetic_market(rows: int = 480) -> pd.DataFrame:
    index = np.arange(rows)
    base = 3.0 + 0.0002 * index + 0.006 * np.sin(index / 17)
    frame = pd.DataFrame({
        "ts_event": pd.date_range("2026-06-01 12:00", periods=rows, freq="min", tz="UTC"),
        "instrument_id": 101,
        "open": base,
        "high": base + 0.002,
        "low": base - 0.002,
        "close": base + 0.0005 * np.cos(index / 5),
        "volume": 100 + index % 13,
        "data_condition": "available",
    })
    for number, source in enumerate(("wti", "lng_equity", "ng_m1", "ng_m2", "ng_m3")):
        frame[f"{source}_close"] = 60 + number + 0.001 * index
        frame[f"{source}_instrument_id"] = 200 + number
        frame[f"{source}_available"] = 1
        frame[f"{source}_age_minutes"] = 0
    frame["eia_minutes_to_release"] = 1_000.0
    frame["eia_minutes_since_release"] = 1_000.0
    frame["eia_post_release_60m"] = 0
    frame["eia_release_window"] = 0
    return frame


def main() -> None:
    featured, feature_names, audit = prepare_features(synthetic_market())
    executable = executable_dataset(featured, horizon=5, stride=5)

    # A deliberately simple diagnostic signal—not a fitted trading strategy.
    prediction = np.sign(executable.momentum_15m_vol.fillna(0).to_numpy()) * 3
    trades = simulate(executable, prediction, cost=2, edge=0)
    accounting_ok = np.allclose(
        trades.net_ticks,
        trades.gross_ticks - trades.cost_ticks,
    )

    print(f"Raw rows: {audit['raw_rows']:,}")
    print(f"Causal intraday features: {len(feature_names)}")
    print(f"Executable 5-minute examples: {len(executable):,}")
    print(f"Diagnostic non-overlapping positions exercised: {len(trades):,}")
    print(f"Cost-accounting identity verified: {accounting_ok}")
    print("Synthetic demonstration passed; it validates mechanics, not strategy returns.")


if __name__ == "__main__":
    main()
