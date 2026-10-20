# Input Data Contract

The full trainer expects one Parquet table sorted by `ts_event`. Timestamps must
be timezone-aware UTC bar-start times. Rows represent observed one-minute bars;
missing intervals should remain missing rather than being synthesized.

## Required NG target fields

| Field | Meaning |
|---|---|
| `ts_event` | UTC bar-start timestamp |
| `instrument_id` | Raw NG futures contract identifier |
| `open`, `high`, `low`, `close` | Price in dollars per MMBtu |
| `volume` | Observed bar volume |
| `data_condition` | Vendor condition, e.g. `available` or `degraded` |

OHLC values must be internally consistent, timestamps unique, and the table
strictly increasing after sorting. A degraded target bar breaks the tradable
segment.

## Auxiliary market fields

For each prefix `wti`, `lng_equity`, `ng_m1`, `ng_m2`, and `ng_m3`:

| Field | Meaning |
|---|---|
| `<prefix>_close` | Most recent causally joined close |
| `<prefix>_instrument_id` | Source contract/instrument identifier |
| `<prefix>_available` | 1 when a sufficiently fresh source exists |
| `<prefix>_age_minutes` | Age of joined source observation |

The provided builder uses backward-only `merge_asof`. Curve contracts may be at
most 60 minutes old; WTI and the equity proxy may be at most 10 minutes old.
Each source resets its own return history when its identifier changes or the
source becomes unavailable.

## EIA timing fields

| Field | Meaning |
|---|---|
| `eia_minutes_to_release` | Minutes until the next scheduled release |
| `eia_minutes_since_release` | Minutes since the most recent release |
| `eia_post_release_60m` | Post-release window indicator |
| `eia_release_window` | Scheduled safety-window indicator |

`storage_change_bcf` may exist in an augmented source, but the advanced feature
pipeline intentionally excludes it because a current historical file can contain
later revisions or values not available at the original decision time.

## Data not included

This repository contains no DBN, Parquet, CSV market history, Excel fundamentals,
API keys, trained joblib bundles, or model outputs. Supply data under your own
license. The download scripts estimate cost before acquisition and use hidden
credential prompts or process environment variables.

## Expected scale

The motivating study used roughly five years of one-minute observations and more
than one thousand valid daily sessions. The code does not require those exact
dates, but CLI development and holdout boundaries must exist within the supplied
history, with enough earlier data for training and daily indicator warm-up.
