# Technical Report: Causal Natural-Gas Forecasting

## 1. Objective

The project asks a narrow, testable question: can information available before a
decision improve short-horizon Henry Hub natural-gas forecasts enough to support
an executable strategy after realistic frictions?

The design separates two time scales. A daily layer describes the slower market
regime and produces one- and three-session forecasts. An intraday layer predicts
5-, 15-, and 30-minute open-to-open moves. Several implementations test whether
daily information helps: direct context features, a stacked daily forecast, a
calibrated correction to the intraday model, and a transparent daily-drift
benchmark.

The primary engineering goal is trustworthy research. A candidate is allowed to
abstain, and a profitable-looking evaluation rule cannot retroactively change the
registered choice.

## 2. End-to-end process

1. Acquire licensed one-minute OHLCV bars for the volume-ranked NG lead contract,
   three calendar NG contracts, volume-ranked WTI, and an LNG-related equity.
2. Retain raw instrument identifiers and vendor quality conditions.
3. Join auxiliary markets backward in time with bounded freshness. No future
   quote can be joined to an NG row.
4. Segment every series at gaps, degraded observations, missing sources, and
   contract rolls.
5. Construct causal intraday features and daily/session context.
6. Generate executable labels using delayed future opens only when the complete
   path is continuous, good quality, and on the same contract.
7. Fit daily models on one observation per session using only targets whose
   actual completion timestamps precede the fit cutoff.
8. Compare intraday model families on identical rows and chronological windows.
9. Select a trading threshold on a validation interval, never on the evaluation
   interval.
10. Simulate non-overlapping positions, subtract a fixed round-trip cost, include
    zero-trade days, and measure uncertainty with session blocks.
11. Freeze the selected candidate—or cash—before final evaluation and save data,
    source, feature, cutoff, and model metadata.

## 3. Information timing

One-minute timestamps denote bar starts. A bar beginning at `t` is treated as
complete at `t+1`. With the default one-bar execution delay, a decision based on
that bar enters at the open at `t+2`. A horizon `h` exits at the open at `t+2+h`.

This convention prevents filling at the same price that supplied the decision.
Labels are rounded to integer NG ticks. The path from decision through exit must
belong to one continuous, good-quality target-contract segment. Positions stay
inside the 08:00–16:00 America/New_York research window and avoid the configured
EIA release window.

The daily snapshot is timestamped at 08:00 New York. Static daily indicators use
only earlier completed sessions. Current-session features—return from the
observed opening price, range consumed, elapsed fraction, and distance from
frozen bands—update only after completed minute bars.

## 4. Roll-safe daily construction

Raw futures prices cannot be compared across contract switches as if they were a
single asset. The daily layer therefore builds dimensionless linked log-return
indices. It includes valid session open-to-close returns and includes an
overnight bridge only across consecutive valid sessions on the same contract.
Bridges are omitted at rolls and data gaps.

WTI receives the same independent treatment. The NG–WTI variable is a normalized
relative return index, not a raw dollar subtraction and not an asserted fair-value
relationship.

The research session is 08:00–16:00 New York. A session normally needs at least
400 good minute bars, an opening observation within five minutes of 08:00, a last
observation within five minutes of 16:00, and no internal NG contract switch.

## 5. Feature families

### Intraday features

The base layer contains 73 variables derived from information available at each
decision. They include:

- lagged NG returns and volatility-normalized momentum;
- ATR, wick/body, range, efficiency, volume, and session-VWAP measures;
- minute/day calendar encodings;
- NG calendar spreads and curve changes with independent roll resets;
- fresh WTI and LNG-equity returns;
- EIA release distance/window indicators; and
- source availability, age, quality, and contract-continuity flags.

### Daily and contextual features

The context layer adds 79 variables:

- 3/5/10/20/60-session roll-safe returns;
- distances and slopes relative to 5/20/60-session moving means;
- prior open-to-close changes, high-low ranges, and rolling range statistics;
- Bollinger 20-session z-score, position, bandwidth, and band distances;
- Wilder RSI(14), recent RSI changes, and sustained time above/below the mean;
- pivot divergence confirmed only after two right-hand sessions, never backdated;
- trailing slope-disagreement proxies;
- annual phase and heating/cooling season indicators;
- NG–WTI relative returns and standardized relative levels; and
- current-session progress measured only from completed bars.

Technical indicators are supplied as candidate information. The implementation
does not encode “upper band means short” or “high RSI guarantees reversal.”

## 6. Model architecture

### Daily layer

The daily layer uses strongly regularized ridge regression for normalized
one- and three-session returns and logistic regression for down-move
probabilities. Each session has one training weight. Fits expand monthly and use
only labels completed before the monthly cutoff. Early history falls back to a
zero expected return and an earlier-data direction base rate until enough labels
mature.

The final evaluation period uses frozen daily-model parameters. Indicators still
update causally each morning, but evaluation outcomes do not refit the model.

### Intraday model family

The registered comparison includes:

- intraday-only LightGBM as the primary ablation;
- context LightGBM;
- Explainable Boosting Machine;
- random forest and ExtraTrees with bounded depth and large leaves;
- standardized ridge regression;
- an equal-weight tree/EBM ensemble;
- stacked LightGBM using prior-generated daily scores;
- intraday LightGBM plus a regularized daily correction; and
- a daily-drift benchmark that spreads an expected daily move uniformly across
  the 480-minute research session.

The correction is fitted on one mean residual per session so a daily signal is
not given hundreds of artificial copies merely because minute bars are dense.

## 7. Chronological experiment design

Every monthly fold contains five distinct roles:

- **Training:** estimator fitting, with a bounded 180-day intraday window.
- **Early stop:** LightGBM iteration selection.
- **Calibration:** daily-to-intraday correction and related calibration.
- **Validation:** threshold and policy selection.
- **Evaluation:** untouched until all preceding choices are complete.

Rows are assigned using actual label exit timestamps, not only decision
timestamps. This purges a target that starts before a boundary but finishes after
it. The daily three-session label similarly enters training only after its third
session has ended.

Historical daily scores used by stacked models are generated by prior monthly
fits. Feeding fitted daily predictions back into their own training observations
would be leakage and is explicitly avoided.

## 8. Trading policy and accounting

The policy evaluates a small registered edge-threshold grid. A signal requires
the predicted absolute move to exceed round-trip cost plus the selected edge.
At most one position is open, and an entry may occur exactly when the preceding
position exits.

The default cost is two NG ticks round trip. A full-size NG tick is $10, so this
corresponds to $20 per completed trade. It is a composite spread/commission/
slippage assumption; results are also recalculated at one, two, three, and four
ticks using the same selected trades.

A validation policy must meet minimum trade and active-day counts, be positive
in both validation halves, and have a positive mean-minus-standard-error score.
The research-level selector is stricter: it requires at least 40 development
trades and a positive lower bound from a five-session block bootstrap. Otherwise
the selected model is cash.

## 9. Evaluation and checking

Forecasts are evaluated against both an intraday-only LightGBM and the zero-change
forecast using MAE, RMSE, correlation, R², and nonzero direction accuracy. Daily
models are evaluated once per session with return MAE, direction accuracy, R²,
and Brier score against an earlier-data base rate.

Trading reports include trade count, net ticks, dollar translation for one
full-size contract, win rate, profit factor, drawdown, active/evaluated sessions,
and daily Sharpe. Zero-trade days remain in daily statistics.

Five-session circular block bootstraps reflect serial dependence more honestly
than row-wise resampling. A centered max-mean family check is reported across the
registered candidates. These diagnostics are descriptive safeguards, not proof
of live profitability.

The automated suite contains 38 tests covering:

- future price/quality/roll mutations cannot alter earlier features;
- timestamp duplication fails closed;
- target paths reject gaps, degraded bars, and rolls;
- auxiliary rolls reset only their own return histories;
- decisions, entries, exits, and integer ticks follow the declared convention;
- daily snapshots remain frozen and respect daylight saving time;
- Bollinger, Wilder RSI, and divergence confirmation match explicit calculations;
- missing or incomplete sessions do not forward-fill a stale trend;
- daily target availability uses actual completion times, including weekends;
- final daily models stay frozen across evaluation months;
- daily-to-intraday calibration gives each session equal weight;
- policy cost stress preserves exactly the same trades; and
- the final selector ignores an attractive evaluation-only winner.

Run all checks with:

```bash
python -m unittest discover -s tests -v
```

## 10. Reproducibility

Each run writes a manifest before fitting. It records configuration, source and
data SHA-256 hashes, dates, candidate models, and a deployment-disabled flag.
Output folders cannot be overwritten. Frozen choices and cutoffs are saved before
the final evaluation. Model bundles contain feature lists, policy cutoffs, and a
false deployment flag.

The offline predictor truncates raw input to bars completed at an `as_of` time
before recomputing features. It rejects predictions at or before the policy
cutoff and exports a morning assessment without future labels or current-session
final OHLC values.

## 11. Assumptions and limitations

- OHLCV bars approximate executable prices; there is no order book, queue model,
  partial fill, or market-impact model.
- The flat cost assumption cannot represent every broker, order type, or market
  regime. Zero-cost analysis is only an upper bound.
- EIA timing is used as an event-safety feature; revised storage actuals are not
  allowed into the advanced feature set.
- Calendar terms are proxies. Even several annual cycles do not prove a stable
  seasonal effect.
- NG–WTI relationships are nonstationary and the normalized relative index is
  not itself a tradable spread.
- An LNG-related equity is an imperfect, market-hours-only proxy for physical
  feedgas.
- Point-in-time weather forecasts, storage consensus, production, pipeline flows,
  physical LNG feedgas, and power demand are not included.
- Multiple overlapping three-session targets reduce the effective daily sample
  size.
- Previously inspected dates cannot become a new independent holdout through
  further tuning.
- Serialized joblib files use pickle and must only be loaded from trusted sources.

## 12. Interpretation

The historical work found that the longer-term layer can contain modest daily
directional information, especially at longer horizons, while five-minute price
moves are dominated by noise and event risk. Small forecast improvements do not
automatically survive costs or model-selection uncertainty.

The registered system therefore treats cash as a valid model outcome. That is a
deliberate research conclusion and a practical example of model governance—not a
software failure. A deployment decision would require new prospective data,
measured broker-specific execution costs, and operational controls beyond this
repository.
