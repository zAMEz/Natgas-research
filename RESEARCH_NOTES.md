# Daily context for five-minute natural-gas forecasts

Research date: 2026-10-19. This note describes the rationale, causal design requirements and interpretation limits for the long-term-context experiment. It documents the research method rather than making a performance claim; generated experiment outputs remain the authority for what was run and how it performed.

## What the hypothesis means

The requested hypothesis is sensible to test: the next five-minute return may depend on whether natural gas has risen for several sessions, how unusual its current level is, the season, and its relationship to crude oil. The relevant comparison is a five-minute model with these daily inputs against the same five-minute model without them, using identical dates, fills, costs and selection rules.

Do not encode “high for several days means it must fall” as a guaranteed relationship. EIA identifies weather, production, storage and exports as important drivers, and describes short-run constraints that can amplify price moves. This supports considering slower market conditions, but does not establish a profitable timing rule. Calendar seasonality is also a proxy for those conditions, rather than a measurement of weather surprises or supply changes. [EIA: factors affecting natural-gas prices](https://www.eia.gov/energyexplained/natural-gas/factors-affecting-natural-gas-prices.php).

## Features and their interpretation

| Requested information | Useful causal representation | Interpretation limit |
|---|---|---|
| Historical day-start/day-end swing | Previous completed session open-to-close return; high-low range; absolute return; rolling signed and absolute return summaries; up/down streaks | A wide range describes risk, not necessarily direction. Session high-low can greatly exceed open-to-close movement. |
| Elevated price for several days | Roll-safe 5/10/20/60-session returns, distance from moving means, run length above the mean, recent drawdown and range position | Both reversal and continuation must remain possible. |
| Bollinger bands | Twenty-session standardized price, `%B`, relative band width, and recent band occupancy | Upper-band contact does not itself establish a short signal. |
| RSI and divergence | Fourteen-session RSI; changes in RSI; bullish/bearish divergence between confirmed price pivots and RSI at those pivots; time since confirmation | A pivot requiring future right-hand bars becomes available only after those bars finish. |
| Seasonality | Smooth annual and semiannual sine/cosine terms, plus weekday and explicitly defined seasonal indicators | Even five years supplies only about five annual cycles, which is limited evidence of stable seasonal return predictability. |
| NG–WTI relationship | Lagged daily log returns; rolling return correlation; standardized relative price or log-ratio; causally fitted relative-price residual | Raw NG minus WTI mixes units; correlation and equilibrium relationships can change. |
| Today's progress | Return from the known opening price; distance from prior completed-session range and bands; fraction of historical daily range consumed so far | Only observations available at the current decision time may enter. Today's final range or close cannot. |

John Bollinger's own rules distinguish a relative high/low indicator from a trading signal, explain that prices may persist near a band during trends, and treat default lookbacks as defaults rather than optimal settings. The implementation should therefore supply band information to a model and test its incremental usefulness. [John Bollinger: Bollinger Band Rules](https://www.bollingerbands.com/bollinger-band-rules).

TradingView describes divergence as disagreement between price and an indicator and warns that it does not guarantee a reversal or identify when one will occur. Its platform documentation separately explains how higher-timeframe values and indicators plotted into the past can mislead historical tests. The research implication is to timestamp a divergence at its confirmation, never retrospectively at the earlier turning point. A causal comparison of trailing-window slopes is also possible, but should be called a slope disagreement rather than confirmed pivot divergence. [TradingView: RSI divergence](https://www.tradingview.com/support/solutions/43000589127-rsi-divergence-indicator/); [TradingView: repainting](https://www.tradingview.com/pine-script-docs/concepts/repainting/).

An EIA study of the historical NG–WTI relationship warns that regressions involving nonstationary price levels can produce spurious results. Its findings used 1989–2005 data and must not be treated as a current structural law. A separate EIA analysis documents a substantial subsequent change in the oil/gas ratio. Accordingly, a rolling normalized relative-value feature is a hypothesis to test; it is not an arbitrage condition. [EIA: relationship between crude oil and natural gas prices](https://www.eia.gov/naturalgas/articles/reloilgaspriindex.php); [EIA: changing oil-to-gas price ratio](https://www.eia.gov/todayinenergy/detail.php?id=1030).

## Two levels of forecasting

The following is a project design inference, not a replication of an established profitable strategy.

1. Construct one observation per completed daily session. At the next session's start, use known history to estimate the same-session and three-session directional outlook. Use a small, strongly regularized daily model because even several years contain only hundreds or low thousands of genuinely distinct daily observations.
2. Produce historical daily forecasts sequentially. For any training session, fit the daily model only on earlier examples whose outcome timestamps precede that session's information cutoff. An expanding training window is suitable; exclude early sessions without enough history.
3. Join the daily features and those earlier-generated daily forecasts into the five-minute dataset. Compare an intraday-only LightGBM baseline, context-augmented LightGBM/EBM/random forest, and a model using the daily forecast explicitly.
4. Separately test a calibrated adjustment of the intraday prediction using the daily outlook. Fit adjustment coefficients on an earlier calibration partition and select trading thresholds on the subsequent validation partition. Freeze both before testing.

Training the daily model on all training sessions and feeding its fitted predictions back into those same sessions would leak outcome information through the stacked feature. Scikit-learn's stacking documentation likewise distinguishes predictions generated outside base-model training from fitted predictions and warns of overfitting when prefit models were trained on the same data as the final model. For this time-series application, ordinary shuffled cross-validation is not an adequate substitute for chronological prediction. [Scikit-learn: StackingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html).

A daily expected move is not automatically an executable five-minute expected move. Multiplying it by five divided by the session length assumes a particular intraday allocation of drift. Treat that as a benchmark assumption, or estimate the relationship on an earlier calibration interval. Do not force a positive weight or a bearish sign merely to obtain the desired trading outcome.

## Required timing and contract safeguards

- Define “day” explicitly. The existing intraday pipeline groups session VWAP from 18:00 New York local wall time while trading only during 08:00–16:00. A calendar day, a full futures session and that trading window have different opens and closes. Use local-wall-time grouping so daylight-saving changes do not shift the boundary.
- Freeze the start-of-session assessment from prior completed sessions. The opening price can enter only once it is observed. A completed one-minute opening candle is available at its end, not at its start.
- Keep the daily context and any continuously updated intraday progress features conceptually distinct. Changing future prices must not change an earlier daily assessment.
- Preserve raw futures contract identifiers. Contract switches must not appear as large economic returns. Use a documented causal continuous-series construction or restart/exclude calculations at a roll; a roll-adjusted research index is not itself an executable contract price.
- Protect WTI against its own contract switches, stale observations and missing data independently of NG.
- Associate daily labels with actual completion timestamps. A three-session target is unavailable until the third session closes, including across weekends and holidays; a one-day shift alone is insufficient.
- Do not use final-session completeness or today's eventual data availability as a feature available at the open.

## Evidence to report

Preserve the existing delayed-open execution convention, non-overlapping positions, two-tick round-trip selection cost and one-to-four-tick sensitivity. Reuse identical eligible five-minute rows for baseline and context variants so that a changed sample is not mistaken for improved prediction.

Evaluate daily direction once per session using probability calibration/Brier score and a past-data base-rate comparator, as well as directional accuracy. Evaluate five-minute forecasts against zero change and the intraday-only baseline using MAE, RMSE, correlation and executable net PnL. Report the number of unique sessions separately from the number of intraday rows.


