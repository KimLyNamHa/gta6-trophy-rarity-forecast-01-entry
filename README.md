# GTA VI Trophy Rarity Forecast — week-one earn rates

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/KimLyNamHa/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/KimLyNamHa/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Built for the [make.fun](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) bounty
"Forecast how many GTA VI players earn each trophy in week one".

Forecasts, for every one of GTA VI's 55 PS5 trophies, its earn rate on
[Exophase](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/)
at 2026-11-26T00:00:00Z, as five quantiles (5%, 25%, 50%, 75%, 95%),
scored by mean pinball loss on the logit scale.

## Program

`main.ts` (Node 22, standard library only, no network, deterministic,
~0.2 s per run) reads every `<id>.in` in the input directory and writes
`<id>.out` to the output directory:

```
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## Model (short version; see STRATEGY.md)

- Collapse the history to distinct snapshots — Exophase refreshes on its
  own schedule, and the flat stretches between refreshes carry no signal.
- Model earned counts and players-tracked separately (rate = earned /
  tracked), so a refresh jump in tracked players dilutes rates without
  implying players stopped earning.
- Four candidate medians per trophy (persistence, logit-rate trend vs
  log-time since release, vs linear time, earned/tracked model), slopes
  shrunk toward their tier/rate-band group mean and capped; the candidate
  is chosen per trophy by rolling-origin backtesting inside the case's
  own history, predicting every later distinct snapshot.
- Wide honest spread: logit floor 1.1 for the final horizon, plus
  backtest residuals and refresh-jump volatility growing with horizon.
  The house baseline's narrow band scores ~1.51 on the practice because
  one refresh moves every rate outside it; this spread is built not to
  repeat that.

## Files

- `main.ts` — the forecaster.
- `STRATEGY.md` — the model, the refresh trap in the practice data, and
  the backtest numbers.
- `LICENSE` — MIT.
