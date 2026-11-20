# Strategy — GTA VI trophy rarity forecast

## What the data actually looks like (EDA)

Practice cases p00 (checkpoint 2026-11-19T03:00Z, target 04:30Z) and p01
(03:30Z → 05:00Z), 55 trophies (1 platinum, 2 gold, 6 silver, 46 bronze;
28 secret). Each history is 7–8 reads, all identical:

- players_tracked: 638 in every read, trophies_earned: 943 in every read.
- 0 trophies with any rate change in history. Last rates: one at 100%
  (id 2, Just Like Paradise — complete Gone Fishing), the next highest
  2.04% (Baited My Tackle), then 1.57%, 1.25%, 1.10% …; 40 of 55 below
  1%; the floor is 0.63%, i.e. 4 of 638 players.
- The rates are counts in disguise: 0.63% = 4 players, 0.78% = 5,
  0.94% = 6, 1.10% = 7. At 638 tracked players, one player is 0.16pp.

## The refresh trap

The practice history gives **no trend signal at all** — any line fitted
to it is flat. Their answers are the first refresh (3,868 players
tracked, every rate but the 100% one lower). The house baseline scores
~1.51 on the practice with 90% coverage ~0: its 90-minute band is far
too narrow for a refresh that moves every rate by ~1–1.8 logit.

So the model must be built for the **final** history (reads to
2026-11-21T11:30Z, two days, real growth and real refresh jumps), not
tuned to practice. Nothing in `main.ts` is fitted to the practice
answers — there is no practice answer in the kit to fit to. The
practice cases only check that the program reads a case and writes a
valid forecast, and that its spread can survive a refresh it has not
seen.

## Model

Per trophy, at runtime, from the case's own history alone:

1. **Distinct snapshots.** Collapse consecutive reads that show the same
   players-tracked count and the same rates. A flat stretch between
   Exophase refreshes is one observation, not many.
2. **Earned and tracked separately.** `rate = earned / tracked`.
   A refresh jump in tracked players dilutes every rate while earned
   counts keep rising, so the two are extrapolated separately (log
   earned and log tracked against log-time since release,
   2026-11-19T00:00Z) and divided. Modelling the rate directly would
   read dilution as players un-earning trophies.
3. **Four candidate medians**, per trophy:
   - persistence (last rate),
   - logit-rate trend vs log-time since release (saturating: log-time
     grows slowly, so growth diminishes),
   - logit-rate trend vs linear time,
   - the earned/tracked model.
   Slopes are shrunk toward their group mean (same tier, same
   current-rate band: rare < −4.5 logit, mid, common) with weight
   capped at 8 observations, and capped (log-time slope ≤ 2.0,
   earned slope 0–2.5, tracked slope 0–1.5, linear slope ≤ 0.1/hour),
   so one noisy trophy cannot run away.
4. **Rolling-origin choice.** Cut the distinct snapshots at 50%, 70%,
   85%; fit each candidate on reads before the cut and predict *every*
   later distinct snapshot (long windows — a trend only beats
   persistence over a long window, and the final horizon is 4.52
   days). Score pinball loss on the logit scale; pick the best
   per trophy, ties broken toward earned/tracked, then log-time.
   With a flat or tiny history every candidate is persistence and the
   choice is a tie.
5. **Safety rails.** Median logit is clamped to ±2.5 of the last read's
   logit, and the median rate is capped by what the trophy asks for:
   platinum / 100%-completion 4%, grind-or-collect (every/all/each,
   50–250 counts, 100 locations) 10%, other gold 25%, everything else
   75%; the 100% trophy is floored at 85%.
6. **Spread.** `sigma = sqrt(floor² + residual² + jumpVar × horizonDays)`,
   clamped to [floor, 2.2] logit. Floor is 1.1 for the final forecast
   (one refresh moves a rate ~1.5 logit; the final holds several
   refreshes plus trend uncertainty). `residual` is the rolling-origin
   backtest residual variance (half the trophy's own, half pooled);
   `jumpVar` is the variance of logit changes between distinct
   snapshots in the history — the measured refresh volatility. The
   backtest itself uses a 0.3 floor because its horizons are hours,
   not days; the backtest ranks candidates, it does not set the final
   spread.

## Backtest numbers (all on synthetic cases I built, not on the practice answers, which are not in the kit)

- **Smooth-trend case (s00):** p00 history extended 2 days, tracked
  638 → ~6,000, story trophies rising in logit vs log-time. Model
  follows the trend: chapter trophies 1.25% → median ~49%, final
  mission capped at 25% (gold cap), platinum/100% stay ~0.82%.
  Rolling-origin (long windows) pinball: earned 0.124, logtime 0.135,
  linear 0.286, persistence 0.127; house baseline 0.118 on the same
  origins. Picks: 38 persistence / 17 earned. On smooth short
  horizons the narrow baseline ties — as it should; the advantage
  only appears at the 4.5-day horizon with refreshes.
- **Refresh-dilution case (s04):** tracked ×1.9 per refresh, earned
  ×1.9 for story / ×1.25 for grind, so story rates flat and grind
  rates decay. Model pinball vs the known next state: 0.394, coverage
  0.98; house baseline from the same last rates: 0.251, coverage 0.98.
  The wide spread costs ~0.14 when the world is calm — the price of
  surviving a refresh.
- **Practice sensitivity (p00, assumed answers):** if the refresh
  scales rates by 0.165 (earned frozen) the model scores ~1.02–1.16
  vs the baseline's ~1.69; at ×0.33, ~0.61 vs ~1.01; at ×1.0 (no
  refresh, contradicting the rules) ~0.24 vs ~0.02. A sigma floor of
  1.1 beat 0.9 across the plausible refresh scales, which is why the
  floor is 1.1.
- **Robustness:** flat history, single-read history, and histories
  with null rates all run without crashing and produce valid output.

## Practice outputs (medians = last rates, as they must be on a flat history)

p00/p01: median range 0.63–99.9%; a 0.63% trophy forecasts
[0.10, 0.30, 0.63, 1.31, 3.73] (sigma 1.1); the 100% trophy forecasts
median 99.9%. Output ≤ 65,536 bytes, 55 rows, 5 non-decreasing
quantiles each.

## Tests

- Starter baseline and `main.ts` on p00/p01, each run twice:
  byte-identical outputs, ~0.17 s per run (limit 8 s).
- Synthetic cases s00 (trend), s01 (single read), s02 (null rates),
  s03 (flat, final horizon), s04 (refresh dilution): all valid shape,
  deterministic across two runs.
- Kit check: 4 passed, 1 failed, 14 skipped, 5 checked at submission.
  The one failure is `transcripts_present` (no transcripts bundle was
  supplied to the local check; the parent bundles transcripts at
  submit time).
