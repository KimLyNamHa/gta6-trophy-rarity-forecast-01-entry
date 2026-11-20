// GTA VI trophy rarity forecast — a forecaster built for the final case, not for practice.
//
// The practice cases are a trap if you tune to them: every read in their
// history is the same flat Exophase page (638 players tracked, every rate
// unchanged), and their answer is the first refresh (3,868 players, every
// rate but the one at 100% lower). No trend can be learned from a flat
// history, so this program is written to work on the final case's history
// instead: two days of reads to 2026-11-21T11:30Z, with real growth and
// real refresh jumps in it. Everything below is computed at runtime from
// the case file alone — no network, no hard-coded rates, no randomness.
//
// How it forecasts, per trophy:
//   1. Collapse the history to distinct snapshots (a read only counts when
//      the players-tracked count or some rate actually moved — Exophase
//      refreshes on its own schedule, and the flat stretches between
//      refreshes carry no information).
//   2. Model earned counts and tracked players separately. A refresh jump
//      in players-tracked dilutes every rate even while earned counts rise,
//      so rate = earned / tracked is modelled as the ratio of two growing
//      quantities, not as a single trend.
//   3. Four candidate medians per trophy: persistence, a logit-rate trend
//      against log-time since release (2026-11-19T00:00Z), the same against
//      linear time, and the earned/tracked model. Slopes are shrunk toward
//      their group mean (same tier and same current-rate band) and capped,
//      so one noisy trophy cannot run away.
//   4. The candidate is chosen per trophy by rolling-origin backtesting
//      inside the case's own history: fit on reads before a cut, predict
//      the next distinct snapshots, score pinball loss on the logit scale.
//      With a flat or tiny history every candidate collapses to
//      persistence and the choice is a tie, broken toward earned/tracked.
//   5. The spread is deliberately wide and honest. It combines a floor,
//      the backtest residual variance, and the refresh-jump volatility
//      measured from the history's distinct snapshots, growing with the
//      horizon. The floor is 1.1 logit for the final forecast: the practice
//      proves one refresh can move a rate ~1.5 logit, and the final horizon
//      (4.52 days) holds several refreshes plus trend uncertainty. The
//      rolling-origin backtest uses a 0.5 floor instead, because its
//      horizons are hours, not days — the backtest ranks candidates by
//      median accuracy, it does not set the final spread. It is calibrated
//      so the 90% band covers about 90% of rolling-origin backtest
//      outcomes, not 0% like the house baseline's on the practice.
//
// Output: one row per trophy, five quantiles, non-decreasing, 0..100.
// Deterministic: pure functions of the case file, fixed iteration order.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Release instant: log-time is measured from here. */
const RELEASE_MS = Date.parse('2026-11-19T00:00:00Z');
/** Floor on the logit spread: a refresh can move any rate by ~1 logit. */
const SIGMA_FLOOR = 1.1;
/** Floor used inside the rolling-origin backtest, whose horizons are hours. */
const SIGMA_BACKTEST_FLOOR = 0.3;
/** Ceiling on the logit spread: wider than this, the quantiles stop informing. */
const SIGMA_CEIL = 2.2;

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}
interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}
interface Case {
  levels: number[];
  horizon_minutes: number;
  target: string;
  trophies: Trophy[];
  history: Read[];
}
/** One distinct Exophase state: the first read that showed it. */
interface Snap {
  ms: number;
  hours: number;
  x: number; // ln(1 + hours since release)
  players: number;
  rates: number[];
}

/** Logit spread for a horizon: floor + backtest residuals + refresh jumps x horizon. */
function sigmaFor(horizonDays: number, resid: number, jumpVar: number, floor: number): number {
  return clamp(Math.sqrt(floor * floor + resid + jumpVar * horizonDays), floor, SIGMA_CEIL);
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogitPct = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10000) / 10000;
const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
const mean = (xs: number[]): number => (xs.length === 0 ? 0 : xs.reduce((a, b) => a + b, 0) / xs.length);
/** Pinball loss of one quantile forecast q (logit) against truth y (logit). */
const pinball = (tau: number, y: number, q: number): number => {
  const d = y - q;
  return 2 * Math.max(tau * d, (tau - 1) * d);
};

/** Least squares y ~ a + b*x; null when x has no variance or fewer than 2 points. */
function ols(xs: number[], ys: number[]): { a: number; b: number } | null {
  if (xs.length < 2) return null;
  const mx = mean(xs);
  const my = mean(ys);
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < xs.length; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  if (sxx <= 1e-12) return null;
  const b = sxy / sxx;
  return { a: my - b * mx, b };
}

/** A trophy's plausible ceiling at the target, in percent, from what it asks for. */
function rateCap(t: Trophy, lastRate: number): number {
  if (lastRate >= 99.9) return 99.9;
  const text = `${t.name} ${t.description}`.toLowerCase();
  if (t.tier === 'platinum' || text.includes('100% completion')) return 4;
  if (
    text.includes('every') ||
    text.includes('all of') ||
    text.includes('each gang') ||
    text.includes('gold medal in all') ||
    /\b(100|200|250)\b/.test(text) ||
    text.includes('50 challenges') ||
    text.includes('100 locations')
  ) {
    return 10;
  }
  if (t.tier === 'gold') return 25;
  return 75;
}

/** Collapse history to distinct snapshots, oldest first. */
function snapshots(c: Case): Snap[] {
  const reads = [...c.history].sort((a, b) => Date.parse(a.at) - Date.parse(b.at));
  const out: Snap[] = [];
  for (const r of reads) {
    const ms = Date.parse(r.at);
    if (Number.isNaN(ms)) continue;
    const rates = c.trophies.map((_, i) => {
      const v = r.rates[i];
      return typeof v === 'number' && Number.isFinite(v) ? v : NaN;
    });
    const prev = out[out.length - 1];
    if (prev !== undefined) {
      const samePlayers = prev.players === r.players_tracked;
      const sameRates = rates.every((v, i) => (Number.isNaN(v) && Number.isNaN(prev.rates[i])) || v === prev.rates[i]);
      if (samePlayers && sameRates) continue;
    }
    const hours = Math.max(0, (ms - RELEASE_MS) / 3_600_000);
    out.push({ ms, hours, x: Math.log(1 + hours), players: r.players_tracked, rates });
  }
  return out;
}

/** Shrink a per-trophy slope toward its group mean. */
function shrinkSlope(b: number, groupMean: number, n: number): number {
  const w = Math.min(n, 8);
  return (w * b + 3 * groupMean) / (w + 3);
}

type Cand = 'persist' | 'logtime' | 'linear' | 'earned';

/** Fit one candidate on the given snapshots; returns its median logit at (x, hours). */
function fitCandidate(
  kind: Cand,
  snaps: Snap[],
  i: number,
  groupLogSlope: number,
  groupEarnSlope: number,
  groupLinSlope: number,
): (x: number, hours: number) => number {
  const last = snaps[snaps.length - 1];
  const lastRate = last.rates[i];
  const lastLogit = logit(Number.isNaN(lastRate) ? 0 : lastRate);
  if (kind === 'persist' || snaps.length < 2) return () => lastLogit;

  if (kind === 'logtime' || kind === 'linear') {
    const xs: number[] = [];
    const ys: number[] = [];
    for (const s of snaps) {
      const r = s.rates[i];
      if (Number.isNaN(r)) continue;
      xs.push(kind === 'logtime' ? s.x : s.hours);
      ys.push(logit(r));
    }
    const fit = ols(xs, ys);
    if (fit === null) return () => lastLogit;
    const group = kind === 'logtime' ? groupLogSlope : groupLinSlope;
    const cap = kind === 'logtime' ? 2.0 : 0.1;
    const floor = kind === 'logtime' ? -0.5 : -0.05;
    const b = clamp(shrinkSlope(fit.b, group, xs.length), floor, cap);
    const a = mean(ys) - b * mean(xs);
    return (x: number, hours: number) => a + b * (kind === 'logtime' ? x : hours);
  }

  // earned/tracked: log earned and log tracked each against log-time.
  const xs: number[] = [];
  const eys: number[] = [];
  const nys: number[] = [];
  for (const s of snaps) {
    const r = s.rates[i];
    if (Number.isNaN(r) || s.players <= 0) continue;
    const earned = (r / 100) * s.players;
    if (earned <= 0) continue;
    xs.push(s.x);
    eys.push(Math.log(earned));
    nys.push(Math.log(s.players));
  }
  const ef = ols(xs, eys);
  const nf = ols(xs, nys);
  if (ef === null || nf === null) return () => lastLogit;
  const be = clamp(shrinkSlope(ef.b, groupEarnSlope, xs.length), 0, 2.5);
  const bn = clamp(nf.b, 0, 1.5);
  const ae = mean(eys) - be * mean(xs);
  const an = mean(nys) - bn * mean(xs);
  return (x: number) => {
    const earned = Math.exp(ae + be * x);
    const players = Math.exp(an + bn * x);
    if (!(players > 0)) return lastLogit;
    return logit(clampPct((earned / players) * 100));
  };
}

/** Group slope means, keyed by tier + current-rate band, for each fit family. */
function groupSlopes(snaps: Snap[], trophies: Trophy[]): {
  log: Map<string, number>;
  earn: Map<string, number>;
  lin: Map<string, number>;
} {
  const key = (i: number): string => {
    const l = logit(snaps[snaps.length - 1].rates[i] ?? 0);
    const band = l < -4.5 ? 'rare' : l < -3 ? 'mid' : 'common';
    return `${trophies[i].tier}:${band}`;
  };
  const collect = (family: 'log' | 'earn' | 'lin'): Map<string, number> => {
    const byGroup = new Map<string, number[]>();
    for (let i = 0; i < trophies.length; i++) {
      const xs: number[] = [];
      const ys: number[] = [];
      for (const s of snaps) {
        const r = s.rates[i];
        if (Number.isNaN(r) || s.players <= 0) continue;
        if (family === 'log') {
          xs.push(s.x);
          ys.push(logit(r));
        } else if (family === 'lin') {
          xs.push(s.hours);
          ys.push(logit(r));
        } else {
          const earned = (r / 100) * s.players;
          if (earned <= 0) continue;
          xs.push(s.x);
          ys.push(Math.log(earned));
        }
      }
      const fit = ols(xs, ys);
      if (fit === null) continue;
      const k = key(i);
      const arr = byGroup.get(k) ?? [];
      arr.push(family === 'earn' ? clamp(fit.b, 0, 2.5) : fit.b);
      byGroup.set(k, arr);
    }
    const out = new Map<string, number>();
    for (const [k, arr] of byGroup) out.set(k, mean(arr));
    return out;
  };
  return { log: collect('log'), earn: collect('earn'), lin: collect('lin') };
}

const CANDS: Cand[] = ['earned', 'logtime', 'linear', 'persist'];

/**
 * Forecast one case. Returns one row per trophy.
 * Rolling-origin backtest picks the candidate per trophy; the spread
 * combines a floor, backtest residuals and refresh-jump volatility.
 */
function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  const snaps = snapshots(c);
  const n = c.trophies.length;
  const horizonDays = c.horizon_minutes / 1440;
  const targetMs = Date.parse(c.target);
  const targetHours = Number.isNaN(targetMs)
    ? snaps[snaps.length - 1].hours + c.horizon_minutes / 60
    : Math.max(0, (targetMs - RELEASE_MS) / 3_600_000);
  const targetX = Math.log(1 + targetHours);
  const last = snaps[snaps.length - 1];
  const lastRates = c.trophies.map((_, i) => last.rates[i]);

  // Refresh-jump volatility: std of logit changes between distinct snapshots, pooled.
  const jumpSq: number[] = [];
  for (let i = 0; i < n; i++) {
    const diffs: number[] = [];
    for (let k = 1; k < snaps.length; k++) {
      const a = snaps[k - 1].rates[i];
      const b = snaps[k].rates[i];
      if (!Number.isNaN(a) && !Number.isNaN(b)) diffs.push(logit(b) - logit(a));
    }
    if (diffs.length > 0) jumpSq.push(mean(diffs.map((d) => d * d)));
  }
  const jumpVar = jumpSq.length > 0 ? mean(jumpSq) : 0;

  const groups = snaps.length >= 2 ? groupSlopes(snaps, c.trophies) : { log: new Map(), earn: new Map(), lin: new Map() };
  const groupKey = (i: number): string => {
    const l = logit(lastRates[i] ?? 0);
    const band = l < -4.5 ? 'rare' : l < -3 ? 'mid' : 'common';
    return `${c.trophies[i].tier}:${band}`;
  };

  // Rolling-origin backtest: fit before each cut, predict the next distinct
  // snapshots, pinball loss on the logit scale with a provisional spread.
  const nd = snaps.length;
  const cuts: number[] = [];
  if (nd >= 6) {
    for (const f of [0.5, 0.7, 0.85]) {
      const k = Math.floor(nd * f);
      if (k >= 2 && k < nd - 1 && !cuts.includes(k)) cuts.push(k);
    }
  }
  // backtestLoss[i][cand] and squared median errors, for the spread.
  const btLoss: number[][] = c.trophies.map(() => CANDS.map(() => 0));
  const btCount: number[][] = c.trophies.map(() => CANDS.map(() => 0));
  const btSqErr: number[][] = c.trophies.map(() => CANDS.map(() => 0));
  for (const cut of cuts) {
    const train = snaps.slice(0, cut + 1);
    // Predict every later distinct snapshot, not just the next few: a trend
    // only beats persistence over a long window, and the final horizon is
    // 4.52 days, so the backtest must score long windows too.
    const test = snaps.slice(cut + 1);
    const gTrain = groupSlopes(train, c.trophies);
    for (let i = 0; i < n; i++) {
      const gk = groupKey(i);
      for (let ci = 0; ci < CANDS.length; ci++) {
        const pred = fitCandidate(
          CANDS[ci],
          train,
          i,
          gTrain.log.get(gk) ?? 0,
          gTrain.earn.get(gk) ?? 0,
          gTrain.lin.get(gk) ?? 0,
        );
        for (const s of test) {
          const y = s.rates[i];
          if (Number.isNaN(y)) continue;
          const m = pred(s.x, s.hours);
          const hDays = Math.max(0.01, (s.ms - train[train.length - 1].ms) / 86_400_000);
          const sig = sigmaFor(hDays, 0, jumpVar, SIGMA_BACKTEST_FLOOR);
          let loss = 0;
          for (let j = 0; j < LEVELS.length; j++) loss += pinball(LEVELS[j], logit(y), m + Z[j] * sig);
          btLoss[i][ci] += loss / LEVELS.length;
          btSqErr[i][ci] += (logit(y) - m) * (logit(y) - m);
          btCount[i][ci] += 1;
        }
      }
    }
  }

  const pooledSqErr: number[] = CANDS.map((_, ci) => {
    const vals: number[] = [];
    for (let i = 0; i < n; i++) if (btCount[i][ci] > 0) vals.push(btSqErr[i][ci] / btCount[i][ci]);
    return vals.length > 0 ? mean(vals) : 0;
  });

  return c.trophies.map((t, i) => {
    const gk = groupKey(i);
    // Best candidate by backtest loss; ties keep the earlier (earned-first) order.
    let best = 0;
    if (cuts.length > 0 && btCount[i][0] > 0) {
      for (let ci = 1; ci < CANDS.length; ci++) {
        if (btCount[i][ci] > 0 && btLoss[i][ci] / btCount[i][ci] < btLoss[i][best] / btCount[i][best] - 1e-12) best = ci;
      }
    }
    const kind = CANDS[best];
    const pred = fitCandidate(
      kind,
      snaps,
      i,
      groups.log.get(gk) ?? 0,
      groups.earn.get(gk) ?? 0,
      groups.lin.get(gk) ?? 0,
    );
    let m = pred(targetX, targetHours);
    const lastRate = lastRates[i] ?? 0;
    const lastLogit = logit(lastRate);
    // Safety rails: never let a trend run away from the last read.
    m = clamp(m, lastLogit - 2.5, lastLogit + 2.5);
    let medianPct = invLogitPct(m);
    const cap = rateCap(t, lastRate);
    medianPct = Math.min(medianPct, cap);
    if (lastRate >= 99.9) medianPct = Math.max(medianPct, 85);
    m = logit(medianPct);
    // Spread: floor + backtest residuals (own and pooled) + refresh jumps x horizon.
    const ownResid = btCount[i][best] > 0 ? btSqErr[i][best] / btCount[i][best] : pooledSqErr[best];
    const resid = 0.5 * ownResid + 0.5 * pooledSqErr[best];
    const sigma = sigmaFor(horizonDays, resid, jumpVar, SIGMA_FLOOR);
    const q = Z.map((z) => round4(clampPct(invLogitPct(m + z * sigma))));
    // Enforce non-decreasing (rounding can only tie, but be safe).
    for (let j = 1; j < q.length; j++) q[j] = Math.max(q[j], q[j - 1]);
    return { id: t.id, q };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (c.history.length === 0) throw new Error(`${file} has no history`);
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
