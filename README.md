# NFL Fantasy Draft Tool — 2026/27

A projection engine and draft optimiser for **Imperial Immortals 2026** (ESPN):
14-team, full-PPR, snake, no keepers, `QB / 2RB / 2WR / TE / FLEX / K / DST`
starters over a 15-man roster. Head-to-head, 14 regular-season weeks, six of
fourteen teams reach a three-week bracket.

Built as two tracks behind one interface. **Track 1** is market-anchored and
drives the draft board. **Track 2** is the ML projection model, built to replace
it behind the same `projections.parquet` contract if it could beat ADP out of
sample.

**It couldn't** — see [Track 2](#track-2-the-ml-model-and-why-it-does-not-ship-as-the-projection).
ADP won 2021-2025 against every formulation tried. So Track 1 still drives the
board, and the model ships only as a disagreement flag. The gate did its job:
it stopped a worse projection from shipping.

---

## Quick start

```bash
pip install nflreadpy pytest

python scripts/build_projections.py          # projections + value report
python scripts/cheatsheet.py --slot 8        # printable fallback (open, print to PDF)
python -m src.dashboard.app --slot 8         # live draft board -> localhost:8777
python scripts/draft_plan.py --slot 8        # round-by-round prep
python scripts/pick_reliability.py --slot 8  # how firm each round's pick actually is
python scripts/strategy_table.py             # which strategy wins, by slot
pytest -q                                    # 80 tests

# Analysis (needs the thread cap on a low-memory machine)
POLARS_MAX_THREADS=2 python scripts/eda_yoy.py --write-report
POLARS_MAX_THREADS=2 python scripts/backtest_model.py
POLARS_MAX_THREADS=2 python scripts/disagreements.py
POLARS_MAX_THREADS=2 python scripts/signings_ablation.py --seeds 20
```

`--slot` is your draft position (1-14).

## Using the board during a draft

Type a name, then <kbd>Enter</kbd> to mark them taken by someone else or
<kbd>Shift</kbd>+<kbd>Enter</kbd> to mark them as yours. The centre panel re-ranks
in about a fifth of a second. Search matches team nickname and abbreviation as
well as name, so the defence the room calls "Ravens D/ST" answers to `ravens`,
`bal` or `baltimore`. Read it in this order:

1. **Position urgency** — what you lose by waiting one round at each position.
   This is the pick signal. If QB urgency is 0, taking a QB is pure waste.
2. **Tier + "n left"** — turns red at 2 or fewer. Being last in a tier is the
   only good reason to reach.
3. **VONA** — value over what actually survives to your next pick.
4. **Survives %** — probability that player lasts until you pick again.
5. **"X% best" and the cost line in the verdict** — how often this pick wins
   across plausible projections, and what the runner-up would cost you. When
   that cost is under a point or two, stop deliberating and take either.

---

## How the projections work

There are no hand-made numbers in here. The market supplies the *ordering*;
history supplies what that ordering has been *worth*.

1. Fit, per position, `positional ADP rank -> actual fantasy points` over
   2016-2025 using isotonic regression.
2. Apply the fitted curve to live 2026 ADP.

Fitting against **preseason rank** rather than end-of-season finish is what keeps
this honest. The RB drafted first has averaged **221 PPR points**, not the ~370
the eventual RB1 scores — the difference is winner's curse, and projections that
ignore it systematically overvalue the top of the draft.

The same fit yields empirically calibrated uncertainty, and it is large: **SD ≈
150 points at the top of the RB board**. Every risk calculation downstream uses
that measured spread rather than an assumed one.

Isotonic regression also gives tiers for free. Where the fit plateaus, the data
genuinely cannot separate those players — that plateau *is* the tier.

## Value over replacement

Replacement level is derived, not hardcoded. Base starters come from the league
config; FLEX slots go to whichever RB/WR/TE actually earn them.

For this league that produces:

| Pos | Starters | Replacement |
|-----|----------|-------------|
| QB  | 14       | **235.1**   |
| RB  | 28       | 133.5       |
| WR  | **42**   | 140.8       |
| TE  | 14       | 127.5       |

Two findings worth internalising:

- **All 14 FLEX slots go to WR.** Replacement is WR42 but only RB28, which makes
  the deep WR pool genuinely worthless: past ADP 120, WRs average **-48 VOR** and
  RBs -51. Every late-round position is below replacement, but TE is least below
  it (-16), so a late flier at TE costs you least. That is a statement about
  minimising loss, not about finding value.
- **QB replacement is 235 points.** Josh Allen projects as the highest-scoring
  player in football (318.7) and is still only the *20th* most valuable pick.
  Drake Maye has identical VOR at 17 picks later ADP.

## VONA, and why not VOR

VOR asks "how much better than a waiver pickup is this player?" That is the wrong
question at the table. The alternative to drafting a WR now is not a
replacement-level WR — it's the best WR still there at your next pick.

So player value is **lineup-marginal**: how much they add to your optimal starting
lineup given what you already own. That makes the tool decline a third QB without
any hand-written rule, and prices the FLEX slot correctly. VONA subtracts the
simulated expected best-available at that position when you next pick.

## How much to trust a pick

Every number on the board is an estimate, and the board used to present them as
though they were not. The recommendation is now re-run a few hundred times on
different plausible projections, and reports what survives.

Two sources of error, both measured rather than assumed:

- **The curve.** The isotonic fit is bootstrapped 200 times and the refits are
  kept *whole*, not reduced to a per-rank standard error. Isotonic refits move in
  correlated blocks -- a draw that pushes RB4 down pushes RB5 with it -- and a
  shift common to an entire position changes nothing about which player to take
  within it. Summarising to a standard deviation throws exactly that away.
- **The ordering.** Each player's ADP is perturbed by his own observed `stdev`
  and the board re-ranked, because a player the room disagrees about does not
  have a well-determined positional rank to look up.

That yields two new columns and one new line in the verdict:

| | |
|---|---|
| `p_best` | how often this player tops the board across draws |
| `regret` | what taking him instead of that draw's own best choice costs, in projected points |

**Read `regret`, not `p_best`.** They come apart exactly where it matters. Once
the top of the board is a plateau, "best" is a lottery between players worth the
same amount, and 25% confidence with half a point of regret is not a hard
decision — it is a decision that does not matter. `p_best` says how often a pick
wins; `regret` says what losing costs.

The board also now says when the two ways of ranking disagree. Sorting by VONA
at the mean projection and sorting by mean VONA across draws are not the same
operation, because VONA is a maximum and maxima do not commute with averages.
When they name different players, the dashboard says so — that disagreement is
itself the reliability signal.

Two things this deliberately does not claim:

- **It is a lower bound.** It covers error in the fitted curve and in the draft
  ordering. It cannot cover the market being collectively wrong about a player,
  which is the largest error of all and is not measurable from this data.
- **`sd` and `proj_se` are different columns and are not interchangeable.** `sd`
  (~150 at the top of the RB board) is how far a *season* lands from the curve:
  it is what risk and upside calculations need. `proj_se` is how far the *curve*
  would move if the ten seasons behind it had come out differently. Using the
  first where the second belongs makes every pick look like a coin flip; using
  the second where the first belongs makes a boom-or-bust player look safe.

A board with no error bars on it reports `verdict: unavailable` rather than 100%
confidence — the check is on whether the draws actually move, not on whether an
uncertainty column exists, because kickers and defences carry a stand-in
standard error that was enough to make a certainty-free board look measured.

The pass costs about 0.08s at 200 draws on a 260-player board, so the board
still re-ranks in roughly a fifth of a second.

## Strategy results (slot 8, 600 simulated drafts per policy)

Scored the way the league is actually decided: fourteen head-to-head weeks, top
six seeded into a three-week single-elimination bracket. Baselines are a 0.071
title rate and a 0.429 playoff rate.

| Policy | Title rate | Playoff rate | vs best |
|---|---|---|---|
| Robust-RB | 0.094 ± 0.002 | 0.493 | — |
| BPA | 0.093 ± 0.002 | 0.491 | not significant |
| Hero-RB | 0.093 ± 0.002 | 0.490 | not significant |
| Elite-TE | 0.092 ± 0.002 | 0.494 | not significant |
| Zero-RB | 0.088 ± 0.002 | 0.476 | 2.1σ |
| Late-QB | 0.084 ± 0.002 | 0.483 | **3.6σ** |
| Early-QB | 0.080 ± 0.002 | 0.457 | **5.0σ** |

**Head-to-head compresses everything.** An earlier version of this scored teams
by total season points and produced a 0.070–0.251 spread. That was measuring the
wrong thing: this league is won in a three-week bracket, and a bracket is mostly
noise. Even a team scoring 25 points per week more than everyone else makes the
playoffs 98% of the time but wins the title only 55%.

So the honest summary is that draft strategy moves your title odds from 7.1% to
about 9.4% — real, worth having, and far smaller than the total-points framing
implied. What survives:

- **The top four are indistinguishable.** Don't pre-commit to a script.
- **Both QB extremes still lose.** Early-QB costs ~4 points of playoff rate;
  refusing a QB for eight rounds is also significantly worse. The curve is flat
  then cliffs — Josh Allen and Drake Maye have identical VOR ~17 picks apart, and
  the tier breaks near pick 50. The window is **rounds 4-6**, which is
  independently where `draft_plan.py` puts QB urgency.
- **Zero-RB remains a loser here**, though only at 2.1σ.

**Playoff rate is the better metric.** It has less bracket noise in it than title
rate, and it separates the policies more cleanly.

## Kickers

Your league pays a sixth point for 60+ yard field goals, where most stop at 50+.
Tempting, but: kicker season points have a **year-over-year correlation of
0.109**, and 60+ makes are too rare (three in a season, at most) to show any
persistence at all. The K1-to-K14 gap was 62 points in 2025 and is not capturable.

**Take a kicker in the last round.** The unusual tier is not worth chasing.

## What a prior season actually tells you

`scripts/eda_yoy.py` measures this directly over 2015-2025 — 3,488 player-season
pairs, prior season filtered to 8+ games, next season kept **even when it is a
zero**. Dropping the zeros is the standard way this analysis gets flattered: 9.6%
of the sample never played the following season, and they are the outcomes the
exercise is about. Full tables in [`docs/eda_prior_season.md`](docs/eda_prior_season.md).

Every stat gets three numbers, and the third is the one that matters:

| | |
|---|---|
| predictive | Spearman(prior stat, next-season points) |
| repeatable | Spearman(prior stat, the same stat next season) |
| incremental | predictive, with the player's prior-season fantasy points partialled out |

A stat can be strongly predictive and carry no information: `points` predicts
points, and anything correlated with it inherits that for free.

| stat | predictive | repeatable | incremental |
|---|---:|---:|---:|
| `points` | +0.657 | +0.661 | — *(the control)* |
| `ppg` | +0.646 | +0.713 | +0.063 ± 0.018 |
| **`exp_ppg`** | **+0.638** | **+0.741** | **+0.099 ± 0.015** |
| `touches_pg` | +0.597 | +0.760 | +0.061 ± 0.021 |
| `target_share` | +0.553 | +0.683 | +0.049 ± 0.018 |
| `games` | +0.402 | +0.298 | −0.030 ± 0.017 |
| `points_oe_pg` | +0.047 | — | −0.038 ± 0.018 |

Five things fall out of it.

- **Expected points is the only feature that clearly adds to last year's box
  score.** `exp_ppg` — points per game implied by the opportunity a player was
  actually given — carries +0.099 over prior points, six standard errors from
  zero and comfortably the largest increment in the table. Everything else in the
  top half is mostly re-reading the same season.
- **Opportunity repeats; production does not repeat as well.** `touches_pg`
  predicts itself at +0.760 and `exp_ppg` at +0.741, against +0.661 for fantasy
  points. That is the whole case for modelling opportunity, stated as a number.
- **Availability barely repeats at all.** Games played predicts next season's
  games at **+0.298** — the weakest self-correlation in the table, and lower than
  most stats' correlation with a *different* quantity. Splitting the projection
  into points-per-game × games is right, but the second factor is close to
  unforecastable, and every stat predicts rate far better than availability
  (`ppg`: +0.691 against rate, +0.392 against games).
- **Points over expected regresses, but gently.** The top quintile of
  prior-season points-over-expected does score the most the following year
  (9.32 ppg) — because those players are better, not because the luck repeats.
  Holding prior ppg fixed the partial correlation is **−0.069**: real, negative,
  and far smaller than the "regression candidate" framing implies. It is a
  tie-breaker, not a fade signal.
- **A season two years old is nearly as good as last season.** `ppg` from one
  season out correlates +0.627 with next-season points; from two seasons out,
  +0.580 — **93% retained**, on the same (survivor-only) sample. Last season is
  not special; it is one draw from a stable underlying quality.

### And the spread is enormous

Prior-season points in within-position quintiles, against what happened next:

| quintile | prior | next | sd | p10 | p90 | bust |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 27 | 30 | 54 | 0 | 86 | 49% |
| 3 | 105 | 89 | 76 | 3 | 198 | 35% |
| 5 | 242 | 191 | 92 | 71 | 314 | 20% |

*bust = scored under half of last season's total.*

Even the best-informed bucket — players who just scored 242 points — has a
standard deviation of 92 the following year, a 10th percentile of 71, and a
one-in-five chance of losing half its production. This is the same number the
board's `sd` column reports, arrived at from the other direction, and it is why
the reliability layer exists.

### Age

Change in points per game from one season to the next:

| pos | <24 | 24-26 | 27-29 | 30+ |
|---|---:|---:|---:|---:|
| QB | −0.68 | −1.95 | −2.27 | −1.71 |
| RB | **+0.31** | −1.23 | −2.08 | −2.79 |
| WR | **+0.12** | −0.90 | −1.71 | −2.90 |
| TE | −0.03 | −0.38 | −0.73 | −1.47 |

Only under-24 running backs and receivers improve on average, and the decline at
30+ is worth roughly three points per game at both positions — about 50 points
over a season. Tight ends age most gently, by a wide margin.

## Track 2: the ML model, and why it does not ship as the projection

Built as planned: LightGBM over prior-season production and opportunity, expected
points and points-over-expected, age, NFL experience, draft capital, contract
terms (cap share, years left, contract year), combine athleticism, vacated
opportunity on the new team, and the Vegas implied total.

```bash
python scripts/backtest_model.py     # the ADP gate
python scripts/disagreements.py      # where model and market differ, and whether that pays
```

**It lost to ADP. Repeatedly, and by a consistent margin.**

Backtest 2021-2025, within-position Spearman on drafted players, trained only on
prior seasons:

| Approach | rho | Edge vs ADP | Beat ADP |
|---|---|---|---|
| **ADP (benchmark)** | **+0.504** | — | — |
| Pure model (market-blind) | +0.451 | −0.053 ± 0.019 | 0/5 |
| Market-anchored (ADP as a feature) | +0.473 | −0.031 ± 0.017 | 1/5 |
| Residual correction to ADP | +0.494 | −0.011 ± 0.008 | 2/5 |

The ordering is the finding: **every formulation that respects the market more
does better**, converging on ADP from below without ever crossing it. Feature
importance says why — ADP and its dispersion account for **52.5%** of the model's
total gain. Contract terms, experience, prior production, draft capital and
combine testing *combined* contribute less than the market signal alone.

That is not a bug to tune away. ADP aggregates thousands of real drafters plus
expert consensus plus every injury report up to the moment it was sampled. A
model built from box scores and contracts is not going to strictly dominate that,
and a version of this project that claimed otherwise would be overfitting or
leaking.

### Improving it: what worked and what didn't

Every configuration below is averaged over 20 random seeds. That is not
pedantry: LightGBM's bagging is stochastic, and the **run-to-run spread in edge
is 0.0070** -- larger than most of the effects being tested. Single-seed
before/after comparisons on this dataset measure noise, and each individual run
looks perfectly convincing on its own. Seeds are now pinned, with a regression
test to keep them that way.

| Configuration | Edge vs ADP | Δ vs base |
|---|---|---|
| Base | −0.0142 | — |
| + preseason depth chart | −0.0112 | +0.0030 ± 0.0009 (unstable, see below) |
| + injury history | −0.0074 | +0.0068 ± 0.0009 **real** |
| + college data | −0.0048 | +0.0093 ± 0.0010 **real** |
| **+ rank-residual objective** | **+0.0003 ± 0.0013** | +0.0145 ± 0.0008 **real** |
| Compact ridge, 8 features | −0.0085 | worse |
| Hierarchical (opportunity × games) | −0.0140 | worse |

**The one robust finding, across every experiment run:** anything predicting
*absolute outcomes* loses to ADP; anything predicting a *correction to ADP* wins.
Pure model −0.053, hierarchical −0.014, points-residual −0.005, rank-residual
+0.000. The market is the right prior, and the model's job is to nudge it.

Three hypotheses that were **wrong**:

- *Fewer features and a linear model will beat the booster.* It doesn't — the
  compact ridge is consistently worse. The signal is non-linear enough to need
  the trees.
- *Hierarchical decomposition beats direct prediction.* Not as implemented,
  because predicting opportunity × games throws away the ADP anchor. A fair test
  would rebuild it as a residual; it has not been run.
- *Depth charts are the biggest missing feature.* Its measured effect flips sign
  between runs (+0.0030 at 20 seeds, −0.0002 at 12). Current-season caches expire
  between runs and shift the feature table, so this is **unproven**.

### New signings, and the leak that nearly sold them

The feature table knew about **vacated** opportunity — targets and carries whose
owner is no longer on the roster — and nothing about **arrivals**. That is half
an offseason. A team that loses 120 targets and signs a receiver who saw 140
somewhere else has opened nothing up, and a team that replaces its quarterback
with a better one has improved every pass-catcher it already had.

`src/features/roster_churn.py` adds the other half: 21 features covering arriving
volume at team and position level, the player's own position group net of his own
arrival, incoming draft capital and cap money spent on the competition, and how
the current quarterback room's best prior season compares to what the offense
actually got last year. `scripts/signings_ablation.py` puts them through two
gates — market-blind signal first, then the ADP edge that decides whether
anything ships.

**They clear the first gate, narrowly:** within-position rho against next-season
points goes from **+0.6377 to +0.6413, a gain of +0.0036 ± 0.0009** over 20 seeds
and five seasons. Real, and small — four standard errors from zero, and about a
quarter the size of the rank-residual objective that closed the ADP gap. They
take 13.5% of the model's total gain when offered, led by incoming rookie draft
capital at the position and the quarterback room.

**But the first version of this measured +0.0150 — four times as much — and all
of the difference was a leak.**

`nflreadpy.load_rosters` returns a season-level snapshot taken at the *end* of
the year. A player traded in October is listed with the team that acquired him:
in 2024, **12.3% of skill players sit on a different team there than they did in
week one**. Every feature built on roster membership was therefore reading
midseason transactions on draft day — including `vacated_targets`, which has been
in the model from the start. Rosters now come from the week-one weekly snapshot
for completed seasons and fall back to the live roster for the season being
drafted, which for a season that has not started is the same thing.

| Roster snapshot | without churn | with churn | delta |
|---|---:|---:|---:|
| End-of-season (leaking) | +0.6355 | +0.6505 | +0.0150 ± 0.0012 |
| **Week one (correct)** | **+0.6377** | **+0.6413** | **+0.0036 ± 0.0009** |

Three quarters of the effect was the leak. `--roster-source end-of-season`
reproduces the bad number, and a test in `tests/test_roster_churn.py` fails the
build if the roster snapshot ever drifts back.

The signing features stay **behind a flag** (`use_churn=True`) rather than in the
default model, because clearing a market-blind gate is not the same as beating
ADP. The gate that decides is `signings_ablation.py --gate market`, and the
market gate needs historical ADP — see the note in *Known limitations* about
running it where Fantasy Football Calculator is reachable.

### Where it ends up

Parity, not victory. The final model scores **+0.0003 ± 0.0013 against ADP** --
about 1σ from zero, over a range of −0.0020 to +0.0025. The gap closed from
−0.0142, which is a real engineering result, but a model that ties the market is
not a model that beats it.

It still ships as a disagreement flag, not as the projection.

### Two dead ends worth recording

**ADP velocity** and the **closing-line test** both need dated historical ADP.
Fantasy Football Calculator ignores every date parameter tried (`start_date`,
`end_date`, `date`, `period`) and always returns the same rolling seven-day
window. Neither is testable on free data. Snapshotting ADP daily from now on
would make both available next season.

## Simulator calibration

Perturbing ADP and re-ranking does **not** give you back ADP — ranking is a
nonlinear transform, so high-dispersion players get pulled toward the middle and
the ends of the board compress. Uncorrected, late-round survival was biased by
~10 picks.

`calibrate()` solves for the latent means that reproduce the input ADP. Mean
absolute error: **5.32 → 1.04 picks**. `tests/test_draft.py` enforces this — if
the bots stop reproducing ADP, every survival probability above is wrong, so it
fails the build.

---

## Data sources

All free, all verified live during the build.

| Source | Contents |
|---|---|
| [nflverse](https://nflreadpy.nflverse.com/) via `nflreadpy` | pbp 1999+, weekly stats, rosters, snap counts 2012+, NGS 2016+, depth charts, injuries, PFR advanced |
| nflverse `games.csv` | 2026 schedule **with Vegas spread/total already posted** — 272 games |
| [FFC ADP API](https://help.fantasyfootballcalculator.com/article/42-adp-rest-api) | live PPR ADP with `stdev`/`high`/`low`; historical to 2012 |
| `load_ff_rankings()` | FantasyPros consensus (ranks, not points) |
| nflverse weekly rosters | week-one roster membership — the only snapshot that is knowable on draft day |

ADP data courtesy of [Fantasy Football Calculator](https://fantasyfootballcalculator.com).
Cached to disk and refreshed at most every 12 hours, per their request.

`nfl_data_py` is deprecated — this uses `nflreadpy`, its replacement.

The ffverse ID crosswalk falls back to `raw.githubusercontent.com` when
nflreadpy's `github.com/.../raw/` redirect is unavailable. Same repository, same
file; some networks block one path and not the other, and losing the crosswalk
silently drops every combine feature.

## Identity resolution

Joining FFC names to nflverse IDs is where this kind of project silently breaks: an
unmatched player doesn't error, he just vanishes from your board. Three real
collisions in the live data:

- **Marvin Harrison Jr.** normalises to the same key as his father
- Two **D.J. Moore**s (a WR and a CB)
- Retired duplicate rows carrying no `gsis_id` that win a naive join

Resolution matches on name + position + team, prefers rows that have a `gsis_id`,
then falls back to surname matching. Currently **221/221 skill players resolve,
zero duplicate IDs**, and the test suite fails the build if that ever regresses.

---

## Rookies and college data

CFBD college data is wired in via `CFBD_API_KEY` in a gitignored `.env`: usage
share, receiving and rushing production (final and peak seasons), and pre-draft
scouting grades. The bridge is the draft itself, since nflverse stores a slug
where CFBD stores a numeric id.

Rookies are where the market should be weakest — no NFL snaps to price off. Even
there, it wins. Pooled over 2020-2025, n=128 drafted rookies with an ADP:

| | rho |
|---|---|
| **ADP** | **+0.458** |
| Model, no college data | +0.427 |
| Model, with college data | +0.431 |

College data helps by +0.004, which is noise at this sample size, and the model
still trails ADP by 0.027. ADP already embeds the same draft capital and college
production, plus beat reporting and preseason usage that this pipeline cannot see.

## Known limitations

Stated plainly, because they bound what the output means.

- **Projections carry no independent information about players.** They are ADP
  rank mapped through a historical curve, so by construction they cannot find a
  player the market has mispriced. All edge comes from the *decision* layer —
  VOR, VONA, tiers, roster construction. Track 2 was built to change that and
  **did not**: see above. The market wins on ranking; we win on what to do with
  the ranking.
- **The disagreement flags have not cleared significance** (1.8σ, driven largely
  by one season). Treat them as a prompt to look harder at a player, never as a
  reason to override ADP by several rounds.
- **Pick confidence is a lower bound.** `p_best` and `regret` propagate error in
  the fitted curve and in the draft ordering. They cannot propagate the market
  being wrong about a player, which is the biggest error there is. A 90%
  confident pick is 90% confident *given that ADP is right about him*.
- **The signing features have only cleared the market-blind gate.** +0.0036 ±
  0.0009 rho against next-season points is not the same claim as beating ADP,
  and they stay behind `use_churn=True` until `signings_ablation.py --gate
  market` says otherwise. That run needs historical ADP, which only Fantasy
  Football Calculator supplies; on a network that cannot reach it the gate
  reports SKIPPED rather than inventing a number.
- **Low-memory machine.** The feature pipeline needs `POLARS_MAX_THREADS=2` on a
  laptop with ~1 GB free; polars' per-thread arenas will otherwise exhaust it.
- **The strategy table assumes the projections are true.** It compares policies
  fairly against each other, but its absolute win rates are optimistic and are
  not a forecast of your finish in a real league.
- **Bots follow ADP with roster-awareness.** Real managers reach, panic, and run
  positional runs. Survival probabilities are averages, not guarantees.
- **K and DST are not projected.** They are near-random year to year and go in
  the last two rounds. Take them last.
- **No injury news.** Anything from the last 24 hours is not in here. Check
  before you draft.
- **One live source for ADP.** If Fantasy Football Calculator is unreachable on
  draft morning the board now falls back through the disk cache to the newest
  committed snapshot in `data/adp_history/`, warning loudly about the date it is
  using. A stale board is wrong at the margins; no board is worse. Run
  `scripts/snapshot_adp.py` daily through preseason so that fallback is fresh.

## Layout

```
src/
  config.py            LEAGUE settings + scoring rules -- change these, everything follows
  scoring.py           PPR scoring (pure, cross-checked against nflverse)
  ingest/              nflverse loaders, FFC ADP, CFBD college data, cache, ID resolution
  features/            player-season substrate, the model's feature table,
                       roster churn (arrivals, QB room), year-over-year EDA
  models/              LightGBM points-per-game and games-played models
  project/             actuals, consensus curves + their bootstrap, season Monte Carlo
  draft/               replacement, tiers, VONA, pick reliability, draft simulator,
                       H2H bracket, board state
  dashboard/           stdlib HTTP server + single-page board
scripts/               build_projections, cheatsheet, draft_plan, pick_reliability,
                       strategy_table, backtest_model, disagreements, rookie_model,
                       eda_yoy, signings_ablation
docs/                  eda_prior_season.md -- full year-over-year correlation tables
tests/                 80 tests
```

## Test coverage

- Scoring reproduces nflverse's own `fantasy_points_ppr` across the full 2025
  season (max divergence < 0.02 over ~5,000 stat lines)
- Every skill player in live ADP resolves to a unique nflverse ID
- Draft simulator recovers input ADP after calibration
- Lineup maths: FLEX allocation, and that a QB2 is worth less than a startable WR3
- **Leakage guards**: every production feature is verifiably lagged, `ppg_lag1`
  matches the prior season exactly, no feature reproduces the target, and the
  roster snapshot must match week one — the guard for the end-of-season roster
  leak that inflated the signing features fourfold
- **Pick confidence degrades to "unavailable", never to certainty**: a board
  carrying no error bars reports no confidence rather than 100%
- **The marginal-value interpolation is exact**: the reliability pass evaluates
  lineup gain at its kinks and interpolates, and every confidence number is wrong
  if that is not exact to floating point
- **A signing is not counted as his own competition**
