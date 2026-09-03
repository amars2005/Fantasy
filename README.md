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
pip install -r requirements.txt

python scripts/build_projections.py          # projections + value report
python scripts/cheatsheet.py --slot 8        # printable fallback (open, print to PDF)
python -m src.dashboard.app --slot 8         # live draft board -> localhost:8777
python scripts/draft_plan.py --slot 8        # round-by-round prep
python scripts/strategy_table.py             # which strategy wins, by slot
pytest -q                                    # 52 tests

# Track 2 (needs the thread cap on a low-memory machine)
POLARS_MAX_THREADS=2 python scripts/backtest_model.py
POLARS_MAX_THREADS=2 python scripts/disagreements.py
```

`--slot` is your draft position (1-14).

## Using the board during a draft

Type a name, then <kbd>Enter</kbd> to mark them taken by someone else or
<kbd>Shift</kbd>+<kbd>Enter</kbd> to mark them as yours. The centre panel re-ranks
instantly (~0.13s). Search matches team nickname and abbreviation as well as
name, so the defence the room calls "Ravens D/ST" answers to `ravens`, `bal` or
`baltimore`. Read it in this order:

1. **Position urgency** — what you lose by waiting one round at each position.
   This is the pick signal. If QB urgency is 0, taking a QB is pure waste.
2. **Tier + "n left"** — turns red at 2 or fewer. Being last in a tier is the
   only good reason to reach.
3. **VONA** — value over what actually survives to your next pick.
4. **Survives %** — probability that player lasts until you pick again.

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

### What Proj means, and why a tier shares one number

A projection is the fitted curve read at a player's positional draft rank. It
is a function of that rank and nothing else — no per-player component — so two
players the fit could not separate get the *same* number by construction. That
is not a display bug; it is what a tier is.

The plateaus land where you would want them. On the reference board every tier
is a contiguous run of positional ranks, tight at the top and wide at the
bottom, because that is where the market's information actually runs out:

| | tier 1 | tier 2 | tier 3 | … | widest plateau |
|---|---|---|---|---|---|
| RB | ranks 1-4 | 5-7 | 8-11 | | ranks 47-61 (15) |
| WR | ranks 1-2 | 3-5 | 6-7 | | ranks 71-90 (20) |
| QB | ranks 1-2 | 3-5 | 6 | | ranks 25-32 (8) |
| TE | rank 1 | 2 | 3 | | ranks 9-16 (8) |

Rounding to one decimal merges nothing: tiering the unrounded fit gives an
identical tier count at every position.

The number that qualifies all of this is the **±** column — the measured spread
of what players at that rank went on to score, and it is not small:

| | rank 1 | rank 5 | rank 12 | rank 30 |
|---|---|---|---|---|
| RB | 255 ± 100 | 247 ± 98 | 211 ± 80 | 134 ± 72 |
| WR | 269 ± 84 | 249 ± 77 | 214 ± 75 | 157 ± 69 |
| QB | 319 ± 71 | 284 ± 87 | 243 ± 85 | 184 ± 86 |
| TE | 219 ± 60 | 157 ± 58 | 128 ± 57 | — |

The gap between adjacent tiers is routinely smaller than the spread within one.
Treat a projection as the middle of a range, not a forecast — which is also why
the board is ranked by VONA rather than by projected points.

### Choosing inside a tier

Worth being blunt about, because the board does not do it for you: **inside a
tier, Proj, VOR and VONA are all tied.** VONA is `marginal value − expected best
available at that position`, and both halves are identical for players sharing a
projection, so the order you see within a tier is just the ADP order it arrived
in — not a verdict. Sorting harder would only be inventing a preference the
model does not have.

Three columns still differ, and they are what there is to choose on:

| | what it tells you |
|---|---|
| **Left** | the *timing* signal, not a ranking. At 1 the tier is gone after him, and being last in a tier is the only good reason to reach. |
| **Bye** | red where it collides with players already on your roster. |
| **Playoff** | how much softer weeks 15–17 are for his team than the rest of its season, in opponent points allowed per game. Positive is good. |

Playoff lift is computed per league — a bracket running weeks 14–16 needs a
different number from one running 15–17 — from the opponents each team actually
faces in those weeks, measured by what those defences allowed last season.
Sportsbook lines would be the better measure and do not exist in August, which
is when you draft.

It is deliberately **not** priced into the ranking. It is a team-schedule effect
measured off last season's defences, which is a weaker thing than a projection,
and folding it into VONA would let it quietly outvote the market on players it
has no business separating. It sits in its own column for you to break a tie
with.

A real tier, and everything that distinguishes its members:

| player | Proj | ± | Bye | Playoff |
|---|---|---|---|---|
| Breece Robinson | 278.9 | 20 | 11 | **+3.8** |
| Bijan Robinson | 278.9 | 20 | 12 | +0.9 |
| Ja'Marr Robinson | 278.9 | 20 | 13 | **−2.0** |

### What the Tier and Left columns mean

Tiers are **within a position**, and numbered from 1 down. A WR tier 2 and an RB
tier 2 have nothing to do with each other; the position sits next to the number
on the board for exactly that reason. **Left** is how many players in that tier
are still on the board, the player himself included, so `1` means he is the last
of his kind — the only good reason to reach.

On the reference 14-team board that gives:

| | players | tiers | mean tier size | biggest |
|---|---|---|---|---|
| QB | 32 | 11 | 2.9 | 8 |
| RB | 71 | 16 | 4.4 | 15 |
| WR | 90 | 20 | 4.5 | 20 |
| TE | 27 | 9 | 3.0 | 8 |
| K | 24 | 24 | 1.0 | 1 |
| DST | 27 | 27 | 1.0 | 1 |

Kickers and defences are the exception, and it is not a bug in the fit. Their
curve is fitted on *outcome* rank rather than draft rank — kicker season points
have a year-over-year correlation near zero, so there is nothing better to be
had — and that fit separates all of them. Every kicker lands in a tier of one.

A column reading 1, 2, 3 … 27 down the defences is a rank wearing a tier's
clothes, and "1 left in this tier" against every one of them reads as scarcity
where there is none. So the board prints **—** for the tier and the count at any
position where the fit pooled nobody, rather than a number that cannot mean what
the column says it means.

Tagging a player under **News & risk** moves him to the tier his re-priced
projection earns. The rung is looked up on the existing ladder rather than
recomputed, so tagging one receiver never renumbers the tiers printed against
the others.

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

ADP data courtesy of [Fantasy Football Calculator](https://fantasyfootballcalculator.com).
Cached to disk and refreshed at most every 12 hours, per their request.

### The ADP here is not ESPN's

FFC's ADP is the consensus of their own mock drafts — 8,162 of them in the
2026-08-29 snapshot. ESPN, Yahoo and Sleeper each publish their own, from their
own drafters, and they disagree, most visibly on the positions with the least
separation between them. In that snapshot FFC has Seattle as the first defence
off the board at 82.1 with Houston third at 98.1; ESPN had Houston first. Both
are real numbers about real rooms, and neither is the room you are drafting in.

FFC is used because of what it carries that a ranking list does not: `stdev`,
`high` and `low`, the *distribution* of where a player goes. Pick-survival
probability — and so VONA, the number the board is ranked by — is built on that
distribution, not on the mean. A source without it could not answer "will he
still be there at my next pick", which is the only question the board asks.

If the room you are in prices a player differently, say so: tag him under **News
& risk** with the pick he really slides to. That is exactly what the field is
for.

### ADP is a pick number, and a pick number needs a team count

FFC's API takes a `teams` parameter and ignores it — verified 2026-08-30,
`teams=12` and `teams=14` return byte-identical ADP for all 271 players on every
scoring format. So there is one ADP, and it is a 12-team one.

That matters for kickers and defences, and only for them. A skill player's pick
number is set by how many players are better than him, and every team drafts
skill players continuously from round one, so the 100th-best running back comes
off around the 100th pick whatever the league size. A kicker or a defence is
drafted to fill a roster slot once the starters are done — that is a *round*,
and a round is `teams` picks wide. The first defence at 82.1 is round 7 of a
12-team draft; the same moment in a 14-team draft is pick 96.

Left uncorrected that is a real bias, and it grows with league size: the board
has defences coming off at pick 82 while a 14-team room is still four rounds
from touching one, so they look scarce, survival collapses and VONA spikes on a
position worth almost nothing. `to_league_pick_space` (`src/draft/board.py`, and
`toLeaguePickSpace` in `v2/lib/board.ts`) scales K and DST — `adp`, `adp_mu` and
`stdev` alike — by `teams / 12`. An eight-team league is corrected the other way.

`nfl_data_py` is deprecated — this uses `nflreadpy`, its replacement.

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

## Layout

```
src/
  config.py            LEAGUE settings + scoring rules -- change these, everything follows
  scoring.py           PPR scoring (pure, cross-checked against nflverse)
  ingest/              nflverse loaders, FFC ADP, CFBD college data, cache, ID resolution
  features/            player-season substrate + the model's feature table
  models/              LightGBM points-per-game and games-played models
  project/             actuals, consensus curves, season Monte Carlo
  draft/               replacement, tiers, VONA, draft simulator, H2H bracket, board state
  dashboard/           stdlib HTTP server + single-page board
scripts/               build_projections, cheatsheet, draft_plan,
                       strategy_table, backtest_model, disagreements, rookie_model
tests/                 52 tests
```

## Test coverage

- Scoring reproduces nflverse's own `fantasy_points_ppr` across the full 2025
  season (max divergence < 0.02 over ~5,000 stat lines)
- Every skill player in live ADP resolves to a unique nflverse ID
- Draft simulator recovers input ADP after calibration
- Lineup maths: FLEX allocation, and that a QB2 is worth less than a startable WR3
- **Leakage guards**: every production feature is verifiably lagged, `ppg_lag1`
  matches the prior season exactly, and no feature reproduces the target
