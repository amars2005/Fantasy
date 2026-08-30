# v2: Hosted, configurable draft tool

**Date:** 2026-08-30
**Status:** Approved design, ready for implementation planning

## Goal

Host the draft tool on Vercel so that anyone can run their own league on it, with
league settings (roster shape, arbitrary scoring) configurable per league and
draft state persisted durably across days.

All new code lives in `v2/`. The Python in `src/` is unchanged and remains the
reference implementation.

## Resolved decisions

| Question | Decision |
|---|---|
| Audience | Multi-tenant: many separate leagues, **one user per league**. Not shared boards within a league. |
| Runtime math | Ported to TypeScript. Python becomes a pure offline pipeline. |
| Scoring configurability | Arbitrary per-stat values, including kicker bands and DST bands. |
| League settings entry | Manual form + Sleeper import + ESPN cookie paste. |
| Auth | None. An unguessable league URL is the capability. |
| Sync | Reconcile on load and on window focus. No live polling. |

## Constraint: no regression to the existing tool

v2 consumes `src/` as a library. **No file under `src/` is modified.** The
existing local draft board, the 60 tests, and the Track 2 research line must work
exactly as they do today, before and after every phase.

Specific traps found while checking this, each with its resolution:

- **The `SCORING` column guard belongs in the export script, never in
  `src/scoring.py`.** That module's tolerance of missing columns is
  load-bearing — its own docstring cites position-filtered subsets and older
  seasons. Making it strict would break existing runs. `export_v2_bundle.py`
  performs the assertion on its own frames instead.
- **Do not use `load_adp_history` for the per-format export.** It takes `scoring`
  but no `teams`, so it falls through to `LEAGUE["teams"] = 14` and would
  silently return 14-team history for every format. Call
  `load_adp(year, scoring, teams)` directly in a loop. (`load_adp` is already
  fully parameterised, so this needs no change to `src/`.)
- **Export artefacts go to `data/v2_export/`**, never into `data/processed/`,
  so they cannot collide with `projections.parquet` or the existing draft state.
- **Golden fixtures go to `v2/tests/golden/`.** `.gitignore` excludes all of
  `data/raw/` and `data/processed/`, so fixtures placed under `data/` would be
  silently uncommitted and CI would have nothing to test against.
- **`v2/package.json` stays separate from the root one.** The root declares
  playwright but references it nowhere in the codebase, and its
  `"type": "commonjs"` would fight Next.js. Leave it alone.
- **`.gitignore` and CI changes are additive only** — `v2/node_modules/`,
  `v2/.next/`. `pyproject.toml`'s `testpaths = ["tests"]` is untouched; v2 tests
  run under vitest.

**Verification gate, not a promise:** `pytest -q` must show the same 60 passing
tests at the end of every phase as it does at the start. That is the check that
this constraint actually held.

## Why the current architecture cannot be lifted as-is

Three measured blockers:

1. **Bundle size.** Vercel's limit is 250 MB unzipped per function. Measured
   installed sizes: polars `_polars_runtime_32` 187.5 MB, numpy 31.7 MB,
   scipy + sklearn 157.2 MB. polars is smaller on Linux (~110-130 MB), but
   sklearn — pulled in only by the isotonic fits in `src/project/consensus.py`
   and `src/project/kdst.py` — pushes the bundle over the limit.
2. **Read-only filesystem.** `src/draft/persist.py` writes draft state to
   `data/processed/draft_state/`, and `src/ingest/cache.py` writes parquet to
   `data/raw/`. Vercel functions are read-only outside an ephemeral `/tmp`, so
   every cold start would re-download nflverse and re-hit FFC.
3. **Process-global board.** `src/dashboard/app.py` holds `STATE["board"]` and
   assumes one long-lived server. Board construction was measured at 2.58 s
   (imports 9.7 s, `recommend()` 0.114 s); rebuilding per request is untenable.

Moving the compute to the client resolves all three: nothing heavy is ever in a
request path.

## Architecture: three tiers

### Tier 1 — offline build (Python, GitHub Actions, daily)

New script `scripts/export_v2_bundle.py` reuses the existing pipeline and writes
to Vercel Blob:

- **`stat_components`** — per player-season sums of *raw stat columns* (not
  points) for 2016-2025, covering every key referenced by `SCORING`, with the
  three `FUMBLE_LOST_COLUMNS` pre-summed into one `fumbles_lost`. ~7k rows. This
  is what makes arbitrary scoring possible without re-running Python.
- **`adp_<scoring>_<teams>`** — per FFC format: current-season ADP with
  `stdev`/`high`/`low`/`bye`, historical positional-rank ADP for the fit's
  x-axis, and the precomputed calibrated `adp_mu`.
- **`kdst_components`** — per player-season kicker banded FG/PAT counts, and per
  team-game DST event counts plus `points_allowed` and `yards_allowed`. Per-game
  granularity is required because DST bands are non-linear and cannot be
  recovered from season totals. ~5,400 DST rows.
- **`schedule_difficulty`** — per-team weekly opponent difficulty, so playoff
  lift can be recomputed for whichever weeks a league calls its playoffs.
- **`golden/`** — test fixtures (see Testing).

`calibrate()` in `src/draft/sim_draft.py` (15 iterations x 4000 sims x 271
players) runs **here, not in TypeScript**. It depends only on `adp` and `stdev`,
never on league scoring, so `adp_mu` is precomputed per FFC format.

The job writes to `adp_<date>` and flips a "latest" pointer only on full success.
Stale ADP is acceptable; half-written ADP is not.

### Tier 2 — league setup (TypeScript, server, once per league)

On league creation or config change:

1. Select the nearest FFC format (see ADP anchoring).
2. Apply the league's scoring vector to `stat_components` to get historical
   player-season points.
3. Join to historical ADP, compute positional rank, fit isotonic regression per
   position with the existing `SD_WINDOW = 8` pooling for outcome spread.
4. Apply the curve to current-season ADP for `proj_points`, `sd`, `games`.
5. Project K and DST under the league's kicker and DST scoring.
6. Compute replacement levels and VOR against the league's own starters, derive
   tiers from isotonic plateaus, attach the precomputed `adp_mu`, add playoff
   lift.
7. Cache the resulting board to `leagues.board` (~271 rows for the reference
   league; varies with pool size and roster settings).

Estimated 200-400 ms, run once per league rather than per request.

### Tier 3 — draft board (TypeScript, client)

Loads the cached board and replays the pick log. Every keystroke recomputes
marginal value, VONA and survival locally on typed arrays. Picks append
optimistically and POST in the background.

Client-side compute keeps the 0.13 s re-rank instant, puts zero serverless
compute on the hot path, and keeps the board working if the venue's wifi drops.

`positions >= pick` in `simulate_draft_orders` only requires comparing each
player's score against the pick-th smallest score per simulation, which is an
O(n) selection rather than an O(n log n) argsort — roughly 1M operations in typed
arrays.

## League configuration

One config object, mirroring `src/config.py` key-for-key so the Python stays the
reference implementation and golden fixtures generate directly from it:

```ts
type LeagueConfig = {
  teams; rounds; bench; irSlots
  starters: Record<Slot, number>        // QB RB WR TE FLEX SUPERFLEX K DST
  flexEligible: Position[]
  positionMax: Record<Position, number>
  schedule: { regularSeasonWeeks, playoffWeeks, playoffTeams }
  scoring: Record<StatKey, number>      // 1:1 with SCORING
  kickerScoring: Record<KickerStatKey, number>
  dst: { events, pointsAllowedBands, yardsAllowedBands }
}
```

### Import sources

- **Manual form** — the fallback and the canonical data model.
- **Sleeper** — `GET api.sleeper.app/v1/league/<id>`. Public, unauthenticated.
  Returns `scoring_settings` (flat dict), `roster_positions`, `total_rosters`,
  `playoff_week_start`. Its DST keys (`pts_allow_0`, `pts_allow_1_6`, ...) map
  directly onto the band table. Roughly a 40-entry lookup.
- **ESPN** —
  `lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/<year>/segments/0/leagues/<id>?view=mSettings`.
  Public leagues work unauthenticated; private leagues require the user pasting
  `espn_s2` and `SWID` cookies. Returns `scoringItems` as `{statId, points}` and
  `lineupSlotCounts` keyed by numeric slot id — both undocumented and known to
  shift between seasons. Build this last.

**Every import is a pre-fill, never a commit.** It populates the manual form, the
user confirms, then it saves. If ESPN's id table drifts, the result is a visibly
wrong number on a review screen rather than a silently wrong board.

### ADP anchoring

FFC serves `teams in {8,10,12,14}` and
`scoring in {standard, ppr, half-ppr, 2qb, dynasty}`. Select nearest: teams to
the closest available value; scoring classified by the `receptions` value
(0 / ~0.5 / ~1.0), overridden to `2qb` when the roster permits two starting QBs.

The resulting error is narrow and worth stating precisely. The isotonic fit's
*y-axis* is already the league's own scoring, so projected points are correct.
What is borrowed is the *x-axis* — the draft order real drafters use. Error
therefore concentrates in positions a custom rule revalues, chiefly QB and TE.

It partially self-corrects: replacement and VOR are computed under the league's
own scoring, and the board ranks by VONA rather than ADP, so a league with
6-point passing TDs already shows its QBs with higher VOR. What remains wrong is
**survival probability** — the simulated room drafts to borrowed ADP and will
overestimate how long QBs last in that league.

**Optional, cuttable:** a one-line banner naming the anchor format, roughly 20
lines of work. Declined when offered as a standalone option; recorded here as a
line item, not a requirement.

## The TypeScript port

| Python | `v2/lib/` | Notes |
|---|---|---|
| `src/scoring.py` | `scoring.ts` | Already a pure dot product; near-mechanical |
| `src/project/consensus.py` | `isotonic.ts`, `project.ts` | PAVA, rolling SD, curve application. Highest risk |
| `src/project/kdst.py` | `kdst.ts` | Same machinery over K/DST components |
| `src/draft/replacement.py` | `replacement.ts` | `allocate_flex` -> levels -> VOR |
| `src/draft/tiers.py` | `tiers.ts` | Plateau detection |
| `src/draft/vona.py` | `lineup.ts`, `vona.ts` | Greedy lineup fill; VONA table |
| `src/draft/sim_draft.py` | `simDraft.ts` | Snake picks, sim orders, survival. Not `calibrate()` |
| `src/draft/board.py` | `board.ts` | Orchestration and search |
| `src/features/schedule.py` | `schedule.ts` | Playoff lift from precomputed difficulty |
| `src/draft/persist.py` | — | Replaced by Postgres |

**Not ported:** all of `src/ingest/`, `src/features/build.py`, `src/models/`,
`src/project/actuals.py`, `src/draft/h2h.py`, `src/draft/strategy.py`, and every
script. That is the offline pipeline and the entire Track 2 research line; it
stays in Python, unchanged.

Approximately 800-1000 lines of TypeScript across nine modules, none performing
I/O — preserving the testability property `src/scoring.py` already has.

### Hazard 1: tiers depend on exact float equality

`src/draft/tiers.py` starts a new tier "wherever projected points actually
change" — a literal `!=` on floats. This works only because isotonic regression
emits bit-identical fitted values across a pooled block.

If the TypeScript PAVA computes the block mean per element rather than assigning
one value to the whole block, values within a plateau differ in the last bit,
`!=` fires on every row, and every player becomes his own tier. The board still
renders and the projections are still right, but tiering — which the README calls
the only good reason to reach — is silently destroyed.

**Constraint:** PAVA must assign one computed value per block.

**Test:** golden fixtures assert tier counts and boundaries, not only fitted
values. A 1e-9 value tolerance would pass while tiers were shattered.

Matching sklearn's `IsotonicRegression(increasing=False, out_of_bounds="clip")`
additionally requires weighting duplicate x values by count, linear interpolation
between fitted points when predicting, and clipping out-of-bounds inputs to the
boundary values.

### Hazard 2: the RNG will not match, and need not

numpy's `default_rng(0)` is PCG64 with ziggurat normals. Porting it is possible
and not worthwhile. Two tolerance classes instead:

- **Deterministic** (scoring, isotonic, replacement, tiers, lineup, snake picks):
  exact to 1e-9.
- **Stochastic** (survival, VONA, expected best available): distributional. At
  4000 sims a proportion near 0.5 has SE ~ 0.008, so assert within +/-0.025.
  Separately assert that the VONA table's *ordering* matches, since ordering is
  what the user acts on.

## Testing

`scripts/export_golden.py` emits, for four configurations — 14-team full-PPR (the
existing league), 10-team half-PPR, 12-team superflex with 6-point passing TDs,
and a degenerate no-FLEX league — the input board plus expected output at every
stage, committed to `v2/tests/golden/`.

Division of labour:

- The existing 60 Python tests keep enforcing real invariants against real data
  (calibration MAE 1.04, scoring vs nflverse, 221/221 ID resolution).
- The golden tests enforce only that TypeScript agrees with Python.

CI runs pytest, then vitest.

## Persistence

```sql
leagues
  id             text primary key    -- nanoid; the read/write capability
  config         jsonb not null
  board          jsonb               -- derived, regenerable
  board_hash     text                -- hash(config, adp_as_of)
  adp_as_of      date
  frozen_at      timestamptz
  created_at     timestamptz
picks
  league_id      text references leagues(id) on delete cascade
  seq            int not null        -- insertion sequence, not draft position
  player_id      text                -- null = a player not on our board
  taken_by       text not null       -- 'me' | 'other', matching board.draft()
  voided_at      timestamptz         -- soft delete
  created_at     timestamptz
  primary key (league_id, seq)
```

**The primary key provides idempotent retries.** A POST that times out on bad
venue wifi but succeeded server-side, then is retried, would otherwise
double-record the pick and advance the clock by one — which corrupts every VONA
number downstream. The PK makes the retry a no-op. It also covers a stray second
tab and the laptop-plus-phone case.

`seq` is an insertion counter, not a draft slot. Semantic draft position is
`row_number()` over non-voided rows, so voiding pick 12 does not corrupt picks
13-47. The `skip` case in `src/dashboard/app.py` — someone drafting a player not
on the board — is a row with `player_id` null, handling the existing
`__unknown_` sentinel natively.

**Freeze the board at first pick.** ADP refreshes daily and this league allows
three hours per pick, so a draft runs over days and ADP *will* refresh mid-draft.
Recomputing then would change projections, tiers, replacement levels and every
survival number underneath a draft in progress. `frozen_at` is set on the first
pick; later refreshes leave the board alone and surface a banner offering an
explicit opt-in refresh. Concretely: while `frozen_at` is set, a `board_hash`
mismatch never triggers recomputation — it only enables the banner.

**Sync** is reconcile-on-load and on window focus. Local writes are optimistic;
the server provides durability, not coordination.

**Offline:** board and pick log mirrored to localStorage (~60 KB), unsent picks
queued and flushed on reconnect.

**Recovery:** no accounts, so losing the URL loses the league. localStorage keeps
a "your leagues" list per device, plus a prominent save-this-link step at
creation. Email magic-link recovery is a later addition if users ask for it.

**Consequence worth noting:** users never contact FFC. They read daily Blob
snapshots, and only the GitHub Action calls the API — honouring the caching
request in `src/ingest/adp.py` better than the current per-machine cache does.

## Failure modes

**Loss of the local-only property.** The current tool's best characteristic is
that it runs entirely on a laptop with no network and no install, as
`src/dashboard/app.py` states in its opening comment. Hosting trades that away.
The escape hatch is a **"download cheatsheet" button** rendering the current
board to a self-contained static HTML file — top ~150 players, tiers, per-round
plan, no network required. `scripts/cheatsheet.py` already embodies this
instinct. This is the highest-value robustness feature in the plan.

**Silent scoring drift (a latent bug in the current code).** `src/scoring.py`
deliberately tolerates missing columns: "A stat absent from the frame contributes
nothing." If nflverse renames `receiving_yards`, every WR quietly loses ~140
points with no error and a plausible-looking board. The export job must assert
that every `SCORING` key resolves to a present column (modulo the fumble
special-case) and fail loudly.

**Build job failure.** Write to `adp_<date>`, flip the pointer only on full
success, and surface `adp_as_of` in the UI so staleness is visible.

**Config validation at save.** At least one starter; scoring not identically
zero; roster slots <= rounds; teams within mappable range. If a position has too
few historical observations for a stable isotonic fit, flag it rather than
shipping a curve fitted on noise.

**Degenerate setup.** Board derivation is idempotent and retryable. On failure,
keep the config and show the error; never half-save a board.

**Clock desync.** The remaining source is someone drafting a player not on the
board, already handled by `skip`. `drafted_count` versus expected pick number
stays visible in the header.

**Rate limiting.** League creation is the expensive route (isotonic fit plus DB
write); rate-limit by IP. Blob reads are edge-cached.

**Philosophy:** fail loudly at build time, degrade gracefully at draft time. A
broken build blocks the pointer flip. A draft-time error never blanks the board —
the client keeps its last good state and shows a banner.

## Tech stack

| Layer | Choice | Rationale |
|---|---|---|
| Framework | Next.js (App Router) on Vercel | Server routes for setup and storage; client for the board |
| Language | TypeScript with typed arrays | Needed ops are Box-Muller RNG, O(n) selection, group-max, greedy fill, PAVA |
| Database | Postgres (Neon) + Drizzle | Durable, not a cache. Drizzle avoids Prisma's engine binary |
| Derived boards | `jsonb` on the league row | Regenerable; one less service than Blob on day one |
| Static data | Vercel Blob, daily refresh | ADP changes daily |
| Build job | GitHub Actions cron | Needs nflverse downloads and the 98 MB cache; not a serverless workload |
| UI | React + Tailwind | Porting the 408 lines of `src/dashboard/index.html` |
| Tests | Vitest + golden fixtures from Python | See Testing |

## Layout

```
v2/
  app/                Next.js App Router
  lib/                the nine ported math modules
  db/                 Drizzle schema and migrations
  tests/golden/       fixtures generated by Python
scripts/
  export_v2_bundle.py NEW - emits Tier 1 artefacts
  export_golden.py    NEW - emits test fixtures
src/                  unchanged
```

## Carried-forward limitations

These are properties of the projection method, not of the hosting, and remain
true in v2:

- Projections carry no independent information about players; they are ADP rank
  mapped through a historical curve. All edge is in the decision layer.
- Custom scoring inherits an ADP mismatch, concentrated in QB and TE, with
  survival probabilities most affected.
- K and DST remain near-random year to year.
- No injury news from the last 24 hours.
- The strategy results in the README (14-team full-PPR: Robust-RB 0.094, both QB
  extremes losing) are findings about *that* league. Because each league gets its
  own isotonic fit, replacement levels and tier boundaries genuinely differ, so
  those numbers must not be presented to other leagues as constants.

## Suggested phasing

This is too large for a single implementation plan. Four phases, each
independently verifiable, each leaving the tree working:

1. **Tier 1 export + golden fixtures.** `export_v2_bundle.py`,
   `export_golden.py`, the missing-column guard on `SCORING` keys, and the
   GitHub Action. Pure Python, no `v2/` yet. Verifiable against the existing
   pytest suite.
2. **The TypeScript math port.** The nine `v2/lib/` modules plus Vitest against
   the golden fixtures. No UI, no database, no network. This is where the two
   hazards live and it should not be rushed.
3. **App shell: config, setup, persistence.** Next.js, Drizzle schema, the
   manual config form, Tier 2 derivation, league creation and recovery list.
   Sleeper and ESPN import excluded.
4. **Draft board UI + cheatsheet export.** Port `index.html`, wire optimistic
   picks and reconciliation, offline mirroring, and the static cheatsheet
   download.

Sleeper import, then ESPN import, follow as separate increments once phase 3 is
stable — they are pre-fills over a form that already works without them.

## Out of scope

Accounts and email recovery; shared boards for several people in one league; live
sync; mock-draft mode; in-season lineup tools; porting the Track 2 research line
to TypeScript.
