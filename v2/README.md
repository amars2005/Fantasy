# v2 — the hosted draft board

The Python in [`../src`](../src) is unchanged and remains the reference
implementation. This directory is the hosted, multi-league version of it: the
same projection method and the same decision layer, reimplemented in TypeScript
so they can run in a browser, with league scoring and roster shape configurable
per league.

```bash
npm install
npm test          # 225 unit tests, including agreement with Python
npm run test:e2e  # 45 browser tests (Playwright, Chromium)
npm run test:all  # both
npm run dev       # http://localhost:3000
```

No database and no cloud storage are needed to run it. With no `DATABASE_URL`
the app stores leagues in `.data/leagues.json`, and with no `BUNDLE_BASE_URL` it
reads the static bundle straight off disk. See [Deploying](#deploying) for what
that means once the app is hosted: the checkout the development path reads is
not there, so a deployment needs the bundle pointed at or shipped with it.

## Why it is built this way

Three measured facts drove the architecture.

**The bundle would not fit.** Vercel allows 250 MB unzipped per function.
polars' compiled runtime alone is 187 MB, and scikit-learn plus scipy — pulled
in only by the isotonic fits — add another 157 MB. Precomputing the fit removes
sklearn entirely, and moving the rest to the client removes the question.

**There is nowhere to write.** [`persist.py`](../src/draft/persist.py) writes
draft state to disk and [`cache.py`](../src/ingest/cache.py) writes parquet;
serverless functions are read-only outside an ephemeral `/tmp`, so every cold
start would re-download nflverse.

**The board is a process-global.** [`app.py`](../src/dashboard/app.py) holds one
board in memory and takes 2.6 s to build it. Serverless gives you N instances
and no shared memory.

## The three tiers

**Offline** ([`../scripts/export_v2_bundle.py`](../scripts/export_v2_bundle.py),
daily in Actions). Exports the raw *stat components* per player-season rather
than points, so any scoring vector becomes a dot product. Also runs
`calibrate()` — 15 iterations of 4000 simulations, and dependent only on ADP, so
it never needs to run in a browser — and resolves FFC names to nflverse ids,
which is the step that silently breaks a board and is already solved in Python.
The whole bundle is under a megabyte.

**Setup** (server, once per league). Applies the league's scoring to the
components, refits the rank → points curve, projects, values, tiers. Measured at
~220 ms. Cached on the league row.

**Draft** (browser). Loads the cached board and replays the pick log, then
recomputes marginal value, VONA and survival locally on every keystroke.

## Agreeing with Python

`../scripts/export_golden.py` emits fixtures for four configurations — the
reference 14-team PPR league, a 10-team half-PPR, a 12-team superflex with
6-point passing touchdowns, and a degenerate no-FLEX league — and the tests
assert the TypeScript reproduces them.

Two tolerance classes, because two kinds of maths are involved. Deterministic
work (scoring, the isotonic fit, replacement, tiers, lineups) is asserted to
1e-9. Survival probabilities are asserted distributionally, because numpy's
PCG64 stream is not reproducible here and reproducing it would buy nothing.

Two things are worth knowing about that boundary:

- **Tiers hinge on exact float equality.**
  [`tiers.py`](../src/draft/tiers.py) starts a new tier wherever projected points
  *change*, which works only because isotonic regression assigns a bit-identical
  value across a pooled block. A PAVA that computed each element separately
  would shatter every tier while every number still looked plausible, so
  `isotonic.ts` assigns one value per block by construction and the tests assert
  tier **counts**, not just values.
- **Rounding cannot fully agree.** polars rounds half-to-even on the scaled
  value; `roundTo` reproduces that. But a fitted value can sit exactly on a tie,
  where a 1e-14 difference in summation order flips it by 0.1. Agreement is
  therefore asserted on unrounded values, and the tie set is tracked per
  quantity and capped.

## The browser suite

`npm run test:e2e` builds the app, starts it on port 3210 against its own data
file, and drives Chromium through it. It covers what the unit tests
structurally cannot: anything that depends on *wiring* rather than on a
function's return value.

That is not a theoretical distinction. It found two real bugs that had unit
tests passing either side of them:

- The draft board was constructed without its team-nickname table, so searching
  "ravens" for a defence returned nothing. The unit test passed because it
  supplied the table itself.
- The file-backed store did a read-modify-write per operation with no
  serialisation, so two picks in flight at once could lose one. That is exactly
  the last-write-wins loss the Postgres primary key rules out, and it surfaced
  as a pick vanishing across a page reload.

The largest group is `refresh.spec.ts`, because reloading is not an edge case
here: a draft runs over days, so the page *will* be reloaded, slept, reopened on
a second device, and reopened after the daily ADP refresh has moved underneath
it. It covers picks surviving a reload, the slot being remembered, undo not
renumbering later picks, a second tab reconciling on focus, and the board
staying frozen when newer ADP arrives mid-draft — offered, never applied.

One rule worth knowing when adding to it: the header clock updates
optimistically, so polling it says nothing about whether a write reached the
server. Use `waitForPicksOnServer` before asserting on stored state.

## Layout

```
lib/           the ported maths: isotonic, scoring, projection, replacement,
               tiers, lineup, VONA, simulation, board
lib/import/    Sleeper and ESPN settings importers
db/            Drizzle schema
app/           Next.js routes and pages
tests/         vitest, plus golden fixtures generated by Python
e2e/           Playwright: the app driven in a real browser
```

## Environment

| Variable | Effect when unset |
|---|---|
| `DATABASE_URL` | Leagues stored in `.data/leagues.json` (see [Database](#database) before setting it) |
| `BUNDLE_BASE_URL` | Bundle read off disk: `BUNDLE_DIR`, then `data/v2_export`, then `../data/v2_export` |
| `BUNDLE_DIR` | The two default directories are searched; setting it searches that directory only |
| `FANTASY_DATA_FILE` | The file store writes to `.data/leagues.json` |
| `BLOB_READ_WRITE_TOKEN` | The publish step in CI is skipped |

## Database

The file store is the default and needs nothing, but it writes to disk, and a
serverless filesystem is read-only outside `/tmp`. On a hosted deployment that
leaves two options.

**Postgres** (what a draft running over days needs). It must be Neon:
[`store.ts`](lib/store.ts) uses `drizzle-orm/neon-http` with
`@neondatabase/serverless`, which speaks Neon's HTTP protocol, so an RDS or
Supabase URL will not connect. Create the tables before setting `DATABASE_URL`
-- pointing it at an empty database is worse than leaving it unset, since every
request then fails on `relation "leagues" does not exist`. Either run
`DATABASE_URL=... npm run db:migrate`, or paste
[`db/migrations/0000_moaning_eddie_brock.sql`](db/migrations) into Neon's SQL
editor, which needs no local tooling.

**Ephemeral** (fine for trying it out, not for a real draft). Set
`FANTASY_DATA_FILE=/tmp/leagues.json`. Serverless `/tmp` is writable but
per-instance and short-lived, so leagues vanish on a cold start.

## Deploying

The bundle is not in the repo -- it is a build product of the Python pipeline,
regenerated daily -- so a deployment has to be told where it is. Nothing else
about the app needs provisioning, but this does, and skipping it is the one
failure that takes every board down at once: league creation dies on a missing
`board_ppr.json`, since the projection has nothing to project from.

**Fetched from Blob** (what the daily Action is for). Set `BUNDLE_BASE_URL` to
the Blob store root that `scripts/publish-bundle.mjs` prints. Set it once: the
files live under a dated prefix that moves with every refresh, and the app
resolves the current one through the `bundle/latest.json` pointer, which the
publish step moves only after every file has landed.

**Shipped with the app** (no Blob store, no environment variable). Run
`npm run bundle:stage` to copy the export into `v2/data/v2_export` and commit
it. `next.config.mjs` traces those files into the serverless function -- nothing
imports them, so without that they would not be deployed -- and the loader finds
them there. The trade is that the ADP is then as old as the commit, which the
banner in the UI dates for you.

## Known limitations

- **ADP is anchored, not exact.** Fantasy Football Calculator serves four
  scoring formats and ignores its own `teams` parameter, so a league with custom
  rules borrows the nearest format's draft *order*. Projected points are correct
  under your scoring; what is borrowed is the ordering, so error concentrates in
  positions your rules revalue — mostly QB and TE. VOR and VONA partly
  self-correct, since both are computed under your own scoring. Survival
  probabilities do not: the simulated room drafts to the borrowed ADP. The app
  says which format it anchored to and warns when your rules diverge.
- **The ESPN importer needs checking against a real league.** ESPN's stat and
  lineup-slot ids are undocumented and shift between seasons. The mapping covers
  the well-established ids, reports everything it could not place, and never
  saves without review — but it has not been verified against a live private
  league.
- Everything the parent README lists still applies: projections carry no
  independent information about players, K and DST are near-random year to year,
  and nothing from the last day of injury news is in here.
