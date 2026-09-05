/**
 * League configuration: defaults, validation, and ADP anchoring.
 *
 * `REFERENCE_LEAGUE` mirrors `src/config.py` value-for-value. It is the league
 * the Python pipeline is built around, so it doubles as the fixture every
 * golden test compares against -- if the two ever drift, the tests say so.
 */

import type { Band, LeagueConfig, Position } from "./types";

/** Full PPR. Points per unit of the named stat. Keys are nflverse columns. */
export const PPR_SCORING: Record<string, number> = {
  passing_yards: 0.04, // 1 pt / 25 yds
  passing_tds: 4.0,
  passing_interceptions: -2.0,
  passing_2pt_conversions: 2.0,
  rushing_yards: 0.1, // 1 pt / 10 yds
  rushing_tds: 6.0,
  rushing_2pt_conversions: 2.0,
  receptions: 1.0,
  receiving_yards: 0.1,
  receiving_tds: 6.0,
  receiving_2pt_conversions: 2.0,
  fumbles_lost: -2.0,
  special_teams_tds: 6.0,
  pt_return_tds: 6.0,
  fumble_recovery_tds: 6.0,
};

/**
 * Rules the reference league does not pay, but an imported one might.
 *
 * Mirrors `TD_LENGTH_BANDS` / `LONG_TD_COMPONENTS` and `DST_PBP_EVENTS` in the
 * Python config. These are deliberately *not* in `PPR_SCORING`: the reference
 * league does not pay a long-touchdown bonus, and writing it in at zero would
 * claim a rule it does not have. They are listed here so an import can set
 * them and a manual entry can reach them, and the bundle carries the matching
 * stat columns either way.
 *
 * Both platforms treat a long-touchdown bonus as a counter, so a 55-yard score
 * trips "40+" and "50+" together. The bands below are disjoint and a coarser
 * platform rule is expanded across every band it spans, which is why the
 * bundle can store each touchdown once and still pay both rules.
 */
export const TD_LENGTH_BANDS: [number, number | null][] = [
  [0, 9], [10, 19], [20, 29], [30, 39], [40, 49], [50, null],
];

export function tdBandKey(kind: string, low: number, high: number | null): string {
  return `${kind}_td_${low}_${high ?? ""}`;
}

export const LONG_TD_KINDS = ["passing", "rushing", "receiving"] as const;

/**
 * The bands a cumulative platform rule covers.
 *
 * "40+ yard TD" is not one of our bands; it is every band from 40 up. Spreading
 * it out is what lets the bundle store each touchdown exactly once while a
 * league that sets both a 40+ and a 50+ rule still gets paid for both.
 */
export function tdBandsFromLow(kind: string, low: number): string[] {
  return TD_LENGTH_BANDS.filter(([bandLow]) => bandLow >= low).map(([bandLow, high]) =>
    tdBandKey(kind, bandLow, high),
  );
}

/** The single band starting exactly at `low`, for a rule naming one outright. */
export function tdExactBand(kind: string, low: number): string[] {
  const band = TD_LENGTH_BANDS.find(([bandLow]) => bandLow === low);
  if (!band) throw new Error(`no touchdown band starts at ${low}`);
  return [tdBandKey(kind, band[0], band[1])];
}

export const LONG_TD_COMPONENTS: string[] = LONG_TD_KINDS.flatMap((kind) =>
  TD_LENGTH_BANDS.map(([low, high]) => tdBandKey(kind, low, high)),
);

/** Defensive scores nflverse's team frame does not count. See `src/dst.py`. */
export const DST_PBP_EVENTS = ["def_two_point_returns", "def_one_point_safeties"];

const LONG_TD_LABELS: Record<string, string> = Object.fromEntries(
  LONG_TD_KINDS.flatMap((kind) =>
    TD_LENGTH_BANDS.map(([low, high]) => [
      tdBandKey(kind, low, high),
      `${kind} TD, ${high === null ? `${low}+` : `${low}-${high}`} yds`,
    ]),
  ),
);

const EVENT_LABELS: Record<string, string> = {
  def_two_point_returns: "two-point try returned",
  def_one_point_safeties: "one-point safety",
};

/**
 * A scoring or event key as a person reads it.
 *
 * The raw keys are nflverse column names, and `receiving_td_50_` with the
 * underscores swapped for spaces is not something anyone should have to
 * decode on a review screen.
 */
export function ruleLabel(key: string): string {
  const banded = LONG_TD_LABELS[key] ?? EVENT_LABELS[key];
  if (banded) return banded;

  // `fg_made_60_` and friends: a trailing underscore is an open-ended band, and
  // swapping underscores for spaces leaves it as a stray space at the end.
  const fg = /^fg_(made|missed)_(\d+)_(\d*)$/.exec(key);
  if (fg) {
    const [, kind, low, high] = fg;
    return `FG ${kind}, ${high ? `${low}-${high}` : `${low}+`} yds`;
  }
  return key.replace(/_/g, " ");
}

/** Every scoring key a league may set, whether or not the reference one does. */
export const ALL_SCORING_KEYS: string[] = [
  ...Object.keys(PPR_SCORING),
  ...LONG_TD_COMPONENTS,
];

/**
 * Kicker scoring, banded by distance. The reference league goes further than
 * most: it pays a sixth point at 60+ yards where the common default stops at
 * 50+. Every miss costs a point at any distance.
 */
export const REFERENCE_KICKER_SCORING: Record<string, number> = {
  fg_made_0_19: 3.0,
  fg_made_20_29: 3.0,
  fg_made_30_39: 3.0,
  fg_made_40_49: 4.0,
  fg_made_50_59: 5.0,
  fg_made_60_: 6.0,
  pat_made: 1.0,
  fg_missed_0_19: -1.0,
  fg_missed_20_29: -1.0,
  fg_missed_30_39: -1.0,
  fg_missed_40_49: -1.0,
  fg_missed_50_59: -1.0,
  fg_missed_60_: -1.0,
};

export const REFERENCE_DST_EVENTS: Record<string, number> = {
  def_sacks: 1.0,
  def_interceptions: 2.0,
  def_fumbles: 2.0,
  def_safeties: 2.0,
  def_punt_blocks: 2.0,
  def_pat_blocks: 2.0,
  def_fg_blocks: 2.0,
  def_tds: 6.0,
};

const band = (low: number, high: number, points: number): Band => ({ low, high, points });

export const REFERENCE_POINTS_ALLOWED_BANDS: Band[] = [
  band(0, 0, 5), band(1, 6, 4), band(7, 13, 3), band(14, 17, 1),
  band(18, 27, 0), band(28, 34, -1), band(35, 45, -3), band(46, 999, -5),
];

export const REFERENCE_YARDS_ALLOWED_BANDS: Band[] = [
  band(0, 99, 5), band(100, 199, 3), band(200, 299, 2), band(300, 349, 0),
  band(350, 399, -1), band(400, 449, -3), band(450, 499, -5),
  band(500, 549, -6), band(550, 99999, -7),
];

/** The league `src/config.py` describes: 14-team, full PPR, snake, no keepers. */
export const REFERENCE_LEAGUE: LeagueConfig = {
  teams: 14,
  rounds: 15,
  bench: 6,
  irSlots: 2,
  starters: { QB: 1, RB: 2, WR: 2, TE: 1, FLEX: 1, K: 1, DST: 1 },
  flexEligible: ["RB", "WR", "TE"],
  positionMax: { QB: 4, RB: 8, WR: 8, TE: 3, K: 3, DST: 3 },
  schedule: {
    regularSeasonWeeks: 14,
    playoffWeeks: [15, 16, 17],
    playoffTeams: 6,
  },
  scoring: { ...PPR_SCORING },
  kickerScoring: { ...REFERENCE_KICKER_SCORING },
  dst: {
    events: { ...REFERENCE_DST_EVENTS },
    pointsAllowedBands: REFERENCE_POINTS_ALLOWED_BANDS,
    yardsAllowedBands: REFERENCE_YARDS_ALLOWED_BANDS,
  },
};

/** Scoring formats the bundle carries. `dynasty` is excluded upstream. */
export type FfcFormat = "standard" | "ppr" | "half-ppr" | "2qb";

/**
 * Pick the ADP format a league's board is anchored to.
 *
 * Note what is *not* a parameter here: team count. Fantasy Football Calculator
 * ignores its own `teams` query parameter -- verified 2026-08-30, fetching
 * teams=12 and teams=14 in the same minute returns byte-identical ADP for every
 * player. Team count still drives replacement level, snake order and VONA; it
 * just cannot change the market ordering we anchor to.
 */
export function nearestFfcFormat(league: LeagueConfig): FfcFormat {
  const startingQbs =
    (league.starters.QB ?? 0) + (league.starters.SUPERFLEX ?? 0);
  if (startingQbs >= 2) return "2qb";

  const rec = league.scoring.receptions ?? 0;
  if (rec >= 0.75) return "ppr";
  if (rec >= 0.25) return "half-ppr";
  return "standard";
}

/**
 * How far a league's rules diverge from the format its ADP is borrowed from.
 *
 * The isotonic fit's y-axis is already the league's own scoring, so projected
 * points are right. What is borrowed is the x-axis -- the order real drafters
 * use -- so error concentrates in positions a custom rule revalues. Surfacing
 * that is the difference between the tool being wrong and being confidently
 * wrong.
 */
export function anchorWarnings(league: LeagueConfig): string[] {
  const out: string[] = [];
  const s = league.scoring;

  if ((s.passing_tds ?? 4) !== 4) {
    out.push(
      `Passing touchdowns are worth ${s.passing_tds}, not the 4 the borrowed ADP ` +
        "assumes. Quarterbacks will go earlier in your room than the survival " +
        "estimates expect.",
    );
  }
  if ((s.passing_yards ?? 0.04) !== 0.04) {
    out.push(
      `Passing yards are worth ${s.passing_yards} per yard rather than 0.04, which ` +
        "shifts quarterback value away from the anchored draft order.",
    );
  }
  const startingQbs = (league.starters.QB ?? 0) + (league.starters.SUPERFLEX ?? 0);
  if (startingQbs >= 2) {
    out.push("Superflex detected: ADP is anchored to Fantasy Football Calculator's 2QB board.");
  }
  if ((league.starters.TE ?? 0) > 1) {
    out.push("Multiple starting tight ends will pull tight-end value above the anchored order.");
  }
  return out;
}

export class ConfigError extends Error {}

/**
 * Reject configurations that would produce a meaningless board.
 *
 * Cheap checks, run before anything expensive: a board derived from a
 * degenerate config still renders, which is exactly why it needs catching here.
 */
export function validateLeague(league: LeagueConfig): void {
  const problems: string[] = [];

  if (!Number.isInteger(league.teams) || league.teams < 2 || league.teams > 32) {
    problems.push("teams must be a whole number between 2 and 32");
  }
  if (!Number.isInteger(league.rounds) || league.rounds < 1) {
    problems.push("rounds must be at least 1");
  }

  const starterSlots = Object.values(league.starters).reduce((a, b) => a + (b ?? 0), 0);
  if (starterSlots < 1) problems.push("the league must start at least one player");
  if (starterSlots > league.rounds) {
    problems.push(
      `${starterSlots} starting slots but only ${league.rounds} rounds; ` +
        "a legal lineup could never be drafted",
    );
  }

  const scoringMagnitude = Object.values(league.scoring).reduce(
    (a, b) => a + Math.abs(b ?? 0),
    0,
  );
  if (scoringMagnitude === 0) {
    problems.push("scoring is identically zero, so every player would project the same");
  }

  if (league.flexEligible.length === 0 && (league.starters.FLEX ?? 0) > 0) {
    problems.push("FLEX slots exist but no positions are flex-eligible");
  }

  for (const [pos, max] of Object.entries(league.positionMax)) {
    const starters = league.starters[pos as Position] ?? 0;
    if (max !== undefined && max < starters) {
      problems.push(`positionMax.${pos} (${max}) is below its starting requirement (${starters})`);
    }
  }

  if (problems.length) {
    throw new ConfigError(`Invalid league configuration:\n  - ${problems.join("\n  - ")}`);
  }
}
