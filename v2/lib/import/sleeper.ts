/**
 * Importing a league from Sleeper.
 *
 * The easy one: a public, unauthenticated endpoint that returns the whole
 * scoring block. Everything here is a *pre-fill* -- it populates the config
 * form for a person to confirm, and is never saved directly. That way a
 * mapping that drifts produces a visibly wrong number on a review screen
 * rather than a silently wrong draft board.
 */

import { REFERENCE_LEAGUE, tdBandsFromLow } from "../config";
import type { Band, LeagueConfig, Position, Slot } from "../types";

const API = "https://api.sleeper.app/v1/league";

/** Sleeper's scoring keys to the nflverse column names our rules use. */
const SCORING_MAP: Record<string, string> = {
  pass_yd: "passing_yards",
  pass_td: "passing_tds",
  pass_int: "passing_interceptions",
  pass_2pt: "passing_2pt_conversions",
  rush_yd: "rushing_yards",
  rush_td: "rushing_tds",
  rush_2pt: "rushing_2pt_conversions",
  rec: "receptions",
  rec_yd: "receiving_yards",
  rec_td: "receiving_tds",
  rec_2pt: "receiving_2pt_conversions",
  fum_lost: "fumbles_lost",
  // nflverse folds kickoff-return touchdowns into `special_teams_tds` and keeps
  // punt returns separate, which is exactly how the reference league scores
  // them.
  kr_td: "special_teams_tds",
  pr_td: "pt_return_tds",
  fum_rec_td: "fumble_recovery_tds",
};

/**
 * Long-touchdown bonuses -> the bundle's banded touchdown columns.
 *
 * Sleeper offers only the two cumulative thresholds, with the `p` suffix it
 * uses elsewhere for "or more" (`fgm_60p` above). It has no 40-49 bucket, so
 * "40+" has to mean every band from 40 up, and a league setting both 40+ and
 * 50+ pays both on a 55-yard score -- the same arrangement ESPN uses. That
 * reading is an assumption about a league that sets both thresholds; a league
 * setting one, which is the ordinary case, is unambiguous either way.
 */
const LONG_TD_MAP: Record<string, string[]> = {
  pass_td_40p: tdBandsFromLow("passing", 40),
  pass_td_50p: tdBandsFromLow("passing", 50),
  rush_td_40p: tdBandsFromLow("rushing", 40),
  rush_td_50p: tdBandsFromLow("rushing", 50),
  rec_td_40p: tdBandsFromLow("receiving", 40),
  rec_td_50p: tdBandsFromLow("receiving", 50),
};

const KICKER_MAP: Record<string, string> = {
  fgm_0_19: "fg_made_0_19",
  fgm_20_29: "fg_made_20_29",
  fgm_30_39: "fg_made_30_39",
  fgm_40_49: "fg_made_40_49",
  fgm_50_59: "fg_made_50_59",
  fgm_60p: "fg_made_60_",
  xpm: "pat_made",
  fgmiss_0_19: "fg_missed_0_19",
  fgmiss_20_29: "fg_missed_20_29",
  fgmiss_30_39: "fg_missed_30_39",
  fgmiss_40_49: "fg_missed_40_49",
};

const DST_MAP: Record<string, string> = {
  sack: "def_sacks",
  int: "def_interceptions",
  fum_rec: "def_fumbles",
  safe: "def_safeties",
  def_td: "def_tds",
};

const PA_BANDS: [string, number, number][] = [
  ["pts_allow_0", 0, 0],
  ["pts_allow_1_6", 1, 6],
  ["pts_allow_7_13", 7, 13],
  ["pts_allow_14_20", 14, 20],
  ["pts_allow_21_27", 21, 27],
  ["pts_allow_28_34", 28, 34],
  ["pts_allow_35p", 35, 999],
];

const YA_BANDS: [string, number, number][] = [
  ["yds_allow_0_100", 0, 99],
  ["yds_allow_100_199", 100, 199],
  ["yds_allow_200_299", 200, 299],
  ["yds_allow_300_349", 300, 349],
  ["yds_allow_350_399", 350, 399],
  ["yds_allow_400_449", 400, 449],
  ["yds_allow_450_499", 450, 499],
  ["yds_allow_500_549", 500, 549],
  ["yds_allow_550p", 550, 99999],
];

const SLOT_MAP: Record<string, Slot> = {
  QB: "QB",
  RB: "RB",
  WR: "WR",
  TE: "TE",
  FLEX: "FLEX",
  WRRB_FLEX: "FLEX",
  REC_FLEX: "FLEX",
  SUPER_FLEX: "SUPERFLEX",
  K: "K",
  DEF: "DST",
};

export interface SleeperImport {
  name: string;
  config: LeagueConfig;
  /** Anything the mapping could not place, so the review screen can say so. */
  unmapped: string[];
}

interface SleeperLeague {
  name?: string;
  total_rosters?: number;
  scoring_settings?: Record<string, number>;
  roster_positions?: string[];
  settings?: Record<string, number>;
}

function bandsFrom(
  scoring: Record<string, number>,
  spec: [string, number, number][],
  fallback: Band[],
): Band[] {
  const present = spec.filter(([key]) => key in scoring);
  if (!present.length) return fallback;
  return present.map(([key, low, high]) => ({ low, high, points: scoring[key] }));
}

export function mapSleeperLeague(raw: SleeperLeague): SleeperImport {
  const scoring: Record<string, number> = {};
  const kicker: Record<string, number> = {};
  const dstEvents: Record<string, number> = {};
  const unmapped: string[] = [];
  const settings = raw.scoring_settings ?? {};

  for (const [key, value] of Object.entries(settings)) {
    if (SCORING_MAP[key]) scoring[SCORING_MAP[key]] = value;
    else if (LONG_TD_MAP[key]) {
      // Overlapping thresholds accumulate, so these add rather than replace.
      for (const band of LONG_TD_MAP[key]) scoring[band] = (scoring[band] ?? 0) + value;
    } else if (KICKER_MAP[key]) kicker[KICKER_MAP[key]] = value;
    else if (DST_MAP[key]) dstEvents[DST_MAP[key]] = value;
    else if (!key.startsWith("pts_allow") && !key.startsWith("yds_allow")) {
      unmapped.push(key);
    }
  }

  // Sleeper often carries one blanket "blocked kick" and one blanket field-goal
  // miss rather than the banded versions the reference league uses.
  if ("blk_kick" in settings) {
    for (const k of ["def_punt_blocks", "def_pat_blocks", "def_fg_blocks"]) {
      dstEvents[k] = settings.blk_kick;
    }
  }
  if ("fgmiss" in settings) {
    for (const k of Object.keys(REFERENCE_LEAGUE.kickerScoring)) {
      if (k.startsWith("fg_missed_")) kicker[k] ??= settings.fgmiss;
    }
  }
  if ("fgm_50p" in settings) {
    kicker.fg_made_50_59 ??= settings.fgm_50p;
    kicker.fg_made_60_ ??= settings.fgm_50p;
  }

  const starters: Partial<Record<Slot, number>> = {};
  let bench = 0;
  let irSlots = 0;
  for (const slot of raw.roster_positions ?? []) {
    if (slot === "BN") bench += 1;
    else if (slot === "IR" || slot === "TAXI") irSlots += 1;
    else if (SLOT_MAP[slot]) {
      const mapped = SLOT_MAP[slot];
      starters[mapped] = (starters[mapped] ?? 0) + 1;
    } else unmapped.push(`roster slot ${slot}`);
  }

  const starterCount = Object.values(starters).reduce((a, b) => a + (b ?? 0), 0);
  const playoffStart = raw.settings?.playoff_week_start ?? 15;

  const config: LeagueConfig = {
    teams: raw.total_rosters ?? REFERENCE_LEAGUE.teams,
    rounds: starterCount + bench,
    bench,
    irSlots,
    starters: Object.keys(starters).length ? starters : REFERENCE_LEAGUE.starters,
    flexEligible: ["RB", "WR", "TE"] as Position[],
    positionMax: { ...REFERENCE_LEAGUE.positionMax },
    schedule: {
      regularSeasonWeeks: Math.max(playoffStart - 1, 1),
      playoffWeeks: [playoffStart, playoffStart + 1, playoffStart + 2],
      playoffTeams: raw.settings?.playoff_teams ?? REFERENCE_LEAGUE.schedule.playoffTeams,
    },
    scoring: Object.keys(scoring).length ? scoring : { ...REFERENCE_LEAGUE.scoring },
    kickerScoring: Object.keys(kicker).length
      ? kicker
      : { ...REFERENCE_LEAGUE.kickerScoring },
    dst: {
      events: Object.keys(dstEvents).length
        ? dstEvents
        : { ...REFERENCE_LEAGUE.dst.events },
      pointsAllowedBands: bandsFrom(settings, PA_BANDS, REFERENCE_LEAGUE.dst.pointsAllowedBands),
      yardsAllowedBands: bandsFrom(settings, YA_BANDS, REFERENCE_LEAGUE.dst.yardsAllowedBands),
    },
  };

  return { name: raw.name ?? "Imported league", config, unmapped };
}

export async function importSleeperLeague(leagueId: string): Promise<SleeperImport> {
  const res = await fetch(`${API}/${encodeURIComponent(leagueId)}`, {
    headers: { accept: "application/json" },
  });
  if (!res.ok) {
    throw new Error(
      res.status === 404
        ? "No Sleeper league with that id. It is the long number in the league URL."
        : `Sleeper returned ${res.status}`,
    );
  }
  return mapSleeperLeague((await res.json()) as SleeperLeague);
}
