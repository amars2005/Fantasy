/**
 * Importing a league from ESPN.
 *
 * The fiddly one, and the reason the whole import flow is a *pre-fill* rather
 * than a commit. ESPN's settings endpoint returns scoring as `{statId, points}`
 * and roster slots as numeric ids, both undocumented and both known to shift
 * between seasons. The tables below cover the ids that are well established;
 * anything else is reported as unmapped rather than guessed at.
 *
 * Treat a successful import as a filled-in form to check, never as a
 * configuration to trust. An id table that has drifted should produce a visibly
 * wrong number on a review screen, which is the whole design.
 *
 * Private leagues need `espn_s2` and `SWID` cookies, which the user pastes from
 * their browser. Public leagues need nothing.
 */

import { REFERENCE_LEAGUE } from "../config";
import type { LeagueConfig, Position, Slot } from "../types";

const HOST = "https://lm-api-reads.fantasy.espn.com";

/**
 * ESPN scoring statId -> the nflverse column our rules use.
 *
 * Deliberately conservative. Every id omitted here surfaces to the user rather
 * than being silently dropped or mis-assigned.
 */
const STAT_MAP: Record<number, string> = {
  3: "passing_yards",
  4: "passing_tds",
  19: "passing_2pt_conversions",
  20: "passing_interceptions",
  24: "rushing_yards",
  25: "rushing_tds",
  26: "rushing_2pt_conversions",
  42: "receiving_yards",
  43: "receiving_tds",
  44: "receiving_2pt_conversions",
  53: "receptions",
  72: "fumbles_lost",
};

const KICKER_STAT_MAP: Record<number, string> = {
  74: "fg_made_50_59",
  77: "fg_made_40_49",
  80: "fg_made_30_39",
  86: "pat_made",
};

const DST_STAT_MAP: Record<number, string> = {
  95: "def_interceptions",
  96: "def_fumbles",
  97: "def_fg_blocks",
  98: "def_safeties",
  99: "def_sacks",
};

/** ESPN lineup slot id -> our slot. */
const SLOT_MAP: Record<number, Slot> = {
  0: "QB",
  2: "RB",
  4: "WR",
  6: "TE",
  7: "SUPERFLEX", // "OP" -- offensive player, i.e. superflex
  16: "DST",
  17: "K",
  23: "FLEX",
};

const BENCH_SLOT = 20;
const IR_SLOT = 21;

export interface EspnImport {
  name: string;
  config: LeagueConfig;
  /** Stat and slot ids the table did not cover, for the review screen. */
  unmapped: string[];
  /** Always present: this import needs eyes on it. */
  caution: string;
}

interface EspnSettings {
  settings?: {
    name?: string;
    size?: number;
    scoringSettings?: { scoringItems?: { statId: number; points: number }[] };
    rosterSettings?: { lineupSlotCounts?: Record<string, number> };
    scheduleSettings?: { matchupPeriodCount?: number; playoffTeamCount?: number };
  };
}

export function mapEspnLeague(raw: EspnSettings): EspnImport {
  const s = raw.settings ?? {};
  const scoring: Record<string, number> = {};
  const kicker: Record<string, number> = {};
  const dstEvents: Record<string, number> = {};
  const unmapped: string[] = [];

  for (const item of s.scoringSettings?.scoringItems ?? []) {
    if (STAT_MAP[item.statId]) scoring[STAT_MAP[item.statId]] = item.points;
    else if (KICKER_STAT_MAP[item.statId]) kicker[KICKER_STAT_MAP[item.statId]] = item.points;
    else if (DST_STAT_MAP[item.statId]) dstEvents[DST_STAT_MAP[item.statId]] = item.points;
    else unmapped.push(`statId ${item.statId} (${item.points} pts)`);
  }

  const starters: Partial<Record<Slot, number>> = {};
  let bench = 0;
  let irSlots = 0;
  for (const [id, count] of Object.entries(s.rosterSettings?.lineupSlotCounts ?? {})) {
    const n = Number(count);
    if (!n) continue;
    const slotId = Number(id);
    if (slotId === BENCH_SLOT) bench += n;
    else if (slotId === IR_SLOT) irSlots += n;
    else if (SLOT_MAP[slotId]) starters[SLOT_MAP[slotId]] = (starters[SLOT_MAP[slotId]] ?? 0) + n;
    else unmapped.push(`lineup slot ${slotId} x${n}`);
  }

  const starterCount = Object.values(starters).reduce((a, b) => a + (b ?? 0), 0);
  const regularWeeks = s.scheduleSettings?.matchupPeriodCount ?? 14;

  const config: LeagueConfig = {
    teams: s.size ?? REFERENCE_LEAGUE.teams,
    rounds: starterCount + bench || REFERENCE_LEAGUE.rounds,
    bench,
    irSlots,
    starters: starterCount ? starters : REFERENCE_LEAGUE.starters,
    flexEligible: ["RB", "WR", "TE"] as Position[],
    positionMax: { ...REFERENCE_LEAGUE.positionMax },
    schedule: {
      regularSeasonWeeks: regularWeeks,
      playoffWeeks: [regularWeeks + 1, regularWeeks + 2, regularWeeks + 3],
      playoffTeams: s.scheduleSettings?.playoffTeamCount ?? REFERENCE_LEAGUE.schedule.playoffTeams,
    },
    // Anything the tables did not cover falls back to the reference values, so
    // the form is never left half-empty -- but the caution below says so.
    scoring: { ...REFERENCE_LEAGUE.scoring, ...scoring },
    kickerScoring: { ...REFERENCE_LEAGUE.kickerScoring, ...kicker },
    dst: {
      events: { ...REFERENCE_LEAGUE.dst.events, ...dstEvents },
      pointsAllowedBands: REFERENCE_LEAGUE.dst.pointsAllowedBands,
      yardsAllowedBands: REFERENCE_LEAGUE.dst.yardsAllowedBands,
    },
  };

  return {
    name: s.name ?? "Imported league",
    config,
    unmapped,
    caution:
      "ESPN's stat and slot ids are undocumented and change between seasons. " +
      "Check every scoring value against your league settings before saving -- " +
      "anything this importer could not place is listed above, and the points " +
      "and yards allowed bands for defences were not imported at all.",
  };
}

export interface EspnCredentials {
  espnS2?: string;
  swid?: string;
}

export async function importEspnLeague(
  leagueId: string,
  season: number,
  creds: EspnCredentials = {},
): Promise<EspnImport> {
  const url =
    `${HOST}/apis/v3/games/ffl/seasons/${season}/segments/0/leagues/` +
    `${encodeURIComponent(leagueId)}?view=mSettings`;

  const headers: Record<string, string> = { accept: "application/json" };
  if (creds.espnS2 && creds.swid) {
    headers.cookie = `espn_s2=${creds.espnS2}; SWID=${creds.swid}`;
  }

  const res = await fetch(url, { headers });
  if (!res.ok) {
    throw new Error(
      res.status === 401
        ? "ESPN refused the request. Private leagues need your espn_s2 and SWID cookies."
        : `ESPN returned ${res.status}`,
    );
  }
  return mapEspnLeague((await res.json()) as EspnSettings);
}
