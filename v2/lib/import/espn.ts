/**
 * Importing a league from ESPN.
 *
 * The fiddly one, and the reason the whole import flow is a *pre-fill* rather
 * than a commit. ESPN's settings endpoint returns scoring as `{statId, points}`
 * and roster slots as numeric ids, both undocumented and both known to shift
 * between seasons.
 *
 * Three things about ESPN's model drive the shape of this file.
 *
 * First, **scoring items are counters and they sum**. A 55-yard field goal
 * increments "total FG made", "FG made 50+" and "FG made 50-59" all at once, so
 * a league that sets more than one of them pays all of them. Rules covering
 * overlapping ranges are therefore *added* here, never overwritten. Unused
 * items come back as 0, which is what makes adding them safe.
 *
 * Second, **ESPN returns every item its engine knows about**, including the
 * ones a league does not use, at 0 points. An id worth 0 changes no projection,
 * so reporting it as "not imported" is noise that buries the entries which
 * genuinely need attention. Those two lists are kept apart below.
 *
 * Third, and following from the second: **once ESPN has spoken about a block of
 * scoring, its silence is meaningful**. A league that defines field goals but
 * no missed-field-goal rule does not score misses, so filling that gap from the
 * reference league would invent scoring the league does not have. Gaps inside a
 * block ESPN populated are therefore zeroed, not defaulted.
 *
 * Treat a successful import as a filled-in form to check, never as a
 * configuration to trust. An id table that has drifted should produce a visibly
 * wrong number on a review screen, which is the whole design.
 *
 * Private leagues need `espn_s2` and `SWID` cookies, which the user pastes from
 * their browser. Public leagues need nothing.
 */

import { REFERENCE_LEAGUE, tdBandsFromLow, tdExactBand } from "../config";
import type { Band, LeagueConfig, Position, Slot } from "../types";

const HOST = "https://lm-api-reads.fantasy.espn.com";

/** ESPN position ids, for reading `pointsOverrides`. */
const ESPN_POS_KICKER = "5";
const ESPN_POS_DST = "16";

/**
 * ESPN scoring statId -> the nflverse column our rules use.
 *
 * Deliberately conservative: every id omitted here surfaces to the user rather
 * than being silently dropped or mis-assigned. Ids 41 and 53 are ESPN
 * duplicates for receptions; only one is ever non-zero in a league.
 */
const STAT_MAP: Record<number, string> = {
  3: "passing_yards",
  4: "passing_tds",
  19: "passing_2pt_conversions",
  20: "passing_interceptions",
  24: "rushing_yards",
  25: "rushing_tds",
  26: "rushing_2pt_conversions",
  41: "receptions",
  42: "receiving_yards",
  43: "receiving_tds",
  44: "receiving_2pt_conversions",
  53: "receptions",
  63: "fumble_recovery_tds",
  72: "fumbles_lost",
  // A returner's own touchdowns. The same two ids also pay the defence that
  // scored them, which RETURN_TD_IDS below handles separately.
  101: "special_teams_tds",
  102: "pt_return_tds",
};

const MADE_BANDS = [
  "fg_made_0_19",
  "fg_made_20_29",
  "fg_made_30_39",
  "fg_made_40_49",
  "fg_made_50_59",
  "fg_made_60_",
];
const MISSED_BANDS = [
  "fg_missed_0_19",
  "fg_missed_20_29",
  "fg_missed_30_39",
  "fg_missed_40_49",
  "fg_missed_50_59",
  "fg_missed_60_",
];

/**
 * Kicker ids -> the reference league's distance bands.
 *
 * ESPN's buckets are coarser than ours in both directions: "0-39" spans three
 * of our bands, "50+" spans two, and the totals span all six. Each id pays out
 * across every band it covers, and overlapping ids accumulate.
 *
 * Note 74 against 198/201: ESPN's FG50P means "50 or more" and so includes 60+,
 * while 198 is 50-59 exactly and 201 is 60+. Reading 74 as 50-59 alone -- which
 * this importer used to do -- loses every 60-yard kick's points.
 */
const KICKER_STAT_MAP: Record<number, string[]> = {
  83: MADE_BANDS, // FG: total made, any distance
  85: MISSED_BANDS, // FGM: total missed, any distance
  80: MADE_BANDS.slice(0, 3), // FG0: made 0-39
  82: MISSED_BANDS.slice(0, 3), // FGM0: missed 0-39
  77: ["fg_made_40_49"], // FG40
  79: ["fg_missed_40_49"], // FGM40
  74: ["fg_made_50_59", "fg_made_60_"], // FG50P: made 50+, includes 60+
  76: ["fg_missed_50_59", "fg_missed_60_"], // FGM50P: missed 50+
  198: ["fg_made_50_59"], // FG50: made 50-59 exactly
  200: ["fg_missed_50_59"], // FGM50
  201: ["fg_made_60_"], // FG60
  203: ["fg_missed_60_"], // FGM60
  86: ["pat_made"], // PAT
};

/**
 * Long-touchdown bonuses -> the bundle's banded touchdown columns.
 *
 * ESPN prices these two ways at once. Ids 175-186 name an exact band (0-9,
 * 10-19, 20-29, 30-39); ids 15/35/45 mean "40 or more" and 16/36/46 mean "50 or
 * more". They are counters like everything else here, so a 55-yard touchdown
 * pays the 40+ rule *and* the 50+ rule, and a league that sets both owes both.
 *
 * The bundle stores each touchdown in exactly one disjoint band, so a
 * cumulative rule is expanded across every band it covers and the sums come
 * out right -- the same treatment ESPN's coarse field-goal buckets get above.
 */
const LONG_TD_STAT_MAP: Record<number, string[]> = {
  // Passing.
  175: tdExactBand("passing", 0), 176: tdExactBand("passing", 10),
  177: tdExactBand("passing", 20), 178: tdExactBand("passing", 30),
  15: tdBandsFromLow("passing", 40), 16: tdBandsFromLow("passing", 50),
  // Rushing.
  179: tdExactBand("rushing", 0), 180: tdExactBand("rushing", 10),
  181: tdExactBand("rushing", 20), 182: tdExactBand("rushing", 30),
  35: tdBandsFromLow("rushing", 40), 36: tdBandsFromLow("rushing", 50),
  // Receiving.
  183: tdExactBand("receiving", 0), 184: tdExactBand("receiving", 10),
  185: tdExactBand("receiving", 20), 186: tdExactBand("receiving", 30),
  45: tdBandsFromLow("receiving", 40), 46: tdBandsFromLow("receiving", 50),
};

/**
 * Defensive events that map onto bundle columns.
 *
 * 97 is ESPN's single "blocked punt, PAT or FG" rule; the bundle scores those
 * three apart, so one rule pays all three columns.
 */
const DST_STAT_MAP: Record<number, string[]> = {
  95: ["def_interceptions"],
  96: ["def_fumbles"],
  97: ["def_punt_blocks", "def_pat_blocks", "def_fg_blocks"],
  98: ["def_safeties"],
  99: ["def_sacks"],
  // A returned two-point try and a safety on a try. Both are scored by the
  // defending team, and 206/209 are the generic ids most leagues actually set.
  // The "offensive" variants (204, 207) are left to surface as unmapped: they
  // pay a player rather than a defence, and guessing which would be worse than
  // saying so.
  205: ["def_two_point_returns"],
  206: ["def_two_point_returns"],
  208: ["def_one_point_safeties"],
  209: ["def_one_point_safeties"],
};

/**
 * The touchdown ids, grouped by the kind of touchdown that triggers them.
 *
 * ESPN lets a league price an interception return and a fumble return
 * differently; the bundle carries a single `def_tds` count and cannot. So each
 * kind's total is worked out separately and, when they disagree, the difference
 * is reported rather than averaged away silently.
 */
const RETURN_TD_IDS: Record<string, number[]> = {
  "interception return": [103, 94, 105],
  "fumble return": [104, 94, 105],
  "blocked kick return": [93, 105],
  "kickoff return": [101, 105],
  "punt return": [102, 105],
};

/**
 * The kinds `def_tds` actually counts.
 *
 * nflverse puts a defence's own returns -- interceptions and fumbles -- in
 * `def_tds`, and kick and punt returns in `special_teams_tds`, which ids 101
 * and 102 already score on the returner's line. So when a league prices the
 * kinds apart, the one number the board can carry should be what an
 * interception or fumble return pays. Taking the largest of the five instead
 * priced every defensive score at the rate of a blocked kick -- the rarest of
 * them, and one the column does not even count -- which overstated defences by
 * a few points a season in any league that pays a return premium.
 */
const DEF_TD_KINDS = ["interception return", "fumble return"];

/** statId -> [low, high] for the points-allowed ladder. */
const PA_BAND_IDS: [number, number, number][] = [
  [89, 0, 0], [90, 1, 6], [91, 7, 13], [92, 14, 17], [121, 18, 21],
  [122, 22, 27], [123, 28, 34], [124, 35, 45], [125, 46, 999],
];

/** ESPN emits a second, D/ST-specific copy of the same ladder. */
const DPA_BAND_IDS: [number, number, number][] = [
  [188, 0, 0], [189, 1, 6], [190, 7, 13], [191, 14, 17], [192, 18, 21],
  [193, 22, 27], [194, 28, 34], [195, 35, 45], [196, 46, 999],
];

const YA_BAND_IDS: [number, number, number][] = [
  [128, 0, 99], [129, 100, 199], [130, 200, 299], [131, 300, 349],
  [132, 350, 399], [133, 400, 449], [134, 450, 499], [135, 500, 549],
  [136, 550, 99999],
];

/**
 * Human labels for the ids worth naming in a report.
 *
 * Covers what a football league plausibly leaves unmapped -- bonuses, IDP,
 * punting, head coach. Anything else is reported by number.
 */
const STAT_LABELS: Record<number, string> = {
  0: "each pass attempted", 1: "each pass completed", 2: "each incomplete pass",
  15: "40+ yard TD pass bonus", 16: "50+ yard TD pass bonus",
  17: "300-399 yard passing game", 18: "400+ yard passing game",
  21: "passing completion pct", 22: "passing yards per game",
  23: "rushing attempts", 35: "40+ yard TD rush bonus", 36: "50+ yard TD rush bonus",
  37: "100-199 yard rushing game", 38: "200+ yard rushing game",
  39: "rushing yards per attempt", 40: "rushing yards per game",
  45: "40+ yard TD reception bonus", 46: "50+ yard TD reception bonus",
  56: "100-199 yard receiving game", 57: "200+ yard receiving game",
  58: "receiving target", 59: "receiving yards after catch",
  60: "receiving yards per catch", 61: "receiving yards per game",
  62: "total 2pt conversions", 64: "sacked", 68: "total fumbles",
  73: "total turnovers", 75: "FG attempted (50+)", 78: "FG attempted (40-49)",
  81: "FG attempted (0-39)", 84: "total FG attempted",
  87: "PAT attempted", 88: "PAT missed",
  106: "each fumble forced", 107: "assisted tackles", 108: "solo tackles",
  109: "total tackles", 112: "stuffs", 113: "passes defensed",
  114: "kickoff return yards", 115: "punt return yards",
  120: "points allowed (per point)", 126: "points allowed per game",
  127: "yards allowed (per yard)", 137: "yards allowed per game",
  138: "net punts", 140: "punts inside the 10", 141: "punts inside the 20",
  155: "team win", 156: "team loss", 158: "points scored",
  175: "0-9 yd TD pass bonus", 176: "10-19 yd TD pass bonus",
  177: "20-29 yd TD pass bonus", 178: "30-39 yd TD pass bonus",
  179: "0-9 yd TD rush bonus", 180: "10-19 yd TD rush bonus",
  181: "20-29 yd TD rush bonus", 182: "30-39 yd TD rush bonus",
  183: "0-9 yd TD reception bonus", 184: "10-19 yd TD reception bonus",
  185: "20-29 yd TD reception bonus", 186: "30-39 yd TD reception bonus",
  187: "D/ST points allowed (per point)", 197: "D/ST points allowed per game",
  199: "FG attempted (50-59)", 202: "FG attempted (60+)",
  204: "offensive 2pt return", 205: "defensive 2pt return", 206: "2pt return",
  207: "offensive 1pt safety", 208: "defensive 1pt safety", 209: "1pt safety",
  210: "games played", 211: "passing first down", 212: "rushing first down",
  213: "receiving first down", 214: "FG made yards", 215: "FG missed yards",
};

/** "1 pt", "2 pts", "0.5 pts" -- these lists are read, so they should read. */
function pointsLabel(value: number): string {
  return `${value} ${Math.abs(value) === 1 ? "pt" : "pts"}`;
}

function describeStat(statId: number, value: number): string {
  const label = STAT_LABELS[statId];
  return label
    ? `${label} — ${pointsLabel(value)} (id ${statId})`
    : `statId ${statId} (${pointsLabel(value)})`;
}

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
  /** Rules worth points that this model cannot express. These need eyes. */
  unmapped: string[];
  /** Rules ESPN returned at 0 points, so they change nothing. Informational. */
  ignored: string[];
  /** Places the mapping had to make a judgement call worth reviewing. */
  notes: string[];
  /** Always present: this import needs eyes on it. */
  caution: string;
}

interface ScoringItem {
  statId: number;
  points?: number;
  /** Per-position point values, keyed by ESPN position id. */
  pointsOverrides?: Record<string, number>;
}

interface EspnSettings {
  settings?: {
    name?: string;
    size?: number;
    scoringSettings?: { scoringItems?: ScoringItem[] };
    rosterSettings?: { lineupSlotCounts?: Record<string, number> };
    scheduleSettings?: { matchupPeriodCount?: number; playoffTeamCount?: number };
  };
}

/**
 * Build the bands a league actually defines.
 *
 * Returns null when ESPN mentioned none of the ladder's ids, which is the only
 * case where falling back to the reference bands is right. A ladder ESPN
 * returned entirely at zero is a real configuration -- a league that scores its
 * defences on events alone -- and must be kept as zeros.
 */
function bandsFrom(
  points: Map<number, number>,
  spec: [number, number, number][],
): Band[] | null {
  const present = spec.filter(([id]) => points.has(id));
  if (!present.length) return null;
  return present.map(([id, low, high]) => ({ low, high, points: points.get(id)! }));
}

/**
 * Fill the keys a populated block never mentioned with zero.
 *
 * See the note at the top of the file: inside a block ESPN spoke about, silence
 * means "not scored", so defaulting from the reference league would invent
 * rules. An untouched block returns the reference values instead.
 */
function completeBlock(
  imported: Record<string, number>,
  reference: Record<string, number>,
): Record<string, number> {
  if (!Object.keys(imported).length) return { ...reference };
  const out: Record<string, number> = {};
  for (const key of Object.keys(reference)) out[key] = imported[key] ?? 0;
  for (const [key, value] of Object.entries(imported)) out[key] = value;
  return out;
}

export function mapEspnLeague(raw: EspnSettings): EspnImport {
  const s = raw.settings ?? {};
  const items = s.scoringSettings?.scoringItems ?? [];

  // Index every id once. Position overrides matter for the two spots where one
  // rule means different things to different positions.
  const generic = new Map<number, number>();
  const forKicker = new Map<number, number>();
  const forDst = new Map<number, number>();
  for (const item of items) {
    const base = item.points ?? 0;
    generic.set(item.statId, base);
    forKicker.set(item.statId, item.pointsOverrides?.[ESPN_POS_KICKER] ?? base);
    forDst.set(item.statId, item.pointsOverrides?.[ESPN_POS_DST] ?? base);
  }

  const scoring: Record<string, number> = {};
  const kicker: Record<string, number> = {};
  const dstEvents: Record<string, number> = {};
  const unmapped: string[] = [];
  const ignored: string[] = [];
  const notes: string[] = [];

  const bandIds = new Set<number>(
    [...PA_BAND_IDS, ...DPA_BAND_IDS, ...YA_BAND_IDS].map(([id]) => id),
  );
  const returnTdIds = new Set<number>(Object.values(RETURN_TD_IDS).flat());

  const add = (into: Record<string, number>, keys: string[], points: number) => {
    for (const key of keys) into[key] = (into[key] ?? 0) + points;
  };

  // ESPN returns its items in engine order, which reaches the review screen as
  // a jumble -- a passing bonus, a safety, a rushing bonus. Ids run in football
  // order, so walking them sorted keeps a league's bonuses next to each other.
  const sorted = [...items].sort((a, b) => a.statId - b.statId);

  for (const item of sorted) {
    const { statId } = item;
    const points = generic.get(statId) ?? 0;

    if (STAT_MAP[statId]) {
      // 41 and 53 are duplicate ids for receptions; only one is ever set, so
      // the larger magnitude is the league's actual rule.
      const key = STAT_MAP[statId];
      if (Math.abs(points) >= Math.abs(scoring[key] ?? 0)) scoring[key] = points;
    }
    if (LONG_TD_STAT_MAP[statId]) {
      // Cumulative rules overlap by design, so these add rather than replace:
      // a league paying both 40+ and 50+ owes both on a 55-yard score.
      add(scoring, LONG_TD_STAT_MAP[statId], points);
    }
    if (KICKER_STAT_MAP[statId]) {
      add(kicker, KICKER_STAT_MAP[statId], forKicker.get(statId) ?? points);
    }
    if (DST_STAT_MAP[statId]) {
      add(dstEvents, DST_STAT_MAP[statId], forDst.get(statId) ?? points);
    }

    const handled =
      STAT_MAP[statId] !== undefined ||
      LONG_TD_STAT_MAP[statId] !== undefined ||
      KICKER_STAT_MAP[statId] !== undefined ||
      DST_STAT_MAP[statId] !== undefined ||
      bandIds.has(statId) ||
      returnTdIds.has(statId);

    if (handled) continue;
    // A rule worth nothing changes no projection. Say so quietly.
    if (points === 0) ignored.push(describeStat(statId, points));
    else unmapped.push(describeStat(statId, points));
  }

  // --- defensive touchdowns -------------------------------------------------
  // Work out what each kind of touchdown pays, then collapse to the single
  // count the bundle carries.
  const tdValues = new Map<string, number>();
  for (const [kind, ids] of Object.entries(RETURN_TD_IDS)) {
    const contributing = ids.filter((id) => forDst.has(id));
    if (!contributing.length) continue;
    tdValues.set(kind, contributing.reduce((a, id) => a + (forDst.get(id) ?? 0), 0));
  }
  if (tdValues.size) {
    const distinct = [...new Set(tdValues.values())];
    const counted = DEF_TD_KINDS.filter((k) => tdValues.has(k)).map((k) => tdValues.get(k)!);
    dstEvents.def_tds = Math.max(...(counted.length ? counted : distinct));
    if (distinct.length > 1) {
      const detail = [...tdValues.entries()].map(([kind, v]) => `${kind} ${v}`).join(", ");
      notes.push(
        `Defensive touchdowns are priced differently by type in your league ` +
          `(${detail}). The board carries one value per defensive touchdown, so ` +
          `it used ${dstEvents.def_tds}` +
          (counted.length
            ? ` — what an interception or fumble return pays, which is what the ` +
              `defensive touchdown count is made of. Kickoff and punt returns are ` +
              `scored separately, on the returner's own line.`
            : `, the highest of them.`),
      );
    }
  }

  // --- roster ---------------------------------------------------------------
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

  // ESPN emits the generic points-allowed ladder, the D/ST-specific one, or
  // both. Prefer whichever the league actually defines.
  const paBands = bandsFrom(forDst, DPA_BAND_IDS) ?? bandsFrom(forDst, PA_BAND_IDS);
  const yaBands = bandsFrom(forDst, YA_BAND_IDS);

  if (!paBands) {
    notes.push(
      "ESPN returned no points-allowed ladder, so the reference league's bands " +
        "were kept. Check them if your league scores points allowed.",
    );
  }
  if (!yaBands) {
    notes.push(
      "ESPN returned no yards-allowed ladder, so the reference league's bands " +
        "were kept. Check them if your league scores yards allowed.",
    );
  }

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
    scoring: completeBlock(scoring, REFERENCE_LEAGUE.scoring),
    kickerScoring: completeBlock(kicker, REFERENCE_LEAGUE.kickerScoring),
    dst: {
      events: completeBlock(dstEvents, REFERENCE_LEAGUE.dst.events),
      pointsAllowedBands: paBands ?? REFERENCE_LEAGUE.dst.pointsAllowedBands,
      yardsAllowedBands: yaBands ?? REFERENCE_LEAGUE.dst.yardsAllowedBands,
    },
  };

  return {
    name: s.name ?? "Imported league",
    config,
    unmapped,
    ignored,
    notes,
    caution:
      "ESPN's stat and slot ids are undocumented and change between seasons. " +
      "Check the values below against your league settings before saving.",
  };
}

export interface EspnCredentials {
  espnS2?: string;
  swid?: string;
}

/**
 * Pull the bare numeric league id out of whatever the user pasted.
 *
 * People paste the whole league URL, or `leagueId=123`, far more often than
 * they paste `123`. ESPN answers any of those with a 400 whose body never
 * reaches the user, so normalise here and fail with something actionable.
 */
export class LeagueIdError extends Error {}

export function normaliseLeagueId(input: string): string {
  const trimmed = input.trim();
  const fromQuery = /[?&]leagueId=(\d+)/i.exec(trimmed);
  const id = fromQuery ? fromQuery[1] : /(\d{2,})/.exec(trimmed)?.[1];
  if (!id) {
    throw new LeagueIdError(
      "That does not look like an ESPN league id. It is the number after " +
        "`leagueId=` in your league's URL, for example 1234567.",
    );
  }
  return id;
}

export async function importEspnLeague(
  leagueId: string,
  season: number,
  creds: EspnCredentials = {},
): Promise<EspnImport> {
  const id = normaliseLeagueId(leagueId);
  const url =
    `${HOST}/apis/v3/games/ffl/seasons/${season}/segments/0/leagues/` +
    `${id}?view=mSettings`;

  const headers: Record<string, string> = { accept: "application/json" };
  const s2 = creds.espnS2?.trim();
  const swid = creds.swid?.trim();
  if (s2 && swid) {
    headers.cookie = `espn_s2=${s2}; SWID=${swid}`;
  }

  const res = await fetch(url, { headers });
  if (!res.ok) {
    // Sending neither cookie is the normal public-league case; sending exactly
    // one is always a mistake, and worth its own message.
    const halfCredentials = Boolean(s2) !== Boolean(swid);
    if (res.status === 401 || res.status === 403) {
      throw new Error(
        halfCredentials
          ? "ESPN refused the request, and only one of the two cookies was " +
            "filled in. Private leagues need both espn_s2 and SWID."
          : "ESPN refused the request. Private leagues need your espn_s2 and " +
            "SWID cookies, copied from your browser while signed in to ESPN.",
      );
    }
    if (res.status === 404) {
      throw new Error(
        `ESPN has no league ${id} in the ${season} season. Check the id, and ` +
          "that the league existed that year.",
      );
    }
    throw new Error(`ESPN returned ${res.status} for league ${id}.`);
  }
  return mapEspnLeague((await res.json()) as EspnSettings);
}
