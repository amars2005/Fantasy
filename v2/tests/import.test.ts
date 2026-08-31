/**
 * Platform imports.
 *
 * These map undocumented keys, so the tests pin the mappings that matter and,
 * just as importantly, check that anything unrecognised is *reported* rather
 * than silently dropped. An import is a pre-fill for a person to check; the
 * failure mode it must avoid is looking complete when it is not.
 */

import { describe, expect, it } from "vitest";

import { mapEspnLeague, normaliseLeagueId } from "../lib/import/espn";
import { mapSleeperLeague } from "../lib/import/sleeper";
import { nearestFfcFormat } from "../lib/config";

describe("Sleeper", () => {
  const raw = {
    name: "Dynasty Warriors",
    total_rosters: 12,
    roster_positions: ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF", "BN", "BN", "IR"],
    settings: { playoff_week_start: 15, playoff_teams: 6 },
    scoring_settings: {
      pass_yd: 0.04,
      pass_td: 6,
      pass_int: -2,
      rush_yd: 0.1,
      rush_td: 6,
      rec: 0.5,
      rec_yd: 0.1,
      rec_td: 6,
      fum_lost: -2,
      sack: 1,
      int: 2,
      def_td: 6,
      blk_kick: 2,
      pts_allow_0: 10,
      pts_allow_1_6: 7,
      xpm: 1,
      fgm_50p: 5,
      fgmiss: -1,
      some_future_key: 3,
    },
  };

  const imported = mapSleeperLeague(raw);

  it("maps scoring onto nflverse column names", () => {
    expect(imported.config.scoring.passing_yards).toBe(0.04);
    expect(imported.config.scoring.passing_tds).toBe(6);
    expect(imported.config.scoring.receptions).toBe(0.5);
    expect(imported.config.scoring.fumbles_lost).toBe(-2);
  });

  it("reads the roster into starters, bench and IR", () => {
    expect(imported.config.teams).toBe(12);
    expect(imported.config.starters.RB).toBe(2);
    expect(imported.config.starters.WR).toBe(2);
    expect(imported.config.starters.FLEX).toBe(1);
    expect(imported.config.starters.DST).toBe(1);
    expect(imported.config.bench).toBe(2);
    expect(imported.config.irSlots).toBe(1);
  });

  it("expands blanket blocked-kick and missed-field-goal rules", () => {
    expect(imported.config.dst.events.def_punt_blocks).toBe(2);
    expect(imported.config.dst.events.def_fg_blocks).toBe(2);
    expect(imported.config.kickerScoring.fg_missed_0_19).toBe(-1);
    // Sleeper's 50+ bucket covers both of the reference league's top bands.
    expect(imported.config.kickerScoring.fg_made_50_59).toBe(5);
    expect(imported.config.kickerScoring.fg_made_60_).toBe(5);
  });

  it("takes points-allowed bands from the keys that are present", () => {
    const bands = imported.config.dst.pointsAllowedBands;
    expect(bands.find((b) => b.low === 0 && b.high === 0)?.points).toBe(10);
    expect(bands.find((b) => b.low === 1 && b.high === 6)?.points).toBe(7);
  });

  it("derives the playoff window from playoff_week_start", () => {
    expect(imported.config.schedule.regularSeasonWeeks).toBe(14);
    expect(imported.config.schedule.playoffWeeks).toEqual([15, 16, 17]);
  });

  it("reports keys it could not place instead of dropping them", () => {
    expect(imported.unmapped).toContain("some_future_key");
  });

  it("anchors a half-PPR league to the half-PPR board", () => {
    expect(nearestFfcFormat(imported.config)).toBe("half-ppr");
  });
});

describe("ESPN", () => {
  const raw = {
    settings: {
      name: "Office League",
      size: 10,
      scoringSettings: {
        scoringItems: [
          { statId: 3, points: 0.04 },
          { statId: 4, points: 4 },
          { statId: 53, points: 1 },
          { statId: 42, points: 0.1 },
          { statId: 99, points: 2 },
          { statId: 9999, points: 5 },
        ],
      },
      rosterSettings: {
        lineupSlotCounts: { "0": 1, "2": 2, "4": 2, "6": 1, "23": 1, "16": 1, "17": 1, "20": 6, "21": 1 },
      },
      scheduleSettings: { matchupPeriodCount: 13, playoffTeamCount: 4 },
    },
  };

  const imported = mapEspnLeague(raw);

  it("maps the stat ids it knows", () => {
    expect(imported.config.scoring.passing_yards).toBe(0.04);
    expect(imported.config.scoring.passing_tds).toBe(4);
    expect(imported.config.scoring.receptions).toBe(1);
    expect(imported.config.dst.events.def_sacks).toBe(2);
  });

  it("reads lineup slot ids into starters", () => {
    expect(imported.config.teams).toBe(10);
    expect(imported.config.starters.QB).toBe(1);
    expect(imported.config.starters.RB).toBe(2);
    expect(imported.config.starters.FLEX).toBe(1);
    expect(imported.config.bench).toBe(6);
    expect(imported.config.irSlots).toBe(1);
  });

  it("reports unknown stat ids rather than guessing", () => {
    expect(imported.unmapped.some((u) => u.includes("9999"))).toBe(true);
  });

  it("always carries a caution, because the id tables drift", () => {
    expect(imported.caution).toMatch(/undocumented/i);
  });

  it("derives the playoff window from the matchup period count", () => {
    expect(imported.config.schedule.regularSeasonWeeks).toBe(13);
    expect(imported.config.schedule.playoffWeeks).toEqual([14, 15, 16]);
  });
});

/**
 * A real ESPN response, reduced to its scoring block.
 *
 * Taken from a live 2026 league whose import produced 31 "unmapped" entries,
 * of which only five were worth any points. The ids are exactly as ESPN
 * returned them, including the ladders it reports entirely at zero -- which is
 * the case that used to be silently replaced with the reference league's bands.
 */
describe("ESPN, against a real league's scoring block", () => {
  const scoringItems = [
    // Passing, rushing, receiving.
    { statId: 3, points: 0.04 }, { statId: 4, points: 4 },
    { statId: 19, points: 2 }, { statId: 20, points: -2 },
    { statId: 24, points: 0.1 }, { statId: 25, points: 6 },
    { statId: 26, points: 2 }, { statId: 42, points: 0.1 },
    { statId: 43, points: 6 }, { statId: 44, points: 2 },
    { statId: 53, points: 1 }, { statId: 72, points: -2 },
    // Long-touchdown bonuses this model cannot express.
    { statId: 16, points: 1 }, { statId: 36, points: 0.5 },
    { statId: 46, points: 1 },
    // Returns and defensive touchdowns, all worth six.
    { statId: 63, points: 6 }, { statId: 93, points: 6 },
    { statId: 101, points: 6 }, { statId: 102, points: 6 },
    { statId: 103, points: 6 }, { statId: 104, points: 6 },
    // Defensive events.
    { statId: 95, points: 2 }, { statId: 96, points: 2 },
    { statId: 98, points: 2 }, { statId: 99, points: 1 },
    { statId: 106, points: 0 },
    // Kicking: banded made, blanket and short-range misses.
    { statId: 80, points: 3 }, { statId: 77, points: 4 },
    { statId: 198, points: 6 }, { statId: 201, points: 7 },
    { statId: 86, points: 1 }, { statId: 85, points: -2 },
    { statId: 82, points: -10 },
    // Both ladders, returned in full at zero.
    { statId: 89, points: 0 }, { statId: 90, points: 0 },
    { statId: 91, points: 0 }, { statId: 92, points: 0 },
    { statId: 123, points: 0 }, { statId: 124, points: 0 },
    { statId: 125, points: 0 },
    { statId: 128, points: 0 }, { statId: 129, points: 0 },
    { statId: 130, points: 0 }, { statId: 132, points: 0 },
    { statId: 133, points: 0 }, { statId: 134, points: 0 },
    { statId: 135, points: 0 }, { statId: 136, points: 0 },
    // Odds and ends worth a point or two.
    { statId: 206, points: 2 }, { statId: 209, points: 1 },
  ];

  const imported = mapEspnLeague({
    settings: {
      name: "Imperial Immortals",
      size: 14,
      scoringSettings: { scoringItems },
      rosterSettings: {
        lineupSlotCounts: { "0": 1, "2": 2, "4": 2, "6": 1, "23": 1, "16": 1, "17": 1, "20": 6 },
      },
      scheduleSettings: { matchupPeriodCount: 14, playoffTeamCount: 6 },
    },
  });

  it("keeps a zeroed points-allowed ladder instead of inventing one", () => {
    const bands = imported.config.dst.pointsAllowedBands;
    expect(bands.every((b) => b.points === 0)).toBe(true);
    // The reference league pays 5 for a shutout. Importing that here would add
    // scoring the league does not have.
    expect(bands.find((b) => b.low === 0 && b.high === 0)?.points).toBe(0);
  });

  it("keeps a zeroed yards-allowed ladder too", () => {
    expect(imported.config.dst.yardsAllowedBands.every((b) => b.points === 0)).toBe(true);
  });

  it("maps the touchdown ids that have a home", () => {
    expect(imported.config.scoring.fumble_recovery_tds).toBe(6);
    expect(imported.config.scoring.special_teams_tds).toBe(6);
    expect(imported.config.scoring.pt_return_tds).toBe(6);
    expect(imported.config.dst.events.def_tds).toBe(6);
  });

  it("reads ESPN's coarse field-goal buckets across our finer bands", () => {
    const k = imported.config.kickerScoring;
    // One rule for everything inside 40 pays all three short bands.
    expect(k.fg_made_0_19).toBe(3);
    expect(k.fg_made_20_29).toBe(3);
    expect(k.fg_made_30_39).toBe(3);
    expect(k.fg_made_40_49).toBe(4);
    expect(k.fg_made_50_59).toBe(6);
    expect(k.fg_made_60_).toBe(7);
    expect(k.pat_made).toBe(1);
  });

  it("sums the overlapping miss rules, because ESPN's counters both fire", () => {
    const k = imported.config.kickerScoring;
    // A missed 30-yarder trips both "any miss" (-2) and "miss inside 40" (-10).
    expect(k.fg_missed_30_39).toBe(-12);
    // Beyond 40 only the blanket rule applies.
    expect(k.fg_missed_40_49).toBe(-2);
    expect(k.fg_missed_60_).toBe(-2);
  });

  it("expands the single blocked-kick rule across all three columns", () => {
    // This league does not score blocked kicks, and silence inside a block ESPN
    // populated means zero, not the reference league's 2.
    expect(imported.config.dst.events.def_punt_blocks).toBe(0);
    expect(imported.config.dst.events.def_fg_blocks).toBe(0);
  });

  it("separates rules worth points from rules worth nothing", () => {
    // Only the five genuine bonuses need a person's attention.
    expect(imported.unmapped).toHaveLength(5);
    expect(imported.unmapped.join(" ")).toMatch(/50\+ yard TD pass bonus/);
    expect(imported.unmapped.join(" ")).toMatch(/1pt safety/);
    // The 0-point entries are reported, but separately and quietly.
    expect(imported.ignored.join(" ")).toMatch(/fumble forced/);
    expect(imported.unmapped.join(" ")).not.toMatch(/fumble forced/);
  });

  it("names ids in words rather than as bare numbers", () => {
    expect(imported.unmapped.join(" ")).not.toMatch(/^statId 16 /);
  });
});

describe("ESPN league ids", () => {
  it("takes the bare number", () => {
    expect(normaliseLeagueId("1234567")).toBe("1234567");
    expect(normaliseLeagueId("  1234567  ")).toBe("1234567");
  });

  it("takes what people actually paste", () => {
    expect(
      normaliseLeagueId("https://fantasy.espn.com/football/league?leagueId=1234567&seasonId=2026"),
    ).toBe("1234567");
    expect(normaliseLeagueId("leagueId=1234567")).toBe("1234567");
  });

  it("rejects a string with no id in it, rather than letting ESPN 400", () => {
    expect(() => normaliseLeagueId("my league")).toThrow(/does not look like/i);
  });
});
