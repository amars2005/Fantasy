/**
 * Platform imports.
 *
 * These map undocumented keys, so the tests pin the mappings that matter and,
 * just as importantly, check that anything unrecognised is *reported* rather
 * than silently dropped. An import is a pre-fill for a person to check; the
 * failure mode it must avoid is looking complete when it is not.
 */

import { describe, expect, it } from "vitest";

import { mapEspnLeague } from "../lib/import/espn";
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
