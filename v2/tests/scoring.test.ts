/**
 * Scoring, banding, and league configuration.
 *
 * `scoreLine` is the load-bearing primitive: every projection, replacement
 * level and VOR number downstream is denominated in what it returns.
 */

import { describe, expect, it } from "vitest";

import {
  anchorWarnings,
  ConfigError,
  nearestFfcFormat,
  PPR_SCORING,
  REFERENCE_LEAGUE,
  validateLeague,
} from "../lib/config";
import { roundTo } from "../lib/bundle";
import { scoreBand, scoreColumnar, scoreLine } from "../lib/scoring";
import type { LeagueConfig } from "../lib/types";

describe("scoreLine", () => {
  it("scores a full PPR stat line", () => {
    const line = { receptions: 100, receiving_yards: 1400, receiving_tds: 10 };
    expect(scoreLine(line, PPR_SCORING)).toBeCloseTo(100 + 140 + 60, 9);
  });

  it("treats absent stats as zero", () => {
    // A receiver has no passing stats; the line must still score correctly
    // without the caller padding it.
    expect(scoreLine({ receptions: 5 }, PPR_SCORING)).toBeCloseTo(5, 9);
    expect(scoreLine({}, PPR_SCORING)).toBe(0);
  });

  it("applies negative rules", () => {
    expect(scoreLine({ passing_interceptions: 3, fumbles_lost: 2 }, PPR_SCORING)).toBeCloseTo(
      -10,
      9,
    );
  });

  it("honours a custom scoring vector", () => {
    const sixPoint = { ...PPR_SCORING, passing_tds: 6 };
    const line = { passing_tds: 30, passing_yards: 4500 };
    expect(scoreLine(line, PPR_SCORING)).toBeCloseTo(120 + 180, 9);
    expect(scoreLine(line, sixPoint)).toBeCloseTo(180 + 180, 9);
  });
});

describe("scoreColumnar", () => {
  it("matches scoreLine over a columnar table", () => {
    const columns = ["name", "receptions", "receiving_yards"];
    const rows = [
      ["A", 80, 1000],
      ["B", 20, 250],
    ];
    expect(scoreColumnar(rows, columns, PPR_SCORING)).toEqual([180, 45]);
  });

  it("ignores rules with no matching column", () => {
    expect(scoreColumnar([[5]], ["receptions"], { receptions: 1, nonsense: 99 })).toEqual([5]);
  });
});

describe("scoreBand", () => {
  const bands = [
    { low: 0, high: 0, points: 5 },
    { low: 1, high: 6, points: 4 },
    { low: 7, high: 13, points: 3 },
  ];

  it("picks the band the value falls in, inclusive at both ends", () => {
    expect(scoreBand(0, bands)).toBe(5);
    expect(scoreBand(1, bands)).toBe(4);
    expect(scoreBand(6, bands)).toBe(4);
    expect(scoreBand(7, bands)).toBe(3);
    expect(scoreBand(13, bands)).toBe(3);
  });

  it("scores zero outside every band", () => {
    expect(scoreBand(50, bands)).toBe(0);
  });
});

describe("roundTo", () => {
  it("reproduces polars' banker's rounding on the scaled value", () => {
    // Every one of these was read off polars 1.44 rather than assumed.
    expect(roundTo(14.25, 1)).toBe(14.2); // 142.5 -> 142, even
    expect(roundTo(14.35, 1)).toBe(14.4); // 143.5 -> 144, even
    expect(roundTo(14.45, 1)).toBe(14.4);
    expect(roundTo(-14.25, 1)).toBe(-14.2);
    expect(roundTo(0.125, 1)).toBe(0.1);
    expect(roundTo(0.375, 1)).toBe(0.4);
  });

  it("rounds half to even at zero decimal places too", () => {
    expect(roundTo(0.5, 0)).toBe(0);
    expect(roundTo(1.5, 0)).toBe(2);
    expect(roundTo(2.5, 0)).toBe(2);
    expect(roundTo(-2.5, 0)).toBe(-2);
  });

  it("leaves values that are not ties alone", () => {
    expect(roundTo(123.456, 1)).toBe(123.5);
    expect(roundTo(15.3, 1)).toBe(15.3);
  });
});

describe("ADP anchoring", () => {
  it("classifies by reception value", () => {
    expect(nearestFfcFormat(REFERENCE_LEAGUE)).toBe("ppr");
    expect(
      nearestFfcFormat({ ...REFERENCE_LEAGUE, scoring: { ...PPR_SCORING, receptions: 0.5 } }),
    ).toBe("half-ppr");
    expect(
      nearestFfcFormat({ ...REFERENCE_LEAGUE, scoring: { ...PPR_SCORING, receptions: 0 } }),
    ).toBe("standard");
  });

  it("prefers the 2QB board when two quarterbacks start", () => {
    const superflex: LeagueConfig = {
      ...REFERENCE_LEAGUE,
      starters: { ...REFERENCE_LEAGUE.starters, SUPERFLEX: 1 },
    };
    expect(nearestFfcFormat(superflex)).toBe("2qb");
  });

  it("says nothing when a league matches its anchor", () => {
    expect(anchorWarnings(REFERENCE_LEAGUE)).toEqual([]);
  });

  it("warns when custom rules revalue a position", () => {
    const sixPoint: LeagueConfig = {
      ...REFERENCE_LEAGUE,
      scoring: { ...PPR_SCORING, passing_tds: 6 },
    };
    const warnings = anchorWarnings(sixPoint);
    expect(warnings.length).toBeGreaterThan(0);
    expect(warnings[0]).toMatch(/quarterback/i);
  });
});

describe("league validation", () => {
  it("accepts the reference league", () => {
    expect(() => validateLeague(REFERENCE_LEAGUE)).not.toThrow();
  });

  it("rejects a lineup that could never be drafted", () => {
    expect(() =>
      validateLeague({ ...REFERENCE_LEAGUE, rounds: 3 }),
    ).toThrow(ConfigError);
  });

  it("rejects scoring that is identically zero", () => {
    expect(() =>
      validateLeague({ ...REFERENCE_LEAGUE, scoring: { receptions: 0 } }),
    ).toThrow(/identically zero/);
  });

  it("rejects a roster with no starters", () => {
    expect(() => validateLeague({ ...REFERENCE_LEAGUE, starters: {} })).toThrow(
      /at least one player/,
    );
  });

  it("rejects a position cap below its own starting requirement", () => {
    expect(() =>
      validateLeague({ ...REFERENCE_LEAGUE, positionMax: { ...REFERENCE_LEAGUE.positionMax, RB: 1 } }),
    ).toThrow(/below its starting requirement/);
  });

  it("rejects an impossible team count", () => {
    expect(() => validateLeague({ ...REFERENCE_LEAGUE, teams: 1 })).toThrow(/teams/);
  });
});
