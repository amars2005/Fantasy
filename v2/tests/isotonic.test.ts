/**
 * The isotonic fit, against Python's.
 *
 * Deterministic maths, so the tolerance is 1e-9 and not a hair looser. If these
 * drift, every projection drifts with them.
 */

import { describe, expect, it } from "vitest";

import { fitIsotonic, predictIsotonic } from "../lib/isotonic";
import { fitCurves } from "../lib/project";
import { FIXTURE_NAMES, loadFixture, loadTable } from "./golden";

const TOL = 1e-9;

describe("pool-adjacent-violators", () => {
  it("leaves an already-monotone sequence untouched", () => {
    const x = [1, 2, 3, 4, 5];
    const y = [10, 8, 6, 4, 2];
    expect(predictIsotonic(fitIsotonic(x, y, false), x)).toEqual(y);
  });

  it("pools a violating pair into their mean", () => {
    const fit = fitIsotonic([1, 2, 3], [10, 2, 6], false);
    const got = predictIsotonic(fit, [1, 2, 3]);
    expect(got[0]).toBeCloseTo(10, 12);
    // 2 and 6 violate the decreasing constraint and pool to 4.
    expect(got[1]).toBeCloseTo(4, 12);
    expect(got[2]).toBeCloseTo(4, 12);
  });

  it("collapses duplicate x into a weighted mean before fitting", () => {
    // Rank 1 observed three times, averaging 12; rank 2 once at 5.
    const fit = fitIsotonic([1, 1, 1, 2], [9, 12, 15, 5], false);
    expect(predictIsotonic(fit, [1])[0]).toBeCloseTo(12, 12);
    expect(predictIsotonic(fit, [2])[0]).toBeCloseTo(5, 12);
  });

  it("clips outside the observed range rather than extrapolating", () => {
    const fit = fitIsotonic([2, 3, 4], [10, 8, 6], false);
    expect(predictIsotonic(fit, [0, 1])).toEqual([10, 10]);
    expect(predictIsotonic(fit, [5, 99])).toEqual([6, 6]);
  });

  it("assigns a bit-identical value across a pooled block", () => {
    // This is the property tiering depends on. Not "close to" -- identical.
    //
    // [10, 5, 5.5, 4.5, 5, 1] pools into two blocks, not one: 5/5.5 average to
    // 5.25 and 4.5/5 average to 4.75. Asserting the real structure is the
    // point -- distinct blocks must stay distinct, and each must be internally
    // exact, because tiering reads both facts off this array.
    const x = [1, 2, 3, 4, 5, 6];
    const y = [10, 5, 5.5, 4.5, 5, 1];
    const fitted = predictIsotonic(fitIsotonic(x, y, false), x);

    expect(fitted[0]).toBeCloseTo(10, 12);
    expect(fitted[1]).toBeCloseTo(5.25, 12);
    expect(fitted[3]).toBeCloseTo(4.75, 12);
    expect(fitted[5]).toBeCloseTo(1, 12);

    // Within a block: identical, bit for bit.
    expect(Object.is(fitted[1], fitted[2])).toBe(true);
    expect(Object.is(fitted[3], fitted[4])).toBe(true);
    // Between blocks: genuinely different, so the tier boundary survives.
    expect(fitted[2]).not.toBe(fitted[3]);
    expect(new Set(fitted).size).toBe(4);
  });

  it("rejects mismatched inputs rather than fitting nonsense", () => {
    expect(() => fitIsotonic([1, 2], [1], false)).toThrow(/length mismatch/);
    expect(() => fitIsotonic([], [], false)).toThrow(/no observations/);
  });
});

describe.each(FIXTURE_NAMES)("curves match Python: %s", (name) => {
  const fixture = loadFixture(name);
  const training = loadTable(`training_${fixture.format}`);
  const curves = fitCurves(training, fixture.league.scoring);

  it("fits the same positions", () => {
    expect(Object.keys(curves).sort()).toEqual(Object.keys(fixture.curves).sort());
  });

  for (const pos of Object.keys(fixture.curves)) {
    describe(pos, () => {
      const expected = fixture.curves[pos];

      it("uses the same observation count and grid", () => {
        expect(curves[pos].n).toBe(expected.n);
        expect(curves[pos].grid).toEqual(expected.grid);
      });

      it("matches expected points", () => {
        curves[pos].mean.forEach((v, i) => {
          expect(Math.abs(v - expected.mean[i])).toBeLessThan(TOL);
        });
      });

      it("matches outcome spread", () => {
        curves[pos].sd.forEach((v, i) => {
          expect(Math.abs(v - expected.sd[i])).toBeLessThan(TOL);
        });
      });

      it("matches games played", () => {
        curves[pos].games.forEach((v, i) => {
          expect(Math.abs(v - expected.games[i])).toBeLessThan(TOL);
        });
      });

      it("keeps plateaus exactly flat", () => {
        // A plateau in Python must be a plateau here, bit for bit -- otherwise
        // tiering shatters. Compare the *structure*, not just the values.
        const boundaries = (xs: number[]) =>
          xs.map((v, i) => (i === 0 ? true : v !== xs[i - 1]));
        expect(boundaries(curves[pos].mean)).toEqual(boundaries(expected.mean));
      });
    });
  }
});
