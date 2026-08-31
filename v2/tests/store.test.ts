/**
 * Storage behaviour, with the idempotency property front and centre.
 *
 * A doubled pick is not a cosmetic bug: it advances the clock one past the
 * room, which puts "picks until my next turn" -- and therefore every VONA
 * number -- permanently out of step for the rest of the draft.
 */

import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { REFERENCE_LEAGUE } from "../lib/config";
import { FileStore, type LeagueRecord, type PickRecord } from "../lib/store";

let dir: string;
let store: FileStore;

const league = (id = "test-league"): LeagueRecord => ({
  id,
  name: "Test",
  config: REFERENCE_LEAGUE,
  format: "ppr",
  board: [],
  boardHash: "hash",
  adpAsOf: "2026-08-30",
  frozenAt: null,
  createdAt: new Date().toISOString(),
});

const pick = (seq: number, playerId: string | null = "p1"): PickRecord => ({
  leagueId: "test-league",
  seq,
  playerId,
  takenBy: "other",
  voidedAt: null,
  createdAt: new Date().toISOString(),
});

beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), "fantasy-store-"));
  store = new FileStore(join(dir, "leagues.json"));
});

afterEach(async () => {
  await rm(dir, { recursive: true, force: true });
});

describe("leagues", () => {
  it("round-trips a league", async () => {
    await store.createLeague(league());
    const got = await store.getLeague("test-league");
    expect(got?.name).toBe("Test");
    expect(got?.config.teams).toBe(REFERENCE_LEAGUE.teams);
  });

  it("returns null for a league that does not exist", async () => {
    expect(await store.getLeague("nope")).toBeNull();
  });

  it("patches without clobbering untouched fields", async () => {
    await store.createLeague(league());
    const updated = await store.updateLeague("test-league", { frozenAt: "2026-08-30T12:00:00Z" });
    expect(updated?.frozenAt).toBe("2026-08-30T12:00:00Z");
    expect(updated?.name).toBe("Test");
    expect(updated?.format).toBe("ppr");
  });
});

describe("picks", () => {
  beforeEach(async () => {
    await store.createLeague(league());
  });

  it("appends in sequence", async () => {
    await store.appendPick(pick(1, "a"));
    await store.appendPick(pick(2, "b"));
    const picks = await store.listPicks("test-league");
    expect(picks.map((p) => p.playerId)).toEqual(["a", "b"]);
  });

  it("is idempotent on seq", async () => {
    // The retry case: a POST that timed out on bad wifi but actually landed.
    const first = await store.appendPick(pick(1, "a"));
    const second = await store.appendPick(pick(1, "a"));

    expect(first.created).toBe(true);
    expect(second.created).toBe(false);
    expect((await store.listPicks("test-league")).length).toBe(1);
  });

  it("does not let a conflicting retry overwrite the stored pick", async () => {
    await store.appendPick(pick(1, "a"));
    const clash = await store.appendPick({ ...pick(1, "b"), takenBy: "me" });

    expect(clash.created).toBe(false);
    expect(clash.pick.playerId).toBe("a");
    const stored = await store.listPicks("test-league");
    expect(stored.length).toBe(1);
    expect(stored[0].playerId).toBe("a");
  });

  it("lists only picks after a cursor", async () => {
    for (let i = 1; i <= 5; i++) await store.appendPick(pick(i, `p${i}`));
    const since = await store.listPicks("test-league", 3);
    expect(since.map((p) => p.seq)).toEqual([4, 5]);
  });

  it("soft-deletes, keeping the audit trail and later sequence numbers", async () => {
    for (let i = 1; i <= 3; i++) await store.appendPick(pick(i, `p${i}`));
    await store.voidPick("test-league", 2);

    const all = await store.listPicks("test-league");
    expect(all.length).toBe(3);
    expect(all.find((p) => p.seq === 2)?.voidedAt).not.toBeNull();
    // Voiding the middle pick must not renumber the ones after it.
    expect(all.map((p) => p.seq)).toEqual([1, 2, 3]);

    const live = all.filter((p) => p.voidedAt === null);
    expect(live.map((p) => p.playerId)).toEqual(["p1", "p3"]);
  });

  it("records a pick with no player, so the clock still moves", async () => {
    await store.appendPick(pick(1, null));
    const picks = await store.listPicks("test-league");
    expect(picks[0].playerId).toBeNull();
  });
});
