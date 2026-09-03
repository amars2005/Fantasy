/**
 * League creation and board derivation -- the Tier 2 half of the system.
 *
 * Expensive relative to a request (a few hundred milliseconds of isotonic
 * fitting), cheap relative to a draft: it runs once per league and the result
 * is cached on the league row. Everything after this happens in the browser.
 */

import { createHash } from "node:crypto";
import { nanoid } from "nanoid";

import { deriveBoard, toLeaguePickSpace } from "./board";
import { adpAsOf, loadBundleFor } from "./bundleLoader";
import { anchorWarnings, nearestFfcFormat, validateLeague } from "./config";
import { getStore, type LeagueRecord, type PickRecord } from "./store";
import type { LeagueConfig, Player } from "./types";

/**
 * Identifies the inputs a derived board depends on.
 *
 * A change to either the league's rules or the ADP snapshot invalidates it --
 * though see `frozenAt`: a draft in progress deliberately keeps its stale
 * board rather than being re-anchored underneath.
 */
export function boardHash(config: LeagueConfig, adp: string | null): string {
  return createHash("sha256")
    .update(JSON.stringify(config))
    .update("|")
    .update(adp ?? "")
    .digest("hex")
    .slice(0, 32);
}

export interface CreateLeagueInput {
  name?: string;
  config: LeagueConfig;
}

export async function createLeague(input: CreateLeagueInput): Promise<LeagueRecord> {
  validateLeague(input.config);

  const format = nearestFfcFormat(input.config);
  const bundle = await loadBundleFor(format);
  const board = deriveBoard(bundle, input.config);
  const asOf = await adpAsOf(format);

  const record: LeagueRecord = {
    id: nanoid(12),
    name: input.name?.trim() || "My league",
    config: input.config,
    format,
    board,
    boardHash: boardHash(input.config, asOf),
    adpAsOf: asOf,
    frozenAt: null,
    createdAt: new Date().toISOString(),
  };

  return getStore().createLeague(record);
}

export interface LeagueView {
  league: LeagueRecord;
  picks: PickRecord[];
  warnings: string[];
  /** True when newer ADP exists but the board is frozen mid-draft. */
  refreshAvailable: boolean;
}

export async function getLeagueView(id: string): Promise<LeagueView | null> {
  const store = getStore();
  const league = await store.getLeague(id);
  if (!league) return null;

  const picks = await store.listPicks(id);
  const asOf = await adpAsOf(league.format);
  const currentHash = boardHash(league.config, asOf);

  // A stale board on a frozen league is correct, not a bug: re-deriving mid
  // draft would move every projection, tier and survival number underneath a
  // draft already in progress.
  const stale = currentHash !== league.boardHash;

  return {
    // The stored board is in the bundle's own 12-team pick space, which is what
    // the bundle means and what the golden fixtures pin. Converting on the way
    // out rather than at write time means every league gets the correction at
    // once, with no stored board to migrate and no board moving underneath a
    // draft that is already running.
    league: {
      ...league,
      board: league.board
        ? toLeaguePickSpace(league.board, league.config.teams)
        : null,
    },
    picks,
    warnings: anchorWarnings(league.config),
    refreshAvailable: stale && league.frozenAt !== null,
  };
}

/**
 * Re-derive a league's board against the current bundle.
 *
 * Only ever on explicit request. Clears the freeze, because the user has
 * decided the newer ADP is worth the discontinuity.
 */
export async function refreshBoard(id: string): Promise<LeagueRecord | null> {
  const store = getStore();
  const league = await store.getLeague(id);
  if (!league) return null;

  const format = nearestFfcFormat(league.config);
  const bundle = await loadBundleFor(format);
  const board: Player[] = deriveBoard(bundle, league.config);
  const asOf = await adpAsOf(format);

  return store.updateLeague(id, {
    format,
    board,
    boardHash: boardHash(league.config, asOf),
    adpAsOf: asOf,
    frozenAt: null,
  });
}

/**
 * Record a pick.
 *
 * `seq` comes from the client, which knows how many picks it has. That is what
 * makes the write idempotent: a retry after a timed-out request carries the
 * same `seq` and lands on the primary key instead of advancing the clock twice.
 */
export async function recordPick(
  leagueId: string,
  seq: number,
  playerId: string | null,
  takenBy: string,
): Promise<{ pick: PickRecord; created: boolean } | null> {
  const store = getStore();
  const league = await store.getLeague(leagueId);
  if (!league) return null;

  // First pick freezes the board.
  if (!league.frozenAt) {
    await store.updateLeague(leagueId, { frozenAt: new Date().toISOString() });
  }

  return store.appendPick({
    leagueId,
    seq,
    playerId,
    takenBy,
    voidedAt: null,
    createdAt: new Date().toISOString(),
  });
}

export async function undoPick(leagueId: string, seq: number): Promise<PickRecord | null> {
  return getStore().voidPick(leagueId, seq);
}

/** Live picks, in draft order, with voided rows dropped. */
export function activePicks(picks: PickRecord[]): PickRecord[] {
  return picks.filter((p) => p.voidedAt === null).sort((a, b) => a.seq - b.seq);
}
