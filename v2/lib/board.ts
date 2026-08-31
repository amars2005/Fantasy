/**
 * Deriving a league's board, and driving it during a draft.
 *
 * Port of `src/draft/board.py`, split in two because the hosting model splits
 * it in two:
 *
 *   `deriveBoard` is the expensive half -- fit the curves under this league's
 *   scoring, project, value, tier. It runs once per league on the server and
 *   its result is cached.
 *
 *   `DraftBoard` is the live half. It holds who is gone and what you own, and
 *   answers the only question that matters when the clock is running: who
 *   should I take right now, and why. It runs in the browser on every
 *   keystroke.
 */

import { columnIndex, roundTo, type ColumnarTable } from "./bundle";
import { fitCurves, projectPlayers, type BoardEntry } from "./project";
import { projectKdst } from "./kdst";
import { addPlayoffLift, teamScheduleStrength } from "./schedule";
import { addTiers } from "./tiers";
import { addVor, replacementLevels } from "./replacement";
import { snakePicks } from "./simDraft";
import { positionDropoff, vonaTable, type Dropoff } from "./vona";
import { mulberry32 } from "./rng";
import type { LeagueConfig, Player, Position, RosterSlot } from "./types";

/** Everything `deriveBoard` needs from the static bundle. */
export interface Bundle {
  board: ColumnarTable;
  training: ColumnarTable;
  kickerComponents: ColumnarTable;
  dstComponents: ColumnarTable;
  schedule: ColumnarTable;
  teamNicknames: Record<string, string>;
}

function toBoardEntries(table: ColumnarTable): BoardEntry[] {
  const i = columnIndex(table);
  return table.rows.map((r) => ({
    player_id: String(r[i.player_id]),
    name: String(r[i.name]),
    pos: String(r[i.pos]) as Position,
    tm: String(r[i.tm] ?? ""),
    adp: Number(r[i.adp]),
    stdev: Number(r[i.stdev] ?? 0),
    bye: r[i.bye] === null ? null : Number(r[i.bye]),
    pos_rank: Number(r[i.pos_rank]),
    adp_mu: Number(r[i.adp_mu]),
  }));
}

/**
 * Build a league's board: projections, value, tiers.
 *
 * This is the only place league scoring enters. Everything downstream reads
 * points that were computed here.
 */
export function deriveBoard(bundle: Bundle, league: LeagueConfig): Player[] {
  const entries = toBoardEntries(bundle.board);

  const curves = fitCurves(bundle.training, league.scoring);
  const skill = projectPlayers(curves, entries);

  // Kickers and defences must be on the board even though they are barely
  // worth projecting: a pick that cannot be marked desynchronises the pick
  // counter, and every "picks until my next turn" number drifts with it.
  const kdst = projectKdst(
    entries.filter((e) => e.pos === "K" || e.pos === "DST"),
    bundle.kickerComponents,
    bundle.dstComponents,
    league.kickerScoring,
    league.dst,
  );

  let players = addTiers(addVor([...skill, ...kdst], league));

  // Weeks that decide the title are not interchangeable, and ADP is
  // format-blind so it cannot price them.
  try {
    players = addPlayoffLift(players, teamScheduleStrength(bundle.schedule, league));
  } catch {
    players = players.map((p) => ({ ...p, playoff_lift: 0 }));
  }

  return players.sort((a, b) => a.adp - b.adp);
}

/** One searchable haystack per player: name, team, and -- for defences -- nickname. */
export function searchKey(p: Player, nicknames: Record<string, string>): string {
  // Nicknames stay off the skill players deliberately. Nobody types "ravens"
  // to reach a quarterback, and attaching it to all nine players on a team
  // would bury the defence the word was typed to find.
  const nick = p.pos === "DST" ? (nicknames[p.tm] ?? "") : "";
  return `${p.name} ${p.tm ?? ""} ${nick}`.toLowerCase();
}

export interface Recommendation {
  on_the_clock: number;
  my_next_pick: number | null;
  picks_until_next: number;
  roster: RosterSlot[];
  replacement: Record<string, number>;
  dropoff: Dropoff[];
  recommendations: Player[];
}

/** Marker prefix for picks we could not identify. */
export const UNKNOWN_PREFIX = "__unknown_";

export class DraftBoard {
  readonly players: Player[];
  readonly league: LeagueConfig;
  readonly slot: number;
  readonly picks: number[];
  /** player_id -> "me" | "other", in pick order. */
  drafted: Map<string, string>;

  private byId: Map<string, Player>;
  private keys: Map<string, string>;

  constructor(
    players: Player[],
    league: LeagueConfig,
    slot: number,
    nicknames: Record<string, string> = {},
  ) {
    this.players = players;
    this.league = league;
    this.slot = slot;
    this.picks = snakePicks(slot, league.teams, league.rounds);
    this.drafted = new Map();
    this.byId = new Map(players.map((p) => [p.player_id, p]));
    this.keys = new Map(players.map((p) => [p.player_id, searchKey(p, nicknames)]));
  }

  /** Overall pick currently on the clock (1-based). */
  get pickNumber(): number {
    return this.drafted.size + 1;
  }

  get available(): Player[] {
    return this.players.filter((p) => !this.drafted.has(p.player_id));
  }

  get myRoster(): RosterSlot[] {
    const out: RosterSlot[] = [];
    for (const [id, who] of this.drafted) {
      if (who !== "me") continue;
      const p = this.byId.get(id);
      if (p) out.push({ pos: p.pos, proj_points: p.proj_points });
    }
    return out;
  }

  /** My next pick, counting the one on the clock if it is mine. */
  nextPick(): number | null {
    return this.picks.find((p) => p >= this.pickNumber) ?? null;
  }

  /** How many players come off the board before I choose again. */
  picksUntilNext(): number {
    const current = this.nextPick();
    if (current === null) return 0;
    const following = this.picks.find((p) => p > current);
    if (following === undefined) return this.available.length;
    return following - current;
  }

  draft(playerId: string, by: "me" | "other" = "other"): void {
    this.drafted.set(playerId, by);
  }

  undo(playerId: string): void {
    this.drafted.delete(playerId);
  }

  /**
   * Advance the clock for picks we cannot identify.
   *
   * A safety net for when someone drafts a player who is not on the board at
   * all. Without it the counter silently falls behind the room, and every
   * VONA number drifts with it.
   */
  skip(count = 1): void {
    for (let i = 0; i < count; i++) {
      this.drafted.set(`${UNKNOWN_PREFIX}${this.drafted.size}_${i}`, "other");
    }
  }

  undoSkip(): void {
    const unknown = [...this.drafted.keys()].filter((k) => k.startsWith(UNKNOWN_PREFIX));
    if (unknown.length) this.drafted.delete(unknown[unknown.length - 1]);
  }

  find(query: string, limit = 8): Player[] {
    const q = query.trim().toLowerCase();
    if (!q) return [];
    // Substring, not regex: a stray "(" typed mid-draft must match nothing
    // rather than throw.
    return this.available
      .filter((p) => (this.keys.get(p.player_id) ?? "").includes(q))
      .sort((a, b) => a.adp - b.adp)
      .slice(0, limit);
  }

  recommend(n = 12, nSims = 3000): Recommendation {
    const board = this.available;
    const roster = this.myRoster;
    const gap = Math.max(this.picksUntilNext(), 1);

    const table = vonaTable(board, roster, gap, this.league, nSims, mulberry32(7));
    const dropoff = positionDropoff(table);

    // Tier scarcity: how many players remain in each player's own tier.
    const tierLeft = new Map<string, number>();
    for (const p of board) {
      const key = `${p.pos}|${p.tier}`;
      tierLeft.set(key, (tierLeft.get(key) ?? 0) + 1);
    }

    // Bye collisions: how many players already on our roster share this
    // candidate's bye. Stacking starters on one week is worth seeing, not
    // worth reaching for.
    const myByes = new Map<number, number>();
    for (const [id, who] of this.drafted) {
      if (who !== "me") continue;
      const bye = this.byId.get(id)?.bye;
      if (bye != null) myByes.set(bye, (myByes.get(bye) ?? 0) + 1);
    }

    const recommendations = table.slice(0, n).map((p) => ({
      ...p,
      tier_left: tierLeft.get(`${p.pos}|${p.tier}`) ?? 0,
      bye_conflicts: p.bye != null ? (myByes.get(p.bye) ?? 0) : 0,
    }));

    const replacement: Record<string, number> = {};
    for (const [pos, level] of Object.entries(replacementLevels(board, this.league))) {
      replacement[pos] = roundTo(level, 1);
    }

    return {
      on_the_clock: this.pickNumber,
      my_next_pick: this.nextPick(),
      picks_until_next: gap,
      roster,
      replacement,
      dropoff,
      recommendations,
    };
  }
}
