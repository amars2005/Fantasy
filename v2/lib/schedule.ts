/**
 * Schedule strength, with the playoff weeks weighted separately.
 *
 * Port of `src/features/schedule.py`.
 *
 * Season-long strength of schedule is close to useless for fantasy: over
 * seventeen games it mostly averages out, and the market has already priced it.
 * The weeks that are *not* interchangeable are the ones the title is decided
 * in. A player facing soft defences in exactly those weeks is worth more than
 * his season projection says, and ADP cannot price that because ADP is
 * format-blind -- the leagues it is sampled from have different playoff
 * windows.
 *
 * Which is also why this is computed per league rather than baked into the
 * bundle: a league whose bracket runs weeks 14-16 needs a different number from
 * one whose bracket runs 15-17.
 *
 * Method note: the obvious measure would be the Vegas implied total for those
 * weeks, but sportsbooks only post lines a few weeks out, so the playoff weeks
 * are empty in August when the draft happens. Opponents, however, are known for
 * every game from the day the schedule is released. So difficulty is measured
 * by how many points those specific opponents allowed last season.
 */

import { columnIndex, type ColumnarTable } from "./bundle";
import type { LeagueConfig, Player } from "./types";

export interface TeamStrength {
  team: string;
  sosRegular: number;
  sosPlayoff: number;
  /** Positive means the weeks that decide the title are softer than the rest. */
  playoffLift: number;
}

/**
 * Matchup difficulty for each team, split by season phase.
 *
 * `schedule` carries (week, team, opponent) for the coming season; its
 * `strength` sub-table carries each defence's points allowed per game last
 * season, where higher means softer.
 */
export function teamScheduleStrength(
  schedule: ColumnarTable,
  league: LeagueConfig,
): Map<string, TeamStrength> {
  const strengthTable = schedule.strength as ColumnarTable | undefined;
  if (!strengthTable) throw new Error("schedule bundle has no strength table");

  const sIdx = columnIndex(strengthTable);
  const paPerGame = new Map<string, number>();
  for (const row of strengthTable.rows) {
    paPerGame.set(String(row[sIdx.opponent]), Number(row[sIdx.pa_per_game] ?? 0));
  }

  const playoffWeeks = new Set(league.schedule.playoffWeeks);
  const regularWeeks = new Set<number>();
  for (let w = 1; w <= league.schedule.regularSeasonWeeks; w++) regularWeeks.add(w);

  const idx = columnIndex(schedule);
  const acc = new Map<string, { reg: number[]; post: number[] }>();

  for (const row of schedule.rows) {
    const team = String(row[idx.team]);
    const week = Number(row[idx.week]);
    const pa = paPerGame.get(String(row[idx.opponent]));
    // A team whose opponent has no prior-season record contributes nothing,
    // matching the Python's drop_nulls on the join.
    if (pa === undefined) continue;

    const entry = acc.get(team) ?? { reg: [], post: [] };
    if (regularWeeks.has(week)) entry.reg.push(pa);
    if (playoffWeeks.has(week)) entry.post.push(pa);
    acc.set(team, entry);
  }

  const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);

  const out = new Map<string, TeamStrength>();
  for (const [team, { reg, post }] of acc) {
    const sosRegular = mean(reg);
    const sosPlayoff = mean(post);
    out.set(team, {
      team,
      sosRegular,
      sosPlayoff,
      playoffLift: post.length && reg.length ? sosPlayoff - sosRegular : 0,
    });
  }
  return out;
}

/** Attach each player's team playoff-schedule lift to the board. */
export function addPlayoffLift(
  players: Player[],
  strength: Map<string, TeamStrength>,
): Player[] {
  return players.map((p) => ({
    ...p,
    playoff_lift: strength.get(p.tm)?.playoffLift ?? 0,
  }));
}
