/** Shared shapes. Mirrors `src/config.py` key-for-key so the Python stays the
 *  reference implementation and golden fixtures generate straight from it. */

export type Position = "QB" | "RB" | "WR" | "TE" | "K" | "DST";
export type Slot = Position | "FLEX" | "SUPERFLEX";

export const SKILL_POSITIONS: Position[] = ["QB", "RB", "WR", "TE"];
export const DRAFTABLE_POSITIONS: Position[] = ["QB", "RB", "WR", "TE", "K", "DST"];

/** A scoring band, e.g. 7-13 points allowed is worth 3. Inclusive on both ends. */
export interface Band {
  low: number;
  high: number;
  points: number;
}

export interface DstScoring {
  events: Record<string, number>;
  pointsAllowedBands: Band[];
  yardsAllowedBands: Band[];
}

export interface LeagueSchedule {
  regularSeasonWeeks: number;
  playoffWeeks: number[];
  playoffTeams: number;
}

export interface LeagueConfig {
  teams: number;
  rounds: number;
  bench: number;
  irSlots: number;
  starters: Partial<Record<Slot, number>>;
  flexEligible: Position[];
  positionMax: Partial<Record<Position, number>>;
  schedule: LeagueSchedule;
  /** Points per unit of each stat. Keys match nflverse column names. */
  scoring: Record<string, number>;
  kickerScoring: Record<string, number>;
  dst: DstScoring;
}

export interface Player {
  player_id: string;
  name: string;
  pos: Position;
  tm: string;
  adp: number;
  stdev: number;
  bye: number | null;
  pos_rank: number;
  proj_points: number;
  sd: number;
  games: number;
  source: string;
  /** Calibrated latent draft mean, precomputed in Python. */
  adp_mu: number;

  /**
   * The same three quantities before rounding to one decimal.
   *
   * Rounding is cosmetic, but it lands on an exact tie often enough that a
   * 1e-14 difference in float summation order flips a projection by 0.1. These
   * are what agreement between implementations is actually measured on.
   */
  proj_raw?: number;
  sd_raw?: number;
  games_raw?: number;

  // Attached by the value layer.
  replacement?: number;
  vor?: number;
  vor_rank?: number;
  tier?: number;
  tier_size?: number;
  tier_points?: number;
  playoff_lift?: number;

  // Attached per pick by the live board.
  marginal?: number;
  next_best?: number;
  vona?: number;
  p_survives?: number;
}

/** A fitted positional curve: expected points, spread and games by rank. */
export interface Curve {
  grid: number[];
  mean: number[];
  sd: number[];
  games: number[];
  n: number;
}

/** A roster entry as the lineup optimiser sees it. */
export interface RosterSlot {
  pos: Position;
  proj_points: number;
}
