/**
 * Durable draft state.
 *
 * Postgres rather than a cache, deliberately. The reference league allows three
 * hours per pick, so a draft runs over days; a Redis free tier that evicts keys
 * would lose a draft in progress, which is the one failure this system cannot
 * have.
 */

import {
  date,
  index,
  integer,
  jsonb,
  pgTable,
  primaryKey,
  text,
  timestamp,
} from "drizzle-orm/pg-core";

import type { LeagueConfig, Player } from "../lib/types";

export const leagues = pgTable("leagues", {
  /** nanoid. Unguessable, and the only capability needed to read or write. */
  id: text("id").primaryKey(),
  name: text("name").notNull().default("My league"),
  config: jsonb("config").$type<LeagueConfig>().notNull(),
  /** Which Fantasy Football Calculator board the ADP was anchored to. */
  format: text("format").notNull(),
  /** Derived board. Regenerable, so this column is a cache, not a source. */
  board: jsonb("board").$type<Player[]>(),
  /** hash(config, adpAsOf): a mismatch means the cache is stale. */
  boardHash: text("board_hash"),
  adpAsOf: date("adp_as_of"),
  /**
   * Set on the first pick. While set, a boardHash mismatch never triggers a
   * recompute -- it only offers the refresh. Silently re-anchoring a draft
   * already in progress would change every projection, tier and survival
   * number underneath it.
   */
  frozenAt: timestamp("frozen_at", { withTimezone: true }),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
});

export const picks = pgTable(
  "picks",
  {
    leagueId: text("league_id")
      .notNull()
      .references(() => leagues.id, { onDelete: "cascade" }),
    /**
     * Insertion sequence, NOT draft position.
     *
     * Semantic draft position is row_number() over non-voided rows, so voiding
     * pick 12 does not corrupt picks 13 onward. Its real job is the primary
     * key below.
     */
    seq: integer("seq").notNull(),
    /** Null means a player who was not on our board -- the clock still moves. */
    playerId: text("player_id"),
    takenBy: text("taken_by").notNull().default("other"),
    /** Soft delete, so an undo keeps the audit trail. */
    voidedAt: timestamp("voided_at", { withTimezone: true }),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (t) => ({
    // The concurrency control. A POST that times out on bad venue wifi but
    // succeeded server-side, then gets retried, would otherwise double-record
    // the pick and advance the clock by one -- which corrupts every VONA
    // number downstream. This makes the retry a no-op instead.
    pk: primaryKey({ columns: [t.leagueId, t.seq] }),
    byLeague: index("picks_by_league").on(t.leagueId, t.seq),
  }),
);

export type LeagueRow = typeof leagues.$inferSelect;
export type PickRow = typeof picks.$inferSelect;
