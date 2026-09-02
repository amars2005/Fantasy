/**
 * Storage, behind one interface with two implementations.
 *
 * Postgres is what ships. The file-backed store is what makes the app runnable
 * without provisioning a database -- `npm run dev` with no `DATABASE_URL` gets
 * a working draft board against the local bundle, which matters because the
 * thing being replaced was a tool that ran on a laptop with no setup at all.
 *
 * Both honour the same contract, and the interesting half of that contract is
 * `appendPick`: it must be idempotent on `seq`, returning the existing row
 * rather than writing a second one.
 */

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

import type { LeagueConfig, Player } from "./types";

export interface LeagueRecord {
  id: string;
  name: string;
  config: LeagueConfig;
  format: string;
  board: Player[] | null;
  boardHash: string | null;
  adpAsOf: string | null;
  frozenAt: string | null;
  createdAt: string;
}

export interface PickRecord {
  leagueId: string;
  seq: number;
  playerId: string | null;
  takenBy: string;
  voidedAt: string | null;
  createdAt: string;
}

export interface Store {
  createLeague(record: LeagueRecord): Promise<LeagueRecord>;
  getLeague(id: string): Promise<LeagueRecord | null>;
  updateLeague(id: string, patch: Partial<LeagueRecord>): Promise<LeagueRecord | null>;
  listPicks(leagueId: string, sinceSeq?: number): Promise<PickRecord[]>;
  /** Idempotent on (leagueId, seq): a repeat returns the stored row. */
  appendPick(pick: PickRecord): Promise<{ pick: PickRecord; created: boolean }>;
  voidPick(leagueId: string, seq: number): Promise<PickRecord | null>;
}

// --- file-backed (development) ----------------------------------------------

interface FileShape {
  leagues: Record<string, LeagueRecord>;
  picks: Record<string, PickRecord[]>;
}

export class FileStore implements Store {
  constructor(private readonly path: string) {}

  /**
   * Serialises every operation.
   *
   * Each one is a read-modify-write over a single JSON file, so two picks in
   * flight at once will interleave their reads and the second write silently
   * drops the first. That is precisely the last-write-wins data loss the
   * `(league_id, seq)` primary key rules out on Postgres, and it showed up as
   * a pick vanishing across a page reload. Node is single-threaded, so a
   * promise chain is enough of a mutex here.
   */
  private queue: Promise<unknown> = Promise.resolve();

  private serialise<T>(operation: () => Promise<T>): Promise<T> {
    // Chain off the tail regardless of whether it settled, so one failed
    // operation cannot wedge the queue for everything behind it.
    const next = this.queue.then(operation, operation);
    this.queue = next.catch(() => undefined);
    return next;
  }

  private async read(): Promise<FileShape> {
    try {
      return JSON.parse(await readFile(this.path, "utf-8")) as FileShape;
    } catch {
      return { leagues: {}, picks: {} };
    }
  }

  private async write(data: FileShape): Promise<void> {
    try {
      await mkdir(dirname(this.path), { recursive: true });
      // Write-then-rename, so a crash mid-save cannot leave a truncated file
      // that would read back as an empty draft.
      const tmp = `${this.path}.tmp`;
      await writeFile(tmp, JSON.stringify(data, null, 1), "utf-8");
      const { rename } = await import("node:fs/promises");
      await rename(tmp, this.path);
    } catch (err) {
      const code = (err as NodeJS.ErrnoException | null)?.code;
      // A serverless filesystem is read-only outside /tmp, so the store that
      // exists to need no provisioning is the one thing that cannot work
      // there. Unhandled, this surfaced as `ENOENT: mkdir '/var/task/v2/.data'`
      // -- a path the deployer never chose, about a database they never knew
      // they needed.
      if (code === "EROFS" || code === "EACCES" || code === "ENOENT") {
        throw new Error(
          `Cannot write ${this.path}: the filesystem is read-only. Set ` +
            `DATABASE_URL to store leagues in Postgres, which is what a draft ` +
            `running over days needs, or FANTASY_DATA_FILE to a writable path ` +
            `such as /tmp/leagues.json -- ephemeral, so leagues there are lost ` +
            `on every cold start.`,
        );
      }
      throw err;
    }
  }

  async createLeague(record: LeagueRecord): Promise<LeagueRecord> {
    return this.serialise(async () => {
      const data = await this.read();
      data.leagues[record.id] = record;
      data.picks[record.id] ??= [];
      await this.write(data);
      return record;
    });
  }

  async getLeague(id: string): Promise<LeagueRecord | null> {
    return this.serialise(async () => (await this.read()).leagues[id] ?? null);
  }

  async updateLeague(id: string, patch: Partial<LeagueRecord>): Promise<LeagueRecord | null> {
    return this.serialise(async () => {
      const data = await this.read();
      const existing = data.leagues[id];
      if (!existing) return null;
      const merged = { ...existing, ...patch };
      data.leagues[id] = merged;
      await this.write(data);
      return merged;
    });
  }

  async listPicks(leagueId: string, sinceSeq = 0): Promise<PickRecord[]> {
    return this.serialise(async () => {
      const data = await this.read();
      return (data.picks[leagueId] ?? []).filter((p) => p.seq > sinceSeq);
    });
  }

  async appendPick(pick: PickRecord): Promise<{ pick: PickRecord; created: boolean }> {
    return this.serialise(async () => {
      const data = await this.read();
      const list = (data.picks[pick.leagueId] ??= []);
      const existing = list.find((p) => p.seq === pick.seq);
      // The seq is the idempotency key: a retried request returns what is
      // already stored rather than writing a second row.
      if (existing) return { pick: existing, created: false };
      list.push(pick);
      list.sort((a, b) => a.seq - b.seq);
      await this.write(data);
      return { pick, created: true };
    });
  }

  async voidPick(leagueId: string, seq: number): Promise<PickRecord | null> {
    return this.serialise(async () => {
      const data = await this.read();
      const row = (data.picks[leagueId] ?? []).find((p) => p.seq === seq);
      if (!row) return null;
      row.voidedAt = new Date().toISOString();
      await this.write(data);
      return row;
    });
  }
}

// --- Postgres ---------------------------------------------------------------

/**
 * Lazily constructed so the file store path never imports the driver, and so a
 * missing `DATABASE_URL` is a fallback rather than a crash at import time.
 */
export class PostgresStore implements Store {
  constructor(private readonly url: string) {}

  private async db() {
    const { drizzle } = await import("drizzle-orm/neon-http");
    const { neon } = await import("@neondatabase/serverless");
    const schema = await import("../db/schema");
    return { db: drizzle(neon(this.url)), schema };
  }

  async createLeague(record: LeagueRecord): Promise<LeagueRecord> {
    const { db, schema } = await this.db();
    await db.insert(schema.leagues).values({
      id: record.id,
      name: record.name,
      config: record.config,
      format: record.format,
      board: record.board,
      boardHash: record.boardHash,
      adpAsOf: record.adpAsOf,
      frozenAt: record.frozenAt ? new Date(record.frozenAt) : null,
    });
    return record;
  }

  async getLeague(id: string): Promise<LeagueRecord | null> {
    const { db, schema } = await this.db();
    const { eq } = await import("drizzle-orm");
    const rows = await db.select().from(schema.leagues).where(eq(schema.leagues.id, id));
    if (!rows.length) return null;
    const r = rows[0];
    return {
      id: r.id,
      name: r.name,
      config: r.config,
      format: r.format,
      board: r.board,
      boardHash: r.boardHash,
      adpAsOf: r.adpAsOf,
      frozenAt: r.frozenAt ? r.frozenAt.toISOString() : null,
      createdAt: r.createdAt.toISOString(),
    };
  }

  async updateLeague(id: string, patch: Partial<LeagueRecord>): Promise<LeagueRecord | null> {
    const { db, schema } = await this.db();
    const { eq } = await import("drizzle-orm");
    const values: Record<string, unknown> = {};
    if (patch.name !== undefined) values.name = patch.name;
    if (patch.config !== undefined) values.config = patch.config;
    if (patch.format !== undefined) values.format = patch.format;
    if (patch.board !== undefined) values.board = patch.board;
    if (patch.boardHash !== undefined) values.boardHash = patch.boardHash;
    if (patch.adpAsOf !== undefined) values.adpAsOf = patch.adpAsOf;
    if (patch.frozenAt !== undefined) {
      values.frozenAt = patch.frozenAt ? new Date(patch.frozenAt) : null;
    }
    if (Object.keys(values).length) {
      await db.update(schema.leagues).set(values).where(eq(schema.leagues.id, id));
    }
    return this.getLeague(id);
  }

  async listPicks(leagueId: string, sinceSeq = 0): Promise<PickRecord[]> {
    const { db, schema } = await this.db();
    const { and, eq, gt } = await import("drizzle-orm");
    const rows = await db
      .select()
      .from(schema.picks)
      .where(and(eq(schema.picks.leagueId, leagueId), gt(schema.picks.seq, sinceSeq)))
      .orderBy(schema.picks.seq);
    return rows.map((r) => ({
      leagueId: r.leagueId,
      seq: r.seq,
      playerId: r.playerId,
      takenBy: r.takenBy,
      voidedAt: r.voidedAt ? r.voidedAt.toISOString() : null,
      createdAt: r.createdAt.toISOString(),
    }));
  }

  async appendPick(pick: PickRecord): Promise<{ pick: PickRecord; created: boolean }> {
    const { db, schema } = await this.db();
    // ON CONFLICT DO NOTHING is the idempotency: a retried POST that already
    // landed writes nothing and we hand back what is stored.
    const inserted = await db
      .insert(schema.picks)
      .values({
        leagueId: pick.leagueId,
        seq: pick.seq,
        playerId: pick.playerId,
        takenBy: pick.takenBy,
      })
      .onConflictDoNothing()
      .returning();

    if (inserted.length) return { pick, created: true };

    const existing = (await this.listPicks(pick.leagueId, pick.seq - 1)).find(
      (p) => p.seq === pick.seq,
    );
    return { pick: existing ?? pick, created: false };
  }

  async voidPick(leagueId: string, seq: number): Promise<PickRecord | null> {
    const { db, schema } = await this.db();
    const { and, eq } = await import("drizzle-orm");
    const rows = await db
      .update(schema.picks)
      .set({ voidedAt: new Date() })
      .where(and(eq(schema.picks.leagueId, leagueId), eq(schema.picks.seq, seq)))
      .returning();
    if (!rows.length) return null;
    const r = rows[0];
    return {
      leagueId: r.leagueId,
      seq: r.seq,
      playerId: r.playerId,
      takenBy: r.takenBy,
      voidedAt: r.voidedAt ? r.voidedAt.toISOString() : null,
      createdAt: r.createdAt.toISOString(),
    };
  }
}

let cached: Store | null = null;

/** Where the file store keeps its data. Overridable so tests get their own. */
export function fileStorePath(): string {
  return process.env.FANTASY_DATA_FILE ?? join(process.cwd(), ".data", "leagues.json");
}

/** Postgres when `DATABASE_URL` is set, a local file otherwise. */
export function getStore(): Store {
  if (cached) return cached;
  const url = process.env.DATABASE_URL;
  cached = url ? new PostgresStore(url) : new FileStore(fileStorePath());
  return cached;
}

/** Test seam. */
export function setStore(store: Store | null): void {
  cached = store;
}
