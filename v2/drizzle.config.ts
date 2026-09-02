import { defineConfig } from "drizzle-kit";

/**
 * Migrations for the Postgres store.
 *
 * The schema had no config and no migrations, so `DATABASE_URL` pointed at an
 * empty database was worse than leaving it unset: the file store at least
 * works, whereas empty Postgres fails every request with `relation "leagues"
 * does not exist`.
 */
export default defineConfig({
  schema: "./db/schema.ts",
  out: "./db/migrations",
  dialect: "postgresql",
  dbCredentials: { url: process.env.DATABASE_URL ?? "" },
});
