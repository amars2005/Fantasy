/**
 * Server-side loading of the static bundle.
 *
 * Three places it can come from, tried in this order:
 *
 * 1. `BUNDLE_BASE_URL` -- the published bundle, refreshed daily by the GitHub
 *    Action that runs `scripts/export_v2_bundle.py`. This is the production
 *    path. The variable may point either at the Blob store root, in which case
 *    the dated prefix is resolved through the `bundle/latest.json` pointer that
 *    `scripts/publish-bundle.mjs` moves only once every file has landed, or
 *    straight at a prefix that already holds the files.
 * 2. `BUNDLE_DIR`, or `data/v2_export` beside the app -- a bundle deployed
 *    with the app rather than fetched. `next.config.mjs` traces those files
 *    into the serverless function.
 * 3. `../data/v2_export` -- the repo checkout in development, so the app runs
 *    with nothing provisioned.
 *
 * Tables are cached in module memory. They are immutable for the life of a
 * deployment, and re-parsing a megabyte of JSON per league creation would be
 * pure waste.
 */

import { readFile } from "node:fs/promises";
import { isAbsolute, join, resolve } from "node:path";

import type { Bundle } from "./board";
import type { ColumnarTable, Manifest } from "./bundle";

/** Where `publish-bundle.mjs` keeps the pointer at the newest complete upload. */
const POINTER_PATH = "bundle/latest.json";

const cache = new Map<string, unknown>();

/** Resolved once per deployment: the prefix the pointer names, or the base. */
let resolvedBase: Promise<string> | null = null;

function trimSlash(url: string): string {
  return url.replace(/\/+$/, "");
}

/** Directories to try, most explicit first. */
function localDirs(): string[] {
  const dirs: string[] = [];
  const explicit = process.env.BUNDLE_DIR;
  if (explicit) dirs.push(isAbsolute(explicit) ? explicit : resolve(process.cwd(), explicit));
  // Deployed with the app: `next.config.mjs` traces this into the function.
  dirs.push(join(process.cwd(), "data", "v2_export"));
  // The repo checkout, in development.
  dirs.push(join(process.cwd(), "..", "data", "v2_export"));
  return dirs;
}

function isMissing(err: unknown): boolean {
  const code = (err as NodeJS.ErrnoException | null)?.code;
  return code === "ENOENT" || code === "ENOTDIR";
}

/**
 * Turn `BUNDLE_BASE_URL` into the prefix the files actually sit under.
 *
 * A base that carries the pointer is a store root and the files are under the
 * dated prefix it names -- which is the whole point of the pointer, since that
 * prefix changes every time the bundle is refreshed. A base without one is
 * taken at face value, so pointing the variable straight at a prefix still
 * works.
 */
async function resolveBase(base: string): Promise<string> {
  const root = trimSlash(base);
  try {
    const res = await fetch(`${root}/${POINTER_PATH}`, { cache: "no-store" });
    if (res.ok) {
      const pointer = (await res.json()) as { prefix?: string };
      if (pointer.prefix) return `${root}/${trimSlash(pointer.prefix)}`;
    }
  } catch {
    // No pointer, or it is unreachable. Fall through to the base itself: the
    // files may well be sitting directly under it.
  }
  return root;
}

async function fetchTable<T>(base: string, name: string): Promise<T> {
  if (!resolvedBase) resolvedBase = resolveBase(base);
  const prefix = await resolvedBase;
  const res = await fetch(`${prefix}/${name}.json`);
  if (!res.ok) throw new Error(`bundle: ${name} responded ${res.status} from ${prefix}`);
  return (await res.json()) as T;
}

async function readLocal<T>(name: string): Promise<T> {
  const dirs = localDirs();
  for (const dir of dirs) {
    try {
      return JSON.parse(await readFile(join(dir, `${name}.json`), "utf-8")) as T;
    } catch (err) {
      if (!isMissing(err)) throw err;
    }
  }
  // The unhelpful version of this is a bare ENOENT from a path nobody
  // configured, which is what a deployment with no bundle used to report.
  throw new Error(
    `bundle: ${name}.json not found. Set BUNDLE_BASE_URL to the published ` +
      `bundle, or BUNDLE_DIR to a directory holding it, or run ` +
      `scripts/export_v2_bundle.py. Looked in: ${dirs.join(", ")}`,
  );
}

async function load<T>(name: string): Promise<T> {
  const hit = cache.get(name);
  if (hit) return hit as T;

  const base = process.env.BUNDLE_BASE_URL;
  const parsed = base ? await fetchTable<T>(base, name) : await readLocal<T>(name);

  cache.set(name, parsed);
  return parsed;
}

export async function loadManifest(): Promise<Manifest & { team_nicknames: Record<string, string> }> {
  return load("manifest");
}

export async function loadBundleFor(format: string): Promise<Bundle> {
  const [board, training, kickerComponents, dstComponents, schedule, manifest] =
    await Promise.all([
      load<ColumnarTable>(`board_${format}`),
      load<ColumnarTable>(`training_${format}`),
      load<ColumnarTable>("kicker_components"),
      load<ColumnarTable>("dst_components"),
      load<ColumnarTable>("schedule_difficulty"),
      loadManifest(),
    ]);

  return {
    board,
    training,
    kickerComponents,
    dstComponents,
    schedule,
    teamNicknames: manifest.team_nicknames ?? {},
  };
}

/** When the ADP in the bundle was sampled, for the staleness banner. */
export async function adpAsOf(format: string): Promise<string | null> {
  const board = await load<ColumnarTable>(`board_${format}`);
  return (board.adp_as_of as string | undefined) ?? null;
}

/** Test seam. */
export function clearBundleCache(): void {
  cache.clear();
  resolvedBase = null;
}
