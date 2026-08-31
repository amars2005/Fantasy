/**
 * Server-side loading of the static bundle.
 *
 * In production the bundle lives in Vercel Blob, refreshed daily by the
 * GitHub Action that runs `scripts/export_v2_bundle.py`. In development it is
 * read straight off disk from `data/v2_export`, so the app runs with nothing
 * provisioned.
 *
 * Tables are cached in module memory. They are immutable for the life of a
 * deployment, and re-parsing a megabyte of JSON per league creation would be
 * pure waste.
 */

import { readFile } from "node:fs/promises";
import { join } from "node:path";

import type { Bundle } from "./board";
import type { ColumnarTable, Manifest } from "./bundle";

const LOCAL_DIR = join(process.cwd(), "..", "data", "v2_export");

const cache = new Map<string, unknown>();

async function load<T>(name: string): Promise<T> {
  const hit = cache.get(name);
  if (hit) return hit as T;

  const base = process.env.BUNDLE_BASE_URL;
  let parsed: T;

  if (base) {
    const res = await fetch(`${base.replace(/\/$/, "")}/${name}.json`);
    if (!res.ok) throw new Error(`bundle: ${name} responded ${res.status}`);
    parsed = (await res.json()) as T;
  } else {
    parsed = JSON.parse(await readFile(join(LOCAL_DIR, `${name}.json`), "utf-8")) as T;
  }

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
}
