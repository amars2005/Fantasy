/**
 * Copy the exported bundle into the app, for deploying without a Blob store.
 *
 * `next.config.mjs` traces `v2/data/v2_export` into the serverless function, so
 * a bundle staged here (and committed) deploys with the app and the loader
 * finds it with no environment variable set. Otherwise the bundle lives only in
 * the repo root, which is outside the build's file tracing, and the function
 * comes up with no board at all.
 */

import { cp, mkdir, readdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const SRC = join(HERE, "..", "..", "data", "v2_export");
const DEST = join(HERE, "..", "data", "v2_export");

let files;
try {
  files = (await readdir(SRC)).filter((f) => f.endsWith(".json"));
} catch {
  console.error(`No bundle at ${SRC}. Run scripts/export_v2_bundle.py first.`);
  process.exit(1);
}

if (!files.includes("manifest.json")) {
  console.error("No manifest.json in the export; refusing to stage a partial bundle.");
  process.exit(1);
}

await mkdir(DEST, { recursive: true });
for (const name of files) {
  await cp(join(SRC, name), join(DEST, name));
}

console.log(`Staged ${files.length} files into v2/data/v2_export.`);
