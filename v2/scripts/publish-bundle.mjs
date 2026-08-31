/**
 * Publish the exported bundle to Vercel Blob.
 *
 * Files are written under a dated prefix first, and only when every one of them
 * has landed is the `latest` pointer moved. Stale ADP is acceptable -- the UI
 * shows its date -- but a half-written bundle would parse cleanly and be
 * silently wrong, which is the failure this ordering exists to prevent.
 *
 * Set BUNDLE_BASE_URL in the app to the value this prints.
 */

import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";

const DIR = join(process.cwd(), "..", "data", "v2_export");
const token = process.env.BLOB_READ_WRITE_TOKEN;

if (!token) {
  console.error("BLOB_READ_WRITE_TOKEN is not set; nothing published.");
  process.exit(0);
}

const { put } = await import("@vercel/blob");

const files = (await readdir(DIR)).filter((f) => f.endsWith(".json"));
if (!files.length) {
  console.error("No bundle files found. Run scripts/export_v2_bundle.py first.");
  process.exit(1);
}

// A manifest that does not parse means the export did not finish.
const manifest = JSON.parse(await readFile(join(DIR, "manifest.json"), "utf-8"));
if (!manifest.boards?.length) {
  console.error("Manifest lists no boards; refusing to publish.");
  process.exit(1);
}

const stamp = new Date().toISOString().slice(0, 10);
const uploaded = [];

for (const name of files) {
  const body = await readFile(join(DIR, name));
  const { url } = await put(`bundle/${stamp}/${name}`, body, {
    access: "public",
    token,
    contentType: "application/json",
    addRandomSuffix: false,
  });
  uploaded.push(name);
  console.log(`  ${name} -> ${url}`);
}

// Every file landed. Move the pointer.
await put(
  "bundle/latest.json",
  JSON.stringify({ prefix: `bundle/${stamp}`, files: uploaded, publishedAt: new Date().toISOString() }),
  { access: "public", token, contentType: "application/json", addRandomSuffix: false },
);

console.log(`\nPublished ${uploaded.length} files under bundle/${stamp} and moved latest.`);
