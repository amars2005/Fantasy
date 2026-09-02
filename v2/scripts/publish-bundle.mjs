/**
 * Publish the exported bundle to Vercel Blob.
 *
 * Files are written under a dated prefix first, and only when every one of them
 * has landed is the `latest` pointer moved. Stale ADP is acceptable -- the UI
 * shows its date -- but a half-written bundle would parse cleanly and be
 * silently wrong, which is the failure this ordering exists to prevent.
 *
 * Set BUNDLE_BASE_URL in the app to the store root this prints -- not to the
 * dated prefix, which moves with every refresh. The app resolves the prefix
 * through the pointer on each cold start, so the variable is set once.
 */

import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";

const DIR = join(process.cwd(), "..", "data", "v2_export");

// A secret pasted with its surrounding quotes, or as the whole `KEY="value"`
// line, is a valid string as far as the environment is concerned and reaches
// the API as a token that simply is not one. What comes back is an access
// denial indistinguishable from a revoked token, so check the shape here.
const raw = process.env.BLOB_READ_WRITE_TOKEN;

if (!raw || !raw.trim()) {
  console.error("BLOB_READ_WRITE_TOKEN is not set; nothing published.");
  process.exit(0);
}

const token = raw.trim().replace(/^BLOB_READ_WRITE_TOKEN=/, "").replace(/^["']|["']$/g, "");

// vercel_blob_rw_<storeId>_<secret>
const parts = token.split("_");
if (!token.startsWith("vercel_blob_rw_") || parts.length < 5 || !parts[3]) {
  console.error(
    "BLOB_READ_WRITE_TOKEN does not look like a Blob token. Expected " +
      "vercel_blob_rw_<storeId>_<secret>; got " +
      `${token.length} characters starting "${token.slice(0, 12)}". Check the ` +
      "secret holds the token alone -- no quotes, no KEY= prefix, no newline.",
  );
  process.exit(1);
}

// Which store this run is writing to. The id is public -- it is in the URL of
// every file published -- and printing it is how you tell a stale secret from
// a revoked one when the API says only "access denied".
console.log(`Publishing to Blob store ${parts[3]}.`);

const { put } = await import("@vercel/blob");

let files;
try {
  files = (await readdir(DIR)).filter((f) => f.endsWith(".json"));
} catch {
  files = [];
}
if (!files.length) {
  console.error(`No bundle files in ${DIR}. Run scripts/export_v2_bundle.py first.`);
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
    // Without this the SDK refuses to write a pathname that already exists,
    // which would make a same-day re-run fail on its first file -- and a
    // re-run is exactly what happens after a flake or a fix.
    allowOverwrite: true,
  });
  uploaded.push(name);
  console.log(`  ${name} -> ${url}`);
}

// Every file landed. Move the pointer.
const { url: pointerUrl } = await put(
  "bundle/latest.json",
  JSON.stringify({ prefix: `bundle/${stamp}`, files: uploaded, publishedAt: new Date().toISOString() }),
  // The pointer is rewritten by every refresh; that is its whole job.
  { access: "public", token, contentType: "application/json", addRandomSuffix: false, allowOverwrite: true },
);

console.log(`\nPublished ${uploaded.length} files under bundle/${stamp} and moved latest.`);
console.log(`Set BUNDLE_BASE_URL to ${pointerUrl.replace(/\/bundle\/latest\.json$/, "")}`);
