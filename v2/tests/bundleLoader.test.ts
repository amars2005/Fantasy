/**
 * Where the bundle comes from on a deployed instance.
 *
 * This is the seam that took the hosted app down: with no BUNDLE_BASE_URL the
 * loader fell back to the development path, `../data/v2_export`, which is not
 * in the repo and not in the serverless function either, so every league
 * creation died on `ENOENT .../data/v2_export/board_ppr.json` -- an error that
 * names a path nobody configured and no variable to set.
 */

import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { adpAsOf, clearBundleCache, loadManifest } from "../lib/bundleLoader";

const table = (name: string, extra: Record<string, unknown> = {}) => ({
  name,
  generated_at: "2026-09-01T06:00:00Z",
  columns: ["player_id"],
  rows: [["00-0000001"]],
  ...extra,
});

let dir: string;

beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), "bundle-"));
  clearBundleCache();
  delete process.env.BUNDLE_BASE_URL;
  delete process.env.BUNDLE_DIR;
});

afterEach(async () => {
  vi.unstubAllGlobals();
  delete process.env.BUNDLE_BASE_URL;
  delete process.env.BUNDLE_DIR;
  clearBundleCache();
  await rm(dir, { recursive: true, force: true });
});

describe("local bundles", () => {
  it("reads the directory BUNDLE_DIR names", async () => {
    await writeFile(join(dir, "manifest.json"), JSON.stringify({ season: 2026, team_nicknames: { BAL: "ravens" } }));
    process.env.BUNDLE_DIR = dir;

    expect((await loadManifest()).team_nicknames).toEqual({ BAL: "ravens" });
  });

  it("says what to configure when there is no bundle anywhere", async () => {
    // BUNDLE_DIR is exclusive, so this holds whether or not the checkout this
    // runs in happens to have an exported bundle sitting at the repo root --
    // in CI it does, and a first version of this test passed locally and
    // failed there for exactly that reason.
    process.env.BUNDLE_DIR = join(dir, "absent");

    // The failure a deployment actually hits, so it has to name the fix.
    await expect(adpAsOf("ppr")).rejects.toThrow(/board_ppr\.json not found/);
    await expect(adpAsOf("ppr")).rejects.toThrow(/BUNDLE_BASE_URL/);
  });

  it("does not swallow a corrupt file as a missing one", async () => {
    await writeFile(join(dir, "manifest.json"), "{ truncated");
    process.env.BUNDLE_DIR = dir;

    await expect(loadManifest()).rejects.toThrow(SyntaxError);
  });
});

describe("published bundles", () => {
  it("follows the latest pointer to the dated prefix", async () => {
    // The prefix moves every refresh, which is why the pointer exists: the
    // deployment holds a fixed store root and finds today's upload through it.
    const fetchMock = vi.fn(async (url: string) => {
      if (url.endsWith("/bundle/latest.json")) {
        return new Response(JSON.stringify({ prefix: "bundle/2026-09-02" }), { status: 200 });
      }
      if (url === "https://blob.example/bundle/2026-09-02/board_ppr.json") {
        return new Response(JSON.stringify(table("board_ppr", { adp_as_of: "2026-09-02" })), { status: 200 });
      }
      return new Response("not found", { status: 404 });
    });
    vi.stubGlobal("fetch", fetchMock);
    process.env.BUNDLE_BASE_URL = "https://blob.example/";

    expect(await adpAsOf("ppr")).toBe("2026-09-02");
  });

  it("resolves the pointer once, not per table", async () => {
    const fetchMock = vi.fn(async (url: string) =>
      url.endsWith("/bundle/latest.json")
        ? new Response(JSON.stringify({ prefix: "bundle/2026-09-02" }), { status: 200 })
        : new Response(JSON.stringify(table("t")), { status: 200 }),
    );
    vi.stubGlobal("fetch", fetchMock);
    process.env.BUNDLE_BASE_URL = "https://blob.example";

    await Promise.all([adpAsOf("ppr"), adpAsOf("half_ppr"), loadManifest()]);

    const pointerCalls = fetchMock.mock.calls.filter(([url]) => String(url).endsWith("latest.json"));
    expect(pointerCalls).toHaveLength(1);
  });

  it("still accepts a base that points straight at the files", async () => {
    const fetchMock = vi.fn(async (url: string) =>
      url === "https://blob.example/bundle/2026-09-02/board_ppr.json"
        ? new Response(JSON.stringify(table("board_ppr", { adp_as_of: "2026-09-02" })), { status: 200 })
        : new Response("not found", { status: 404 }),
    );
    vi.stubGlobal("fetch", fetchMock);
    process.env.BUNDLE_BASE_URL = "https://blob.example/bundle/2026-09-02";

    expect(await adpAsOf("ppr")).toBe("2026-09-02");
  });

  it("reports the status and the prefix when a table is missing", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("nope", { status: 403 })));
    process.env.BUNDLE_BASE_URL = "https://blob.example";

    await expect(adpAsOf("ppr")).rejects.toThrow(/board_ppr responded 403 from https:\/\/blob\.example/);
  });
});
