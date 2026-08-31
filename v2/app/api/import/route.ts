import { NextResponse } from "next/server";

import { importEspnLeague } from "../../../lib/import/espn";
import { importSleeperLeague } from "../../../lib/import/sleeper";

export const runtime = "nodejs";

/**
 * Import a league's settings from a platform.
 *
 * This returns a config to *review*, never a saved league. Both importers map
 * undocumented keys, so a table that has drifted must surface as a visibly
 * wrong number on a form rather than a silently wrong draft board.
 */
export async function POST(request: Request) {
  let body: {
    source?: string;
    leagueId?: string;
    season?: number;
    espnS2?: string;
    swid?: string;
  };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Body must be JSON" }, { status: 400 });
  }

  if (!body.leagueId) {
    return NextResponse.json({ error: "A league id is required" }, { status: 400 });
  }

  try {
    if (body.source === "sleeper") {
      return NextResponse.json(await importSleeperLeague(body.leagueId));
    }
    if (body.source === "espn") {
      const season = body.season ?? new Date().getFullYear();
      return NextResponse.json(
        await importEspnLeague(body.leagueId, season, {
          espnS2: body.espnS2,
          swid: body.swid,
        }),
      );
    }
    return NextResponse.json({ error: "source must be sleeper or espn" }, { status: 400 });
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Import failed" },
      { status: 502 },
    );
  }
}
