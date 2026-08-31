import { NextResponse } from "next/server";

import { loadManifest } from "../../../../lib/bundleLoader";
import { getLeagueView } from "../../../../lib/leagues";

export const runtime = "nodejs";

/** Everything the client needs to run a draft: config, board, picks. */
export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const view = await getLeagueView(id);
  if (!view) return NextResponse.json({ error: "No such league" }, { status: 404 });

  // The board's search box matches defences on team nickname -- nobody types
  // "BAL" to find the Ravens -- so the client needs this table.
  const manifest = await loadManifest();

  return NextResponse.json({
    id: view.league.id,
    name: view.league.name,
    config: view.league.config,
    format: view.league.format,
    board: view.league.board,
    adpAsOf: view.league.adpAsOf,
    frozenAt: view.league.frozenAt,
    refreshAvailable: view.refreshAvailable,
    warnings: view.warnings,
    picks: view.picks,
    teamNicknames: manifest.team_nicknames ?? {},
  });
}
