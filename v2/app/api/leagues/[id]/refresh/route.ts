import { NextResponse } from "next/server";

import { toLeaguePickSpace } from "../../../../../lib/board";

import { refreshBoard } from "../../../../../lib/leagues";

export const runtime = "nodejs";

/**
 * Re-derive the board against the current bundle, and clear the freeze.
 *
 * Only ever on explicit request. ADP refreshes daily and a draft runs over
 * days, so re-deriving automatically would move every projection, tier and
 * survival number underneath a draft already in progress.
 */
export async function POST(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const league = await refreshBoard(id);
  if (!league) return NextResponse.json({ error: "No such league" }, { status: 404 });

  return NextResponse.json({
    id: league.id,
    adpAsOf: league.adpAsOf,
    board: league.board ? toLeaguePickSpace(league.board, league.config.teams) : null,
  });
}
