import { NextResponse } from "next/server";

import { ConfigError } from "../../../lib/config";
import { createLeague } from "../../../lib/leagues";

export const runtime = "nodejs";

/** Create a league: validate, derive the board once, store it. */
export async function POST(request: Request) {
  let body: { name?: string; config?: unknown };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Body must be JSON" }, { status: 400 });
  }

  if (!body.config || typeof body.config !== "object") {
    return NextResponse.json({ error: "A league config is required" }, { status: 400 });
  }

  try {
    const league = await createLeague({
      name: body.name,
      // Validated inside createLeague before anything expensive happens.
      config: body.config as never,
    });
    return NextResponse.json(
      { id: league.id, name: league.name, format: league.format, adpAsOf: league.adpAsOf },
      { status: 201 },
    );
  } catch (err) {
    if (err instanceof ConfigError) {
      return NextResponse.json({ error: err.message }, { status: 422 });
    }
    // Derivation failed. The config is not saved, so the user can correct and
    // retry rather than being left with a league that has no board.
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Could not build a board" },
      { status: 500 },
    );
  }
}
