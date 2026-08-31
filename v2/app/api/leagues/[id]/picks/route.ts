import { NextResponse } from "next/server";

import { getStore } from "../../../../../lib/store";
import { recordPick, undoPick } from "../../../../../lib/leagues";

export const runtime = "nodejs";

/** Picks after `since`, for reconciling a second tab or a returning device. */
export async function GET(
  request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const since = Number(new URL(request.url).searchParams.get("since") ?? 0);
  const picks = await getStore().listPicks(id, Number.isFinite(since) ? since : 0);
  return NextResponse.json({ picks });
}

/**
 * Record a pick.
 *
 * `seq` is supplied by the client and is what makes this idempotent: a retry
 * after a timed-out request carries the same `seq`, hits the primary key, and
 * returns the stored row instead of advancing the clock a second time. A
 * doubled pick would put every "picks until my next turn" number -- and so
 * every VONA number -- permanently out of step with the room.
 */
export async function POST(
  request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;

  let body: { seq?: number; playerId?: string | null; takenBy?: string };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Body must be JSON" }, { status: 400 });
  }

  if (!Number.isInteger(body.seq) || (body.seq as number) < 1) {
    return NextResponse.json({ error: "seq must be a positive integer" }, { status: 400 });
  }

  const result = await recordPick(
    id,
    body.seq as number,
    body.playerId ?? null,
    body.takenBy === "me" ? "me" : "other",
  );
  if (!result) return NextResponse.json({ error: "No such league" }, { status: 404 });

  // 200 rather than 201 on a repeat, so the client can tell the difference
  // without it being an error.
  return NextResponse.json(
    { pick: result.pick, created: result.created },
    { status: result.created ? 201 : 200 },
  );
}

/** Undo: a soft delete, so the pick log stays an audit trail. */
export async function DELETE(
  request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const seq = Number(new URL(request.url).searchParams.get("seq"));
  if (!Number.isInteger(seq)) {
    return NextResponse.json({ error: "seq is required" }, { status: 400 });
  }

  const voided = await undoPick(id, seq);
  if (!voided) return NextResponse.json({ error: "No such pick" }, { status: 404 });
  return NextResponse.json({ pick: voided });
}
