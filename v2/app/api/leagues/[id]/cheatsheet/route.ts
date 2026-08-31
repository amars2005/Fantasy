import { renderCheatsheet } from "../../../../../lib/cheatsheet";
import { getLeagueView } from "../../../../../lib/leagues";

export const runtime = "nodejs";

/**
 * The offline escape hatch: one self-contained HTML file, no scripts, no
 * fetches. Served as a download so it lands on disk before the draft rather
 * than living in a tab that needs the network to reopen.
 */
export async function GET(
  request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const view = await getLeagueView(id);
  if (!view || !view.league.board) {
    return new Response("No such league", { status: 404 });
  }

  const slot = Number(new URL(request.url).searchParams.get("slot") ?? 1);
  const html = renderCheatsheet({
    leagueName: view.league.name,
    league: view.league.config,
    players: view.league.board,
    slot: Number.isFinite(slot) && slot > 0 ? slot : 1,
    adpAsOf: view.league.adpAsOf,
    warnings: view.warnings,
  });

  const safeName = view.league.name.replace(/[^a-z0-9]+/gi, "-").toLowerCase();
  return new Response(html, {
    headers: {
      "content-type": "text/html; charset=utf-8",
      "content-disposition": `attachment; filename="${safeName}-cheatsheet.html"`,
    },
  });
}
