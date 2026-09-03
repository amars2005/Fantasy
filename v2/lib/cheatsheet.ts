/**
 * A self-contained cheatsheet, for when the network is not there.
 *
 * Hosting this tool trades away the best property of the thing it replaces:
 * `src/dashboard/app.py` runs on a laptop with no network, no install and
 * nothing to go wrong on the morning of the draft. This buys part of that back.
 * One HTML file, no scripts, no fetches, no fonts -- open it or print it, and
 * it works if Vercel is down, if the database is down, or if the draft venue's
 * wifi is the usual disaster.
 *
 * It is deliberately not a copy of the live board. It carries what you cannot
 * recompute in your head: the ordering, the tier breaks, and where replacement
 * sits.
 */

import { untieredPositions } from "./tiers";
import type { LeagueConfig, Player } from "./types";

const DEFAULT_LIMIT = 150;

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function snakeRounds(slot: number, teams: number, rounds: number): number[] {
  const picks: number[] = [];
  for (let r = 0; r < rounds; r++) {
    picks.push(r % 2 === 0 ? r * teams + slot : r * teams + (teams - slot + 1));
  }
  return picks;
}

export interface CheatsheetInput {
  leagueName: string;
  league: LeagueConfig;
  players: Player[];
  slot: number;
  adpAsOf: string | null;
  /** Anchor-format warnings, so the printed copy carries them too. */
  warnings?: string[];
  limit?: number;
}

export function renderCheatsheet(input: CheatsheetInput): string {
  const { league, slot } = input;
  const limit = input.limit ?? DEFAULT_LIMIT;

  const board = [...input.players].sort((a, b) => (b.vor ?? 0) - (a.vor ?? 0));
  const top = board.slice(0, limit);

  const replacement = new Map<string, number>();
  for (const p of input.players) {
    if (p.replacement !== undefined) replacement.set(p.pos, p.replacement);
  }

  const picks = snakeRounds(slot, league.teams, league.rounds);

  // The heavy rule marks the last player of a tier -- the one case worth
  // reaching for.
  //
  // Stated as "the next printed row is a different tier" this is very nearly
  // right, and was: tier-mates share a projection by definition, so at a fixed
  // position they share a VOR and sort into one run. It is right by luck of the
  // ordering rather than by construction, though, and it misses the last row on
  // the page and any tier still running past the cut. Counting a tier's members
  // and marking the row that exhausts it says the same thing on purpose.
  const tierTotal = new Map<string, number>();
  for (const p of input.players) {
    const key = `${p.pos}|${p.tier}`;
    tierTotal.set(key, (tierTotal.get(key) ?? 0) + 1);
  }
  const tierSeen = new Map<string, number>();
  // Positions the fit could not pool at all -- every kicker his own tier. The
  // rule would mark every one of their rows, which is not a tier break, it is
  // just the list continuing.
  const untiered = untieredPositions(input.players);

  const rows = top
    .map((p, i) => {
      const key = `${p.pos}|${p.tier}`;
      const seen = (tierSeen.get(key) ?? 0) + 1;
      tierSeen.set(key, seen);
      const tiered = !untiered.has(p.pos);
      // Counted against the whole board, not the printed slice: a tier that
      // runs on past the cut has not been exhausted by the last row shown.
      const tierBreak = tiered && seen === tierTotal.get(key);
      return `<tr class="${tierBreak ? "tier-end" : ""}">
      <td class="num">${i + 1}</td>
      <td>${escapeHtml(p.name)}</td>
      <td class="pos ${p.pos}">${p.pos}</td>
      <td>${escapeHtml(p.tm ?? "")}</td>
      <td class="num">${p.adp.toFixed(1)}</td>
      <td class="num">${p.proj_points.toFixed(0)}</td>
      <td class="num dim">${(p.sd ?? 0).toFixed(0)}</td>
      <td class="num">${(p.vor ?? 0).toFixed(0)}</td>
      <td class="num">${tiered ? `${p.pos}${p.tier}` : "&mdash;"}</td>
      <td class="num">${p.bye ?? "-"}</td>
      <td class="num dim">${(p.playoff_lift ?? 0) > 0 ? "+" : ""}${(p.playoff_lift ?? 0).toFixed(1)}</td>
    </tr>`;
    })
    .join("\n");

  const replacementRows = [...replacement.entries()]
    .sort((a, b) => (a[0] < b[0] ? -1 : 1))
    .map(([pos, level]) => `<tr><td>${pos}</td><td class="num">${level.toFixed(1)}</td></tr>`)
    .join("\n");

  const pickList = picks
    .map((p, i) => `<li><span class="rd">R${i + 1}</span> pick ${p}</li>`)
    .join("\n");

  const warnings = (input.warnings ?? [])
    .map((w) => `<li>${escapeHtml(w)}</li>`)
    .join("\n");

  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>${escapeHtml(input.leagueName)} - draft cheatsheet</title>
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px;
    font: 12px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    color: #16181d; background: #fff;
  }
  h1 { font-size: 18px; margin: 0 0 2px; }
  h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .08em;
       color: #6b7280; margin: 0 0 8px; }
  .meta { color: #6b7280; font-size: 11px; margin-bottom: 18px; }
  .layout { display: grid; grid-template-columns: 1fr 210px; gap: 28px; align-items: start; }
  table { border-collapse: collapse; width: 100%; }
  th, td { text-align: left; padding: 3px 6px; border-bottom: 1px solid #eef0f3; }
  th { font-size: 10px; text-transform: uppercase; letter-spacing: .06em; color: #6b7280;
       border-bottom: 1px solid #d6dae1; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  tr.tier-end td { border-bottom: 2px solid #b9c0cb; }
  td.dim { color: #8a919c; }
  .pos { font-weight: 600; font-size: 10px; }
  .QB { color: #b3452f; } .RB { color: #1f7a4d; } .WR { color: #2454a6; }
  .TE { color: #8a5a1a; } .K  { color: #6b7280; } .DST { color: #55507a; }
  aside section { margin-bottom: 22px; }
  ul { margin: 0; padding-left: 16px; }
  li { margin-bottom: 2px; }
  .rd { display: inline-block; width: 26px; color: #6b7280; }
  .warn { border-left: 3px solid #d08b2c; padding-left: 10px; color: #6b5426; }
  @media print {
    body { padding: 0; font-size: 10px; }
    .layout { gap: 18px; }
    tr { page-break-inside: avoid; }
  }
</style>
</head>
<body>
  <h1>${escapeHtml(input.leagueName)}</h1>
  <div class="meta">
    ${league.teams}-team &middot; slot ${slot} &middot; ${league.rounds} rounds
    &middot; ADP as of ${escapeHtml(input.adpAsOf ?? "unknown")}
    &middot; ranked by value over replacement
  </div>

  <div class="layout">
    <main>
      <h2>Top ${top.length} by VOR</h2>
      <table>
        <thead>
          <tr>
            <th class="num">#</th><th>Player</th><th>Pos</th><th>Tm</th>
            <th class="num">ADP</th><th class="num">Proj</th><th class="num">&plusmn;</th>
            <th class="num">VOR</th><th class="num">Tier</th><th class="num">Bye</th>
            <th class="num">Playoff</th>
          </tr>
        </thead>
        <tbody>
${rows}
        </tbody>
      </table>
    </main>

    <aside>
      <section>
        <h2>Your picks</h2>
        <ul>${pickList}</ul>
      </section>
      <section>
        <h2>Replacement level</h2>
        <table><tbody>${replacementRows}</tbody></table>
      </section>
      ${
        warnings
          ? `<section class="warn"><h2>Read first</h2><ul>${warnings}</ul></section>`
          : ""
      }
      <section>
        <h2>Reading it</h2>
        <ul>
          <li>A heavy rule under a player means he is the last of his tier. Being last in a tier is the only good reason to reach.</li>
          <li>Proj is read off a curve of draft rank against what players at that rank actually scored, so a tier shares one number. &plusmn; is the measured spread around it — wide enough that the projection is the middle of a range, not a forecast.</li>
          <li>Inside a tier the projection is tied, so there is nothing to choose on but Bye and Playoff — how much softer weeks 15-17 are for his team than the rest of its season. Neither is priced into the ranking.</li>
          <li>VOR is against replacement, not against the field.</li>
          <li>K and DST last. They are near-random year to year.</li>
        </ul>
      </section>
    </aside>
  </div>
</body>
</html>`;
}
