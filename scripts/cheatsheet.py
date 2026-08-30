"""Printable tiered cheat sheet -- the offline fallback.

If the laptop dies, the wifi drops, or the draft app eats the browser tab, this
is what you draft from. Open the HTML and print to PDF.
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl

from src.config import DATA_PROCESSED, LEAGUE, SEASON
from src.draft.replacement import add_vor, replacement_levels
from src.draft.sim_draft import snake_picks
from src.draft.tiers import add_tiers
from src.project.consensus import project
from src.project.kdst import project_kdst

POS_COLORS = {"QB": "#7c3aed", "RB": "#059669", "WR": "#2563eb",
              "TE": "#d97706", "DST": "#475569"}


def _tier_blocks(board: pl.DataFrame, pos: str, max_players: int) -> str:
    sub = board.filter(pl.col("pos") == pos).sort("proj_points", descending=True).head(max_players)
    out = []
    for tier, group in sub.group_by("tier", maintain_order=True):
        rows = "".join(
            f"<tr><td>{html.escape(r['name'])}</td><td class='t'>{r['tm']}</td>"
            f"<td class='n'>{r['adp']:.0f}</td><td class='n'>{r['proj_points']:.0f}</td>"
            f"<td class='n b'>{r['vor']:.0f}</td><td class='n'>{r['bye'] or '-'}</td></tr>"
            for r in group.sort("adp").iter_rows(named=True)
        )
        out.append(
            f"<div class='tier'><div class='tierhead'>Tier {tier[0]}"
            f"<span>{group.height} player{'s' if group.height > 1 else ''}</span></div>"
            f"<table><tr><th>Player</th><th>Tm</th><th>ADP</th><th>Proj</th>"
            f"<th>VOR</th><th>Bye</th></tr>{rows}</table></div>"
        )
    return "".join(out)


def build(season: int, slot: int | None, out_path: Path) -> Path:
    proj = project(season)
    try:
        kdst = project_kdst(season)
        if kdst.height:
            proj = pl.concat([proj, kdst], how="diagonal_relaxed")
    except Exception:
        pass
    board = add_tiers(add_vor(proj))
    levels = replacement_levels(board)

    picks = snake_picks(slot, LEAGUE["teams"], LEAGUE["rounds"]) if slot else []
    picks_html = (
        "<div class='picks'><b>Your picks (slot " + str(slot) + "):</b> "
        + " &middot; ".join(str(p) for p in picks) + "</div>"
        if picks else ""
    )

    cols = "".join(
        f"<section><h2 style='border-color:{POS_COLORS[p]}'>"
        f"<span style='color:{POS_COLORS[p]}'>{p}</span></h2>"
        f"<div class='repl'>replacement: {levels.get(p, 0):.0f} pts</div>"
        f"{_tier_blocks(board, p, n)}</section>"
        # Defences earn a column now: year-over-year persistence is 0.276, more
        # than twice the kicker's 0.109. Kickers stay off the sheet deliberately.
        for p, n in [("RB", 42), ("WR", 50), ("TE", 20), ("QB", 20), ("DST", 16)]
    )

    doc = f"""<!doctype html><meta charset="utf-8"><title>Cheat Sheet {season}</title>
<style>
 @page{{size:A4 landscape;margin:10mm}}
 body{{font:10px/1.3 ui-sans-serif,system-ui,sans-serif;color:#111;margin:0}}
 h1{{font-size:15px;margin:0 0 2px}}
 .sub{{color:#666;font-size:10px;margin-bottom:6px}}
 .picks{{background:#f4f6f8;border:1px solid #dde3ea;padding:5px 8px;
        border-radius:5px;margin-bottom:8px;font-size:10px}}
 .grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}
 section{{break-inside:avoid}}
 h2{{font-size:12px;margin:0 0 2px;padding-bottom:2px;border-bottom:2px solid}}
 .repl{{color:#888;font-size:9px;margin-bottom:5px}}
 .tier{{margin-bottom:7px;break-inside:avoid}}
 .tierhead{{display:flex;justify-content:space-between;background:#eef1f5;
   padding:2px 5px;font-weight:700;font-size:9px;border-radius:3px}}
 .tierhead span{{color:#7a8593;font-weight:400}}
 table{{width:100%;border-collapse:collapse;margin-top:2px}}
 th{{font-size:8px;color:#8a94a0;text-align:right;font-weight:600;padding:1px 3px}}
 th:first-child,td:first-child{{text-align:left}}
 td{{padding:1px 3px;border-bottom:1px solid #f0f2f5}}
 .n{{text-align:right;font-variant-numeric:tabular-nums}}
 .t{{color:#8a94a0;font-size:9px}}
 .b{{font-weight:700}}
</style>
<h1>{season} Draft Cheat Sheet &mdash; {LEAGUE['teams']}-team full PPR</h1>
<div class="sub">Tiers are groups the historical data cannot separate: take any
player in a tier over reaching for one in the tier below. VOR is value over this
league's replacement level. Never take the last player in a tier if the next tier
is still deep.</div>
{picks_html}
<div class="grid">{cols}</div>"""

    out_path.write_text(doc, encoding="utf-8")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=SEASON)
    ap.add_argument("--slot", type=int, default=None)
    args = ap.parse_args()

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    path = build(args.season, args.slot, DATA_PROCESSED / "cheatsheet.html")
    print(f"Wrote {path}")
    print("Open it in a browser and print to PDF (A4 landscape).")


if __name__ == "__main__":
    main()
