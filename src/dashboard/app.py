"""Local draft-day dashboard.

Deliberately built on the standard library alone. This runs once a year, under a
60-second clock, on a laptop that must not surprise you: no framework, no build
step, nothing to install the morning of the draft.
"""

from __future__ import annotations

import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.config import LEAGUE
from src.draft.board import DraftBoard

HERE = Path(__file__).parent
STATE: dict = {}


def _payload(board: DraftBoard) -> dict:
    rec = board.recommend(n=14)
    rec["league"] = {
        "teams": LEAGUE["teams"],
        "slot": board.slot,
        "picks": board.picks,
        "rounds": LEAGUE["rounds"],
    }
    rec["drafted_count"] = len(board.drafted)
    rec["on_the_clock_is_mine"] = board.pick_number == board.next_pick()
    # Two different numbers that were being conflated in the header: how long
    # until our turn, and how many players come off the board between our own
    # consecutive picks. The second is what VONA reasons about; the first is what
    # a person watching the draft wants to see.
    nxt = board.next_pick()
    rec["picks_away"] = (nxt - board.pick_number) if nxt else 0
    rec["unknown_picks"] = sum(1 for k in board.drafted if k.startswith("__unknown_"))
    rec["resumed"] = board.resumed or {}
    recent = [
        (pid, by) for pid, by in list(board.drafted.items())
        if not pid.startswith("__unknown_")
    ][-12:][::-1]
    lookup = {
        r["player_id"]: r
        for r in board.players.select(["player_id", "name", "pos", "tm"]).to_dicts()
    }
    rec["recent"] = [
        {**lookup.get(pid, {"player_id": pid, "name": pid, "pos": "?", "tm": ""}), "by": by}
        for pid, by in recent
    ]
    return rec


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the console quiet during a draft
        pass

    def _send(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        board: DraftBoard = STATE["board"]

        if url.path in ("/", "/index.html"):
            body = (HERE / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if url.path == "/api/state":
            self._send(_payload(board))
            return

        if url.path == "/api/search":
            q = (parse_qs(url.query).get("q") or [""])[0]
            rows = board.find(q, limit=10) if q else board.available.sort("adp").head(10)
            self._send({
                "results": rows.select(
                    ["player_id", "name", "pos", "tm", "adp", "proj_points", "tier"]
                ).to_dicts()
            })
            return

        self._send({"error": "not found"}, 404)

    def do_POST(self) -> None:
        url = urlparse(self.path)
        board: DraftBoard = STATE["board"]
        length = int(self.headers.get("Content-Length") or 0)
        data = json.loads(self.rfile.read(length) or b"{}")

        if url.path == "/api/draft":
            board.draft(data["player_id"], data.get("by", "other"))
            self._send(_payload(board))
            return

        if url.path == "/api/undo":
            pid = data.get("player_id")
            if pid:
                board.undo(pid)
            self._send(_payload(board))
            return

        if url.path == "/api/skip":
            # Someone drafted a player who is not on our board. Advance the clock
            # anyway, or every "picks until my next turn" number drifts.
            count = int(data.get("count", 1))
            if count > 0:
                board.skip(count)
            else:
                board.undo_skip()
            self._send(_payload(board))
            return

        self._send({"error": "not found"}, 404)


def serve(slot: int, port: int = 8777, open_browser: bool = True,
          fresh: bool = False) -> None:
    print(f"Building board for slot {slot} of {LEAGUE['teams']}...")
    board = DraftBoard(slot=slot, resume=not fresh)
    STATE["board"] = board
    if board.resumed.get("picks"):
        print(f"Resumed {board.resumed['picks']} picks saved at "
              f"{board.resumed.get('saved_at')}")
        print(f"  (start with --fresh to discard and begin a new draft)")
    elif board.resumed.get("mismatch"):
        print("Found saved state for a different league shape; ignoring it.")
    url = f"http://localhost:{port}"
    print(f"Draft board ready at {url}")
    if open_browser:
        webbrowser.open(url)
    ThreadingHTTPServer(("localhost", port), Handler).serve_forever()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Live fantasy draft board")
    ap.add_argument("--slot", type=int, required=True, help="your draft slot (1-14)")
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                    help="discard saved state and start a new draft")
    args = ap.parse_args()
    serve(args.slot, args.port, open_browser=not args.no_browser, fresh=args.fresh)
