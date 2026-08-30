"""Live draft state.

Holds the board, tracks who is gone and what you own, and answers the only
question that matters when the clock is running: who should I take right now,
and why.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src.config import LEAGUE, SEASON
from src.draft import persist
from src.draft.replacement import add_vor, replacement_levels
from src.draft.sim_draft import add_calibrated_adp, snake_picks
from src.draft.tiers import add_tiers
from src.draft.vona import position_dropoff, vona_table
from src.ingest.ids import TEAM_NICKNAMES
from src.project.consensus import project
from src.features.schedule import add_playoff_lift
from src.project.kdst import project_kdst


class DraftBoard:
    def __init__(self, slot: int, projections: pl.DataFrame | None = None,
                 league: dict | None = None, resume: bool = True,
                 season: int = SEASON):
        self.league = league or LEAGUE
        self.slot = slot
        self.season = season
        proj = projections if projections is not None else project()
        # Kickers and defences must be on the board even though they are barely
        # worth projecting. A pick that cannot be marked desynchronises the pick
        # counter from the real draft, and every "picks until my next turn"
        # number -- which is what VONA is built on -- drifts with it.
        try:
            kdst = project_kdst()
            if kdst.height:
                proj = pl.concat([proj, kdst], how="diagonal_relaxed")
        except Exception:
            pass
        board = add_calibrated_adp(add_tiers(add_vor(proj, self.league)))
        try:
            # Weeks 15-17 decide the title, and ADP is format-blind so it cannot
            # price them. Positive lift means softer matchups exactly then.
            board = add_playoff_lift(board)
        except Exception:
            board = board.with_columns(pl.lit(0.0).alias("playoff_lift"))
        # One searchable haystack per player: name, team abbreviation, and -- for
        # defences only -- the team nickname. Nicknames stay off the skill players
        # deliberately. Enter marks the first row, and nobody types "ravens" to
        # reach Lamar Jackson; attaching it to all nine would bury the defence the
        # word was typed to find.
        self.players = board.with_columns(
            (
                pl.col("name").str.to_lowercase()
                + pl.lit(" ")
                + pl.col("tm").fill_null("").str.to_lowercase()
                + pl.lit(" ")
                + pl.when(pl.col("pos") == "DST")
                .then(pl.col("tm").fill_null("").replace_strict(TEAM_NICKNAMES, default=""))
                .otherwise(pl.lit(""))
            ).alias("search_key")
        )
        self.picks = snake_picks(slot, self.league["teams"], self.league["rounds"])
        # Picks survive a restart: a draft that runs for days must not be lost
        # to a closed terminal.
        self.drafted: dict[str, str] = {}  # player_id -> "me" | "other"
        self.resumed: dict = {}
        if resume:
            restored, meta = persist.load(slot, season)
            self.drafted.update(restored)
            self.resumed = meta

    # --- state ------------------------------------------------------------
    @property
    def pick_number(self) -> int:
        """Overall pick currently on the clock (1-based)."""
        return len(self.drafted) + 1

    @property
    def my_roster(self) -> list[dict]:
        ids = [pid for pid, who in self.drafted.items() if who == "me"]
        if not ids:
            return []
        return self.players.filter(pl.col("player_id").is_in(ids)).select(
            ["player_id", "name", "pos", "proj_points"]
        ).to_dicts()

    @property
    def available(self) -> pl.DataFrame:
        return self.players.filter(~pl.col("player_id").is_in(list(self.drafted)))

    def next_pick(self) -> int | None:
        """My next pick, counting the one on the clock if it is mine."""
        return next((p for p in self.picks if p >= self.pick_number), None)

    def picks_until_next(self) -> int:
        """How many players come off the board before I choose again."""
        current = self.next_pick()
        if current is None:
            return 0
        following = next((p for p in self.picks if p > current), None)
        if following is None:
            return len(self.available)
        return following - current

    def _persist(self) -> None:
        try:
            persist.save(self.slot, self.season, self.drafted)
        except Exception:
            # Never let a failed save block a pick during a live draft.
            pass

    def draft(self, player_id: str, by: str = "other") -> None:
        self.drafted[player_id] = by
        self._persist()

    def skip(self, count: int = 1) -> None:
        """Advance the clock for picks we cannot identify.

        A safety net for when someone drafts a player who is not on the board at
        all. Without it the counter silently falls behind the room.
        """
        for i in range(count):
            self.drafted[f"__unknown_{len(self.drafted)}_{i}"] = "other"
        self._persist()

    def undo_skip(self) -> None:
        unknown = [k for k in self.drafted if k.startswith("__unknown_")]
        if unknown:
            self.drafted.pop(unknown[-1])
            self._persist()

    def undo(self, player_id: str) -> None:
        self.drafted.pop(player_id, None)
        self._persist()

    def find(self, query: str, limit: int = 8) -> pl.DataFrame:
        # literal=True is not a detail: str.contains defaults to regex, so a
        # stray "(" raised mid-draft and killed the request rather than simply
        # matching nothing.
        q = query.strip().lower()
        return (
            self.available.filter(pl.col("search_key").str.contains(q, literal=True))
            .sort("adp")
            .head(limit)
        )

    # --- recommendation ---------------------------------------------------
    def recommend(self, n: int = 12, n_sims: int = 3000) -> dict:
        board = self.available
        roster = self.my_roster
        gap = max(self.picks_until_next(), 1)

        table = vona_table(
            board, roster, gap, self.league, n_sims=n_sims,
            rng=np.random.default_rng(7),
        )
        dropoff = position_dropoff(table)

        # Tier scarcity: how many players remain in each player's own tier.
        left = board.group_by(["pos", "tier"]).len().rename({"len": "tier_left"})
        table = table.join(left, on=["pos", "tier"], how="left")

        # Bye collisions: how many players already on our roster share this
        # candidate's bye. Stacking six starters on one week costs roughly 17
        # points over a season -- worth seeing, not worth reaching for.
        my_byes = [
            r["bye"] for r in self.players
            .filter(pl.col("player_id").is_in([p["player_id"] for p in roster] or [""]))
            .select("bye").to_dicts()
            if r.get("bye") is not None
        ] if roster else []
        counts = {b: my_byes.count(b) for b in set(my_byes)}
        table = table.with_columns(
            pl.col("bye").replace_strict(counts, default=0).alias("bye_conflicts")
        )

        return {
            "on_the_clock": self.pick_number,
            "my_next_pick": self.next_pick(),
            "picks_until_next": gap,
            "roster": roster,
            "replacement": {k: round(v, 1) for k, v in replacement_levels(board, self.league).items()},
            "dropoff": dropoff.to_dicts(),
            "recommendations": table.select(
                ["player_id", "name", "pos", "tm", "adp", "proj_points", "sd",
                 "tier", "tier_left", "marginal", "next_best", "vona",
                 "p_survives", "bye", "bye_conflicts", "playoff_lift"]
            ).head(n).to_dicts(),
        }
