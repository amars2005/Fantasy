"""Draft-board state guards.

All of these construct the board with `resume=False`: the dashboard resumes a
saved draft by default, which is right for a live draft and wrong for a test that
asserts on pick numbers.

The bug these exist for: the board originally held only skill players, but about
28 kickers and defences come off the board in a 210-pick draft. A pick that
cannot be marked leaves the counter behind the room, and every "picks until my
next turn" number -- which is the entire basis of VONA's survival horizon --
drifts silently with it. Nothing errors; the recommendations just quietly become
answers to the wrong question.
"""

import polars as pl
import pytest


@pytest.mark.slow
def test_board_includes_kickers_and_defences():
    from src.draft.board import DraftBoard

    board = DraftBoard(slot=8, resume=False)
    positions = set(board.players["pos"].to_list())
    assert "K" in positions, "kickers missing: their picks cannot be marked"
    assert "DST" in positions, "defences missing: their picks cannot be marked"


@pytest.mark.slow
def test_every_position_has_a_replacement_level():
    from src.draft.board import DraftBoard
    from src.draft.replacement import replacement_levels

    board = DraftBoard(slot=8, resume=False)
    levels = replacement_levels(board.players)
    for pos in ("QB", "RB", "WR", "TE", "K", "DST"):
        assert pos in levels, f"no replacement level for {pos}"


@pytest.mark.slow
def test_skip_advances_the_clock_and_undoes():
    from src.draft.board import DraftBoard

    board = DraftBoard(slot=8, resume=False)
    start = board.pick_number
    board.skip(5)
    assert board.pick_number == start + 5
    board.undo_skip()
    assert board.pick_number == start + 4


@pytest.mark.slow
def test_skipped_picks_do_not_appear_as_drafted_players():
    """Placeholder picks advance the clock but must not pollute the roster."""
    from src.draft.board import DraftBoard

    board = DraftBoard(slot=8, resume=False)
    board.skip(3)
    assert board.my_roster == []
    # And they must not be removable from the player pool.
    assert board.available.height == board.players.height


@pytest.mark.slow
def test_on_the_clock_matches_our_slot_at_the_right_pick():
    from src.draft.board import DraftBoard

    board = DraftBoard(slot=8, resume=False)
    board.skip(7)
    assert board.pick_number == 8
    assert board.next_pick() == 8, "pick 8 belongs to slot 8 in round one"


@pytest.mark.slow
def test_bye_conflicts_are_flagged_after_drafting():
    """Stacking byes costs ~17 points a season; the board must surface it."""
    from src.draft.board import DraftBoard

    board = DraftBoard(slot=8, resume=False)
    first = board.recommend(n=1)["recommendations"][0]
    assert first["bye_conflicts"] == 0, "empty roster cannot have a bye conflict"

    board.draft(first["player_id"], "me")
    after = board.recommend(n=40)["recommendations"]
    same_bye = [r for r in after if r["bye"] == first["bye"]]
    assert same_bye, "expected some candidate sharing that bye"
    assert all(r["bye_conflicts"] >= 1 for r in same_bye)
    assert all(r["bye_conflicts"] == 0 for r in after if r["bye"] != first["bye"])


@pytest.mark.slow
def test_playoff_lift_is_attached_and_varies():
    """Weeks 15-17 matchup quality: real signal ADP cannot price."""
    from src.draft.board import DraftBoard

    board = DraftBoard(slot=8, resume=False)
    lifts = board.players["playoff_lift"].drop_nulls().to_list()
    assert lifts, "playoff_lift missing from the board"
    assert max(lifts) - min(lifts) > 2.0, "playoff lift should vary across teams"


@pytest.mark.slow
def test_search_survives_regex_special_characters():
    """A stray bracket must not kill the search request.

    `find` matched with polars `str.contains`, which is a regex by default, so
    typing "(" raised ComputeError inside the request thread. The dashboard's
    fetch then rejected with no response and the previous result list stayed on
    screen -- so Enter marked whoever was still displayed. A silent wrong pick.
    """
    from src.draft.board import DraftBoard

    board = DraftBoard(slot=8, resume=False)
    for query in ("(", "[", "a[", "*", "+", "?", ")("):
        rows = board.find(query)
        assert rows.height == 0, f"{query!r} should match no one, literally"


@pytest.mark.slow
def test_search_finds_defences_by_nickname():
    """The board announces "Ravens D/ST"; the feed names it "Baltimore Defense".

    K and DST are on the board so every pick can be marked and the counter stays
    with the room. A defence you cannot type is a pick you cannot mark.
    """
    from src.draft.board import DraftBoard

    board = DraftBoard(slot=8, resume=False)
    for nickname, team in (("ravens", "BAL"), ("49ers", "SF"), ("bills", "BUF")):
        rows = board.find(nickname)
        teams = rows.filter(pl.col("pos") == "DST")["tm"].to_list()
        assert team in teams, f"{nickname!r} did not find the {team} defence"


@pytest.mark.slow
def test_search_finds_players_by_team_abbreviation():
    """"TB" is deliberate: no player name contains it, so this cannot pass by
    accidentally matching a name substring the way "BUF" does in "Buffalo
    Defense"."""
    from src.draft.board import DraftBoard

    board = DraftBoard(slot=8, resume=False)
    rows = board.find("TB")
    assert rows.height > 0, "team abbreviation should match that team's players"
    assert all(t == "TB" for t in rows["tm"].to_list())


@pytest.mark.slow
def test_defence_nickname_puts_the_defence_first():
    """Enter marks the *first* row.

    Nobody types "ravens" to find Lamar Jackson -- a team nickname is how you
    reach a D/ST, which is the only roster slot identified by team rather than
    by a person's name. Surfacing nine team-mates above it would rebuild the
    silent wrong-pick this whole fix exists to remove.
    """
    from src.draft.board import DraftBoard

    board = DraftBoard(slot=8, resume=False)
    for nickname, team in (("ravens", "BAL"), ("niners", "SF"), ("bucs", "TB")):
        top = board.find(nickname).row(0, named=True)
        assert top["pos"] == "DST", f"{nickname!r} surfaced {top['name']} first"
        assert top["tm"] == team
