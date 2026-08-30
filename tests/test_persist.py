"""Draft state must survive the process dying.

This league allows three hours per pick, so a draft runs across days. Keeping
picks only in memory meant a closed terminal or a slept laptop discarded the
whole thing with no recovery -- the worst possible failure, at the worst possible
time, with no warning that it could happen.
"""

import json

import pytest

from src.draft import persist


@pytest.fixture
def temp_state(tmp_path, monkeypatch):
    monkeypatch.setattr(persist, "STATE_DIR", tmp_path / "draft_state")
    return tmp_path


def test_save_then_load_round_trips(temp_state):
    drafted = {"00-001": "me", "00-002": "other", "__unknown_2_0": "other"}
    persist.save(slot=8, season=2026, drafted=drafted)
    restored, meta = persist.load(slot=8, season=2026)
    assert restored == drafted
    assert meta["picks"] == 3


def test_missing_state_returns_empty(temp_state):
    restored, meta = persist.load(slot=8, season=2026)
    assert restored == {}
    assert meta == {}


def test_corrupt_state_does_not_raise(temp_state):
    persist.save(slot=8, season=2026, drafted={"00-001": "me"})
    path = persist.state_path(8, 2026)
    path.write_text("{ truncated", encoding="utf-8")
    restored, meta = persist.load(slot=8, season=2026)
    assert restored == {}, "a corrupt file must not crash a live draft"


def test_state_from_a_different_slot_is_refused(temp_state):
    """Pick numbers mean different things at different slots."""
    persist.save(slot=8, season=2026, drafted={"00-001": "me"})
    restored, meta = persist.load(slot=3, season=2026)
    assert restored == {}


def test_write_is_atomic_and_leaves_no_temp_files(temp_state):
    for i in range(5):
        persist.save(slot=8, season=2026, drafted={f"p{i}": "other"})
    leftovers = list(persist.STATE_DIR.glob("*.tmp"))
    assert not leftovers, f"temp files left behind: {leftovers}"


def test_clear_removes_state(temp_state):
    persist.save(slot=8, season=2026, drafted={"00-001": "me"})
    persist.clear(slot=8, season=2026)
    assert persist.load(slot=8, season=2026)[0] == {}


@pytest.mark.slow
def test_board_reloads_its_picks(temp_state):
    """The end-to-end guarantee: a new board picks up where the old one died."""
    from src.draft.board import DraftBoard

    board = DraftBoard(slot=8, resume=False)
    first = board.available.sort("adp")["player_id"][0]
    board.draft(first, "me")
    board.skip(3)
    expected = board.pick_number

    revived = DraftBoard(slot=8, resume=True)
    assert revived.pick_number == expected
    assert [r["player_id"] for r in revived.my_roster] == [first]


@pytest.mark.slow
def test_fresh_start_ignores_saved_state(temp_state):
    from src.draft.board import DraftBoard

    board = DraftBoard(slot=8, resume=False)
    board.skip(6)
    fresh = DraftBoard(slot=8, resume=False)
    assert fresh.pick_number == 1
