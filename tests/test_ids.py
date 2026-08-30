"""Identity resolution guards.

The live-data test is the important one: if the crosswalk or ADP source shifts
under us, an unresolved star player silently disappears from the draft board.
That must fail the build, not degrade quietly.
"""

import polars as pl
import pytest

from src.ingest.ids import normalise_name, normalise_position, normalise_team


def test_suffixes_are_stripped():
    assert normalise_name("Marvin Harrison Jr.") == "marvin harrison"
    assert normalise_name("Kenneth Walker III") == "kenneth walker"


def test_punctuation_and_accents_are_stripped():
    assert normalise_name("Ja'Marr Chase") == "jamarr chase"
    assert normalise_name("D.J. Moore") == "dj moore"


def test_nickname_aliases_are_applied():
    assert normalise_name("Chig Okonkwo") == "chigoziem okonkwo"
    assert normalise_name("Kenny Gainwell") == "kenneth gainwell"


def test_rams_normalise_to_nflverse_convention():
    # nflverse uses LA; FFC uses LAR; the crosswalk uses RAM.
    assert normalise_team("LAR") == "LA"
    assert normalise_team("RAM") == "LA"
    assert normalise_team("LA") == "LA"


def test_team_abbreviations_are_unified():
    assert normalise_team("KCC") == "KC"
    assert normalise_team("JAC") == "JAX"
    assert normalise_team("LVR") == "LV"
    assert normalise_team("") == "FA"


def test_ffc_position_labels_are_mapped():
    assert normalise_position("DEF") == "DST"
    assert normalise_position("PK") == "K"


@pytest.mark.slow
def test_every_skill_player_in_live_adp_resolves():
    from src.ingest.adp import load_adp
    from src.ingest.ids import resolve, unresolved

    resolved = resolve(load_adp())
    skill = resolved.filter(pl.col("pos").is_in(["QB", "RB", "WR", "TE"]))
    missing = unresolved(skill)
    names = missing["name"].to_list()
    assert not names, f"unresolved skill players: {names}"


@pytest.mark.slow
def test_no_two_players_share_a_gsis_id():
    """A duplicated id means two different players collapsed into one."""
    from src.ingest.adp import load_adp
    from src.ingest.ids import resolve

    resolved = resolve(load_adp()).filter(pl.col("gsis_id").is_not_null())
    dupes = resolved.group_by("gsis_id").len().filter(pl.col("len") > 1)
    assert dupes.height == 0, f"duplicate gsis ids: {dupes.to_dicts()}"
