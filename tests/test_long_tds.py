"""Touchdowns banded by the length of the scoring play.

The one block of components this project derives itself rather than reading
from nflverse, which is why it carries its own reconciliation. The fast tests
pin the band vocabulary and prove the guard actually fires; the slow one does
the thing that matters -- checks the derivation against nflverse's own
touchdown totals over real seasons.
"""

import polars as pl
import pytest

from scripts.export_v2_bundle import (
    TD_TOTALS,
    _reconcile_long_tds,
    build_stat_components,
    ExportError,
)
from src.config import LONG_TD_COMPONENTS, TD_LENGTH_BANDS, td_band_key


def test_band_keys_match_the_kicker_convention():
    assert td_band_key("passing", 40, 49) == "passing_td_40_49"
    # An open-ended band keeps the trailing underscore, as `fg_made_60_` does.
    assert td_band_key("receiving", 50, None) == "receiving_td_50_"


def test_bands_cover_every_length_exactly_once():
    """Disjoint and gapless: every touchdown lands in one band and no other."""
    for yards in range(0, 120):
        matched = [
            (low, high)
            for low, high in TD_LENGTH_BANDS
            if yards >= low and (high is None or yards <= high)
        ]
        assert len(matched) == 1, f"{yards} yards matched {matched}"


def test_three_kinds_of_touchdown_are_banded():
    assert len(LONG_TD_COMPONENTS) == 3 * len(TD_LENGTH_BANDS)
    for kind in ("passing", "rushing", "receiving"):
        assert f"{kind}_td_50_" in LONG_TD_COMPONENTS


def _frame(**overrides) -> pl.DataFrame:
    row = {c: 0 for c in LONG_TD_COMPONENTS}
    row.update({total: 0 for total in TD_TOTALS.values()})
    row.update(overrides)
    return pl.DataFrame([row])


def test_reconciliation_passes_when_the_bands_add_up():
    _reconcile_long_tds(_frame(passing_td_50_=2, passing_td_0_9=1, passing_tds=3))


def test_reconciliation_fails_when_a_touchdown_goes_missing():
    """The failure a renamed play-by-play column would actually produce.

    Nothing else in the bundle would look wrong: the player still has his
    touchdowns, they just stop earning the bonus. So this has to raise.
    """
    with pytest.raises(ExportError, match="bands sum to 1 but nflverse counts 3"):
        _reconcile_long_tds(_frame(passing_td_50_=1, passing_tds=3))


@pytest.mark.slow
def test_derivation_reconciles_against_nflverse_over_real_seasons():
    """The real check: bands must sum to nflverse's own touchdown totals.

    `build_stat_components` reconciles internally and raises on a mismatch, so
    reaching the assertions at all is most of the test. Laterals are the case
    that made this necessary -- nflverse credits a touchdown scored after a
    lateral to the player who took it, and crediting the original receiver left
    the bands disagreeing with the receiving touchdowns they are a bonus on.
    """
    df = build_stat_components([2023, 2024])

    for kind, total in TD_TOTALS.items():
        banded = sum(
            df[td_band_key(kind, low, high)].sum() for low, high in TD_LENGTH_BANDS
        )
        assert banded == df[total].sum(), kind

    # And the bands carry real signal, not a column of zeros.
    assert df["receiving_td_50_"].sum() > 0
