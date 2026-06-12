"""Unit tests for src.data.team_mapping."""

import polars as pl
import pytest

from src.data.team_mapping import (
    CANONICAL_TEAM_IDS,
    canonicalize_team_columns,
    canonicalize_team_name,
    validate_team_mapping_complete,
)


class TestCanonicalizeTeamName:
    def test_canonical_abbreviations_pass_through(self) -> None:
        for team_id in CANONICAL_TEAM_IDS:
            assert canonicalize_team_name(team_id) == team_id

    @pytest.mark.parametrize(
        ("full_name", "expected"),
        [
            ("Boston Celtics", "BOS"),
            ("Golden State Warriors", "GSW"),
            ("Philadelphia 76ers", "PHI"),
            ("Portland Trail Blazers", "POR"),
            ("LA Clippers", "LAC"),
            ("Los Angeles Clippers", "LAC"),
            ("New Orleans Pelicans", "NOP"),
        ],
    )
    def test_full_names_map(self, full_name: str, expected: str) -> None:
        assert canonicalize_team_name(full_name) == expected

    @pytest.mark.parametrize(
        "variant", ["boston celtics", "BOSTON CELTICS", "Boston celtics", "bos"]
    )
    def test_case_insensitive(self, variant: str) -> None:
        assert canonicalize_team_name(variant) == "BOS"

    @pytest.mark.parametrize(
        "variant", ["  BOS  ", "\tBoston Celtics\n", " boston   celtics "]
    )
    def test_whitespace_ignored(self, variant: str) -> None:
        assert canonicalize_team_name(variant) == "BOS"

    @pytest.mark.parametrize(
        ("alias", "expected"),
        [
            ("GS", "GSW"),
            ("SA", "SAS"),
            ("NO", "NOP"),
            ("NY", "NYK"),
            ("BRK", "BKN"),
            ("CHO", "CHA"),
            ("PHO", "PHX"),
            ("UTAH", "UTA"),
            ("WSH", "WAS"),
            ("L.A. Lakers", "LAL"),
        ],
    )
    def test_common_aliases_map(self, alias: str, expected: str) -> None:
        assert canonicalize_team_name(alias) == expected

    def test_unknown_team_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown team name"):
            canonicalize_team_name("Seattle SuperSonics")  # deferred relocation

    def test_null_raises(self) -> None:
        with pytest.raises(ValueError, match="null"):
            canonicalize_team_name(None)

    @pytest.mark.parametrize("empty", ["", "   ", "\t"])
    def test_empty_raises(self, empty: str) -> None:
        with pytest.raises(ValueError, match="empty"):
            canonicalize_team_name(empty)


class TestCanonicalizeTeamColumns:
    def test_single_column(self) -> None:
        df = pl.DataFrame({"team": ["Boston Celtics", "GS", "phx"]})
        result = canonicalize_team_columns(df, ["team"])
        assert result.get_column("team").to_list() == ["BOS", "GSW", "PHX"]

    def test_multiple_columns(self) -> None:
        df = pl.DataFrame(
            {
                "home_team": ["Boston Celtics", "NY"],
                "away_team": ["SA", "Miami Heat"],
                "other": [1, 2],
            }
        )
        result = canonicalize_team_columns(df, ["home_team", "away_team"])
        assert result.get_column("home_team").to_list() == ["BOS", "NYK"]
        assert result.get_column("away_team").to_list() == ["SAS", "MIA"]
        assert result.get_column("other").to_list() == [1, 2]  # untouched

    def test_missing_column_raises(self) -> None:
        df = pl.DataFrame({"team": ["BOS"]})
        with pytest.raises(ValueError, match="missing required columns"):
            canonicalize_team_columns(df, ["home_team"])

    def test_unknown_team_in_dataframe_raises_naming_column(self) -> None:
        df = pl.DataFrame({"home_team": ["BOS", "Vancouver Grizzlies"]})
        with pytest.raises(ValueError, match="'home_team'.*unknown team"):
            canonicalize_team_columns(df, ["home_team"])

    def test_null_in_dataframe_raises(self) -> None:
        df = pl.DataFrame({"home_team": ["BOS", None]})
        with pytest.raises(ValueError, match="'home_team'.*null"):
            canonicalize_team_columns(df, ["home_team"])

    def test_row_count_preserved(self) -> None:
        df = pl.DataFrame({"team": ["BOS"] * 5 + ["GS"] * 5})
        result = canonicalize_team_columns(df, ["team"])
        assert result.height == 10


class TestMappingCompleteness:
    def test_mapping_is_complete(self) -> None:
        validate_team_mapping_complete()  # no raise

    def test_thirty_canonical_ids(self) -> None:
        assert len(CANONICAL_TEAM_IDS) == 30
        assert len(set(CANONICAL_TEAM_IDS)) == 30

    def test_every_full_name_maps_to_distinct_franchise(self) -> None:
        # All 30 franchises reachable via at least one full name.
        from src.data.team_mapping import _FULL_NAMES

        assert set(_FULL_NAMES.values()) == set(CANONICAL_TEAM_IDS)
