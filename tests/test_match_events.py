"""Unit tests for src.data.match_events."""

import polars as pl
import pytest

from src.data.match_events import (
    derive_event_date_from_odds,
    match_odds_to_games,
    validate_matched_odds,
)
from src.features.market_features import add_moneyline_market_features


@pytest.fixture()
def games() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id": ["0022300001", "0022300002"],
            "season": ["2023-24", "2023-24"],
            "game_date": ["2023-10-24", "2023-10-25"],
            "home_team": ["BOS", "GSW"],
            "away_team": ["NYK", "PHX"],
            "home_score": [108, 99],
            "away_score": [104, 112],
            "home_win": [1, 0],
        }
    )


def odds_rows(**overrides) -> pl.DataFrame:
    """Two odds rows (one per game), captured the day before each game."""
    base = {
        "provider_event_id": ["evt1", "evt2"],
        "game_id": [None, None],
        "sportsbook": ["draftkings", "draftkings"],
        "market": ["h2h", "h2h"],
        "timestamp": ["2023-10-23T18:00:00", "2023-10-24T18:00:00"],
        "home_team": ["Boston Celtics", "GS"],
        "away_team": ["New York Knicks", "Phoenix Suns"],
        "home_american_odds": [-150, 120],
        "away_american_odds": [130, -140],
        "is_opening_line": [True, True],
        "is_closing_line": [False, False],
        "commence_time": ["2023-10-24T23:10:00", "2023-10-25T02:10:00"],
    }
    base.update(overrides)
    return pl.DataFrame(base, schema_overrides={"game_id": pl.String})


class TestDeriveEventDate:
    def test_uses_existing_event_date(self) -> None:
        odds = odds_rows().drop("commence_time").with_columns(
            pl.Series("event_date", ["2023-10-24", "2023-10-25"])
        )
        result = derive_event_date_from_odds(odds)
        assert result.schema["event_date"] == pl.Date

    def test_derives_from_commence_time(self) -> None:
        result = derive_event_date_from_odds(odds_rows())
        assert result.schema["event_date"] == pl.Date
        assert str(result.get_column("event_date")[0]) == "2023-10-24"

    def test_derives_from_event_datetime(self) -> None:
        odds = odds_rows().rename({"commence_time": "event_datetime"})
        result = derive_event_date_from_odds(odds)
        assert result.schema["event_date"] == pl.Date

    def test_no_event_date_source_raises(self) -> None:
        odds = odds_rows().drop("commence_time")
        with pytest.raises(ValueError, match="cannot be used as the event date"):
            derive_event_date_from_odds(odds)

    def test_snapshot_timestamp_not_used_as_fallback(self) -> None:
        # timestamp present but no event-date column: must raise, never
        # silently fall back to the capture timestamp.
        odds = odds_rows().drop("commence_time")
        assert "timestamp" in odds.columns
        with pytest.raises(ValueError, match="snapshot 'timestamp'"):
            derive_event_date_from_odds(odds)


class TestMatchOddsToGames:
    def test_populates_internal_game_id(self, games: pl.DataFrame) -> None:
        matched = match_odds_to_games(odds_rows(), games)
        assert matched.height == 2
        by_event = {
            r["provider_event_id"]: r["game_id"]
            for r in matched.iter_rows(named=True)
        }
        assert by_event == {"evt1": "0022300001", "evt2": "0022300002"}

    def test_provider_event_id_preserved(self, games: pl.DataFrame) -> None:
        matched = match_odds_to_games(odds_rows(), games)
        assert sorted(matched.get_column("provider_event_id").to_list()) == [
            "evt1",
            "evt2",
        ]

    def test_teams_canonicalized_before_matching(self, games: pl.DataFrame) -> None:
        # Odds use 'Boston Celtics'/'GS' full/alias forms; games use BOS/GSW.
        matched = match_odds_to_games(odds_rows(), games)
        assert matched.get_column("game_id").null_count() == 0
        assert set(matched.get_column("home_team").to_list()) == {"BOS", "GSW"}

    def test_unmatched_row_raises(self, games: pl.DataFrame) -> None:
        # Wrong date: commence_time a week later than any game.
        odds = odds_rows(
            commence_time=["2023-11-01T23:10:00", "2023-10-25T02:10:00"]
        )
        with pytest.raises(ValueError, match="failed to match any game"):
            match_odds_to_games(odds, games)

    def test_duplicate_odds_rows_raise(self, games: pl.DataFrame) -> None:
        odds = pl.concat([odds_rows(), odds_rows().head(1)])
        with pytest.raises(ValueError, match="duplicate key"):
            match_odds_to_games(odds, games)

    def test_multiple_game_matches_raise(self, games: pl.DataFrame) -> None:
        # Two games with identical date/teams but different IDs.
        dup_games = pl.concat(
            [games, games.head(1).with_columns(pl.lit("0022399999").alias("game_id"))]
        )
        with pytest.raises(ValueError, match="match multiple games"):
            match_odds_to_games(odds_rows(), dup_games)

    def test_missing_event_date_column_raises(self, games: pl.DataFrame) -> None:
        with pytest.raises(ValueError, match="event_date, commence_time"):
            match_odds_to_games(odds_rows().drop("commence_time"), games)

    def test_missing_required_columns_raise(self, games: pl.DataFrame) -> None:
        with pytest.raises(ValueError, match="odds is missing"):
            match_odds_to_games(odds_rows().drop("sportsbook"), games)
        with pytest.raises(ValueError, match="games is missing"):
            match_odds_to_games(odds_rows(), games.drop("game_date"))


class TestValidateMatchedOdds:
    def test_valid_matched_odds_pass(self, games: pl.DataFrame) -> None:
        matched = match_odds_to_games(odds_rows(), games)
        validate_matched_odds(matched)  # no raise

    def test_null_game_id_rejected(self, games: pl.DataFrame) -> None:
        matched = match_odds_to_games(odds_rows(), games).with_columns(
            pl.lit(None, dtype=pl.String).alias("game_id")
        )
        with pytest.raises(ValueError, match="'game_id' contains"):
            validate_matched_odds(matched)

    def test_duplicate_matched_key_rejected(self, games: pl.DataFrame) -> None:
        matched = match_odds_to_games(odds_rows(), games)
        with pytest.raises(ValueError, match="duplicate key"):
            validate_matched_odds(pl.concat([matched, matched.head(1)]))

    def test_invalid_odds_rejected(self, games: pl.DataFrame) -> None:
        matched = match_odds_to_games(odds_rows(), games).with_columns(
            pl.lit(50).alias("home_american_odds")
        )
        with pytest.raises(ValueError, match="impossible American odds"):
            validate_matched_odds(matched)


class TestDownstreamCompatibility:
    def test_output_feeds_market_features(self, games: pl.DataFrame) -> None:
        matched = match_odds_to_games(odds_rows(), games)
        validate_matched_odds(matched)
        features = add_moneyline_market_features(matched)
        assert features.height == matched.height
        assert "home_fair_market_prob" in features.columns
        # game_id present and populated for the downstream join to games.
        assert features.get_column("game_id").null_count() == 0
