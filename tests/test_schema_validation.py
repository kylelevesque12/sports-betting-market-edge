"""Unit tests for src.data.schema_validation."""

import polars as pl
import pytest

from src.data.schema_validation import (
    parse_date_column,
    parse_datetime_column,
    validate_american_odds_column,
    validate_binary_column,
    validate_required_columns,
)


class TestValidateRequiredColumns:
    def test_all_present_passes(self) -> None:
        df = pl.DataFrame({"a": [1], "b": [2]})
        validate_required_columns(df, ["a", "b"], "games")  # no raise

    def test_missing_columns_raise_with_dataset_name(self) -> None:
        df = pl.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match=r"games is missing required columns.*'b'"):
            validate_required_columns(df, ["a", "b"], "games")

    def test_default_dataset_name(self) -> None:
        with pytest.raises(ValueError, match="dataset is missing"):
            validate_required_columns(pl.DataFrame({"a": [1]}), ["z"])


class TestParseDateColumn:
    def test_iso_strings_parse_to_date(self) -> None:
        df = pl.DataFrame({"game_date": ["2025-11-01", "2025-11-03"]})
        result = parse_date_column(df, "game_date")
        assert result.schema["game_date"] == pl.Date
        assert result.height == 2

    def test_existing_date_column_unchanged(self) -> None:
        df = pl.DataFrame({"game_date": ["2025-11-01"]}).with_columns(
            pl.col("game_date").str.to_date()
        )
        result = parse_date_column(df, "game_date")
        assert result.schema["game_date"] == pl.Date
        assert result.equals(df)

    def test_null_dates_raise(self) -> None:
        df = pl.DataFrame({"game_date": ["2025-11-01", None]})
        with pytest.raises(ValueError, match="null value"):
            parse_date_column(df, "game_date")

    def test_bad_date_strings_raise(self) -> None:
        df = pl.DataFrame({"game_date": ["2025-11-01", "11/03/2025"]})
        with pytest.raises(ValueError, match="non-parseable date"):
            parse_date_column(df, "game_date")

    def test_nonsense_date_raises(self) -> None:
        df = pl.DataFrame({"game_date": ["2025-13-45"]})
        with pytest.raises(ValueError, match="non-parseable date"):
            parse_date_column(df, "game_date")

    def test_missing_column_raises(self) -> None:
        with pytest.raises(ValueError, match="missing required columns"):
            parse_date_column(pl.DataFrame({"x": [1]}), "game_date")

    def test_unsupported_dtype_raises(self) -> None:
        df = pl.DataFrame({"game_date": [1, 2]})
        with pytest.raises(ValueError, match="Date or ISO date string"):
            parse_date_column(df, "game_date")


class TestParseDatetimeColumn:
    def test_iso_strings_parse_to_datetime(self) -> None:
        df = pl.DataFrame({"timestamp": ["2025-11-01T18:30:00", "2025-11-02T15:00:00"]})
        result = parse_datetime_column(df, "timestamp")
        assert isinstance(result.schema["timestamp"], pl.Datetime)

    def test_existing_datetime_column_unchanged(self) -> None:
        df = pl.DataFrame({"timestamp": ["2025-11-01T18:30:00"]}).with_columns(
            pl.col("timestamp").str.to_datetime()
        )
        result = parse_datetime_column(df, "timestamp")
        assert result.equals(df)

    def test_null_timestamps_raise(self) -> None:
        df = pl.DataFrame({"timestamp": ["2025-11-01T18:30:00", None]})
        with pytest.raises(ValueError, match="null value"):
            parse_datetime_column(df, "timestamp")

    def test_bad_timestamp_strings_raise(self) -> None:
        df = pl.DataFrame({"timestamp": ["not a timestamp"]})
        with pytest.raises(ValueError, match="non-parseable datetime"):
            parse_datetime_column(df, "timestamp")

    def test_unsupported_dtype_raises(self) -> None:
        df = pl.DataFrame({"timestamp": [1.5]})
        with pytest.raises(ValueError, match="Datetime or ISO datetime"):
            parse_datetime_column(df, "timestamp")


class TestValidateAmericanOddsColumn:
    def test_valid_odds_pass(self) -> None:
        df = pl.DataFrame({"odds": [-110, 100, -100, 150, -2000, 2000]})
        validate_american_odds_column(df, "odds")  # no raise

    def test_zero_odds_raise(self) -> None:
        df = pl.DataFrame({"odds": [-110, 0]})
        with pytest.raises(ValueError, match="American odds of 0"):
            validate_american_odds_column(df, "odds")

    @pytest.mark.parametrize("bad", [50, -50, 99, -99, 1])
    def test_odds_inside_open_interval_raise(self, bad: int) -> None:
        df = pl.DataFrame({"odds": [-110, bad]})
        with pytest.raises(ValueError, match="impossible American odds"):
            validate_american_odds_column(df, "odds")

    def test_boundary_values_pass(self) -> None:
        # Exactly -100 and +100 are valid (even money).
        validate_american_odds_column(pl.DataFrame({"odds": [-100, 100]}), "odds")

    def test_null_odds_raise(self) -> None:
        df = pl.DataFrame({"odds": [-110, None]})
        with pytest.raises(ValueError, match="null value"):
            validate_american_odds_column(df, "odds")

    def test_missing_column_raises(self) -> None:
        with pytest.raises(ValueError, match="missing required columns"):
            validate_american_odds_column(pl.DataFrame({"x": [1]}), "odds")


class TestValidateBinaryColumn:
    def test_zero_one_passes(self) -> None:
        validate_binary_column(pl.DataFrame({"home_win": [0, 1, 1, 0]}), "home_win")

    @pytest.mark.parametrize("bad", [2, -1])
    def test_invalid_values_raise(self, bad: int) -> None:
        df = pl.DataFrame({"home_win": [0, 1, bad]})
        with pytest.raises(ValueError, match="only 0 and 1"):
            validate_binary_column(df, "home_win")

    def test_nulls_raise(self) -> None:
        df = pl.DataFrame({"home_win": [0, None, 1]})
        with pytest.raises(ValueError, match="null value"):
            validate_binary_column(df, "home_win")

    def test_missing_column_raises(self) -> None:
        with pytest.raises(ValueError, match="missing required columns"):
            validate_binary_column(pl.DataFrame({"x": [1]}), "home_win")
