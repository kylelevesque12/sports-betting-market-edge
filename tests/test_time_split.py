"""Unit tests for src.models.time_split."""

import polars as pl
import pytest

from src.models.time_split import time_based_split

DATES = [
    "2025-11-01",
    "2025-11-03",
    "2025-11-05",
    "2025-11-08",
    "2025-11-10",
    "2025-11-12",
    "2025-11-15",
    "2025-11-17",
]


@pytest.fixture()
def string_df() -> pl.DataFrame:
    # Deliberately unsorted input to prove outputs are sorted by the split.
    return pl.DataFrame(
        {"game_date": list(reversed(DATES)), "value": list(range(len(DATES)))}
    )


@pytest.fixture()
def date_df(string_df: pl.DataFrame) -> pl.DataFrame:
    return string_df.with_columns(pl.col("game_date").str.to_date())


def split(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    return time_based_split(
        df,
        date_col="game_date",
        train_end_date="2025-11-08",
        test_start_date="2025-11-10",
        test_end_date="2025-11-15",
    )


class TestSplitBoundaries:
    def test_train_rows_on_or_before_train_end(self, string_df: pl.DataFrame) -> None:
        train, _ = split(string_df)
        assert train.height == 4
        assert max(train.get_column("game_date").to_list()) <= "2025-11-08"

    def test_test_rows_within_window_inclusive(self, string_df: pl.DataFrame) -> None:
        _, test = split(string_df)
        dates = test.get_column("game_date").to_list()
        assert dates == ["2025-11-10", "2025-11-12", "2025-11-15"]  # both ends inclusive

    def test_no_test_period_rows_in_train(self, string_df: pl.DataFrame) -> None:
        train, test = split(string_df)
        train_dates = set(train.get_column("game_date").to_list())
        test_dates = set(test.get_column("game_date").to_list())
        assert train_dates.isdisjoint(test_dates)
        assert max(train_dates) < min(test_dates)

    def test_gap_rows_excluded_entirely(self, string_df: pl.DataFrame) -> None:
        # 2025-11-17 is after the test window; it must appear nowhere.
        train, test = split(string_df)
        all_rows = train.height + test.height
        assert all_rows == 7  # 8 input rows minus the post-test-window row


class TestSorting:
    def test_outputs_sorted_by_date(self, string_df: pl.DataFrame) -> None:
        train, test = split(string_df)  # input was reverse-sorted
        for frame in (train, test):
            dates = frame.get_column("game_date").to_list()
            assert dates == sorted(dates)


class TestDtypes:
    def test_works_with_iso_strings(self, string_df: pl.DataFrame) -> None:
        train, test = split(string_df)
        assert train.height == 4
        assert test.height == 3

    def test_works_with_polars_date(self, date_df: pl.DataFrame) -> None:
        train, test = split(date_df)
        assert train.height == 4
        assert test.height == 3
        assert train.schema["game_date"] == pl.Date

    def test_string_and_date_columns_agree(
        self, string_df: pl.DataFrame, date_df: pl.DataFrame
    ) -> None:
        train_s, test_s = split(string_df)
        train_d, test_d = split(date_df)
        assert train_s.get_column("value").to_list() == train_d.get_column("value").to_list()
        assert test_s.get_column("value").to_list() == test_d.get_column("value").to_list()


class TestValidation:
    def test_missing_date_column_raises(self, string_df: pl.DataFrame) -> None:
        with pytest.raises(ValueError, match="not found"):
            time_based_split(
                string_df, "no_such_col", "2025-11-08", "2025-11-10", "2025-11-15"
            )

    def test_empty_train_raises(self, string_df: pl.DataFrame) -> None:
        with pytest.raises(ValueError, match="Training split is empty"):
            time_based_split(
                string_df, "game_date", "2025-10-01", "2025-11-10", "2025-11-15"
            )

    def test_empty_test_raises(self, string_df: pl.DataFrame) -> None:
        with pytest.raises(ValueError, match="Test split is empty"):
            time_based_split(
                string_df, "game_date", "2025-11-08", "2025-12-01", "2025-12-15"
            )

    def test_test_start_after_test_end_raises(self, string_df: pl.DataFrame) -> None:
        with pytest.raises(ValueError, match="after test_end_date"):
            time_based_split(
                string_df, "game_date", "2025-11-08", "2025-11-15", "2025-11-10"
            )

    @pytest.mark.parametrize("train_end", ["2025-11-10", "2025-11-12"])
    def test_train_end_on_or_after_test_start_raises(
        self, string_df: pl.DataFrame, train_end: str
    ) -> None:
        with pytest.raises(ValueError, match="strictly before"):
            time_based_split(
                string_df, "game_date", train_end, "2025-11-10", "2025-11-15"
            )

    def test_non_iso_boundary_raises(self, string_df: pl.DataFrame) -> None:
        with pytest.raises(ValueError, match="ISO date string"):
            time_based_split(
                string_df, "game_date", "11/08/2025", "2025-11-10", "2025-11-15"
            )

    def test_unsupported_dtype_raises(self, string_df: pl.DataFrame) -> None:
        bad = string_df.with_columns(pl.lit(1).alias("game_date"))
        with pytest.raises(ValueError, match="Date or ISO string"):
            time_based_split(
                bad, "game_date", "2025-11-08", "2025-11-10", "2025-11-15"
            )
