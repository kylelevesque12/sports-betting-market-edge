"""Schema validation and parsing utilities for real-data ingestion.

Milestone M1 of docs/research_plan.md (Section 7, data validation rules).
These utilities run before real NBA game and odds data enter the pipeline,
addressing known real-data risks: string dates, null dates/timestamps,
impossible American odds, and malformed binary columns.
"""

import polars as pl

# Real American odds satisfy |odds| >= 100; anything strictly inside
# (-100, 100) is malformed source data (research_plan.md Section 7).
MIN_ABS_AMERICAN_ODDS = 100


def validate_required_columns(
    df: pl.DataFrame,
    required_columns: list[str],
    dataset_name: str = "dataset",
) -> None:
    """Raise if any required column is missing from ``df``.

    Args:
        df: DataFrame to check.
        required_columns: Column names that must be present.
        dataset_name: Name used in the error message.

    Raises:
        ValueError: Naming the dataset and the missing columns.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


def _reject_nulls(df: pl.DataFrame, column: str) -> None:
    null_count = df.get_column(column).null_count()
    if null_count > 0:
        raise ValueError(f"column {column!r} contains {null_count} null value(s).")


def parse_date_column(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """Return ``df`` with ``column`` parsed to a Polars ``Date``.

    ISO date strings (YYYY-MM-DD) are converted; an existing ``Date`` column
    is returned unchanged.

    Args:
        df: DataFrame containing the column.
        column: Name of the date column.

    Raises:
        ValueError: If the column is missing, contains nulls, contains
            non-parseable date strings, or has an unsupported dtype.
    """
    validate_required_columns(df, [column], "dataframe")
    _reject_nulls(df, column)

    dtype = df.schema[column]
    if dtype == pl.Date:
        return df
    if dtype in (pl.Utf8, pl.String):
        try:
            return df.with_columns(pl.col(column).str.to_date(strict=True))
        except pl.exceptions.PolarsError as exc:
            raise ValueError(
                f"column {column!r} contains non-parseable date strings: {exc}"
            ) from exc
    raise ValueError(
        f"column {column!r} must be a Date or ISO date string column, "
        f"got dtype {dtype}."
    )


def parse_datetime_column(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """Return ``df`` with ``column`` parsed to a Polars ``Datetime``.

    ISO datetime strings (e.g. ``2025-11-01T18:30:00``) are converted; an
    existing ``Datetime`` column is returned unchanged.

    Args:
        df: DataFrame containing the column.
        column: Name of the timestamp column.

    Raises:
        ValueError: If the column is missing, contains nulls, contains
            non-parseable datetime strings, or has an unsupported dtype.
    """
    validate_required_columns(df, [column], "dataframe")
    _reject_nulls(df, column)

    dtype = df.schema[column]
    if isinstance(dtype, pl.Datetime):
        return df
    if dtype in (pl.Utf8, pl.String):
        try:
            return df.with_columns(pl.col(column).str.to_datetime(strict=True))
        except pl.exceptions.PolarsError as exc:
            # Timezone-suffixed ISO strings (e.g. The Odds API's UTC "Z"
            # timestamps) need an explicit zone; parse them as UTC-aware.
            if "time zone" in str(exc):
                try:
                    return df.with_columns(
                        pl.col(column).str.to_datetime(strict=True, time_zone="UTC")
                    )
                except pl.exceptions.PolarsError as utc_exc:
                    raise ValueError(
                        f"column {column!r} contains non-parseable datetime "
                        f"strings: {utc_exc}"
                    ) from utc_exc
            raise ValueError(
                f"column {column!r} contains non-parseable datetime strings: {exc}"
            ) from exc
    raise ValueError(
        f"column {column!r} must be a Datetime or ISO datetime string column, "
        f"got dtype {dtype}."
    )


def validate_american_odds_column(df: pl.DataFrame, column: str) -> None:
    """Raise if ``column`` contains impossible American odds.

    Valid American odds satisfy ``odds <= -100`` or ``odds >= +100``.
    Zero and values strictly between -100 and 100 are malformed.

    Args:
        df: DataFrame containing the odds column.
        column: Name of the American odds column.

    Raises:
        ValueError: If the column is missing, contains nulls, contains 0,
            or contains values strictly between -100 and 100.
    """
    validate_required_columns(df, [column], "dataframe")
    _reject_nulls(df, column)

    odds = pl.col(column)
    zeros = df.filter(odds == 0)
    if zeros.height > 0:
        raise ValueError(
            f"column {column!r} contains {zeros.height} row(s) with American "
            f"odds of 0."
        )

    inside = df.filter(
        (odds > -MIN_ABS_AMERICAN_ODDS) & (odds < MIN_ABS_AMERICAN_ODDS)
    )
    if inside.height > 0:
        bad = sorted(inside.get_column(column).unique().to_list())
        raise ValueError(
            f"column {column!r} contains impossible American odds (must be "
            f"<= -100 or >= +100), found: {bad}"
        )


def validate_binary_column(df: pl.DataFrame, column: str) -> None:
    """Raise unless ``column`` contains only 0 and 1 with no nulls.

    Args:
        df: DataFrame containing the binary column.
        column: Name of the column to check.

    Raises:
        ValueError: If the column is missing, contains nulls, or contains
            values other than 0 and 1.
    """
    validate_required_columns(df, [column], "dataframe")
    _reject_nulls(df, column)

    bad_rows = df.filter(~pl.col(column).is_in([0, 1]))
    if bad_rows.height > 0:
        bad = sorted(bad_rows.get_column(column).unique().to_list())
        raise ValueError(
            f"column {column!r} must contain only 0 and 1, found: {bad}"
        )
