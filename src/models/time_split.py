"""Time-based train/test splitting for leakage-safe model evaluation.

CLAUDE.md rule: final evaluation always uses time-based splits, never random
ones. Random splits let a model train on games played *after* its test games,
which inflates apparent skill. This utility enforces a strict chronological
gap: training data must end before the test window begins.
"""

from datetime import date

import polars as pl


def _parse_iso(name: str, value: str) -> date:
    try:
        return date.fromisoformat(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"{name} must be an ISO date string (YYYY-MM-DD), got {value!r}."
        ) from exc


def time_based_split(
    df: pl.DataFrame,
    date_col: str,
    train_end_date: str,
    test_start_date: str,
    test_end_date: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split rows chronologically into train and test sets.

    Train contains rows with ``date_col <= train_end_date``; test contains
    rows with ``test_start_date <= date_col <= test_end_date`` (inclusive).
    No shuffling — both outputs are sorted by ``date_col``.

    Args:
        df: DataFrame to split.
        date_col: Name of the date column. May be an ISO date string column
            or a Polars ``Date`` column.
        train_end_date: Last date (inclusive) of the training period, ISO format.
        test_start_date: First date (inclusive) of the test period, ISO format.
        test_end_date: Last date (inclusive) of the test period, ISO format.

    Returns:
        ``(train, test)`` DataFrames, each sorted by ``date_col``.

    Raises:
        ValueError: If ``date_col`` is missing or an unsupported dtype, any
            boundary date is not ISO format, ``test_start_date`` is after
            ``test_end_date``, ``train_end_date`` is on or after
            ``test_start_date`` (no chronological separation), or either
            split is empty.
    """
    if date_col not in df.columns:
        raise ValueError(f"date column {date_col!r} not found in DataFrame.")

    train_end = _parse_iso("train_end_date", train_end_date)
    test_start = _parse_iso("test_start_date", test_start_date)
    test_end = _parse_iso("test_end_date", test_end_date)

    if test_start > test_end:
        raise ValueError(
            f"test_start_date ({test_start}) is after test_end_date ({test_end})."
        )
    if train_end >= test_start:
        raise ValueError(
            f"train_end_date ({train_end}) must be strictly before "
            f"test_start_date ({test_start}) for a clean chronological split."
        )

    dtype = df.schema[date_col]
    if dtype == pl.Date:
        train_end_lit: object = train_end
        test_start_lit: object = test_start
        test_end_lit: object = test_end
    elif dtype in (pl.Utf8, pl.String):
        # ISO strings compare correctly lexicographically.
        train_end_lit = train_end.isoformat()
        test_start_lit = test_start.isoformat()
        test_end_lit = test_end.isoformat()
    else:
        raise ValueError(
            f"date column {date_col!r} must be a Date or ISO string column, "
            f"got dtype {dtype}."
        )

    col = pl.col(date_col)
    train = df.filter(col <= train_end_lit).sort(date_col)
    test = df.filter((col >= test_start_lit) & (col <= test_end_lit)).sort(date_col)

    if train.height == 0:
        raise ValueError(f"Training split is empty (no rows on or before {train_end}).")
    if test.height == 0:
        raise ValueError(
            f"Test split is empty (no rows between {test_start} and {test_end})."
        )

    return train, test
