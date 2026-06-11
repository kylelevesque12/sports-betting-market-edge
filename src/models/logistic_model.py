"""Logistic regression win-probability model.

First trainable model in the v1 ladder (docs/tech_stack.md): market baseline
-> logistic regression -> regularized logistic regression -> calibration.
Polars is converted to numpy at this boundary, per the architecture rule.

Training data must come from a time-based split (src.models.time_split) —
never a random split — and features must be pre-game safe (e.g. the shifted
rolling features from src.features.rolling_stats).
"""

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _extract_features(df, feature_cols: list[str], frame_name: str) -> np.ndarray:
    """Convert the feature columns of a Polars or pandas DataFrame to numpy.

    Raises ValueError on missing columns or nulls/NaNs.
    """
    if isinstance(df, pl.DataFrame):
        columns = df.columns
    else:  # pandas or pandas-like
        columns = list(df.columns)

    missing = [col for col in feature_cols if col not in columns]
    if missing:
        raise ValueError(f"{frame_name} is missing feature columns: {missing}")

    if isinstance(df, pl.DataFrame):
        matrix = df.select(feature_cols).to_numpy().astype(float)
    else:
        matrix = df[feature_cols].to_numpy(dtype=float)

    if np.isnan(matrix).any():
        bad_cols = [
            col
            for col, has_nan in zip(feature_cols, np.isnan(matrix).any(axis=0))
            if has_nan
        ]
        raise ValueError(
            f"{frame_name} contains null values in feature columns: {bad_cols}"
        )
    return matrix


def _extract_target(df, target_col: str) -> np.ndarray:
    """Extract and validate a binary 0/1 target column."""
    if isinstance(df, pl.DataFrame):
        if target_col not in df.columns:
            raise ValueError(f"train_df is missing target column: {target_col!r}")
        target = df.get_column(target_col).to_numpy().astype(float)
    else:
        if target_col not in list(df.columns):
            raise ValueError(f"train_df is missing target column: {target_col!r}")
        target = df[target_col].to_numpy(dtype=float)

    if np.isnan(target).any():
        raise ValueError(f"target column {target_col!r} contains null values.")
    if not np.isin(target, (0.0, 1.0)).all():
        bad = sorted(set(target[~np.isin(target, (0.0, 1.0))].tolist()))
        raise ValueError(f"target column {target_col!r} must contain only 0 and 1, found: {bad}")
    return target.astype(int)


def train_logistic_model(
    train_df,
    feature_cols: list[str],
    target_col: str = "team_win",
    random_state: int = 42,
) -> Pipeline:
    """Fit a standardized logistic regression on precomputed features.

    Args:
        train_df: Polars or pandas DataFrame of training rows (from a
            time-based split).
        feature_cols: Names of pre-game feature columns to train on.
        target_col: Binary 0/1 outcome column. Defaults to ``team_win``.
        random_state: Seed for the LogisticRegression solver.

    Returns:
        Fitted sklearn Pipeline (StandardScaler -> LogisticRegression). The
        training feature order is stored on the pipeline as
        ``feature_cols_`` and enforced at prediction time.

    Raises:
        ValueError: If feature or target columns are missing, the target is
            not strictly 0/1, or features/target contain nulls.
    """
    features = _extract_features(train_df, feature_cols, "train_df")
    target = _extract_target(train_df, target_col)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(max_iter=1000, random_state=random_state),
            ),
        ]
    )
    model.fit(features, target)
    # The pipeline sees positional arrays, so column names are lost at the
    # numpy boundary. Persist the training order to guard predictions.
    model.feature_cols_ = list(feature_cols)
    return model


def predict_logistic_model(
    model: Pipeline,
    test_df,
    feature_cols: list[str],
) -> np.ndarray:
    """Predict win probabilities (class 1) for each row of ``test_df``.

    Args:
        model: Fitted pipeline from :func:`train_logistic_model`.
        test_df: Polars or pandas DataFrame of rows to score.
        feature_cols: The same feature columns used in training.

    Returns:
        1-D numpy array of probabilities in [0, 1], one per input row.

    Raises:
        ValueError: If feature columns are missing or contain nulls, or if
            ``feature_cols`` does not exactly match the order the model was
            trained with (positional arrays make a reordering silently wrong).
    """
    trained_cols = getattr(model, "feature_cols_", None)
    if trained_cols is not None and list(feature_cols) != trained_cols:
        raise ValueError(
            "Prediction feature columns must exactly match the training "
            f"feature order. Trained with {trained_cols}, got {list(feature_cols)}."
        )

    features = _extract_features(test_df, feature_cols, "test_df")
    return model.predict_proba(features)[:, 1]
