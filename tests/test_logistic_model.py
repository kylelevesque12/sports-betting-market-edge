"""Unit tests for src.models.logistic_model."""

import numpy as np
import polars as pl
import pytest

from src.models.logistic_model import predict_logistic_model, train_logistic_model

FEATURES = ["rolling_win_pct_3", "rolling_point_diff_3"]


def synthetic_data(n: int = 40, seed: int = 7) -> dict[str, list]:
    """Linearly separable-ish synthetic team-game rows."""
    rng = np.random.default_rng(seed)
    win_pct = rng.uniform(0, 1, n)
    point_diff = rng.uniform(-10, 10, n)
    # Outcome driven by features plus noise — learnable but not trivial.
    logits = 3 * (win_pct - 0.5) + 0.3 * point_diff + rng.normal(0, 0.5, n)
    wins = (logits > 0).astype(int)
    return {
        "rolling_win_pct_3": win_pct.tolist(),
        "rolling_point_diff_3": point_diff.tolist(),
        "team_win": wins.tolist(),
    }


@pytest.fixture()
def train_pl() -> pl.DataFrame:
    return pl.DataFrame(synthetic_data())


@pytest.fixture()
def test_pl() -> pl.DataFrame:
    return pl.DataFrame(synthetic_data(n=15, seed=11))


class TestTraining:
    def test_trains_on_synthetic_data(self, train_pl: pl.DataFrame) -> None:
        model = train_logistic_model(train_pl, FEATURES)
        assert hasattr(model, "predict_proba")
        # Steps as specified: StandardScaler then LogisticRegression.
        assert [name for name, _ in model.steps] == ["scaler", "logistic"]

    def test_learns_signal(self, train_pl: pl.DataFrame, test_pl: pl.DataFrame) -> None:
        model = train_logistic_model(train_pl, FEATURES)
        probs = predict_logistic_model(model, test_pl, FEATURES)
        accuracy = np.mean((probs >= 0.5) == test_pl.get_column("team_win").to_numpy())
        assert accuracy > 0.6  # better than chance on learnable synthetic data

    def test_reproducible_with_random_state(self, train_pl: pl.DataFrame) -> None:
        m1 = train_logistic_model(train_pl, FEATURES, random_state=42)
        m2 = train_logistic_model(train_pl, FEATURES, random_state=42)
        np.testing.assert_allclose(
            m1.named_steps["logistic"].coef_, m2.named_steps["logistic"].coef_
        )


class TestPredictions:
    def test_correct_length(self, train_pl: pl.DataFrame, test_pl: pl.DataFrame) -> None:
        model = train_logistic_model(train_pl, FEATURES)
        probs = predict_logistic_model(model, test_pl, FEATURES)
        assert probs.shape == (test_pl.height,)

    def test_probabilities_in_range(
        self, train_pl: pl.DataFrame, test_pl: pl.DataFrame
    ) -> None:
        model = train_logistic_model(train_pl, FEATURES)
        probs = predict_logistic_model(model, test_pl, FEATURES)
        assert ((probs >= 0) & (probs <= 1)).all()


class TestTrainingValidation:
    def test_missing_feature_column_raises(self, train_pl: pl.DataFrame) -> None:
        with pytest.raises(ValueError, match="missing feature columns"):
            train_logistic_model(train_pl, FEATURES + ["no_such_feature"])

    def test_missing_target_column_raises(self, train_pl: pl.DataFrame) -> None:
        with pytest.raises(ValueError, match="missing target column"):
            train_logistic_model(train_pl, FEATURES, target_col="no_such_target")

    def test_non_binary_target_raises(self, train_pl: pl.DataFrame) -> None:
        bad = train_pl.with_columns(
            (pl.col("team_win") * 2).alias("team_win")  # 0/2 instead of 0/1
        )
        with pytest.raises(ValueError, match="only 0 and 1"):
            train_logistic_model(bad, FEATURES)

    def test_null_feature_raises(self, train_pl: pl.DataFrame) -> None:
        bad = train_pl.with_columns(
            pl.when(pl.arange(0, pl.len()) == 0)
            .then(None)
            .otherwise(pl.col("rolling_win_pct_3"))
            .alias("rolling_win_pct_3")
        )
        with pytest.raises(ValueError, match="null values in feature columns"):
            train_logistic_model(bad, FEATURES)

    def test_null_target_raises(self, train_pl: pl.DataFrame) -> None:
        bad = train_pl.with_columns(
            pl.when(pl.arange(0, pl.len()) == 0)
            .then(None)
            .otherwise(pl.col("team_win"))
            .alias("team_win")
        )
        with pytest.raises(ValueError, match="contains null values"):
            train_logistic_model(bad, FEATURES)


class TestPredictionValidation:
    def test_missing_feature_column_raises(
        self, train_pl: pl.DataFrame, test_pl: pl.DataFrame
    ) -> None:
        model = train_logistic_model(train_pl, FEATURES)
        with pytest.raises(ValueError, match="missing feature columns"):
            predict_logistic_model(model, test_pl.drop(FEATURES[0]), FEATURES)

    def test_null_feature_raises(
        self, train_pl: pl.DataFrame, test_pl: pl.DataFrame
    ) -> None:
        model = train_logistic_model(train_pl, FEATURES)
        bad = test_pl.with_columns(
            pl.when(pl.arange(0, pl.len()) == 0)
            .then(None)
            .otherwise(pl.col("rolling_point_diff_3"))
            .alias("rolling_point_diff_3")
        )
        with pytest.raises(ValueError, match="null values in feature columns"):
            predict_logistic_model(model, bad, FEATURES)


class TestFeatureOrderGuard:
    def test_same_order_works(self, train_pl: pl.DataFrame, test_pl: pl.DataFrame) -> None:
        model = train_logistic_model(train_pl, FEATURES)
        assert model.feature_cols_ == FEATURES
        probs = predict_logistic_model(model, test_pl, FEATURES)
        assert probs.shape == (test_pl.height,)

    def test_reordered_columns_raise(
        self, train_pl: pl.DataFrame, test_pl: pl.DataFrame
    ) -> None:
        model = train_logistic_model(train_pl, FEATURES)
        with pytest.raises(ValueError, match="match the training feature order"):
            predict_logistic_model(model, test_pl, list(reversed(FEATURES)))

    def test_missing_feature_column_raises(
        self, train_pl: pl.DataFrame, test_pl: pl.DataFrame
    ) -> None:
        model = train_logistic_model(train_pl, FEATURES)
        with pytest.raises(ValueError, match="match the training feature order"):
            predict_logistic_model(model, test_pl, FEATURES[:1])

    def test_extra_feature_column_raises(
        self, train_pl: pl.DataFrame, test_pl: pl.DataFrame
    ) -> None:
        model = train_logistic_model(train_pl, FEATURES)
        with pytest.raises(ValueError, match="match the training feature order"):
            predict_logistic_model(model, test_pl, FEATURES + ["team_win"])


class TestPandasSupport:
    def test_works_with_pandas(self) -> None:
        pd = pytest.importorskip("pandas")
        train_pd = pd.DataFrame(synthetic_data())
        test_pd = pd.DataFrame(synthetic_data(n=15, seed=11))
        model = train_logistic_model(train_pd, FEATURES)
        probs = predict_logistic_model(model, test_pd, FEATURES)
        assert probs.shape == (len(test_pd),)
        assert ((probs >= 0) & (probs <= 1)).all()

    def test_pandas_and_polars_agree(self, train_pl: pl.DataFrame) -> None:
        pd = pytest.importorskip("pandas")
        data = synthetic_data()
        m_pl = train_logistic_model(pl.DataFrame(data), FEATURES)
        m_pd = train_logistic_model(pd.DataFrame(data), FEATURES)
        np.testing.assert_allclose(
            m_pl.named_steps["logistic"].coef_, m_pd.named_steps["logistic"].coef_
        )

    def test_pandas_null_feature_raises(self) -> None:
        pd = pytest.importorskip("pandas")
        data = synthetic_data()
        data["rolling_win_pct_3"][0] = None
        with pytest.raises(ValueError, match="null values in feature columns"):
            train_logistic_model(pd.DataFrame(data), FEATURES)
