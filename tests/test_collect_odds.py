"""Unit tests for src.data.collect_odds (mocked snapshots only, no network)."""

import json
from pathlib import Path

import polars as pl
import pytest

from src.data.collect_odds import (
    CLEAN_ODDS_COLUMNS,
    mark_opening_and_closing_lines,
    normalize_the_odds_api_h2h_snapshot,
    save_odds_parquet,
    save_raw_odds_json,
)

SNAPSHOT_TS = "2023-10-24T18:00:00"


def make_event(
    event_id: str = "evt1",
    home: str = "Boston Celtics",
    away: str = "New York Knicks",
    books: list[dict] | None = None,
) -> dict:
    if books is None:
        books = [
            make_bookmaker("draftkings", home, away, -150, 130),
            make_bookmaker("fanduel", home, away, -148, 128),
        ]
    return {
        "id": event_id,
        "home_team": home,
        "away_team": away,
        "commence_time": "2023-10-25T00:10:00Z",
        "bookmakers": books,
    }


def make_bookmaker(
    key: str, home: str, away: str, home_price: int, away_price: int
) -> dict:
    return {
        "key": key,
        "title": key.title(),
        "markets": [
            {
                "key": "h2h",
                "outcomes": [
                    {"name": home, "price": home_price},
                    {"name": away, "price": away_price},
                ],
            }
        ],
    }


def snapshot(events: list[dict]) -> dict:
    return {"timestamp": SNAPSHOT_TS, "data": events}


class TestNormalizeSnapshot:
    def test_required_columns(self) -> None:
        odds = normalize_the_odds_api_h2h_snapshot(snapshot([make_event()]), SNAPSHOT_TS)
        assert tuple(odds.columns) == CLEAN_ODDS_COLUMNS

    def test_one_row_per_event_bookmaker(self) -> None:
        events = [make_event("evt1"), make_event("evt2", "Miami Heat", "LA Clippers")]
        odds = normalize_the_odds_api_h2h_snapshot(snapshot(events), SNAPSHOT_TS)
        assert odds.height == 4  # 2 events x 2 books
        keys = odds.select(["provider_event_id", "sportsbook"]).unique()
        assert keys.height == 4

    def test_teams_canonicalized(self) -> None:
        odds = normalize_the_odds_api_h2h_snapshot(snapshot([make_event()]), SNAPSHOT_TS)
        assert odds.get_column("home_team").unique().to_list() == ["BOS"]
        assert odds.get_column("away_team").unique().to_list() == ["NYK"]

    def test_timestamp_parsed_to_datetime(self) -> None:
        odds = normalize_the_odds_api_h2h_snapshot(snapshot([make_event()]), SNAPSHOT_TS)
        assert isinstance(odds.schema["timestamp"], pl.Datetime)

    def test_home_away_odds_mapped_correctly(self) -> None:
        # Outcomes listed away-first to prove mapping is by name, not order.
        book = {
            "key": "draftkings",
            "markets": [
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": "New York Knicks", "price": 130},
                        {"name": "Boston Celtics", "price": -150},
                    ],
                }
            ],
        }
        odds = normalize_the_odds_api_h2h_snapshot(
            snapshot([make_event(books=[book])]), SNAPSHOT_TS
        )
        row = odds.row(0, named=True)
        assert row["home_american_odds"] == -150
        assert row["away_american_odds"] == 130

    def test_game_id_present_but_null(self) -> None:
        odds = normalize_the_odds_api_h2h_snapshot(snapshot([make_event()]), SNAPSHOT_TS)
        assert "game_id" in odds.columns
        assert odds.get_column("game_id").null_count() == odds.height

    def test_flags_default_false(self) -> None:
        odds = normalize_the_odds_api_h2h_snapshot(snapshot([make_event()]), SNAPSHOT_TS)
        assert not odds.get_column("is_opening_line").any()
        assert not odds.get_column("is_closing_line").any()

    def test_missing_teams_raise(self) -> None:
        event = make_event()
        del event["home_team"]
        with pytest.raises(ValueError, match="missing home/away teams"):
            normalize_the_odds_api_h2h_snapshot(snapshot([event]), SNAPSHOT_TS)

    def test_missing_h2h_market_raises(self) -> None:
        book = {"key": "draftkings", "markets": [{"key": "spreads", "outcomes": []}]}
        with pytest.raises(ValueError, match="market 'h2h' is missing"):
            normalize_the_odds_api_h2h_snapshot(
                snapshot([make_event(books=[book])]), SNAPSHOT_TS
            )

    def test_missing_home_outcome_raises(self) -> None:
        # Two outcomes, but neither is the home team.
        book = {
            "key": "draftkings",
            "markets": [
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": "New York Knicks", "price": 130},
                        {"name": "Miami Heat", "price": 500},
                    ],
                }
            ],
        }
        with pytest.raises(ValueError, match="no h2h outcome for home team"):
            normalize_the_odds_api_h2h_snapshot(
                snapshot([make_event(books=[book])]), SNAPSHOT_TS
            )

    def test_wrong_outcome_count_raises(self) -> None:
        # NBA h2h has exactly two outcomes; a third means corrupt data.
        book = {
            "key": "draftkings",
            "markets": [
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Boston Celtics", "price": -150},
                        {"name": "New York Knicks", "price": 130},
                        {"name": "Miami Heat", "price": 500},
                    ],
                }
            ],
        }
        with pytest.raises(ValueError, match="exactly 2 outcomes, got 3"):
            normalize_the_odds_api_h2h_snapshot(
                snapshot([make_event(books=[book])]), SNAPSHOT_TS
            )

    def test_invalid_american_odds_raise(self) -> None:
        book = make_bookmaker("draftkings", "Boston Celtics", "New York Knicks", -50, 130)
        with pytest.raises(ValueError, match="impossible American odds"):
            normalize_the_odds_api_h2h_snapshot(
                snapshot([make_event(books=[book])]), SNAPSHOT_TS
            )

    def test_decimal_format_prices_raise(self) -> None:
        # 1.91 is a decimal-odds price; whole-number American is required.
        book = make_bookmaker("draftkings", "Boston Celtics", "New York Knicks", 1.91, 130)
        with pytest.raises(ValueError, match="decimal format"):
            normalize_the_odds_api_h2h_snapshot(
                snapshot([make_event(books=[book])]), SNAPSHOT_TS
            )

    def test_wrong_snapshot_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="'data' key"):
            normalize_the_odds_api_h2h_snapshot({"events": []}, SNAPSHOT_TS)

    def test_unknown_team_raises(self) -> None:
        event = make_event(home="Seattle SuperSonics")
        with pytest.raises(ValueError, match="unknown team"):
            normalize_the_odds_api_h2h_snapshot(snapshot([event]), SNAPSHOT_TS)


class TestMarkOpeningAndClosing:
    def make_multi_snapshot_odds(self) -> pl.DataFrame:
        frames = [
            normalize_the_odds_api_h2h_snapshot(snapshot([make_event()]), ts)
            for ts in ("2023-10-24T10:00:00", "2023-10-24T15:00:00", "2023-10-24T23:30:00")
        ]
        return pl.concat(frames)

    def test_earliest_and_latest_marked(self) -> None:
        marked = mark_opening_and_closing_lines(self.make_multi_snapshot_odds())
        dk = marked.filter(pl.col("sportsbook") == "draftkings").sort("timestamp")
        assert dk.get_column("is_opening_line").to_list() == [True, False, False]
        assert dk.get_column("is_closing_line").to_list() == [False, False, True]

    def test_all_rows_preserved(self) -> None:
        odds = self.make_multi_snapshot_odds()
        assert mark_opening_and_closing_lines(odds).height == odds.height

    def test_single_snapshot_is_both(self) -> None:
        odds = normalize_the_odds_api_h2h_snapshot(snapshot([make_event()]), SNAPSHOT_TS)
        marked = mark_opening_and_closing_lines(odds)
        assert marked.get_column("is_opening_line").all()
        assert marked.get_column("is_closing_line").all()

    def test_groups_are_independent_per_book(self) -> None:
        # fanduel has only the middle snapshot -> both flags for fanduel there.
        odds = self.make_multi_snapshot_odds().filter(
            (pl.col("sportsbook") == "draftkings")
            | (pl.col("timestamp") == pl.lit("2023-10-24T15:00:00").str.to_datetime())
        )
        marked = mark_opening_and_closing_lines(odds)
        fd = marked.filter(pl.col("sportsbook") == "fanduel")
        assert fd.height == 1
        assert fd.get_column("is_opening_line").all()
        assert fd.get_column("is_closing_line").all()

    def test_missing_columns_raise(self) -> None:
        with pytest.raises(ValueError, match="missing required columns"):
            mark_opening_and_closing_lines(pl.DataFrame({"timestamp": ["x"]}))

    def test_null_timestamp_raises(self) -> None:
        odds = normalize_the_odds_api_h2h_snapshot(
            snapshot([make_event()]), SNAPSHOT_TS
        ).with_columns(pl.lit(None, dtype=pl.Datetime("us")).alias("timestamp"))
        with pytest.raises(ValueError, match="'timestamp' contains"):
            mark_opening_and_closing_lines(odds)


class TestSavers:
    def test_save_raw_json_and_refuse_overwrite(self, tmp_path: Path) -> None:
        raw = snapshot([make_event()])
        out = tmp_path / "raw" / "snap.json"
        save_raw_odds_json(raw, str(out))
        assert json.loads(out.read_text())["data"][0]["id"] == "evt1"
        with pytest.raises(FileExistsError, match="already exists"):
            save_raw_odds_json(raw, str(out))
        save_raw_odds_json({"data": []}, str(out), overwrite=True)  # explicit ok

    def test_save_parquet_and_refuse_overwrite(self, tmp_path: Path) -> None:
        odds = normalize_the_odds_api_h2h_snapshot(snapshot([make_event()]), SNAPSHOT_TS)
        out = tmp_path / "processed" / "odds.parquet"
        save_odds_parquet(odds, str(out))
        assert pl.read_parquet(out).equals(odds)
        with pytest.raises(FileExistsError, match="already exists"):
            save_odds_parquet(odds, str(out))
