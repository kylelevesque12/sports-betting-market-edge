"""Team name normalization to canonical NBA abbreviations.

Milestone M2 of docs/research_plan.md. Real game and odds sources spell
teams differently ("Golden State Warriors", "GS", "GSW", "Phoenix"/"PHO");
merging on raw names silently drops games. Everything is mapped to one
canonical abbreviation per franchise before any join.

Conventions and known aliases are documented in docs/team_mapping.md.
Historical relocations/renames (e.g. SEA->OKC, NJN->BKN) are deferred until
the real data window requires them.
"""

import polars as pl

from src.data.schema_validation import validate_required_columns

CANONICAL_TEAM_IDS: tuple[str, ...] = (
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
)

_FULL_NAMES: dict[str, str] = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "LA Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

# Common sportsbook / data-source abbreviations that differ from canonical.
_EXTRA_ALIASES: dict[str, str] = {
    "GS": "GSW",     # common sportsbook form
    "SA": "SAS",
    "NO": "NOP",
    "NY": "NYK",
    "BRK": "BKN",    # basketball-reference
    "CHO": "CHA",    # basketball-reference
    "PHO": "PHX",    # basketball-reference / older sources
    "UTAH": "UTA",   # ESPN
    "WSH": "WAS",    # ESPN
    "Golden State": "GSW",
    "Okla City": "OKC",  # sportsbookreviewsonline form
}


def _normalize(raw: str) -> str:
    """Lowercase, strip, collapse whitespace, and drop periods."""
    return " ".join(raw.replace(".", "").lower().split())


def _build_alias_table() -> dict[str, str]:
    table: dict[str, str] = {}
    for canonical in CANONICAL_TEAM_IDS:
        table[_normalize(canonical)] = canonical
    for name, canonical in {**_FULL_NAMES, **_EXTRA_ALIASES}.items():
        table[_normalize(name)] = canonical
    return table


_ALIAS_TABLE: dict[str, str] = _build_alias_table()


def canonicalize_team_name(team: str) -> str:
    """Map a team name or abbreviation to its canonical NBA abbreviation.

    Matching is case-insensitive, ignores leading/trailing and repeated
    whitespace, and ignores periods (so ``"L.A. Lakers"`` matches).

    Args:
        team: Team name, abbreviation, or known alias.

    Returns:
        Canonical three-letter abbreviation (e.g. ``"GSW"``).

    Raises:
        ValueError: If ``team`` is None, empty, or not a known team.
    """
    if team is None:
        raise ValueError("team name is null.")
    if not isinstance(team, str) or not team.strip():
        raise ValueError(f"team name is empty or not a string: {team!r}")

    canonical = _ALIAS_TABLE.get(_normalize(team))
    if canonical is None:
        raise ValueError(
            f"unknown team name: {team!r}. Add an alias to "
            f"src/data/team_mapping.py if this is a legitimate new variant."
        )
    return canonical


def canonicalize_team_columns(
    df: pl.DataFrame,
    columns: list[str],
) -> pl.DataFrame:
    """Return ``df`` with each named column mapped to canonical abbreviations.

    Args:
        df: DataFrame containing team name columns.
        columns: Column names to canonicalize (e.g.
            ``["home_team", "away_team"]``).

    Returns:
        New DataFrame with the same column names, values canonicalized.

    Raises:
        ValueError: If a named column is missing, or any value is null,
            empty, or an unknown team (the error names the column).
    """
    validate_required_columns(df, columns, "dataframe")

    result = df
    for column in columns:
        unique_values = result.get_column(column).unique().to_list()
        mapping: dict[str, str] = {}
        for value in unique_values:
            try:
                mapping[value] = canonicalize_team_name(value)
            except ValueError as exc:
                raise ValueError(f"column {column!r}: {exc}") from exc
        result = result.with_columns(
            pl.col(column).replace_strict(mapping).alias(column)
        )
    return result


def validate_team_mapping_complete() -> None:
    """Verify the alias table covers all 30 NBA franchises consistently.

    Raises:
        ValueError: If any canonical ID is missing from the table, or a
            canonical abbreviation does not map to itself.
    """
    mapped_ids = set(_ALIAS_TABLE.values())
    missing = sorted(set(CANONICAL_TEAM_IDS) - mapped_ids)
    if missing:
        raise ValueError(f"team mapping is missing canonical IDs: {missing}")

    unexpected = sorted(mapped_ids - set(CANONICAL_TEAM_IDS))
    if unexpected:
        raise ValueError(
            f"team mapping contains non-canonical target IDs: {unexpected}"
        )

    hijacked = [
        canonical
        for canonical in CANONICAL_TEAM_IDS
        if _ALIAS_TABLE.get(_normalize(canonical)) != canonical
    ]
    if hijacked:
        raise ValueError(
            f"canonical IDs do not map to themselves: {hijacked}"
        )
