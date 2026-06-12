# Event Matching (M5)

## Why provider_event_id is not game_id

Odds providers assign their own event identifiers (e.g. The Odds API's hex
event ids); the project's internal `game_id` comes from NBA.com via M3
ingestion. The two namespaces share nothing. Matching therefore uses what
the datasets actually have in common: **the event's start date and the
canonicalized home/away teams**.

## Why the snapshot timestamp is not the game date

An odds row's `timestamp` records **when the price was captured**, not when
the game is played. An opening line for a Tuesday game is typically captured
Monday — matching on the snapshot timestamp would attach those odds to the
wrong day. For this reason `derive_event_date_from_odds` requires a true
event-start column and **never falls back to the snapshot timestamp**; if no
event-date column exists, it raises with an explanation rather than
guessing.

## Matching keys

A match requires all three, exactly:

- `event_date == game_date` (both as true `pl.Date`)
- canonical `home_team` equality
- canonical `away_team` equality

Both frames are canonicalized through `src/data/team_mapping.py` before
joining, so source spelling differences ("Boston Celtics", "GS") cannot
break matches. No fuzzy matching: near-misses are bugs to investigate, not
distances to tolerate.

## Event-date requirements

Accepted source columns, in priority order: `event_date` (parsed as Date),
`commence_time` (Datetime; date part used — The Odds API provides this), or
`event_datetime` (same handling). Note for late-night games: commence times
near midnight UTC can land on the next calendar day relative to US game
dates. If the selected provider reports UTC commence times, timezone
conversion to US/Eastern before date extraction belongs in the ingestion
layer — verify against known games during real-data validation.

## Duplicate detection

Two layers, both hard failures:

- **Input odds:** no duplicate (provider_event_id, sportsbook, market,
  timestamp) rows — the same price captured twice means a collection bug.
- **Matched odds:** no duplicate (game_id, sportsbook, market, timestamp)
  rows — the post-match analogue, enforced by `validate_matched_odds`.

A multi-game match (one odds row joining two games) is also a hard failure;
it indicates duplicate date/team keys in the games table.

## Unmatched events

Unmatched odds rows **raise, never drop**. Silent drops are the worst
failure mode in this pipeline — a backtest on quietly thinned data looks
healthier than it is. The error reports the first unmatched examples
(provider_event_id, event_date, teams) so the cause — date offset, team
alias gap, missing game — can be diagnosed and fixed at the source. If real
data eventually requires tolerating known-unmatchable events (e.g.
preseason games in the odds feed), that exclusion must be explicit,
documented, and counted — not a silent join artifact.

## Historical relocations

As with team mapping (docs/team_mapping.md), relocations and renames
(SEA→OKC, NJN→BKN, etc.) are deferred until the selected data window
requires them. Unknown historical names raise during canonicalization, so
an older window cannot silently mis-match.
