# Odds Ingestion (M4)

## Provider

v1 provider is **The Odds API**. Its `h2h` market key is the pre-game
moneyline — the only market in scope (research_plan.md, Section 2). The
historical endpoint returns point-in-time snapshots: a `data` list of events,
each with `home_team`, `away_team`, and per-bookmaker `markets` containing
`h2h` outcomes priced per team. Snapshots are requested with
`oddsFormat=american`; the normalizer rejects non-whole-number prices, which
catches accidentally-decimal snapshots immediately.

## commence_time is preserved

The clean schema carries the provider's `commence_time` (parsed as a true
Datetime), because M5 event matching requires the **event start time** — the
snapshot `timestamp` only records when odds were captured and cannot locate
the game on a calendar (see docs/event_matching.md). Events missing
`commence_time` are rejected at normalization rather than failing later at
matching. Note: The Odds API reports commence times in UTC; the timezone
handling caveat in docs/event_matching.md applies before date-based
matching against US game dates.

## provider_event_id vs. game_id

Provider event IDs do **not** match internal NBA game IDs. The clean odds
schema therefore carries `provider_event_id` from the provider, and `game_id`
is null at this stage by design. Event matching (provider event → NBA game,
by date + canonicalized home/away teams) is deferred to **M5**, where it can
be validated as its own milestone — the merge-validation rules in
research_plan.md Section 7 (every odds row maps to exactly one game) apply
there, not here.

## Opening vs. closing lines

A single snapshot says nothing about line position; opening/closing are
properties of a snapshot *series*. `mark_opening_and_closing_lines` groups by
(provider_event_id, sportsbook, market) and marks each group's earliest
timestamp as the opening line and latest as the closing line. A group with
one snapshot is both. Downstream, the research-plan timing rules apply:
closing lines are the benchmark; opening/earlier snapshots are the only
simulatable bet prices.

## Secrets

No API keys in code, ever. `fetch_the_odds_api_historical_snapshot` reads
`THE_ODDS_API_KEY` from the environment (e.g. via an uncommitted `.env`),
and raises if unset. Historical snapshots consume paid quota — every fetched
response is saved immediately via `save_raw_odds_json` to `data/raw/`
(immutable, per CLAUDE.md) so no quota is ever spent twice on the same data.

## Testing and the no-network rule

Unit tests use mocked raw dictionaries matching the provider shape — no
live API calls, no API key required to run the suite. The fetch helper is
the only networked function and is never invoked by tests.

## Fallback plan

Normalization is decoupled from fetching: any saved raw JSON snapshot with
the provider shape can be normalized without API access. If The Odds API is
unavailable or unfunded, a small set of saved snapshots (committed under
`data/external/` or supplied locally) drives the same pipeline — the
ingestion interface stays API-ready while the project runs on saved samples
(research_plan.md, Section 6 fallback).
