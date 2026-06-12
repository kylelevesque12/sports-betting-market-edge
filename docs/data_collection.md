# Real Data Collection

## NBA games (M7)

`scripts/collect_real_nba_games.py` collects regular-season game results via
`nba_api` (LeagueGameLog) for the initial window **2023-24 and 2024-25** —
deliberately small for the first real run; widen the `SEASONS` list when the
pipeline has proven itself on two seasons.

Flow per season: fetch raw team-game rows → save untouched to
`data/raw/nba_games/leaguegamelog_{season}.parquet` → normalize via
`normalize_nba_games` (pairing, canonical teams, true dates, tie rejection)
→ validate (string game_ids with leading zeros preserved, canonical teams,
binary home_win, no duplicate game_ids) → write
`data/processed/nba_games.parquet`.

Properties worth knowing:

- **Idempotent / quota-polite:** existing raw season files are loaded, not
  refetched. `--overwrite` refetches raw and replaces processed output.
- **Raw immutability:** raw files are never modified after saving
  (CLAUDE.md); normalization always reads from them.
- **Fail loud, never partial:** an nba_api failure aborts with a clear
  message; nothing partial is saved.
- **No secrets:** nba_api requires no API key. Odds collection (M4/M5
  scripts, later) reads THE_ODDS_API_KEY from the environment only.
- **stats.nba.com access:** NBA.com sometimes blocks cloud/datacenter IPs.
  If the fetch times out, run the script from a residential connection;
  raw files committed to a local machine can then be reused anywhere
  because of the load-don't-refetch behavior.

### Neutral-site games

Real data exposed a source quirk: NBA.com lists **both** teams with `@` for
neutral-site games (international games in Mexico City/Paris, NBA Cup
semifinals in Las Vegas) — no home team exists in the LeagueGameLog data.
The collection script excludes these games explicitly: they are counted,
their game_ids printed, and the exclusion appears in the summary (5 games
across 2023-24/2024-25, all in 2024-25). This follows the research plan's
exclusion policy — explicit, documented, counted, never silent. If
neutral-site games are ever needed, official home designations are
available from other nba_api endpoints.

## Odds sample (M8)

`scripts/collect_real_odds_sample.py` is a **tiny** connectivity/key/shape
test against The Odds API's historical endpoint — not a historical pull.
It requests two h2h snapshots (afternoon + near-tip UTC) for the busiest
game date in the collected games table, limited to two bookmakers
(draftkings, fanduel), with `oddsFormat=american` and `dateFormat=iso`.

Budget and safety properties:

- ~2 historical requests total; the script prints the API's
  `x-requests-remaining` header after each call.
- `THE_ODDS_API_KEY` comes from the environment only; errors print HTTP
  status and a truncated response body but never the key or request URL.
- Raw JSON saved untouched to `data/raw/odds/the_odds_api/` before any
  processing; existing raw files are loaded instead of refetched, so
  reruns spend no credits. `--overwrite` (announced loudly) refetches.
- Normalized output (M4 path + opening/closing marking) goes to
  `data/processed/odds_sample.parquet`.
- Empty snapshots or zero h2h rows fail with actionable messages.

Run it manually from your terminal — never from tests:

    export THE_ODDS_API_KEY=...   # uncommitted; e.g. via .env
    python scripts/collect_real_odds_sample.py

## Matching the odds sample (M9)

`scripts/match_real_odds_sample.py` joins the odds sample to the games table
via the strict M5 matcher. First real run finding: a historical snapshot
contains **every upcoming event at capture time** — the 2024-04-12 sample
included 17 events spanning Apr 12–17, and the two unmatched ones were the
ATL/CHI **play-in tournament** games (Apr 16/17), which the regular-season
games table excludes by definition. 15 of 17 events match; strict matching
correctly refused the other two and the script's diagnostic table identified
them precisely.

Resolution (implemented): historical snapshots can include future events
outside the currently collected games table, so the matching script performs
an explicit **scope pre-filter** before strict matching — odds events whose
(event_date, home_team, away_team) key does not exist in the games table are
excluded, with the count, provider_event_ids, dates, and teams printed.
Excluded events are reported, never silently dropped. This scope filter is
deliberately separate from strict event matching: `match_events.py` remains
strict and unweakened, and every in-scope row must still match exactly one
game or the run fails loudly.

## First real baseline evaluation (sample)

`scripts/evaluate_real_market_baseline_sample.py` runs the closing-line
market baseline over the matched sample (15 games, 2 books). First-run
metrics were *worse than a coin flip* (log loss ~0.71 vs ln 2 ≈ 0.693) —
which is exactly what a 15-game sample can do, and a useful reminder of why
the research plan forbids conclusions at this scale. The sample also lands
on the season's final weekend (the busiest-date heuristic picks late-season
days), where resting/tanking makes favorites unusually unreliable; the
full-season pull will not share that bias. Pipeline validation only.

## One-month odds collection

`scripts/collect_real_odds_month.py` scales the M8 pattern to one
mid-season month (January 2024 — far from opening week, All-Star, and the
play-in boundary), two snapshots per game date (18:00Z early, 23:30Z
near-tip), two bookmakers. The default run is a **dry plan** that prints
the date range, snapshot count, and estimated credit cost and exits with
zero network calls; `--confirm` fetches only the snapshots whose raw JSON
is not already saved (resume-safe — an interrupted run continues where it
left off at no extra cost); `--confirm --overwrite` refetches everything.
Credits remaining are printed after each live call. Output:
`data/processed/odds_month_sample.parquet`.

## Matching the one-month sample

`scripts/match_real_odds_month.py` applies the scope-filter + strict-match
pattern at month scale. January 2024 result: **1,201 of 1,203 odds rows
matched (234 games)**, zero duplicates, zero unmatched in-scope events. The
single scope-excluded event (UTA vs GSW, 2024-01-17) is the Warriors-Jazz
game **postponed** after the death of Warriors assistant coach Dejan
Milojević — the odds existed pre-game, but no game occurred on that date,
so the exclusion is correct behavior, not data loss. Postponements are
therefore a real category the scope filter handles naturally: the
rescheduled game appears in the games table on its actual date and matches
the odds captured for that later date.

Tests never call the live API — `tests/test_collect_nba_games.py` covers
normalization and saving with mocked frames only.
