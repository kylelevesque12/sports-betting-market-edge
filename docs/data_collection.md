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

Tests never call the live API — `tests/test_collect_nba_games.py` covers
normalization and saving with mocked frames only.
