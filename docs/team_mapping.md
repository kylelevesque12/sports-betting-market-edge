# Team Mapping

## Why this exists

Real NBA game data and sportsbook odds data spell teams differently:
"Golden State Warriors" (NBA.com), "GSW" (most books), "GS" (some books),
"Golden State" (scores archives), "PHO" vs "PHX" (basketball-reference vs.
common usage). Joining datasets on raw names silently drops games — the worst
kind of data loss, because nothing errors. Every team identifier is therefore
mapped to one canonical ID **before any merge** (research_plan.md, M2), and
unknown names fail loudly rather than passing through.

## Canonical convention

Internal team IDs are the standard three-letter NBA abbreviations:

ATL, BOS, BKN, CHA, CHI, CLE, DAL, DEN, DET, GSW, HOU, IND, LAC, LAL, MEM,
MIA, MIL, MIN, NOP, NYK, OKC, ORL, PHI, PHX, POR, SAC, SAS, TOR, UTA, WAS

Matching is case-insensitive, ignores leading/trailing/repeated whitespace,
and ignores periods ("L.A. Lakers" matches).

## Known aliases

Beyond full team names and the canonical abbreviations themselves:

| Alias | Canonical | Source convention |
|---|---|---|
| GS | GSW | common sportsbook form |
| SA | SAS | common sportsbook form |
| NO | NOP | common sportsbook form |
| NY | NYK | common sportsbook form |
| BRK | BKN | basketball-reference |
| CHO | CHA | basketball-reference |
| PHO | PHX | basketball-reference / older sources |
| UTAH | UTA | ESPN |
| WSH | WAS | ESPN |
| Golden State | GSW | scores/odds archives |
| Okla City | OKC | sportsbookreviewsonline |
| LA Clippers / Los Angeles Clippers | LAC | both official forms |

New variants found during real ingestion are added to `_EXTRA_ALIASES` in
`src/data/team_mapping.py` with a test — never worked around at the call
site.

## Historical relocations and renames — deferred

Franchise moves and renames (SEA→OKC 2008, NJN→BKN 2012, NOH/NOK→NOP
2013, CHA Bobcats→Hornets 2014) are **not yet mapped**. They are deferred
until the real data window is chosen: if ingestion starts at 2014-15 or
later, none are needed; an earlier window requires season-aware mapping
(the same name can mean different things in different years), which is a
deliberate design decision to make then, not a default to guess now.
Unknown historical names currently raise ValueError — by design, so an
older data window cannot silently mis-join.
