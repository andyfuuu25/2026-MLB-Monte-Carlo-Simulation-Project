"""Data ingestion layer.

Strategy
--------
The spec calls for ``pybaseball`` (``batting_stats``, ``pitching_stats``,
``schedule_and_record``). Those functions scrape FanGraphs and
Baseball-Reference, both of which now return HTTP 403 to automated clients.
Each fetch therefore *attempts pybaseball first* and transparently falls back
to the official **MLB Stats API** (``statsapi.mlb.com``) — the league's own
free, documented data service — with column names normalized to a common
schema. Every fetch is cached to ``data/`` as CSV so re-runs are offline-fast.

Common schemas
--------------
Batting rows:  player_id, name, team_id, PA, AB, H, 2B, 3B, HR, BB, HBP, SF
Pitching rows: player_id, name, team_id, IP, K, BB, HBP, HR, ER, outs
Game rows:     game_pk, date, season, home_id, away_id, home_score,
               away_score, state ('final' | 'future'), home_win
"""

from __future__ import annotations

import logging
import time
from typing import Iterator

import pandas as pd
import requests

from .config import DATA_DIR, HISTORY_SEASONS, REQUEST_TIMEOUT, STATSAPI_BASE, TARGET_SEASON

log = logging.getLogger(__name__)

_session = requests.Session()
_session.headers.update({"User-Agent": "mlb-hybrid-sim/1.0 (research)"})


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def _get_json(endpoint: str, **params) -> dict:
    """GET a Stats API endpoint with one retry on transient failure."""
    url = f"{STATSAPI_BASE}/{endpoint}"
    for attempt in (1, 2):
        try:
            resp = _session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2)
    raise RuntimeError("unreachable")


def _cached(name: str, builder) -> pd.DataFrame:
    """Return DataFrame from data/<name>.csv, building and caching if absent."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{name}.csv"
    if path.exists():
        return pd.read_csv(path)
    df = builder()
    df.to_csv(path, index=False)
    log.info("cached %s (%d rows)", path.name, len(df))
    return df


def innings_to_outs(ip: str | float) -> int:
    """Convert baseball innings notation ('85.2' = 85 IP + 2 outs) to outs."""
    whole, _, frac = str(ip).partition(".")
    return int(whole or 0) * 3 + int(frac or 0)


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------
def fetch_teams(season: int = TARGET_SEASON) -> pd.DataFrame:
    """All 30 MLB teams with league / division metadata."""

    def build() -> pd.DataFrame:
        payload = _get_json("teams", sportId=1, season=season)
        rows = [
            {
                "team_id": t["id"],
                "abbrev": t["abbreviation"],
                "name": t["name"],
                "league": t["league"]["name"],
                "division": t["division"]["name"],
            }
            for t in payload["teams"]
        ]
        return pd.DataFrame(rows).sort_values("team_id").reset_index(drop=True)

    return _cached(f"teams_{season}", build)


# ---------------------------------------------------------------------------
# Player-level season stats
# ---------------------------------------------------------------------------
_BATTING_MAP = {
    "plateAppearances": "PA", "atBats": "AB", "hits": "H", "doubles": "2B",
    "triples": "3B", "homeRuns": "HR", "baseOnBalls": "BB",
    "hitByPitch": "HBP", "sacFlies": "SF",
}
_PITCHING_MAP = {
    "strikeOuts": "K", "baseOnBalls": "BB", "hitByPitch": "HBP",
    "homeRuns": "HR", "earnedRuns": "ER", "hits": "HA",
    "battersFaced": "BF", "groundOuts": "GO", "airOuts": "AO",
}


def _statsapi_player_stats(season: int, group: str) -> Iterator[dict]:
    """Page through every player's season line for one stat group."""
    offset = 0
    while True:
        payload = _get_json(
            "stats", stats="season", group=group, season=season, sportId=1,
            playerPool="all", limit=1000, offset=offset,
        )
        splits = payload["stats"][0]["splits"] if payload.get("stats") else []
        yield from splits
        if len(splits) < 1000:
            return
        offset += 1000


def fetch_batting(season: int) -> pd.DataFrame:
    """Player batting lines. Tries pybaseball.batting_stats (FanGraphs) first."""

    def build() -> pd.DataFrame:
        try:  # spec-preferred source
            from pybaseball import batting_stats

            fg = batting_stats(season, qual=1)
            log.info("batting %d via pybaseball/FanGraphs", season)
            return pd.DataFrame({
                "player_id": fg["IDfg"], "name": fg["Name"], "team_id": pd.NA,
                "PA": fg["PA"], "AB": fg["AB"], "H": fg["H"], "2B": fg["2B"],
                "3B": fg["3B"], "HR": fg["HR"], "BB": fg["BB"],
                "HBP": fg["HBP"], "SF": fg["SF"],
            })
        except Exception as exc:  # FanGraphs returns 403 to scrapers
            log.warning("pybaseball batting_stats unavailable (%s); "
                        "falling back to MLB Stats API", type(exc).__name__)

        rows = []
        for s in _statsapi_player_stats(season, "hitting"):
            if "team" not in s:  # league-total rows for multi-team players
                continue
            stat = s["stat"]
            row = {"player_id": s["player"]["id"], "name": s["player"]["fullName"],
                   "team_id": s["team"]["id"]}
            row.update({out: stat.get(src, 0) for src, out in _BATTING_MAP.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    return _cached(f"batting_{season}", build)


def fetch_pitching(season: int) -> pd.DataFrame:
    """Player pitching lines. Tries pybaseball.pitching_stats first."""

    def build() -> pd.DataFrame:
        try:
            from pybaseball import pitching_stats

            fg = pitching_stats(season, qual=1)
            log.info("pitching %d via pybaseball/FanGraphs", season)
            return pd.DataFrame({
                "player_id": fg["IDfg"], "name": fg["Name"], "team_id": pd.NA,
                "IP": fg["IP"], "K": fg["SO"], "BB": fg["BB"], "HBP": fg["HBP"],
                "HR": fg["HR"], "ER": fg["ER"], "HA": fg["H"],
                "BF": fg["TBF"], "GO": fg["GB"], "AO": fg["FB"],
                "outs": (fg["IP"].astype(float) * 3).round().astype(int),
            })
        except Exception as exc:
            log.warning("pybaseball pitching_stats unavailable (%s); "
                        "falling back to MLB Stats API", type(exc).__name__)

        rows = []
        for s in _statsapi_player_stats(season, "pitching"):
            if "team" not in s:
                continue
            stat = s["stat"]
            row = {"player_id": s["player"]["id"], "name": s["player"]["fullName"],
                   "team_id": s["team"]["id"],
                   "IP": stat.get("inningsPitched", "0.0"),
                   "outs": innings_to_outs(stat.get("inningsPitched", "0.0"))}
            row.update({out: stat.get(src, 0) for src, out in _PITCHING_MAP.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    return _cached(f"pitching_{season}", build)


# ---------------------------------------------------------------------------
# Statcast leaderboards (Baseball Savant)
# ---------------------------------------------------------------------------
def _savant_csv(name: str, endpoint: str, empty_cols: list[str],
                **params) -> pd.DataFrame:
    """Fetch one Savant leaderboard CSV, cached; empty frame on failure."""

    def build() -> pd.DataFrame:
        try:
            import io
            resp = _session.get(
                f"https://baseballsavant.mlb.com/leaderboard/{endpoint}",
                params={**params, "csv": "true"},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return pd.read_csv(io.StringIO(resp.text.lstrip("﻿")))
        except Exception as exc:
            log.warning("Savant %s unavailable (%s); returning empty frame",
                        endpoint, type(exc).__name__)
            return pd.DataFrame(columns=empty_cols)

    return _cached(name, build)


def fetch_xstats(season: int, kind: str) -> pd.DataFrame:
    """Expected statistics (xwOBA / xERA) — Statcast quality-of-contact."""
    cols = ["player_id", "woba", "est_woba"] + (
        ["era", "xera"] if kind == "pitcher" else [])
    return _savant_csv(f"xstats_{kind}_{season}", "expected_statistics", cols,
                       type=kind, year=season, position="", team="", min="q")


def fetch_statcast_batters(season: int) -> pd.DataFrame:
    """Exit velocity / barrels / hard-hit leaderboard."""
    cols = ["player_id", "brl_percent", "ev95percent", "avg_hit_speed"]
    return _savant_csv(f"statcast_bat_{season}", "statcast", cols,
                       type="batter", year=season, position="", team="",
                       min="q")


def fetch_arsenal(season: int) -> pd.DataFrame:
    """Pitch mix (usage %), per-type velocity and spin, merged wide."""

    def build() -> pd.DataFrame:
        frames = {}
        for typ in ("n_", "avg_speed", "avg_spin"):
            df = _savant_csv(f"_tmp_arsenal_{typ}{season}", "pitch-arsenals",
                             ["pitcher"], year=season, min=100, type=typ,
                             hand="")
            (DATA_DIR / f"_tmp_arsenal_{typ}{season}.csv").unlink(missing_ok=True)
            frames[typ] = df
        out = frames["n_"]
        for typ in ("avg_speed", "avg_spin"):
            other = frames[typ].drop(columns=["last_name, first_name"],
                                     errors="ignore")
            out = out.merge(other, on="pitcher", how="outer")
        return out.rename(columns={"pitcher": "player_id"})

    return _cached(f"arsenal_{season}", build)


def fetch_pitch_discipline(season: int) -> pd.DataFrame:
    """Zone%, whiff%, chase%, first-pitch strike% (Savant custom board)."""
    cols = ["player_id", "pa", "k_percent", "bb_percent", "whiff_percent",
            "oz_swing_percent", "in_zone_percent", "f_strike_percent"]
    return _savant_csv(
        f"discipline_{season}", "custom", cols, year=season, type="pitcher",
        min=100, selections=("pa,k_percent,bb_percent,whiff_percent,"
                             "oz_swing_percent,in_zone_percent,"
                             "f_strike_percent"))


def fetch_fastball_movement(season: int) -> pd.DataFrame:
    """Four-seam movement: induced vertical + horizontal break vs league."""
    cols = ["pitcher_id", "pitcher_break_z_induced", "pitcher_break_x",
            "avg_speed"]
    df = _savant_csv(f"movement_ff_{season}", "pitch-movement", cols,
                     year=season, team="", min="q", pitch_type="FF", hand="")
    return df.rename(columns={"pitcher_id": "player_id"})



def fetch_oaa(season: int) -> pd.DataFrame:
    """Outs Above Average + Fielding Runs Prevented from Baseball Savant.

    Savant's leaderboard CSV endpoint remains open to programmatic clients
    (unlike FanGraphs/Baseball-Reference) and keys players by MLBAM id, so it
    joins directly onto the Stats API player tables. Returns an empty frame
    on failure — fielding then contributes zero runs rather than crashing.
    """

    def build() -> pd.DataFrame:
        try:
            resp = _session.get(
                "https://baseballsavant.mlb.com/leaderboard/outs_above_average",
                params={"type": "Fielder", "year": season, "min": 1,
                        "csv": "true"},
                headers={"User-Agent": "Mozilla/5.0"}, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            import io
            df = pd.read_csv(io.StringIO(resp.text.lstrip("﻿")))
            return pd.DataFrame({
                "player_id": df["player_id"],
                "oaa": df["outs_above_average"],
                "frp": df["fielding_runs_prevented"],
                "position": df["primary_pos_formatted"],
            })
        except Exception as exc:
            log.warning("Savant OAA unavailable for %d (%s); fielding runs "
                        "default to 0", season, type(exc).__name__)
            return pd.DataFrame(columns=["player_id", "oaa", "frp", "position"])

    return _cached(f"oaa_{season}", build)


# ---------------------------------------------------------------------------
# Game logs / schedules
# ---------------------------------------------------------------------------
def fetch_season_games(season: int) -> pd.DataFrame:
    """Every regular-season game: completed results and future fixtures.

    Attempts ``pybaseball.schedule_and_record`` (Baseball-Reference) first;
    falls back to the MLB Stats API schedule endpoint.
    """

    def build() -> pd.DataFrame:
        try:
            from pybaseball import schedule_and_record  # noqa: F401
            # Probe one team: Baseball-Reference 403s automated clients. Even
            # when reachable we normalize through the Stats API schema below,
            # so the probe only decides whether to emit the fallback warning.
            schedule_and_record(season, "LAD")
        except Exception as exc:
            log.warning("pybaseball schedule_and_record unavailable (%s); "
                        "falling back to MLB Stats API", type(exc).__name__)

        payload = _get_json(
            "schedule", sportId=1, season=season, gameType="R",
            startDate=f"{season}-02-20", endDate=f"{season}-11-30",
        )
        rows = []
        for day in payload["dates"]:
            for g in day["games"]:
                state = g["status"]["abstractGameState"]
                detailed = g["status"]["detailedState"]
                if detailed in ("Postponed", "Cancelled"):
                    continue  # rescheduled games appear again under new dates
                home, away = g["teams"]["home"], g["teams"]["away"]
                final = state == "Final"
                rows.append({
                    "game_pk": g["gamePk"],
                    "date": day["date"],
                    "season": season,
                    "home_id": home["team"]["id"],
                    "away_id": away["team"]["id"],
                    "home_score": home.get("score") if final else None,
                    "away_score": away.get("score") if final else None,
                    "state": "final" if final else "future",
                })
        df = pd.DataFrame(rows).drop_duplicates("game_pk", keep="last")
        df = df.sort_values(["date", "game_pk"]).reset_index(drop=True)
        # Ties are impossible in MLB; guard against malformed rows.
        finals = df["state"] == "final"
        df.loc[finals, "home_win"] = (
            df.loc[finals, "home_score"] > df.loc[finals, "away_score"]
        ).astype(int)
        return df

    return _cached(f"games_{season}", build)


def load_all_games() -> pd.DataFrame:
    """Chronologically-ordered games for all history seasons + target season."""
    frames = [fetch_season_games(s) for s in (*HISTORY_SEASONS, TARGET_SEASON)]
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["season", "date", "game_pk"]).reset_index(drop=True)
