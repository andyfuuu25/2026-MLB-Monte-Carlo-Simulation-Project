"""Fantasy draft mode: build a roster from real players, simulate its season.

Player projections
------------------
Each player's talent is a simplified **Marcel** forecast (Tango, 2004 — the
"Marcel the Monkey" forecasting system): the last three seasons blended with
recency weights 5/4/3, then regressed toward the league mean by playing time
(200 PA of league-average ballast for batters, 50 IP for pitchers). Batters
are rated by Runs Created per PA; pitchers by projected FIP.

Team construction
-----------------
- 9 hitters share the ~38 team PA per game equally (a DH-style lineup;
  fielding positions are intentionally ignored).
- 5 starting pitchers cover ~65% of innings (IP-weighted rotation FIP);
  the remaining 35% is a league-average bullpen.
- The roster's expected RS/G and RA/G flow through the same machinery as
  real teams, and its starting Elo is imputed from the league-wide linear
  relationship between roster run differential and season-start Elo.

The fantasy club takes over one franchise's schedule slot and the full
season is replayed from game 1 (10,000 trials), against a memoized baseline
replay with the real club in place — same seed, so deltas are causal.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .features import GAMES_PER_SEASON, fip, league_fip_constant, runs_created

log = logging.getLogger(__name__)

MARCEL_WEIGHTS = (5, 4, 3)        # target season, -1, -2
REGRESS_PA = 200.0                # league-average ballast, batters
REGRESS_IP = 50.0                 # league-average ballast, pitchers
PA_PER_GAME = 38.0                # league team plate appearances per game
LINEUP_SLOTS = 9
ROTATION_SLOTS = 5
ROTATION_IP_SHARE = 0.65          # starters' share of innings
RA9_TO_ERA = 1.08                 # unearned-run scaling (as in features.py)


# ---------------------------------------------------------------------------
# Draft pool
# ---------------------------------------------------------------------------
def build_hitter_pool(batting: dict[int, pd.DataFrame],
                      target_season: int) -> pd.DataFrame:
    """Marcel-blended RC/PA for every batter with a meaningful sample."""
    frames = []
    for offset, w in enumerate(MARCEL_WEIGHTS):
        season = target_season - offset
        if season not in batting:
            continue
        df = batting[season].copy()
        df["RC"] = runs_created(df)
        df["w"] = w
        df["is_target"] = offset == 0
        frames.append(df)
    allb = pd.concat(frames, ignore_index=True)

    lg_rc_pa = allb.loc[allb["is_target"], "RC"].sum() / max(
        allb.loc[allb["is_target"], "PA"].sum(), 1)

    g = allb.assign(wPA=allb["w"] * allb["PA"], wRC=allb["w"] * allb["RC"])
    agg = g.groupby("player_id").agg(
        name=("name", "last"), wPA=("wPA", "sum"), wRC=("wRC", "sum"),
        PA=("PA", "sum"), w=("w", "sum"))
    # Latest team label for display (target season preferred).
    latest = (g.sort_values(["is_target", "w"])
              .groupby("player_id")["team_id"].last())
    agg["team_id"] = latest

    n_eff = agg["wPA"] / (agg["w"] / len(MARCEL_WEIGHTS)).clip(lower=1)
    raw = agg["wRC"] / agg["wPA"].clip(lower=1)
    agg["rc_pa"] = (raw * n_eff + lg_rc_pa * REGRESS_PA) / (n_eff + REGRESS_PA)
    agg["rc650"] = agg["rc_pa"] * 650          # per-full-season scale, display

    pool = agg[agg["PA"] >= 200].reset_index()
    pool.attrs["lg_rc_pa"] = float(lg_rc_pa)
    return pool.sort_values("rc_pa", ascending=False)


def build_pitcher_pool(pitching: dict[int, pd.DataFrame],
                       target_season: int) -> pd.DataFrame:
    """Marcel-blended FIP for every pitcher with a meaningful sample."""
    frames = []
    for offset, w in enumerate(MARCEL_WEIGHTS):
        season = target_season - offset
        if season not in pitching:
            continue
        df = pitching[season].copy()
        df["w"] = w
        df["is_target"] = offset == 0
        frames.append(df)
    allp = pd.concat(frames, ignore_index=True)

    target = allp[allp["is_target"]]
    c_fip = league_fip_constant(target)
    lg_fip = float(np.average(fip(target, c_fip), weights=target["outs"]))

    g = allp.assign(**{c: allp["w"] * allp[c]
                       for c in ("outs", "HR", "BB", "HBP", "K")})
    agg = g.groupby("player_id").agg(
        name=("name", "last"), outs=("outs", "sum"), HR=("HR", "sum"),
        BB=("BB", "sum"), HBP=("HBP", "sum"), K=("K", "sum"), w=("w", "sum"))
    agg["raw_outs"] = allp.groupby("player_id")["outs"].sum()
    latest = (g.sort_values(["is_target", "w"])
              .groupby("player_id")["team_id"].last())
    agg["team_id"] = latest

    agg["fip_raw"] = fip(agg, c_fip)
    ip_eff = (agg["outs"] / 3.0) / (agg["w"] / len(MARCEL_WEIGHTS)).clip(lower=1)
    agg["fip"] = ((agg["fip_raw"] * ip_eff + lg_fip * REGRESS_IP)
                  / (ip_eff + REGRESS_IP))
    agg["ip"] = agg["raw_outs"] / 3.0          # unweighted career-window IP

    pool = agg[agg["raw_outs"] >= 150].reset_index()   # >= 50 innings
    pool.attrs["lg_fip"] = lg_fip
    return pool.sort_values("fip", ascending=True)


# ---------------------------------------------------------------------------
# Roster -> team profile
# ---------------------------------------------------------------------------
def fantasy_profile(hitter_ids: list[int], pitcher_ids: list[int],
                    hitters: pd.DataFrame, pitchers: pd.DataFrame,
                    lg_fip: float) -> dict:
    """Expected RS/G, RA/G and run diff for a drafted roster."""
    h = hitters.set_index("player_id").loc[hitter_ids]
    p = pitchers.set_index("player_id").loc[pitcher_ids]
    if len(h) != LINEUP_SLOTS or len(p) != ROTATION_SLOTS:
        raise ValueError(
            f"roster must be exactly {LINEUP_SLOTS} hitters "
            f"and {ROTATION_SLOTS} pitchers")

    exp_rs = float(h["rc_pa"].sum() * PA_PER_GAME / LINEUP_SLOTS)
    rot_fip = float(np.average(p["fip"], weights=p["outs"].clip(lower=1)))
    team_fip = ROTATION_IP_SHARE * rot_fip + (1 - ROTATION_IP_SHARE) * lg_fip
    exp_ra = team_fip * RA9_TO_ERA
    return {
        "exp_rs": round(exp_rs, 3),
        "exp_ra": round(exp_ra, 3),
        "run_diff": round(exp_rs - exp_ra, 3),
        "rotation_fip": round(rot_fip, 2),
        "hitters": h.reset_index()[["player_id", "name", "rc650"]]
                    .round({"rc650": 1}).to_dict(orient="records"),
        "pitchers": p.reset_index()[["player_id", "name", "fip"]]
                     .round({"fip": 2}).to_dict(orient="records"),
    }


def implied_elo(run_diff: float, profiles: pd.DataFrame,
                season_start_elo: dict[int, float]) -> float:
    """Impute a starting Elo from the league's run-diff -> Elo relationship."""
    x = profiles["run_diff"].to_numpy(dtype=float)
    y = np.array([season_start_elo[t] for t in profiles["team_id"]])
    slope, intercept = np.polyfit(x, y, 1)
    return float(np.clip(slope * run_diff + intercept, 1350.0, 1750.0))
