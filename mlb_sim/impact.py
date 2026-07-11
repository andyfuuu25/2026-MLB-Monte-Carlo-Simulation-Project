"""General player-impact engine: value ANY player and simulate removing them.

Composite metric model — weights from the literature
----------------------------------------------------
The requested metrics (fWAR, bWAR, wRC+, wOBA, OAA, DRS, FIP, SIERA, WPA) are
not independent signals: the research consensus organizes them into the WAR
component framework (runs above replacement, ~10 runs = 1 win — Tango,
*The Book*; FanGraphs WAR docs). This module weights them accordingly:

===========  ==================================================================
Metric       Role & weight (with source)
===========  ==================================================================
wOBA         THE offensive driver. Linear-weights on-base average (Tango);
             batting runs = (wOBA − lgwOBA)/wOBA_scale × PA.
wRC+         Same information as wOBA, index-scaled to league 100 — computed
             and displayed, but not double-counted in the composite.
OAA / DRS    Fielding runs. Statcast **Fielding Runs Prevented** (the runs
             conversion of OAA) is used directly; DRS lives on 403-blocked
             Baseball-Reference/Fielding Bible and OAA is its modern granular
             replacement (MLB Statcast methodology).
SIERA        60% of pitcher talent. ERA estimators rank SIERA ≥ xFIP > FIP for
             predicting *future* ERA (Swartz; FanGraphs library; Pitcher List
             replication studies).
FIP          40% of pitcher talent — keeps real HR information SIERA regresses.
fWAR / bWAR  Not fetchable (FanGraphs/B-R block automated clients); the
             composite below IS the fWAR recipe computed in-house —
             (batting runs + fielding runs + replacement offset) / 10, or
             (replacement RA − talent RA) × IP/9/10 for pitchers — so the
             engine reports its own **WAR estimate** on the same scale.
WPA          Weight **zero** by design: the literature is unanimous that WPA
             is a descriptive "story stat" with no predictive validity for
             talent (FanGraphs library; Hardball Times).
===========  ==================================================================

Replacement level: batters −20 runs / 600 PA; pitchers ≈ league FIP + 1.00
(FanGraphs replacement-level framework). Runs per win: 10.

Removal simulation
------------------
This module handles player *valuation*; the roster surgery itself (handing a
player's PA/innings to replacement level and rebuilding team RS/RA) lives in
``features.apply_player_injury``, and the Monte Carlo engine replays the
remaining season under both rosters with identical random seeds — so every
delta is causally attributable to that one player.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .features import (MIN_OUTS, MIN_PA, fip, league_fip_constant,
                       runs_created)

log = logging.getLogger(__name__)

RUNS_PER_WIN = 10.0                 # Tango, The Book; FanGraphs WAR docs
REPL_BAT_RUNS_PER_600 = 20.0        # replacement offset, position players
REPL_FIP_GAP = 1.00                 # replacement pitcher = league FIP + 1
SIERA_WEIGHT, FIP_WEIGHT = 0.60, 0.40   # predictive-validity ordering
GAMES_FULL = 162

# FanGraphs "guts" linear weights (stable across recent run environments).
W_BB, W_HBP, W_1B, W_2B, W_3B, W_HR = 0.689, 0.720, 0.882, 1.254, 1.590, 2.050
WOBA_SCALE = 1.24


# ---------------------------------------------------------------------------
# Rate metrics
# ---------------------------------------------------------------------------
def add_batting_metrics(bat: pd.DataFrame) -> pd.DataFrame:
    """wOBA, wRAA, and (non-park) wRC+ for every batter."""
    df = bat[bat["PA"] >= MIN_PA].copy()
    singles = df["H"] - df["2B"] - df["3B"] - df["HR"]
    num = (W_BB * df["BB"] + W_HBP * df["HBP"] + W_1B * singles
           + W_2B * df["2B"] + W_3B * df["3B"] + W_HR * df["HR"])
    den = (df["AB"] + df["BB"] + df["HBP"] + df["SF"]).clip(lower=1)
    df["woba"] = num / den

    lg_woba = float(num.sum() / den.sum())
    df["wraa"] = (df["woba"] - lg_woba) / WOBA_SCALE * df["PA"]
    # League runs/PA proxied by league Runs Created per PA.
    lg_r_pa = float(runs_created(df).sum() / df["PA"].sum())
    df["wrc_plus"] = 100 * ((df["wraa"] / df["PA"] + lg_r_pa) / lg_r_pa)
    df.attrs["lg_woba"] = lg_woba
    return df


def add_pitching_metrics(pit: pd.DataFrame) -> pd.DataFrame:
    """FIP and (approximate) SIERA for every pitcher.

    SIERA follows Swartz's published formula. True batted-ball counts need
    Statcast pitch data; ground/air OUT totals stand in for GB/FB and pop-ups
    are unavailable, so this is an approximation (documented in README).
    """
    df = pit[pit["outs"] >= MIN_OUTS].copy()
    c_fip = league_fip_constant(df)
    df["fip"] = fip(df, c_fip)

    bf = df["BF"].clip(lower=1).astype(float)
    so_pa = df["K"] / bf
    bb_pa = df["BB"] / bf
    net_gb = (df["GO"] - df["AO"]) / bf
    df["siera"] = (6.145 - 16.986 * so_pa + 11.434 * bb_pa - 1.858 * net_gb
                   + 7.653 * so_pa**2
                   + np.where(net_gb < 0, 6.664, -6.664) * net_gb**2
                   + 10.130 * so_pa * net_gb - 5.195 * bb_pa * net_gb)
    df["siera"] = df["siera"].clip(2.0, 7.5)
    df.attrs["c_fip"] = c_fip
    df.attrs["lg_fip"] = float(np.average(df["fip"], weights=df["outs"]))
    return df


# ---------------------------------------------------------------------------
# Composite value (runs above replacement -> WAR estimate)
# ---------------------------------------------------------------------------
def player_value_table(bat: pd.DataFrame, pit: pd.DataFrame,
                       oaa: pd.DataFrame) -> pd.DataFrame:
    """One row per player-role with composite runs above replacement."""
    b = add_batting_metrics(bat)
    p = add_pitching_metrics(pit)
    field = oaa.set_index("player_id")[["frp", "oaa", "position"]]

    b = b.join(field, on="player_id")
    b["frp"] = b["frp"].fillna(0.0)
    b["raa"] = b["wraa"] + b["frp"]                    # bat + glove, runs
    b["rar"] = b["raa"] + REPL_BAT_RUNS_PER_600 * b["PA"] / 600
    b["war_est"] = b["rar"] / RUNS_PER_WIN
    b["role"] = "bat"

    lg_fip = p.attrs["lg_fip"]
    p["talent_ra"] = SIERA_WEIGHT * p["siera"] + FIP_WEIGHT * p["fip"]
    p["rar"] = (lg_fip + REPL_FIP_GAP - p["talent_ra"]) * (p["outs"] / 3) / 9
    p["war_est"] = p["rar"] / RUNS_PER_WIN
    p["role"] = "pitch"

    cols_b = ["player_id", "name", "team_id", "role", "PA", "woba",
              "wrc_plus", "frp", "oaa", "position", "rar", "war_est"]
    cols_p = ["player_id", "name", "team_id", "role", "outs", "fip", "siera",
              "talent_ra", "rar", "war_est"]
    out = pd.concat([b[cols_b], p[cols_p]], ignore_index=True)
    out.attrs["lg_fip"] = lg_fip
    return out


def composite_leaderboard(values: pd.DataFrame, teams: pd.DataFrame,
                          top: int = 30) -> pd.DataFrame:
    """Rank players by composite full-season value (WAR estimate).

    Two-way players' batting and pitching rows are combined. The Δwins column
    is the classical runs-to-wins conversion (RAR / 10) — the Monte Carlo
    endpoint gives the schedule-aware version for any selected player.
    """
    agg = (values.groupby(["player_id", "team_id"], as_index=False)
           .agg(name=("name", "first"), rar=("rar", "sum"),
                war_est=("war_est", "sum"), roles=("role", "count")))
    merged = values.set_index(["player_id", "team_id"])
    lead = agg.sort_values("war_est", ascending=False).head(top).copy()
    lead["d_wins"] = lead["rar"] / RUNS_PER_WIN

    abbrev = teams.set_index("team_id")["abbrev"]
    lead["team"] = lead["team_id"].map(abbrev)

    def metrics_for(pid, tid):
        rows = merged.loc[[(pid, tid)]]
        m = {}
        for _, r in rows.iterrows():
            if r["role"] == "bat":
                m.update({"woba": round(r["woba"], 3),
                          "wrc_plus": round(r["wrc_plus"], 0),
                          "frp": round(r["frp"], 0)})
            else:
                m.update({"fip": round(r["fip"], 2),
                          "siera": round(r["siera"], 2)})
        return m

    lead["metrics"] = [metrics_for(r.player_id, r.team_id)
                       for r in lead.itertuples()]
    return lead
