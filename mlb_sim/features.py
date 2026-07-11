"""Bottom-up roster feature engineering.

Player talent is aggregated into two team-level numbers per season:

- **Expected Runs Scored / game** — sum of each batter's Basic Runs Created,
  RC = (H + BB) * TB / (AB + BB)  (Bill James, *Baseball Abstract*), divided
  by 162 team games.
- **Expected Runs Allowed / game** — innings-weighted staff FIP,
  FIP = (13*HR + 3*(BB+HBP) - 2*K) / IP + cFIP, with the constant cFIP set so
  league FIP equals league ERA (Tango; FanGraphs glossary), scaled from an
  earned-run to a total-run basis using the league RA9/ERA ratio.

Shohei Ohtani's two-way value is explicitly split: his batting line lives in
the Dodgers' offense matrix and his pitching line in their run-prevention
matrix, so an injury scenario can zero out one half independently.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import (DODGERS_TEAM_ID, MIN_OUTS, MIN_PA, OHTANI_MLBAM_ID,
                     REPLACEMENT_FIP_PENALTY)

log = logging.getLogger(__name__)

GAMES_PER_SEASON = 162


# ---------------------------------------------------------------------------
# Player-level metrics
# ---------------------------------------------------------------------------
def runs_created(bat: pd.DataFrame) -> pd.Series:
    """Basic Runs Created (James): (H + BB) * TB / (AB + BB)."""
    tb = bat["H"] + bat["2B"] + 2 * bat["3B"] + 3 * bat["HR"]
    denom = (bat["AB"] + bat["BB"]).clip(lower=1)
    return (bat["H"] + bat["BB"]) * tb / denom


def fip(pit: pd.DataFrame, c_fip: float) -> pd.Series:
    """Fielding Independent Pitching (Tango): (13HR + 3(BB+HBP) - 2K)/IP + C."""
    ip = (pit["outs"] / 3.0).clip(lower=1e-9)
    return (13 * pit["HR"] + 3 * (pit["BB"] + pit["HBP"]) - 2 * pit["K"]) / ip + c_fip


def league_fip_constant(pit: pd.DataFrame) -> float:
    """cFIP chosen so that league FIP == league ERA."""
    ip = pit["outs"].sum() / 3.0
    lg_era = 9.0 * pit["ER"].sum() / ip
    raw = (13 * pit["HR"].sum() + 3 * (pit["BB"].sum() + pit["HBP"].sum())
           - 2 * pit["K"].sum()) / ip
    return lg_era - raw


# ---------------------------------------------------------------------------
# Team profiles
# ---------------------------------------------------------------------------
def build_team_profiles(bat: pd.DataFrame, pit: pd.DataFrame,
                        games_played: pd.Series | None = None,
                        ra9_to_era: float = 1.08) -> pd.DataFrame:
    """Aggregate player lines into per-team expected RS/G, RA/G, run diff.

    ``games_played`` (indexed by team_id) prorates counting-stat offense for
    partial seasons; defaults to a full 162. ``ra9_to_era`` converts earned
    runs to total runs (league RA9/ERA is historically ~1.08).
    """
    bat = bat[bat["PA"] >= MIN_PA].copy()
    pit = pit[pit["outs"] >= MIN_OUTS].copy()

    # Offense: playing-time-implicit (counting stat) Runs Created per team.
    bat["RC"] = runs_created(bat)
    rc_total = bat.groupby("team_id")["RC"].sum()
    if games_played is not None:
        offense = rc_total / games_played.reindex(rc_total.index).clip(lower=1)
    else:
        offense = rc_total / GAMES_PER_SEASON

    # Run prevention: innings-weighted staff FIP -> expected RA/9.
    c_fip = league_fip_constant(pit)
    pit["FIP"] = fip(pit, c_fip)
    weighted = pit.groupby("team_id").apply(
        lambda g: np.average(g["FIP"], weights=g["outs"]), include_groups=False
    )
    defense = weighted * ra9_to_era  # ERA-scale FIP -> total runs allowed / 9

    prof = pd.DataFrame({"exp_rs": offense, "exp_ra": defense}).dropna()
    prof["run_diff"] = prof["exp_rs"] - prof["exp_ra"]
    prof.index.name = "team_id"
    return prof.reset_index()


# ---------------------------------------------------------------------------
# Ohtani two-way handling
# ---------------------------------------------------------------------------
def ohtani_split_report(bat: pd.DataFrame, pit: pd.DataFrame) -> dict:
    """Verify and report Ohtani's presence in BOTH matrices for the Dodgers."""
    b = bat[(bat["player_id"] == OHTANI_MLBAM_ID)
            & (bat["team_id"] == DODGERS_TEAM_ID)]
    p = pit[(pit["player_id"] == OHTANI_MLBAM_ID)
            & (pit["team_id"] == DODGERS_TEAM_ID)]
    report = {
        "hitting_rows": len(b),
        "pitching_rows": len(p),
        "PA": int(b["PA"].iloc[0]) if len(b) else 0,
        "RC": float(runs_created(b).iloc[0]) if len(b) else 0.0,
        "IP": float(p["outs"].iloc[0] / 3.0) if len(p) else 0.0,
    }
    log.info("Ohtani split — hitting: %d PA (%.1f RC) | pitching: %.1f IP",
             report["PA"], report["RC"], report["IP"])
    return report


# Replacement-level bat: ~20 runs per 600 PA below league average at
# severity 1.0 (FanGraphs replacement-level framework); the severity dial
# scales that gap, exactly as it scales the pitching FIP gap.
REPLACEMENT_RUNS_PER_600: float = 20.0


def apply_player_injury(profiles: pd.DataFrame, bat: pd.DataFrame,
                        pit: pd.DataFrame, player_id: int, team_id: int,
                        kind: str = "pitching",
                        severity: float = REPLACEMENT_FIP_PENALTY
                        ) -> pd.DataFrame:
    """Simulate losing one player's contribution to replacement level.

    ``kind`` selects which half of the player's value disappears:
    - ``"pitching"``: his innings go to a replacement arm with
      FIP = league FIP + ``severity`` runs.
    - ``"batting"``: his plate appearances go to a replacement bat whose
      RC/PA sits ``severity`` x 20-runs-per-600-PA below league average.
    - ``"both"``: both substitutions (a two-way player fully lost).

    The team's expected RS/G and/or RA/G are rescaled by the ratio of the
    substituted aggregate to the healthy aggregate, preserving the base
    profile's proration and league scaling.
    """
    if kind not in ("pitching", "batting", "both"):
        raise ValueError(f"unknown injury kind: {kind!r}")
    out = profiles.copy()
    mask = out["team_id"] == team_id
    if not mask.any():
        raise ValueError(f"team {team_id} has no roster profile")

    if kind in ("pitching", "both"):
        p = pit[pit["outs"] >= MIN_OUTS].copy()
        c_fip = league_fip_constant(p)
        p["FIP"] = fip(p, c_fip)
        staff = p[p["team_id"] == team_id].copy()
        hit_row = staff["player_id"] == player_id
        if not hit_row.any():
            raise ValueError("player has no qualifying pitching line "
                             "on this team")
        league_fip = np.average(p["FIP"], weights=p["outs"])
        healthy_fip = np.average(staff["FIP"], weights=staff["outs"])
        staff.loc[hit_row, "FIP"] = league_fip + severity
        injured_fip = np.average(staff["FIP"], weights=staff["outs"])
        base_ra = float(out.loc[mask, "exp_ra"].iloc[0])
        new_ra = base_ra * injured_fip / healthy_fip
        out.loc[mask, "exp_ra"] = new_ra
        log.info("injury (%s pitching): exp RA/G %.3f -> %.3f "
                 "(staff FIP %.2f -> %.2f)", player_id, base_ra, new_ra,
                 healthy_fip, injured_fip)

    if kind in ("batting", "both"):
        b = bat[bat["PA"] >= MIN_PA].copy()
        b["RC"] = runs_created(b)
        lineup = b[b["team_id"] == team_id]
        row = lineup[lineup["player_id"] == player_id]
        if row.empty:
            raise ValueError("player has no qualifying batting line "
                             "on this team")
        lg_rc_pa = b["RC"].sum() / max(b["PA"].sum(), 1)
        repl_rc_pa = max(lg_rc_pa - severity * REPLACEMENT_RUNS_PER_600 / 600.0,
                         0.0)
        healthy_rc = lineup["RC"].sum()
        injured_rc = (healthy_rc - float(row["RC"].iloc[0])
                      + repl_rc_pa * float(row["PA"].iloc[0]))
        base_rs = float(out.loc[mask, "exp_rs"].iloc[0])
        new_rs = base_rs * injured_rc / healthy_rc
        out.loc[mask, "exp_rs"] = new_rs
        log.info("injury (%s batting): exp RS/G %.3f -> %.3f",
                 player_id, base_rs, new_rs)

    out.loc[mask, "run_diff"] = (out.loc[mask, "exp_rs"]
                                 - out.loc[mask, "exp_ra"])
    return out


def apply_ohtani_injury(profiles: pd.DataFrame, bat: pd.DataFrame,
                        pit: pd.DataFrame,
                        fip_penalty: float = REPLACEMENT_FIP_PENALTY
                        ) -> pd.DataFrame:
    """Original Scenario B, now a special case of :func:`apply_player_injury`:
    Ohtani's pitching zeroed to replacement level, his DH bat kept."""
    return apply_player_injury(profiles, bat, pit, OHTANI_MLBAM_ID,
                               DODGERS_TEAM_ID, kind="pitching",
                               severity=fip_penalty)
