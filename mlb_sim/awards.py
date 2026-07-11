"""MVP and Cy Young award simulations from Statcast advanced metrics.

Candidate talent — metrics chosen from the literature
-----------------------------------------------------
**Batters (MVP)** — the research on award voting shows fWAR-style value is
by far the strongest correlate of MVP votes (FanGraphs community voting
study; the "Award Index"), with team success a secondary narrative factor.
Candidate value is therefore the composite runs-above-replacement from
``impact.py`` (wOBA batting runs + Statcast fielding runs), and the
*rest-of-season* talent estimate blends **xwOBA** with actual wOBA
(Statcast expected statistics strip batted-ball luck from quality of
contact). Barrel% / hard-hit% / exit velocity are surfaced as the
quality-of-contact evidence. Two-way players carry their pitching runs too.

**Pitchers (Cy Young)** — recent voting tracks pitcher value metrics
(ERA/FIP/K) almost monotonically. Rest-of-season talent RA9 blends
**SIERA (50%) + FIP (25%) + xERA (25%)**, then a **stuff adjustment**
nudges talent by a z-composite of the physical inputs the Stuff+ research
identifies as most predictive — fastball velocity, usage-weighted spin
rate, four-seam induced vertical break, and whiff% — because stuff models
predict future performance beyond results-based estimators and stabilize
in small samples (Sarris & Bay's Stuff+/Pitching+; FanGraphs primer).
Pitch mix, zone%, and chase% are surfaced on every candidate card.

Award simulation
----------------
Awards are decided *inside* the Monte Carlo: in every trial the candidate's
full-season value is his current runs plus a talent-rate projection over his
remaining workload plus sampling noise, and the ballot score adds a modest
team-success bonus wired to THAT trial's simulated outcome (+6 runs if his
team makes the playoffs, +3 more for a division title) plus Gumbel voter
noise. The award winner in each league is the argmax per trial; reported
probabilities are shares of trials won.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .impact import RUNS_PER_WIN, WOBA_SCALE
from .simulate import SimResult

log = logging.getLogger(__name__)

N_CANDIDATES = 12          # per league per award
XWOBA_WEIGHT = 0.6         # xwOBA vs wOBA in rest-of-season talent
SIERA_W, FIP_W, XERA_W = 0.50, 0.25, 0.25
STUFF_RA9_PER_Z = 0.12     # RA9 credit per stuff-composite z (small nudge)
PLAYOFF_BONUS = 6.0        # runs-equivalent ballot boosts
DIVISION_BONUS = 3.0
VOTER_NOISE = 5.0          # Gumbel scale, runs
BAT_NOISE_PER_PA = 0.5     # sampling sd = this * sqrt(remaining PA)
PIT_NOISE_PER_IP = 0.7


def _z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return ((s - s.mean()) / sd).clip(-3, 3).fillna(0.0)


def _pace(now: pd.Series, gp: int, remaining: pd.Series) -> pd.Series:
    """Project a workload counting stat over each player's remaining games."""
    return now / max(gp, 1) * remaining


# ---------------------------------------------------------------------------
# Candidate pools
# ---------------------------------------------------------------------------
def mvp_pool(values: pd.DataFrame, xbat: pd.DataFrame, scb: pd.DataFrame,
             teams: pd.DataFrame, rem_games: pd.Series,
             team_gp: pd.Series) -> pd.DataFrame:
    """Top MVP candidates per league with projection rates and evidence."""
    b = values[values["role"] == "bat"].copy()
    p = values[values["role"] == "pitch"][["player_id", "team_id", "rar",
                                           "talent_ra", "outs"]]
    p = p.rename(columns={"rar": "pitch_rar", "outs": "pitch_outs"})

    x = xbat.set_index("player_id")[["est_woba"]]
    s = scb.set_index("player_id")[["brl_percent", "ev95percent",
                                    "avg_hit_speed"]]
    b = (b.join(x, on="player_id").join(s, on="player_id")
         .merge(p, on=["player_id", "team_id"], how="left"))

    lg_woba = float((b["woba"] * b["PA"]).sum() / b["PA"].sum())
    talent = (XWOBA_WEIGHT * b["est_woba"].fillna(b["woba"])
              + (1 - XWOBA_WEIGHT) * b["woba"])
    b["rate"] = (talent - lg_woba) / WOBA_SCALE          # runs per PA
    b["rem_pa"] = _pace(b["PA"], int(team_gp.median()),
                        b["team_id"].map(rem_games))
    # Two-way pitching value projected at the same innings pace.
    rem_outs = _pace(b["pitch_outs"].fillna(0), int(team_gp.median()),
                     b["team_id"].map(rem_games))
    b["pitch_proj"] = b["pitch_rar"].fillna(0.0) * np.where(
        b["pitch_outs"].fillna(0) > 0,
        1 + rem_outs / b["pitch_outs"].replace(0, np.nan).fillna(1), 0)

    b["proj_value"] = (b["rar"] + b["rate"] * b["rem_pa"] + b["pitch_proj"])
    b = b.merge(teams[["team_id", "abbrev", "league"]], on="team_id")
    return (b.sort_values("proj_value", ascending=False)
            .groupby("league").head(N_CANDIDATES).reset_index(drop=True))


def cy_pool(values: pd.DataFrame, xpit: pd.DataFrame, disc: pd.DataFrame,
            arsenal: pd.DataFrame, movement: pd.DataFrame,
            teams: pd.DataFrame, rem_games: pd.Series,
            team_gp: pd.Series) -> pd.DataFrame:
    """Top Cy Young candidates per league with stuff-informed talent."""
    p = values[values["role"] == "pitch"].copy()

    p = p.join(xpit.set_index("player_id")[["xera"]], on="player_id")
    p = p.join(disc.set_index("player_id")[
        ["k_percent", "whiff_percent", "oz_swing_percent",
         "in_zone_percent", "f_strike_percent"]], on="player_id")
    p = p.join(movement.set_index("player_id")[
        ["pitcher_break_z_induced"]], on="player_id")

    # Arsenal: fastball velocity, usage-weighted spin, top-3 mix string.
    if len(arsenal):
        ars = arsenal.set_index("player_id")
        usage = ars[[c for c in ars.columns if c.startswith("n_")]]
        speed = ars[[c for c in ars.columns if c.endswith("_avg_speed")]]
        spin = ars[[c for c in ars.columns if c.endswith("_avg_spin")]]
        fb = speed.reindex(columns=["ff_avg_speed", "si_avg_speed"]).max(axis=1)
        u = usage.fillna(0.0)
        u_aligned = u.to_numpy()
        spin_aligned = spin.reindex(
            columns=[c.replace("n_", "") + "_avg_spin" for c in u.columns]
        ).to_numpy()
        with np.errstate(invalid="ignore"):
            wspin = np.nansum(u_aligned * spin_aligned, axis=1) / np.maximum(
                np.nansum(u_aligned * ~np.isnan(spin_aligned), axis=1), 1e-9)

        def mix_str(row) -> str:
            top = row.sort_values(ascending=False).head(3)
            return " · ".join(f"{c[2:].upper()} {v:.0f}%"
                              for c, v in top.items() if v > 0)

        ars_out = pd.DataFrame({
            "fb_velo": fb, "wt_spin": wspin,
            "mix": u.apply(mix_str, axis=1)})
        p = p.join(ars_out, on="player_id")
    else:
        p[["fb_velo", "wt_spin"]] = np.nan
        p["mix"] = ""

    # Stuff composite: velocity, spin, movement, whiff (Stuff+ ingredients).
    p["stuff_z"] = pd.concat([
        _z(p["fb_velo"]), _z(p["wt_spin"]),
        _z(p["pitcher_break_z_induced"]), _z(p["whiff_percent"]),
    ], axis=1).mean(axis=1)

    talent = (SIERA_W * p["siera"] + FIP_W * p["fip"]
              + XERA_W * p["xera"].fillna(p["fip"]))
    p["talent_ra9"] = talent - STUFF_RA9_PER_Z * p["stuff_z"]

    lg_talent = float(np.average(p["talent_ra9"], weights=p["outs"]))
    p["rate9"] = lg_talent + 1.0 - p["talent_ra9"]       # runs saved / 9 IP
    p["rem_ip"] = _pace(p["outs"] / 3.0, int(team_gp.median()),
                        p["team_id"].map(rem_games))
    p["proj_value"] = p["rar"] + p["rate9"] * p["rem_ip"] / 9.0

    p = p.merge(teams[["team_id", "abbrev", "league"]], on="team_id")
    return (p.sort_values("proj_value", ascending=False)
            .groupby("league").head(N_CANDIDATES).reset_index(drop=True))


# ---------------------------------------------------------------------------
# Trial-level award voting
# ---------------------------------------------------------------------------
def _vote(pool: pd.DataFrame, result: SimResult, rng: np.random.Generator,
          value_now: np.ndarray, proj_mean: np.ndarray, noise_sd: np.ndarray
          ) -> dict[str, np.ndarray]:
    """Award-win probability per candidate, by league."""
    n_sims = result.wins.shape[0]
    col_of = {t: i for i, t in enumerate(result.team_ids)}
    cols = pool["team_id"].map(col_of).to_numpy()

    proj = (value_now[:, None]
            + proj_mean[:, None]
            + rng.normal(0.0, noise_sd[:, None], (len(pool), n_sims)))
    ballot = (proj
              + PLAYOFF_BONUS * result.made_playoffs[:, cols].T
              + DIVISION_BONUS * result.won_division[:, cols].T
              + rng.gumbel(0.0, VOTER_NOISE, (len(pool), n_sims)))

    out: dict[str, np.ndarray] = {}
    for lg in ("American League", "National League"):
        idx = np.flatnonzero((pool["league"] == lg).to_numpy())
        winners = idx[np.argmax(ballot[idx], axis=0)]
        probs = np.zeros(len(pool))
        counts = np.bincount(winners, minlength=len(pool))
        probs[: len(counts)] = counts / n_sims
        out[lg] = probs
    return out


def simulate_awards(mvp: pd.DataFrame, cy: pd.DataFrame, result: SimResult,
                    rng: np.random.Generator) -> dict:
    """Run both award votes across every Monte Carlo trial."""
    mvp_probs = _vote(
        mvp, result, rng,
        value_now=(mvp["rar"] + mvp["pitch_rar"].fillna(0)).to_numpy(),
        proj_mean=(mvp["rate"] * mvp["rem_pa"]
                   + mvp["pitch_proj"] - mvp["pitch_rar"].fillna(0)).to_numpy(),
        noise_sd=(BAT_NOISE_PER_PA * np.sqrt(mvp["rem_pa"].clip(lower=1))
                  ).to_numpy())
    cy_probs = _vote(
        cy, result, rng,
        value_now=cy["rar"].to_numpy(),
        proj_mean=(cy["rate9"] * cy["rem_ip"] / 9.0).to_numpy(),
        noise_sd=(PIT_NOISE_PER_IP * np.sqrt(cy["rem_ip"].clip(lower=1))
                  ).to_numpy())

    def rows(pool: pd.DataFrame, probs: dict, lg: str, kind: str) -> list:
        idx = pool.index[pool["league"] == lg]
        out = []
        for i in idx:
            r = pool.loc[i]
            m = ({"woba": round(r["woba"], 3),
                  "xwoba": (None if pd.isna(r["est_woba"])
                            else round(r["est_woba"], 3)),
                  "wrc_plus": round(r["wrc_plus"], 0),
                  "brl": (None if pd.isna(r["brl_percent"])
                          else round(r["brl_percent"], 1)),
                  "hh": (None if pd.isna(r["ev95percent"])
                         else round(r["ev95percent"], 1)),
                  "frp": round(float(r["frp"]), 0),
                  "two_way": bool(r["pitch_rar"] > 0)
                  } if kind == "mvp" else
                 {"fip": round(r["fip"], 2), "siera": round(r["siera"], 2),
                  "xera": (None if pd.isna(r["xera"]) else round(r["xera"], 2)),
                  "k_pct": (None if pd.isna(r["k_percent"])
                            else round(r["k_percent"], 1)),
                  "whiff": (None if pd.isna(r["whiff_percent"])
                            else round(r["whiff_percent"], 1)),
                  "zone": (None if pd.isna(r["in_zone_percent"])
                           else round(r["in_zone_percent"], 1)),
                  "chase": (None if pd.isna(r["oz_swing_percent"])
                            else round(r["oz_swing_percent"], 1)),
                  "velo": (None if pd.isna(r["fb_velo"])
                           else round(r["fb_velo"], 1)),
                  "spin": (None if pd.isna(r["wt_spin"])
                           else int(r["wt_spin"])),
                  "ivb": (None if pd.isna(r["pitcher_break_z_induced"])
                          else round(r["pitcher_break_z_induced"], 1)),
                  "mix": r.get("mix", ""),
                  "stuff_z": round(float(r["stuff_z"]), 2)})
            out.append({
                "player_id": int(r["player_id"]), "name": r["name"],
                "team": r["abbrev"], "prob": round(probs[lg][i] * 100, 1),
                "war_proj": round(float(r["proj_value"]) / RUNS_PER_WIN, 1),
                "metrics": m})
        out.sort(key=lambda x: -x["prob"])
        return out

    return {lgk: rows(pool, probs, lg, kind)
            for lgk, pool, probs, lg, kind in (
                ("al_mvp", mvp, mvp_probs, "American League", "mvp"),
                ("nl_mvp", mvp, mvp_probs, "National League", "mvp"),
                ("al_cy", cy, cy_probs, "American League", "cy"),
                ("nl_cy", cy, cy_probs, "National League", "cy"))}
