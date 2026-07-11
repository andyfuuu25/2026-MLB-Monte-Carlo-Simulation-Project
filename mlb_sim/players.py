"""Per-player rest-of-season Monte Carlo projections and team rate metrics.

Lineup / rotation projection
----------------------------
The "predicted starting lineup" is the team's nine highest-PA batters and the
rotation its five highest-IP arms — playing time observed to date is the
projection of who plays. Each player's remaining season is then simulated
directly:

- **Hitters**: every remaining plate appearance is a draw from a multinomial
  over the player's observed per-PA outcome rates (1B, 2B, 3B, HR, BB+HBP,
  SF, out). ``n_sims`` seasons are drawn at once with
  ``rng.multinomial(rem_PA, rates, size=n_sims)``, added to the current
  line, and summarized as mean and 5th–95th percentile full-season stats.
- **Pitchers**: remaining-season event counts (ER, H, BB, K) are drawn from
  Poisson distributions at the player's observed per-out rates over his
  projected remaining outs — the classical model for rare-event counts in a
  fixed exposure.

Team rate metrics
-----------------
Aggregates player lines into team AVG/OBP/SLG/OPS and ERA/WHIP/K/9 etc.,
plus the normalized indices **OPS+** = 100·(OBP/lgOBP + SLG/lgSLG − 1) and
**ERA+** = 100·(lgERA/ERA). Both are computed *without park factors* (the
official versions are park-adjusted), so treat them as league-relative, not
park-neutral.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _slash(h, bb, hbp, sf, ab, tb, pa):
    """AVG / OBP / SLG from counting stats (vectorized or scalar)."""
    avg = h / np.maximum(ab, 1)
    obp = (h + bb + hbp) / np.maximum(ab + bb + hbp + sf, 1)
    slg = tb / np.maximum(ab, 1)
    return avg, obp, slg


def _band(x: np.ndarray, digits: int = 3) -> dict:
    return {"mean": round(float(np.mean(x)), digits),
            "p5": round(float(np.percentile(x, 5)), digits),
            "p95": round(float(np.percentile(x, 95)), digits)}


# ---------------------------------------------------------------------------
# Hitters
# ---------------------------------------------------------------------------
def project_hitters(bat: pd.DataFrame, team_id: int, team_gp: int,
                    rem_games: int, n_sims: int, rng: np.random.Generator,
                    lineup_size: int = 9) -> list[dict]:
    """Monte Carlo full-season lines for the team's projected lineup."""
    lineup = (bat[bat["team_id"] == team_id]
              .sort_values("PA", ascending=False).head(lineup_size))
    out = []
    for _, r in lineup.iterrows():
        pa, ab = int(r["PA"]), int(r["AB"])
        h, d2, d3, hr = int(r["H"]), int(r["2B"]), int(r["3B"]), int(r["HR"])
        bb, hbp, sf = int(r["BB"]), int(r["HBP"]), int(r["SF"])
        singles = h - d2 - d3 - hr
        tb = h + d2 + 2 * d3 + 3 * hr

        # Per-PA outcome rates -> remaining-PA multinomial draws.
        rates = np.array([singles, d2, d3, hr, bb + hbp, sf], dtype=float) / pa
        rates = np.append(np.clip(rates, 0, None), 0)
        rates[-1] = max(1.0 - rates[:-1].sum(), 0.0)          # outs
        rem_pa = int(round(pa / max(team_gp, 1) * rem_games))
        draws = rng.multinomial(rem_pa, rates, size=n_sims)   # (sims, 7)
        s1, s2, s3, shr, sbb, ssf = (draws[:, i] for i in range(6))

        f_h = h + s1 + s2 + s3 + shr
        f_ab = ab + rem_pa - sbb - ssf
        f_tb = tb + s1 + 2 * s2 + 3 * s3 + 4 * shr
        f_avg, f_obp, f_slg = _slash(f_h, bb + sbb, 0, sf + ssf,
                                     f_ab, f_tb, pa + rem_pa)

        avg, obp, slg = _slash(h, bb, hbp, sf, ab, tb, pa)
        out.append({
            "player_id": int(r["player_id"]), "name": r["name"],
            "pa": pa, "hr": hr,
            "avg": round(float(avg), 3), "obp": round(float(obp), 3),
            "slg": round(float(slg), 3), "ops": round(float(obp + slg), 3),
            "proj_pa": pa + rem_pa,
            "proj_hr": _band(hr + shr, 1),
            "proj_ops": _band(f_obp + f_slg, 3),
        })
    return out


# ---------------------------------------------------------------------------
# Pitchers
# ---------------------------------------------------------------------------
def project_pitchers(pit: pd.DataFrame, team_id: int, team_gp: int,
                     rem_games: int, n_sims: int, rng: np.random.Generator,
                     rotation_size: int = 5) -> list[dict]:
    """Monte Carlo full-season lines for the team's projected rotation."""
    rotation = (pit[pit["team_id"] == team_id]
                .sort_values("outs", ascending=False).head(rotation_size))
    out = []
    for _, r in rotation.iterrows():
        outs = int(r["outs"])
        ip = outs / 3.0
        er, ha, bb, k = int(r["ER"]), int(r["HA"]), int(r["BB"]), int(r["K"])

        rem_outs = int(round(outs / max(team_gp, 1) * rem_games))
        rem_ip = rem_outs / 3.0
        # Poisson event counts at observed per-out rates over remaining outs.
        lam = lambda c: max(c / max(outs, 1) * rem_outs, 1e-9)
        s_er = rng.poisson(lam(er), n_sims)
        s_ha = rng.poisson(lam(ha), n_sims)
        s_bb = rng.poisson(lam(bb), n_sims)
        s_k = rng.poisson(lam(k), n_sims)

        f_ip = ip + rem_ip
        out.append({
            "player_id": int(r["player_id"]), "name": r["name"],
            "ip": round(ip, 1),
            "era": round(9 * er / max(ip, 1e-9), 2),
            "whip": round((ha + bb) / max(ip, 1e-9), 2),
            "k9": round(9 * k / max(ip, 1e-9), 1),
            "proj_ip": round(f_ip, 0),
            "proj_era": _band(9 * (er + s_er) / f_ip, 2),
            "proj_whip": _band((ha + s_ha + bb + s_bb) / f_ip, 2),
            "proj_k": _band(k + s_k, 0),
        })
    return out


# ---------------------------------------------------------------------------
# Team rate metrics
# ---------------------------------------------------------------------------
def _batting_rates(df: pd.DataFrame) -> dict:
    tb = df["H"] + df["2B"] + 2 * df["3B"] + 3 * df["HR"]
    avg, obp, slg = _slash(df["H"].sum(), df["BB"].sum(), df["HBP"].sum(),
                           df["SF"].sum(), df["AB"].sum(), tb.sum(),
                           df["PA"].sum())
    return {"avg": float(avg), "obp": float(obp), "slg": float(slg),
            "ops": float(obp + slg), "hr": int(df["HR"].sum()),
            "bb_pct": float(df["BB"].sum() / max(df["PA"].sum(), 1) * 100)}


def _pitching_rates(df: pd.DataFrame) -> dict:
    ip = df["outs"].sum() / 3.0
    return {"era": float(9 * df["ER"].sum() / ip),
            "whip": float((df["HA"].sum() + df["BB"].sum()) / ip),
            "k9": float(9 * df["K"].sum() / ip),
            "bb9": float(9 * df["BB"].sum() / ip),
            "hr9": float(9 * df["HR"].sum() / ip)}


def team_rate_metrics(bat: pd.DataFrame, pit: pd.DataFrame,
                      team_id: int) -> dict:
    """Team batting/pitching rates vs league, with OPS+ / ERA+ indices."""
    tb_, lb_ = bat[bat["team_id"] == team_id], bat
    tp_, lp_ = pit[pit["team_id"] == team_id], pit
    team_b, lg_b = _batting_rates(tb_), _batting_rates(lb_)
    team_p, lg_p = _pitching_rates(tp_), _pitching_rates(lp_)

    ops_plus = 100 * (team_b["obp"] / lg_b["obp"]
                      + team_b["slg"] / lg_b["slg"] - 1)
    era_plus = 100 * lg_p["era"] / team_p["era"]
    return {
        "batting": {k: round(v, 3 if k in ("avg", "obp", "slg", "ops") else 1)
                    for k, v in team_b.items()},
        "batting_lg": {k: round(v, 3 if k in ("avg", "obp", "slg", "ops") else 1)
                       for k, v in lg_b.items()},
        "pitching": {k: round(v, 2) for k, v in team_p.items()},
        "pitching_lg": {k: round(v, 2) for k, v in lg_p.items()},
        "ops_plus": round(ops_plus, 0),
        "era_plus": round(era_plus, 0),
    }
