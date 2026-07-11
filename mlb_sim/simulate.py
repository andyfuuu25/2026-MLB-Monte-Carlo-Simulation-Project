"""Vectorized Monte Carlo engine: remaining schedule + full postseason.

Regular season
--------------
The N_SIMS trials are carried as NumPy arrays of shape ``(n_sims, 30)``.
Each trial first draws every team's *latent strength* once from
``N(rating, TALENT_SIGMA_ELO)`` and holds it fixed — this propagates
parameter uncertainty (how good is this roster really?) on top of the
binomial game noise, which on its own makes season win distributions far too
narrow. The engine then walks the remaining schedule chronologically; for
each fixture it computes every trial's win probability from the hybrid
classifier (per-trial latent Elo difference + static roster run-differential
difference) and draws Bernoulli outcomes. Games are conditionally
independent given the drawn talent.

The legacy in-trial Elo drift (updating ratings on the trial's own simulated
outcomes) is available via ``ELO_IN_TRIAL_DRIFT`` but off by default: a
simulated result carries no information about true talent, so drift adds an
arbitrary random walk rather than calibrated uncertainty.

Postseason (2022+ MLB format)
-----------------------------
Per league: 3 division winners seeded 1-3 by record, best 3 remaining teams
seeded 4-6. Seeds 1-2 receive byes. Wild Card: 3v6 and 4v5, best-of-3.
Division Series: 1 vs (4/5), 2 vs (3/6), best-of-5. LCS best-of-7, World
Series best-of-7 with home advantage to the better regular-season record.

Simplifications (documented in README): the higher seed is treated as the
home side in every series game (real series alternate 2-3-2 / 2-2-1), and
ties are broken by an i.i.d. random jitter rather than MLB's head-to-head
rules. Each trial's latent strengths carry unchanged into its bracket, so
the postseason is consistent with that trial's regular season by design.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import comb

import numpy as np
import pandas as pd

from .config import (ELO_IN_TRIAL_DRIFT, ELO_K, N_SIMS, RANDOM_SEED,
                     SERIES_DIVISION, SERIES_LCS, SERIES_WILDCARD,
                     SERIES_WORLD, TALENT_SIGMA_ELO)
from .elo import expected_home_score
from .model import WinModel

log = logging.getLogger(__name__)


@dataclass
class SimResult:
    team_ids: np.ndarray          # (30,) MLBAM ids, defines column order
    wins: np.ndarray              # (n_sims, 30) final regular-season wins
    elo_final: np.ndarray         # (n_sims, 30) end-of-season Elo
    made_playoffs: np.ndarray     # (n_sims, 30) bool
    won_division: np.ndarray      # (n_sims, 30) bool
    won_pennant: np.ndarray       # (n_sims, 30) bool
    champion: np.ndarray          # (n_sims,) column index of WS winner


# ---------------------------------------------------------------------------
# Regular season
# ---------------------------------------------------------------------------
def simulate_regular_season(remaining: pd.DataFrame, idx_of: dict[int, int],
                            elo0: np.ndarray, wins0: np.ndarray,
                            rundiff: np.ndarray, model: WinModel,
                            n_sims: int, rng: np.random.Generator,
                            talent_sigma: float = TALENT_SIGMA_ELO,
                            elo_drift: bool = ELO_IN_TRIAL_DRIFT
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Play every remaining fixture across all trials. Returns (wins, elo).

    Each trial's latent team strengths are drawn once (talent uncertainty)
    and held fixed; game outcomes are then conditionally independent unless
    ``elo_drift`` re-enables the legacy in-trial random walk.
    """
    elo = np.tile(elo0, (n_sims, 1))
    if talent_sigma > 0.0:
        elo += rng.normal(0.0, talent_sigma, size=elo.shape)
    wins = np.tile(wins0.astype(np.int32), (n_sims, 1))

    h_idx = remaining["home_id"].map(idx_of).to_numpy()
    a_idx = remaining["away_id"].map(idx_of).to_numpy()
    rd_diff = rundiff[h_idx] - rundiff[a_idx]  # static per fixture

    for h, a, rd in zip(h_idx, a_idx, rd_diff):
        eh, ea = elo[:, h], elo[:, a]
        p_win = model.prob_home_win(eh - ea, rd)
        home_won = rng.random(n_sims) < p_win

        if elo_drift:  # legacy: update ratings on simulated outcomes
            delta = ELO_K * (home_won - expected_home_score(eh, ea))
            elo[:, h] += delta
            elo[:, a] -= delta
        wins[:, h] += home_won
        wins[:, a] += ~home_won
    return wins, elo


# ---------------------------------------------------------------------------
# Postseason
# ---------------------------------------------------------------------------
def _seed_league(jit_wins: np.ndarray, league_cols: np.ndarray,
                 division_cols: list[np.ndarray]) -> np.ndarray:
    """Seeds 1-6 for one league, per trial. Returns (n_sims, 6) column ids.

    ``jit_wins`` is the (n_sims, 30) win matrix plus a small uniform jitter
    that breaks ties (a random coin flip, applied identically everywhere).
    """
    jl = jit_wins[:, league_cols]                       # (S, 15)
    local = {c: i for i, c in enumerate(league_cols)}

    # Division winners -> seeds 1-3, ordered by record.
    winner_local = np.stack(
        [np.array([local[c] for c in div])[
            np.argmax(jit_wins[:, div], axis=1)] for div in division_cols],
        axis=1)                                          # (S, 3) local pos
    w_vals = np.take_along_axis(jl, winner_local, axis=1)
    order = np.argsort(-w_vals, axis=1)
    seeds123 = np.take_along_axis(winner_local, order, axis=1)

    # Wild cards -> best three non-winners, seeds 4-6.
    pool = jl.copy()
    np.put_along_axis(pool, winner_local, -np.inf, axis=1)
    top3 = np.argpartition(-pool, 3, axis=1)[:, :3]
    t_vals = np.take_along_axis(pool, top3, axis=1)
    seeds456 = np.take_along_axis(top3, np.argsort(-t_vals, axis=1), axis=1)

    return league_cols[np.concatenate([seeds123, seeds456], axis=1)]


def _series_win_prob(p_game: np.ndarray, best_of: int) -> np.ndarray:
    """P(higher seed takes a best-of-N) with constant per-game win prob.

    Closed form: probability of >= (N+1)/2 successes in N Bernoulli(p) trials
    (equivalent to playing out all N games).
    """
    need = best_of // 2 + 1
    q = 1.0 - p_game
    total = np.zeros_like(p_game)
    for j in range(need, best_of + 1):
        total += comb(best_of, j) * p_game**j * q ** (best_of - j)
    return total


def _play_series(hi: np.ndarray, lo: np.ndarray, best_of: int,
                 elo: np.ndarray, rundiff: np.ndarray, model: WinModel,
                 rng: np.random.Generator) -> np.ndarray:
    """Resolve one series per trial; ``hi`` holds home advantage throughout."""
    rows = np.arange(len(hi))
    p_game = model.prob_home_win(
        elo[rows, hi] - elo[rows, lo], rundiff[hi] - rundiff[lo])
    hi_wins = rng.random(len(hi)) < _series_win_prob(p_game, best_of)
    return np.where(hi_wins, hi, lo)


def simulate_postseason(wins: np.ndarray, elo: np.ndarray,
                        rundiff: np.ndarray, model: WinModel,
                        league_cols: dict[str, np.ndarray],
                        division_cols: dict[str, list[np.ndarray]],
                        rng: np.random.Generator
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Full bracket for every trial.

    Returns (made_playoffs, won_division, won_pennant, champion).
    """
    n_sims, n_teams = wins.shape
    jit = wins + rng.uniform(0.0, 0.5, size=wins.shape)  # random tie-break
    rows = np.arange(n_sims)

    made = np.zeros((n_sims, n_teams), dtype=bool)
    div_w = np.zeros((n_sims, n_teams), dtype=bool)
    pennant = np.zeros((n_sims, n_teams), dtype=bool)
    league_champ: dict[str, np.ndarray] = {}

    for lg, cols in league_cols.items():
        seeds = _seed_league(jit, cols, division_cols[lg])  # (S, 6)
        np.put_along_axis(made, seeds, True, axis=1)
        np.put_along_axis(div_w, seeds[:, :3], True, axis=1)

        s = [seeds[:, k] for k in range(6)]  # seed 1..6 team columns
        wc_45 = _play_series(s[3], s[4], SERIES_WILDCARD, elo, rundiff, model, rng)
        wc_36 = _play_series(s[2], s[5], SERIES_WILDCARD, elo, rundiff, model, rng)
        ds_1 = _play_series(s[0], wc_45, SERIES_DIVISION, elo, rundiff, model, rng)
        ds_2 = _play_series(s[1], wc_36, SERIES_DIVISION, elo, rundiff, model, rng)
        # LCS home advantage: better regular-season record (jittered).
        hi_first = jit[rows, ds_1] >= jit[rows, ds_2]
        hi = np.where(hi_first, ds_1, ds_2)
        lo = np.where(hi_first, ds_2, ds_1)
        champ = _play_series(hi, lo, SERIES_LCS, elo, rundiff, model, rng)
        pennant[rows, champ] = True
        league_champ[lg] = champ

    al, nl = league_champ["American League"], league_champ["National League"]
    al_home = jit[rows, al] >= jit[rows, nl]
    hi = np.where(al_home, al, nl)
    lo = np.where(al_home, nl, al)
    champion = _play_series(hi, lo, SERIES_WORLD, elo, rundiff, model, rng)
    return made, div_w, pennant, champion


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_simulation(target_games: pd.DataFrame, elo_now: dict[int, float],
                   profiles: pd.DataFrame, teams: pd.DataFrame,
                   model: WinModel, n_sims: int = N_SIMS,
                   seed: int = RANDOM_SEED,
                   talent_sigma: float = TALENT_SIGMA_ELO,
                   elo_drift: bool = ELO_IN_TRIAL_DRIFT) -> SimResult:
    """Simulate the rest of the target season + postseason, n_sims times.

    Completed games in ``target_games`` are locked in as actual results;
    only fixtures with state='future' are simulated. Each trial draws latent
    team strengths once (``talent_sigma``, in Elo points) and carries them
    through both the regular season and the postseason bracket.
    """
    rng = np.random.default_rng(seed)
    team_ids = teams["team_id"].to_numpy()
    idx_of = {t: i for i, t in enumerate(team_ids)}

    elo0 = np.array([elo_now[t] for t in team_ids])
    rundiff = (profiles.set_index("team_id")["run_diff"]
               .reindex(team_ids).fillna(0.0).to_numpy())

    finals = target_games[target_games["state"] == "final"]
    wins0 = np.zeros(len(team_ids))
    for col, val in (("home_id", 1), ("away_id", 0)):
        won = finals.loc[finals["home_win"] == val, col].map(idx_of)
        np.add.at(wins0, won.to_numpy(), 1)

    remaining = target_games[target_games["state"] == "future"]
    log.info("simulating %d remaining games x %d trials "
             "(%d results locked in)", len(remaining), n_sims, len(finals))
    wins, elo = simulate_regular_season(
        remaining, idx_of, elo0, wins0, rundiff, model, n_sims, rng,
        talent_sigma=talent_sigma, elo_drift=elo_drift)

    league_cols = {
        lg: np.flatnonzero((teams["league"] == lg).to_numpy())
        for lg in ("American League", "National League")}
    division_cols = {
        lg: [np.flatnonzero((teams["division"] == d).to_numpy())
             for d in sorted(teams.loc[teams["league"] == lg, "division"].unique())]
        for lg in league_cols}

    made, div_w, pennant, champion = simulate_postseason(
        wins, elo, rundiff, model, league_cols, division_cols, rng)

    return SimResult(team_ids=team_ids, wins=wins, elo_final=elo,
                     made_playoffs=made, won_division=div_w,
                     won_pennant=pennant, champion=champion)


def summarize(result: SimResult, teams: pd.DataFrame) -> pd.DataFrame:
    """Per-team ranking table: expected wins, odds at each October gate."""
    n_sims = result.wins.shape[0]
    champ_counts = np.bincount(result.champion, minlength=len(result.team_ids))
    df = pd.DataFrame({
        "team_id": result.team_ids,
        "exp_wins": result.wins.mean(axis=0),
        "wins_p5": np.percentile(result.wins, 5, axis=0),
        "wins_p95": np.percentile(result.wins, 95, axis=0),
        "div_pct": result.won_division.mean(axis=0) * 100,
        "playoff_pct": result.made_playoffs.mean(axis=0) * 100,
        "pennant_pct": result.won_pennant.mean(axis=0) * 100,
        "ws_pct": champ_counts / n_sims * 100,
        "elo_final": result.elo_final.mean(axis=0),
    })
    df = df.merge(teams[["team_id", "abbrev", "name", "league", "division"]],
                  on="team_id")
    return df.sort_values("ws_pct", ascending=False).reset_index(drop=True)
