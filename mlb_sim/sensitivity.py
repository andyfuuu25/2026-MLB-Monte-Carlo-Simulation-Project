"""Sensitivity analysis: the Dodgers season as a function of Ohtani's arm.

Scenario A (base case)   — Ohtani fully healthy: his batting line contributes
                           to LAD expected runs scored AND his pitching line
                           to LAD expected runs allowed.
Scenario B (stress case) — simulated pitching injury: his innings are handed
                           to a replacement-level arm (league FIP + 1.00);
                           his DH bat stays in the lineup.

Both scenarios re-run the *identical* 10,000-trial engine with the same
random seed, so every difference in the output distributions is attributable
to the single changed parameter (common random numbers variance reduction).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DODGERS_TEAM_ID, N_SIMS, RANDOM_SEED
from .features import apply_ohtani_injury
from .model import WinModel
from .simulate import SimResult, run_simulation

log = logging.getLogger(__name__)


@dataclass
class ScenarioOutcome:
    label: str
    result: SimResult
    lad_wins: np.ndarray       # (n_sims,) LAD regular-season win totals
    lad_playoff_pct: float
    lad_division_pct: float
    lad_pennant_pct: float
    lad_ws_pct: float

    @classmethod
    def from_result(cls, label: str, result: SimResult) -> "ScenarioOutcome":
        col = int(np.flatnonzero(result.team_ids == DODGERS_TEAM_ID)[0])
        n = result.wins.shape[0]
        return cls(
            label=label,
            result=result,
            lad_wins=result.wins[:, col],
            lad_playoff_pct=result.made_playoffs[:, col].mean() * 100,
            lad_division_pct=result.won_division[:, col].mean() * 100,
            lad_pennant_pct=result.won_pennant[:, col].mean() * 100,
            lad_ws_pct=(result.champion == col).sum() / n * 100,
        )


def run_ohtani_sensitivity(target_games: pd.DataFrame, elo_now: dict[int, float],
                           profiles: pd.DataFrame, bat: pd.DataFrame,
                           pit: pd.DataFrame, teams: pd.DataFrame,
                           model: WinModel, n_sims: int = N_SIMS
                           ) -> tuple[ScenarioOutcome, ScenarioOutcome]:
    """Run base and stress scenarios; return (scenario_a, scenario_b)."""
    log.info("Scenario A — Ohtani healthy (two-way)")
    res_a = run_simulation(target_games, elo_now, profiles, teams, model,
                           n_sims=n_sims, seed=RANDOM_SEED)
    a = ScenarioOutcome.from_result("A: Ohtani healthy", res_a)

    log.info("Scenario B — Ohtani pitching injury (bat only)")
    injured = apply_ohtani_injury(profiles, bat, pit)
    res_b = run_simulation(target_games, elo_now, injured, teams, model,
                           n_sims=n_sims, seed=RANDOM_SEED)
    b = ScenarioOutcome.from_result("B: pitching injury", res_b)
    return a, b


def sensitivity_table(a: ScenarioOutcome, b: ScenarioOutcome) -> pd.DataFrame:
    """Side-by-side LAD metrics with the injury-attributable delta."""
    rows = []
    for metric, fn in (
        ("Expected wins", lambda s: s.lad_wins.mean()),
        ("Wins, 5th pct", lambda s: np.percentile(s.lad_wins, 5)),
        ("Wins, 95th pct", lambda s: np.percentile(s.lad_wins, 95)),
        ("Win NL West %", lambda s: s.lad_division_pct),
        ("Make playoffs %", lambda s: s.lad_playoff_pct),
        ("Win pennant %", lambda s: s.lad_pennant_pct),
        ("Win World Series %", lambda s: s.lad_ws_pct),
    ):
        va, vb = fn(a), fn(b)
        rows.append({"metric": metric, "A_healthy": round(va, 2),
                     "B_injured": round(vb, 2), "delta": round(vb - va, 2)})
    return pd.DataFrame(rows)
