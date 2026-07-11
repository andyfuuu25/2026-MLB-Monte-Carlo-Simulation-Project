"""Custom Elo rating engine, written from scratch.

Method (citations in README.md):
- Elo expected score:  E_home = 1 / (1 + 10^-((R_home + HFA - R_away)/400))
  (Elo, *The Rating of Chessplayers*, 1978)
- Per-game update:     R' = R + K * (S - E), K = 20, S ∈ {0, 1}
- Home-field advantage: +24 Elo points to the home side inside the expectation
  (FiveThirtyEight's MLB Elo specification)
- Off-season reversion: each team's rating is pulled 25% back toward the
  league mean of 1500 to model roster turnover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ELO_HFA, ELO_INITIAL, ELO_K, ELO_SEASON_REVERSION


def expected_home_score(home_elo: float | np.ndarray,
                        away_elo: float | np.ndarray,
                        hfa: float = ELO_HFA) -> float | np.ndarray:
    """Probability-scale expected score for the home team (vectorized)."""
    return 1.0 / (1.0 + 10.0 ** (-((home_elo + hfa) - away_elo) / 400.0))


class EloEngine:
    """Rolling Elo tracker replayed chronologically over historical games."""

    def __init__(self, team_ids: list[int]):
        self.ratings: dict[int, float] = {t: ELO_INITIAL for t in team_ids}

    def revert_to_mean(self) -> None:
        """Apply off-season regression: R <- R + f * (1500 - R)."""
        for team, rating in self.ratings.items():
            self.ratings[team] = rating + ELO_SEASON_REVERSION * (ELO_INITIAL - rating)

    def update_game(self, home_id: int, away_id: int, home_win: int) -> None:
        """Transfer Elo points after one completed game."""
        exp = expected_home_score(self.ratings[home_id], self.ratings[away_id])
        delta = ELO_K * (home_win - exp)
        self.ratings[home_id] += delta
        self.ratings[away_id] -= delta

    def replay(self, games: pd.DataFrame) -> pd.DataFrame:
        """Replay completed games chronologically, recording pre-game ratings.

        ``games`` must be sorted by (season, date). Rows with state='future'
        are annotated with the *current* ratings and left un-updated, so the
        returned frame doubles as the Monte Carlo starting grid.
        """
        seasons_seen: set[int] = set()
        home_pre = np.empty(len(games))
        away_pre = np.empty(len(games))

        for i, row in enumerate(games.itertuples(index=False)):
            if row.season not in seasons_seen:
                if seasons_seen:  # not before the first season
                    self.revert_to_mean()
                seasons_seen.add(row.season)
            home_pre[i] = self.ratings[row.home_id]
            away_pre[i] = self.ratings[row.away_id]
            if row.state == "final":
                self.update_game(row.home_id, row.away_id, int(row.home_win))

        out = games.copy()
        out["home_elo_pre"] = home_pre
        out["away_elo_pre"] = away_pre
        return out
