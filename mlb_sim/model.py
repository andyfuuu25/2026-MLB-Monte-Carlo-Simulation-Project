"""Hybrid win-probability classifier.

Each historical game becomes one training row (home-team perspective):

- ``elo_diff``      : home pre-game Elo − away pre-game Elo (the intercept
                      absorbs home-field advantage on top of the Elo HFA)
- ``rundiff_diff``  : home roster run differential − away roster run
                      differential (bottom-up player projections)
- target ``home_win`` ∈ {0, 1}

A scikit-learn ``LogisticRegression`` yields calibrated probabilities (the
logistic link is the canonical calibration map for binary outcomes; Elo itself
is a logistic model, so the features are on compatible scales).

Leakage note: training features for a season-S game use roster profiles built
from season S-1 player stats (the best estimate available *before* the game),
never season-S stats — full-season same-season profiles would leak September
performance into April predictions and inflate every diagnostic. At inference
time the simulator uses current-season profiles, which are equally known
as-of-today. Ablation diagnostics (Elo-only model, constant home-rate
baseline) are reported alongside the hybrid fit so the roster feature's
marginal value stays visible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

log = logging.getLogger(__name__)

FEATURES = ["elo_diff", "rundiff_diff"]


@dataclass
class WinModel:
    """Fitted coefficients, exposed for fast vectorized inference."""
    intercept: float
    coef_elo: float
    coef_rundiff: float
    metrics: dict

    def prob_home_win(self, elo_diff: np.ndarray,
                      rundiff_diff: np.ndarray) -> np.ndarray:
        """Vectorized sigmoid — used inside the Monte Carlo hot loop."""
        z = (self.intercept + self.coef_elo * elo_diff
             + self.coef_rundiff * rundiff_diff)
        return 1.0 / (1.0 + np.exp(-z))


def build_training_frame(games_with_elo: pd.DataFrame,
                         profiles_by_season: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Join pre-game Elo and roster profiles onto completed games.

    ``profiles_by_season[S]`` must contain the profiles to use *for games
    played in season S* — i.e. profiles built from season S-1 stats, so the
    features are knowable before the games they predict (no look-ahead).
    """
    df = games_with_elo[games_with_elo["state"] == "final"].copy()
    parts = []
    for season, prof in profiles_by_season.items():
        rd = prof.set_index("team_id")["run_diff"]
        sub = df[df["season"] == season].copy()
        sub["rundiff_diff"] = (sub["home_id"].map(rd) - sub["away_id"].map(rd))
        parts.append(sub)
    out = pd.concat(parts, ignore_index=True)
    out["elo_diff"] = out["home_elo_pre"] - out["away_elo_pre"]
    out = out.dropna(subset=FEATURES + ["home_win"])
    out["home_win"] = out["home_win"].astype(int)
    return out


def _fit_metrics(X: np.ndarray, y: np.ndarray) -> tuple[LogisticRegression, dict]:
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(X, y)
    p = clf.predict_proba(X)[:, 1]
    return clf, {
        "accuracy": float(accuracy_score(y, p > 0.5)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def train_win_model(train: pd.DataFrame) -> WinModel:
    """Fit logistic regression and report calibration diagnostics.

    Alongside the hybrid fit, two reference points are computed on the same
    rows: an Elo-only logistic (does the roster feature add anything?) and
    the constant home-rate baseline (is there any signal at all?).
    """
    X = train[FEATURES].to_numpy()
    y = train["home_win"].to_numpy()

    clf, fit = _fit_metrics(X, y)
    _, elo_only = _fit_metrics(X[:, :1], y)

    p0 = float(y.mean())  # constant home-rate baseline
    baseline = {
        "accuracy": max(p0, 1 - p0),
        "log_loss": float(-(p0 * np.log(p0) + (1 - p0) * np.log(1 - p0))),
        "brier": float(p0 * (1 - p0)),
    }

    metrics = {
        "n_games": len(train),
        **fit,
        "home_win_rate": p0,
        "implied_hfa_prob": float(1 / (1 + np.exp(-clf.intercept_[0]))),
        "elo_only": elo_only,
        "baseline": baseline,
    }
    log.info("win model: n=%d acc=%.3f logloss=%.4f brier=%.4f "
             "(elo-only logloss=%.4f, baseline logloss=%.4f)",
             metrics["n_games"], metrics["accuracy"], metrics["log_loss"],
             metrics["brier"], elo_only["log_loss"], baseline["log_loss"])

    return WinModel(
        intercept=float(clf.intercept_[0]),
        coef_elo=float(clf.coef_[0][0]),
        coef_rundiff=float(clf.coef_[0][1]),
        metrics=metrics,
    )
