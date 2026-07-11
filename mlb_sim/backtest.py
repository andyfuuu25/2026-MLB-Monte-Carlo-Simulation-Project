"""Walk-forward calibration backtest: does the shipped product tell the truth?

The dashboard ships *season-level distributions* (win bands, playoff odds),
not per-game probabilities — so that is the quantity validated here. For each
backtest season B the entire pipeline is rebuilt exactly as it would have
existed on Opening Day of B:

- Elo replayed over seasons < B only, then off-season reversion applied.
- Win classifier trained on games from seasons < B, with leak-free features
  (a season-S game uses season S-1 roster profiles).
- Team talent estimated from season B-1 stats (best available pre-season).
- The full season-B schedule simulated with every game unplayed.

The simulated distributions are then scored against what actually happened:

- **Coverage** — share of the 30 teams whose actual win total landed inside
  the simulated central 50% / 80% bands (calibrated model: ~50% / ~80%).
  Too-low coverage means the bands are too narrow — the classic symptom of
  simulating schedule luck while treating talent as known.
- **PIT** — randomized probability integral transform per team-season
  (uniform on [0,1] iff the win distributions are calibrated).
- **Win MAE / RMSE** of expected wins vs actual wins.
- **Playoff-odds Brier** vs the climatological baseline (12/30 for every
  team), plus realized rates by predicted-probability bucket.
- **Game-level walk-forward metrics** on season B's actual games: the hybrid
  classifier vs an Elo-only ablation vs the constant home-rate baseline —
  out-of-sample, so the roster feature has to earn its keep.

Actual playoff fields are derived from final standings with the same seeding
rule the simulator uses (division winners + three best remaining records per
league, ties jittered) — MLB's head-to-head tiebreakers are approximated.

Usage
-----
    python -m mlb_sim.backtest                        # default seasons/sigma
    python -m mlb_sim.backtest --sims 4000 --seasons 2024,2025
    python -m mlb_sim.backtest --sweep 0,20,40,60,80  # size TALENT_SIGMA_ELO

Writes ``outputs/backtest_teams.csv`` and ``outputs/backtest_summary.csv``.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from .config import (BACKTEST_N_SIMS, BACKTEST_SEASONS, HISTORY_SEASONS,
                     OUTPUT_DIR, RANDOM_SEED, TALENT_SIGMA_ELO)
from .data import fetch_batting, fetch_pitching, fetch_season_games, fetch_teams
from .elo import EloEngine
from .features import build_team_profiles
from .model import FEATURES, WinModel, build_training_frame, train_win_model
from .simulate import run_simulation

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# As-of-Opening-Day pipeline reconstruction
# ---------------------------------------------------------------------------
def _games_played(finals: pd.DataFrame) -> pd.Series:
    return (finals["home_id"].value_counts()
            .add(finals["away_id"].value_counts(), fill_value=0))


def _profiles_for(season: int) -> pd.DataFrame:
    """Team run profiles from one season's full player stats."""
    games = fetch_season_games(season)
    finals = games[games["state"] == "final"]
    return build_team_profiles(fetch_batting(season), fetch_pitching(season),
                               games_played=_games_played(finals))


@dataclass
class SeasonBacktest:
    """Everything known on Opening Day of ``season``, plus the actuals."""
    season: int
    teams: pd.DataFrame
    model: WinModel
    elo_open: dict[int, float]
    profiles_prior: pd.DataFrame   # season-1 talent estimate
    schedule: pd.DataFrame         # season schedule, every game 'future'
    actual_wins: pd.Series         # team_id -> wins
    actual_playoffs: set[int]      # team_ids (standings-derived)
    game_eval: dict                # walk-forward game-level metrics


def _actual_playoff_teams(teams: pd.DataFrame, wins: pd.Series,
                          seed: int = 0) -> set[int]:
    """Division winners + 3 best remaining per league, from actual wins."""
    rng = np.random.default_rng(seed)
    df = teams.copy()
    df["wins"] = df["team_id"].map(wins).fillna(0.0)
    df["jit"] = df["wins"] + rng.uniform(0.0, 0.5, len(df))

    qualifiers: set[int] = set()
    for _, lg in df.groupby("league"):
        div_winners = lg.loc[lg.groupby("division")["jit"].idxmax()]
        qualifiers |= set(div_winners["team_id"])
        rest = lg[~lg["team_id"].isin(qualifiers)]
        qualifiers |= set(rest.nlargest(3, "jit")["team_id"])
    return qualifiers


def _game_level_eval(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Out-of-sample game metrics: hybrid vs Elo-only vs constant baseline."""
    Xtr = train[FEATURES].to_numpy()
    ytr = train["home_win"].to_numpy()
    Xte = test[FEATURES].to_numpy()
    yte = test["home_win"].to_numpy()

    out = {"n_train": len(train), "n_test": len(test)}
    for name, cols in (("hybrid", [0, 1]), ("elo_only", [0])):
        clf = LogisticRegression(C=1.0, max_iter=1000).fit(Xtr[:, cols], ytr)
        p = clf.predict_proba(Xte[:, cols])[:, 1]
        out[name] = {
            "accuracy": float(accuracy_score(yte, p > 0.5)),
            "log_loss": float(log_loss(yte, p)),
            "brier": float(brier_score_loss(yte, p)),
        }
    p0 = float(ytr.mean())  # train-time home rate, scored on test games
    pb = np.full(len(yte), p0)
    out["baseline"] = {
        "accuracy": float(accuracy_score(yte, pb > 0.5)),
        "log_loss": float(log_loss(yte, pb, labels=[0, 1])),
        "brier": float(brier_score_loss(yte, pb)),
    }
    return out


def prepare_season(season: int) -> SeasonBacktest:
    """Rebuild the pipeline as it existed on Opening Day of ``season``."""
    train_seasons = tuple(s for s in HISTORY_SEASONS if s < season)
    if not train_seasons:
        raise ValueError(f"no history before {season} to train on")
    log.info("backtest %d: training on %s", season, train_seasons)

    teams = fetch_teams()
    train_games = pd.concat([fetch_season_games(s) for s in train_seasons],
                            ignore_index=True)
    train_games = train_games.sort_values(
        ["season", "date", "game_pk"]).reset_index(drop=True)

    engine = EloEngine(teams["team_id"].tolist())
    games_elo = engine.replay(train_games)
    engine.revert_to_mean()  # off-season into the backtest season

    train_profiles = {s: _profiles_for(s - 1) for s in train_seasons}
    model = train_win_model(build_training_frame(games_elo, train_profiles))

    season_games = fetch_season_games(season)
    finals = season_games[season_games["state"] == "final"]
    schedule = season_games.copy()
    schedule["state"] = "future"

    wins = pd.Series(0.0, index=teams["team_id"])
    for col, val in (("home_id", 1), ("away_id", 0)):
        won = finals.loc[finals["home_win"] == val, col].value_counts()
        wins = wins.add(won, fill_value=0.0)

    # Walk-forward game-level eval: pre-game Elo through season B is as-of by
    # construction; the roster feature uses the same B-1 profiles the
    # simulator gets on Opening Day.
    full_replay = EloEngine(teams["team_id"].tolist()).replay(
        pd.concat([train_games, season_games], ignore_index=True)
        .sort_values(["season", "date", "game_pk"]).reset_index(drop=True))
    test_frame = build_training_frame(
        full_replay[full_replay["season"] == season],
        {season: _profiles_for(season - 1)})
    train_frame = build_training_frame(games_elo, train_profiles)
    game_eval = _game_level_eval(train_frame, test_frame)

    return SeasonBacktest(
        season=season, teams=teams, model=model,
        elo_open=dict(engine.ratings),
        profiles_prior=_profiles_for(season - 1),
        schedule=schedule, actual_wins=wins,
        actual_playoffs=_actual_playoff_teams(teams, wins),
        game_eval=game_eval,
    )


# ---------------------------------------------------------------------------
# Scoring simulated distributions against reality
# ---------------------------------------------------------------------------
def evaluate(bt: SeasonBacktest, n_sims: int = BACKTEST_N_SIMS,
             talent_sigma: float = TALENT_SIGMA_ELO,
             seed: int = RANDOM_SEED) -> tuple[dict, pd.DataFrame]:
    """Simulate season ``bt.season`` blind and score it. Returns
    (summary metrics, per-team detail frame)."""
    result = run_simulation(bt.schedule, bt.elo_open, bt.profiles_prior,
                            bt.teams, bt.model, n_sims=n_sims, seed=seed,
                            talent_sigma=talent_sigma)
    rng = np.random.default_rng(seed + 1)

    team_ids = result.team_ids
    actual = bt.actual_wins.reindex(team_ids).to_numpy()
    sim = result.wins  # (n_sims, 30)

    p10, p25, p75, p90 = np.percentile(sim, [10, 25, 75, 90], axis=0)
    in50 = (actual >= p25) & (actual <= p75)
    in80 = (actual >= p10) & (actual <= p90)

    # Randomized PIT for the discrete win distribution.
    below = (sim < actual).mean(axis=0)
    at = (sim == actual).mean(axis=0)
    pit = below + rng.uniform(0.0, 1.0, len(actual)) * at

    playoff_prob = result.made_playoffs.mean(axis=0)
    playoff_actual = np.array([t in bt.actual_playoffs for t in team_ids],
                              dtype=float)

    detail = pd.DataFrame({
        "season": bt.season,
        "team_id": team_ids,
        "abbrev": bt.teams.set_index("team_id")["abbrev"]
                  .reindex(team_ids).to_numpy(),
        "exp_wins": sim.mean(axis=0),
        "wins_p10": p10, "wins_p90": p90,
        "actual_wins": actual,
        "err": sim.mean(axis=0) - actual,
        "pit": pit,
        "playoff_prob": playoff_prob,
        "made_playoffs": playoff_actual.astype(int),
    })

    summary = {
        "season": bt.season,
        "n_sims": n_sims,
        "talent_sigma": talent_sigma,
        "win_mae": float(np.abs(detail["err"]).mean()),
        "win_rmse": float(np.sqrt((detail["err"] ** 2).mean())),
        "sim_sd_mean": float(sim.std(axis=0).mean()),
        "coverage50": float(in50.mean()),
        "coverage80": float(in80.mean()),
        "playoff_brier": float(np.mean((playoff_prob - playoff_actual) ** 2)),
        "playoff_brier_base": float(np.mean((12 / 30 - playoff_actual) ** 2)),
    }
    return summary, detail


def _bucket_calibration(detail: pd.DataFrame) -> pd.DataFrame:
    """Realized playoff rate by predicted-probability bucket (pooled)."""
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    b = pd.cut(detail["playoff_prob"], bins=edges, labels=labels, right=False)
    g = detail.groupby(b, observed=True)
    return pd.DataFrame({
        "teams": g.size(),
        "mean_predicted": g["playoff_prob"].mean().round(3),
        "realized": g["made_playoffs"].mean().round(3),
    })


# ---------------------------------------------------------------------------
# Orchestration / CLI
# ---------------------------------------------------------------------------
def run_backtest(seasons: tuple[int, ...] = BACKTEST_SEASONS,
                 n_sims: int = BACKTEST_N_SIMS,
                 sigmas: list[float] | None = None) -> pd.DataFrame:
    """Backtest each season; optionally sweep talent sigma. Returns the
    summary frame (one row per season x sigma)."""
    sigmas = sigmas if sigmas is not None else [TALENT_SIGMA_ELO]
    backtests = [prepare_season(s) for s in seasons]

    summaries, details = [], []
    for sigma in sigmas:
        for bt in backtests:
            summary, detail = evaluate(bt, n_sims=n_sims, talent_sigma=sigma)
            summaries.append(summary)
            if sigma == sigmas[-1] or len(sigmas) == 1:
                details.append(detail)
    summary_df = pd.DataFrame(summaries)
    detail_df = pd.concat(details, ignore_index=True)

    OUTPUT_DIR.mkdir(exist_ok=True)
    summary_df.to_csv(OUTPUT_DIR / "backtest_summary.csv", index=False)
    detail_df.round(3).to_csv(OUTPUT_DIR / "backtest_teams.csv", index=False)

    # ---- report ----------------------------------------------------------
    print("\n=== Season-level calibration (walk-forward, as-of Opening Day) ===")
    cols = ["season", "talent_sigma", "win_mae", "win_rmse", "sim_sd_mean",
            "coverage50", "coverage80", "playoff_brier", "playoff_brier_base"]
    print(summary_df[cols].round(3).to_string(index=False))
    print("\nNominal coverage is 0.50 / 0.80 -- lower means bands too narrow.")

    if len(sigmas) > 1:
        pooled = (summary_df.groupby("talent_sigma")
                  [["coverage50", "coverage80", "win_mae", "playoff_brier"]]
                  .mean().round(3))
        print("\n=== Talent-sigma sweep (pooled across seasons) ===")
        print(pooled.to_string())
        best = (pooled["coverage80"] - 0.80).abs().idxmin()
        print(f"\n80% coverage closest to nominal at talent_sigma = {best}")

    print("\n=== Playoff-odds calibration (pooled, final sigma) ===")
    print(_bucket_calibration(detail_df).to_string())

    print("\n=== Game-level walk-forward (out-of-sample season) ===")
    for bt in backtests:
        ge = bt.game_eval
        print(f"\n{bt.season} (train n={ge['n_train']}, test n={ge['n_test']}):")
        for name in ("hybrid", "elo_only", "baseline"):
            m = ge[name]
            print(f"  {name:<9} acc={m['accuracy']:.3f} "
                  f"logloss={m['log_loss']:.4f} brier={m['brier']:.4f}")

    print(f"\nDetail written to {OUTPUT_DIR / 'backtest_teams.csv'}")
    return summary_df


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seasons", default=",".join(map(str, BACKTEST_SEASONS)),
                    help="comma-separated seasons to backtest")
    ap.add_argument("--sims", type=int, default=BACKTEST_N_SIMS)
    ap.add_argument("--sweep", default=None,
                    help="comma-separated talent sigmas to sweep, e.g. 0,20,40")
    args = ap.parse_args()

    seasons = tuple(int(s) for s in args.seasons.split(","))
    sigmas = ([float(x) for x in args.sweep.split(",")]
              if args.sweep else None)
    run_backtest(seasons=seasons, n_sims=args.sims, sigmas=sigmas)


if __name__ == "__main__":
    main()
