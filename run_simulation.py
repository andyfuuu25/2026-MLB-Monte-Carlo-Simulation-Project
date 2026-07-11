"""Entry point: hybrid MLB season + postseason Monte Carlo simulation.

Pipeline
--------
1. Ingest 3 historical seasons + the current season (pybaseball first,
   MLB Stats API fallback; everything cached under data/).
2. Replay history chronologically through the from-scratch Elo engine
   (K=20, +24 Elo HFA, 25% off-season reversion to 1500).
3. Build bottom-up roster profiles per season (Runs Created + staff FIP),
   with Ohtani explicitly split across the hitting and pitching matrices.
4. Train the logistic-regression win classifier on Elo diff + roster
   run-differential diff (leak-free: a season-S game uses S-1 profiles).
5. Run the 10,000-trial vectorized Monte Carlo of the remaining schedule
   with a per-trial latent team-strength draw (talent uncertainty, sized by
   the walk-forward backtest), then the full postseason bracket.
6. Repeat under the Ohtani-injury stress case and compare.
7. Emit ranking tables (CSV) and figures (PNG) under outputs/.

Usage:  python run_simulation.py [--sims 10000]
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from mlb_sim.config import HISTORY_SEASONS, N_SIMS, OUTPUT_DIR, TARGET_SEASON
from mlb_sim.data import (fetch_batting, fetch_pitching, fetch_season_games,
                          fetch_teams, load_all_games)
from mlb_sim.elo import EloEngine
from mlb_sim.features import build_team_profiles, ohtani_split_report
from mlb_sim.model import build_training_frame, train_win_model
from mlb_sim.sensitivity import run_ohtani_sensitivity, sensitivity_table
from mlb_sim.simulate import summarize
from mlb_sim.viz import plot_ohtani_sensitivity, plot_season_outlook

log = logging.getLogger("mlb_sim")


def games_played_by_team(games: pd.DataFrame) -> pd.Series:
    """Completed games per team_id for one season's frame."""
    finals = games[games["state"] == "final"]
    return (finals["home_id"].value_counts()
            .add(finals["away_id"].value_counts(), fill_value=0))


def main(n_sims: int) -> None:
    # Windows consoles may default to a legacy codepage (e.g. cp932) that
    # cannot encode em-dashes; force UTF-8 for report output.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    all_seasons = (*HISTORY_SEASONS, TARGET_SEASON)
    # A season-S training game uses S-1 profiles (leak-free), so stats are
    # needed for one season before the earliest training season too.
    profile_seasons = sorted({*all_seasons, *(s - 1 for s in all_seasons)})

    # 1 — ingestion ---------------------------------------------------------
    teams = fetch_teams()
    games = load_all_games()
    batting = {s: fetch_batting(s) for s in profile_seasons}
    pitching = {s: fetch_pitching(s) for s in profile_seasons}

    # 2 — Elo history replay ------------------------------------------------
    engine = EloEngine(teams["team_id"].tolist())
    games_elo = engine.replay(games)

    # 3 — roster profiles (Ohtani split verified on the target season) ------
    ohtani_split_report(batting[TARGET_SEASON], pitching[TARGET_SEASON])

    def season_games(s: int) -> pd.DataFrame:
        in_range = games[games["season"] == s]
        return in_range if len(in_range) else fetch_season_games(s)

    profiles = {
        s: build_team_profiles(
            batting[s], pitching[s],
            games_played=games_played_by_team(season_games(s)))
        for s in profile_seasons
    }

    # 4 — hybrid classifier (leak-free features + ablation diagnostics) -----
    train = build_training_frame(games_elo,
                                 {s: profiles[s - 1] for s in all_seasons})
    model = train_win_model(train)
    print("\n=== Win model diagnostics (train, {n} games) ===".format(
        n=model.metrics["n_games"]))
    for k in ("accuracy", "log_loss", "brier", "home_win_rate",
              "implied_hfa_prob"):
        print(f"  {k:>18}: {model.metrics[k]:.4f}")
    for name in ("elo_only", "baseline"):
        m = model.metrics[name]
        print(f"  {name:>18}: acc={m['accuracy']:.4f} "
              f"logloss={m['log_loss']:.4f} brier={m['brier']:.4f}")

    # 5 & 6 — Monte Carlo, base + stress scenarios ---------------------------
    target_games = games_elo[games_elo["season"] == TARGET_SEASON]
    scenario_a, scenario_b = run_ohtani_sensitivity(
        target_games, engine.ratings, profiles[TARGET_SEASON],
        batting[TARGET_SEASON], pitching[TARGET_SEASON], teams, model,
        n_sims=n_sims)

    # 7 — outputs ------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = summarize(scenario_a.result, teams)
    summary.to_csv(OUTPUT_DIR / "team_rankings.csv", index=False)

    sens = sensitivity_table(scenario_a, scenario_b)
    sens.to_csv(OUTPUT_DIR / "ohtani_sensitivity.csv", index=False)

    pd.set_option("display.width", 140)
    cols = ["abbrev", "name", "exp_wins", "wins_p5", "wins_p95", "div_pct",
            "playoff_pct", "pennant_pct", "ws_pct", "elo_final"]
    print(f"\n=== {TARGET_SEASON} projected standings — top 12 by World "
          f"Series odds ({n_sims:,} trials) ===")
    print(summary[cols].head(12).round(1).to_string(index=False))

    fav = summary.iloc[0]
    print(f"\n>>> Projected World Series favorite: {fav['name']} "
          f"({fav['ws_pct']:.1f}% title odds, {fav['exp_wins']:.1f} expected "
          f"wins, final Elo {fav['elo_final']:.0f})")

    print("\n=== Ohtani sensitivity — Los Angeles Dodgers ===")
    print(sens.to_string(index=False))

    p1 = plot_season_outlook(summary, n_sims, TARGET_SEASON)
    p2 = plot_ohtani_sensitivity(scenario_a, scenario_b, TARGET_SEASON)
    print(f"\nFigures:  {p1}\n          {p2}")
    print(f"Tables:   {OUTPUT_DIR / 'team_rankings.csv'}\n"
          f"          {OUTPUT_DIR / 'ohtani_sensitivity.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sims", type=int, default=N_SIMS,
                        help="number of Monte Carlo trials (default 10000)")
    args = parser.parse_args()
    sys.exit(main(args.sims))
