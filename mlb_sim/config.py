"""Central configuration for the hybrid MLB simulation engine.

All tunable parameters live here so the model can be stress-tested without
touching engine code. Citations for parameter choices are in README.md.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Seasons
# ---------------------------------------------------------------------------
# Historical seasons used to train the Elo engine and the ML classifier.
HISTORY_SEASONS: tuple[int, ...] = (2023, 2024, 2025)
# The season being simulated. Games already completed are locked in as fact;
# the Monte Carlo engine simulates only the remaining schedule.
TARGET_SEASON: int = 2026

# ---------------------------------------------------------------------------
# Elo engine (Elo 1978; MLB adaptation follows FiveThirtyEight's public spec)
# ---------------------------------------------------------------------------
ELO_INITIAL: float = 1500.0     # league mean rating
ELO_K: float = 20.0             # per-game update magnitude (spec requirement)
ELO_HFA: float = 24.0           # home-field advantage in Elo points
                                # (FiveThirtyEight MLB Elo uses ~24 pts)
ELO_SEASON_REVERSION: float = 0.25  # fraction reverted toward the mean each
                                    # off-season for roster turnover

# ---------------------------------------------------------------------------
# Roster / run modeling
# ---------------------------------------------------------------------------
MIN_PA: int = 30                # minimum plate appearances for a batter row
MIN_OUTS: int = 30              # minimum outs recorded (10 IP) for a pitcher
# Replacement level: a freely-available pitcher is roughly one run of FIP
# worse than league average (FanGraphs replacement-level framework).
REPLACEMENT_FIP_PENALTY: float = 1.00

# Shohei Ohtani, MLBAM person id (two-way player, LAD).
OHTANI_MLBAM_ID: int = 660271
DODGERS_TEAM_ID: int = 119

# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
N_SIMS: int = 10_000
RANDOM_SEED: int = 42

# Talent (parameter) uncertainty: each trial draws every team's latent
# strength once from N(rating, TALENT_SIGMA_ELO) and holds it fixed, so games
# are conditionally independent given the drawn talent. This propagates
# uncertainty about how good each roster *really is* — sized empirically so
# the backtest's 80% win-total bands cover ~80% of actual team seasons
# (see mlb_sim/backtest.py). Calibration on 2024-2025 walk-forward: sigma 0
# covers only 70% (binomial schedule noise alone is too narrow, ~6.3-win SD);
# sigma 60 pools to 81.7% coverage and a ~8.8-win SD, matching realized
# season-to-projection dispersion.
TALENT_SIGMA_ELO: float = 60.0

# Legacy behaviour: let Elo drift *inside* each trial by updating on the
# trial's own simulated outcomes. Off by default — a simulated result carries
# no information about true talent, so in-trial drift adds an arbitrary
# random walk rather than calibrated uncertainty (it also contradicted the
# frozen-Elo postseason). Kept as a switch for comparison runs.
ELO_IN_TRIAL_DRIFT: bool = False

# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------
# Seasons replayed as-of Opening Day by mlb_sim/backtest.py. Each needs the
# prior season's stats for leak-free training features, so the earliest
# usable backtest season is min(HISTORY_SEASONS) + 1.
BACKTEST_SEASONS: tuple[int, ...] = (2024, 2025)
BACKTEST_N_SIMS: int = 4_000

# Best-of series lengths for the 2022+ MLB postseason format.
SERIES_WILDCARD: int = 3
SERIES_DIVISION: int = 5
SERIES_LCS: int = 7
SERIES_WORLD: int = 7

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
STATSAPI_BASE = "https://statsapi.mlb.com/api/v1"
REQUEST_TIMEOUT = 45  # seconds
