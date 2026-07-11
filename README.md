# MLB Hybrid Season Simulator

A quantitative engine that projects the MLB season by fusing **team-level Elo
ratings** with **bottom-up player projections**, fused by a machine-learning
classifier and driven through a **10,000-trial vectorized Monte Carlo** of the
schedule and postseason — served as an interactive local dashboard with five
analysis modes: season outlook, player-impact simulation, MVP / Cy Young award
races, fantasy drafting, and per-player projections.

> Educational / research project. Not affiliated with MLB. No credentials or
> personal data are used or stored — all inputs are public league statistics.

---

## Table of contents

1. [Features](#1-features)
2. [Quick start](#2-quick-start)
3. [Repository layout](#3-repository-layout)
4. [Data sources](#4-data-sources)
5. [Modeling methodology](#5-modeling-methodology)
6. [Parameters reference](#6-parameters-reference)
7. [Results snapshot](#7-results-snapshot)
8. [Documented simplifications](#8-documented-simplifications)
9. [Citations](#9-citations)

---

## 1. Features

| Dashboard tab | What it answers |
|---|---|
| **Season outlook** | Full odds table for all 30 clubs — expected wins, division / playoff / pennant / World Series probabilities — plus the focus team's projected lineup & rotation with per-player Monte Carlo stat bands and team rate metrics (OPS+, ERA+, WHIP…) vs league. |
| **Player impact** | Lose ANY player (bat, arm, or both, at an adjustable severity) and measure the causal cost in wins and October odds; league-wide leaderboard ranks players by a literature-weighted composite of wOBA, wRC+, OAA, FIP and SIERA. |
| **Award races** | MVP and Cy Young win probabilities per league, simulated inside every Monte Carlo trial from Statcast advanced metrics (xwOBA, barrels; pitch mix, velocity, spin, movement, zone%). |
| **Fantasy draft** | Draft 9 hitters + 5 starters from every active player (Marcel-projected), drop them into a franchise slot, and replay the full 162-game season. |
| **Headless mode** | `run_simulation.py` reproduces the core pipeline without the UI and writes PNG figures + CSV tables to `outputs/`. |
| **Backtest** | `python -m mlb_sim.backtest` rebuilds the pipeline as-of Opening Day 2024 / 2025 (walk-forward, no future data), simulates each season blind, and scores the shipped product: win-band coverage, PIT, playoff-odds Brier & calibration, and out-of-sample game metrics vs an Elo-only ablation. |

## 2. Quick start

**Windows, one click:** double-click **`Launch MLB Simulator.bat`**. It
checks for Python, installs dependencies on first run, starts the server,
and opens the dashboard in your browser. Close the console window to quit.

**Command line (any OS):**

```bash
pip install -r requirements.txt

python app.py                 # dashboard at http://127.0.0.1:8123 (auto-opens)
python app.py --no-browser    # server only
python run_simulation.py      # headless run -> outputs/
python -m mlb_sim.backtest    # walk-forward calibration backtest -> outputs/
python -m mlb_sim.backtest --sweep 0,30,60,90   # size the talent sigma
```

The first run fetches four seasons of public data (~10 s) and caches it under
`data/` as CSV; subsequent runs are offline-fast. Delete `data/*.csv` to
refresh as the season progresses. Both `data/` and `outputs/` are
git-ignored — they are fully regenerable.

## 3. Repository layout

```
app.py                    Flask server; simulation context built once at startup
run_simulation.py         headless pipeline: report, PNG/CSV outputs
static/index.html         single-page dashboard (vanilla JS + SVG, no frameworks)
requirements.txt          runtime dependencies
mlb_sim/
  config.py               every tunable parameter (see §6)
  data.py                 ingestion: pybaseball first, MLB Stats API fallback,
                          Baseball Savant leaderboards, CSV caching
  elo.py                  from-scratch Elo engine
  features.py             Runs Created + FIP -> team run profiles; player-injury
                          roster surgery
  model.py                logistic-regression win-probability classifier
                          (leak-free features + Elo-only / baseline ablations)
  simulate.py             vectorized Monte Carlo season + postseason bracket
                          (per-trial latent talent draw)
  backtest.py             walk-forward calibration backtest (coverage, PIT,
                          playoff-odds Brier, out-of-sample game metrics)
  impact.py               composite player valuation (wOBA/OAA/FIP/SIERA -> WAR*)
  awards.py               MVP / Cy Young vote simulation from Statcast metrics
  players.py              per-player rest-of-season Monte Carlo + team rates
  fantasy.py              Marcel projections + drafted-roster profiles
  sensitivity.py          scenario comparison utilities (headless mode)
  pipeline.py             SimContext: build-once, simulate-on-demand, memoized
  viz.py                  matplotlib figures (CVD-validated palette)
data/                     cached raw data      (git-ignored, auto-created)
outputs/                  figures and tables   (git-ignored, auto-created)
```

## 4. Data sources

| Source | Used for | Status |
|---|---|---|
| **`pybaseball`** (FanGraphs / Baseball-Reference scrapers) | attempted first per original spec | both backends return **HTTP 403** to automated clients (mid-2026); every fetch falls back automatically |
| **MLB Stats API** (`statsapi.mlb.com`) | game logs & schedules (2023–2026), player batting/pitching lines, team metadata | primary source in practice; official, free, no key required |
| **Baseball Savant** (`baseballsavant.mlb.com`) | Statcast leaderboards: OAA / fielding runs prevented, expected stats (xwOBA/xERA), exit velocity & barrels, pitch arsenal (mix/velocity/spin), plate discipline (zone%, whiff%, chase%), four-seam movement | open to programmatic clients; every fetch degrades gracefully to an empty frame if unavailable |

No API keys, credentials, or personal data are involved anywhere — the
`.gitignore` still guards `data/`, `outputs/`, and common secret-file patterns
defensively.

## 5. Modeling methodology

### 5.1 Elo engine (`elo.py`)

From-scratch implementation replayed chronologically over ~9,700 games:

- Expected score `E_home = 1/(1 + 10^(−(R_home + HFA − R_away)/400))`
- Update `R′ = R + K·(S − E)`, K = 20, S ∈ {0, 1}
- Home-field advantage +24 Elo points inside the expectation
- Off-season reversion: 25% of each rating pulled back toward 1500

Completed target-season games are locked in as fact; the engine's ratings at
"today" seed the Monte Carlo.

### 5.2 Bottom-up roster profiles (`features.py`)

Player talent aggregates into two numbers per team-season:

- **Expected RS/G** — Σ Basic Runs Created, `RC = (H+BB)·TB/(AB+BB)`,
  prorated by games played.
- **Expected RA/G** — innings-weighted staff FIP
  `(13·HR + 3·(BB+HBP) − 2·K)/IP + cFIP` (cFIP set so league FIP = league
  ERA), scaled to total runs by the league RA9/ERA ratio (~1.08).

Two-way players (Ohtani) are explicitly split: bat in the offense matrix, arm
in the run-prevention matrix, independently perturbable.

### 5.3 Hybrid win classifier (`model.py`)

One training row per completed game (home perspective): features
`elo_diff` and `rundiff_diff` (roster run-differential gap), target home-win.
A scikit-learn `LogisticRegression` yields calibrated per-game probabilities —
the logistic link is the same functional form as Elo and Log5.

**Leak-free features**: a season-S training game uses roster profiles built
from season S−1 stats — the best estimate available *before* the game.
(Full-season same-season profiles would leak September performance into
April predictions; fixing this dropped in-sample accuracy from an inflated
0.572 to an honest 0.550 and shrank the roster coefficient accordingly.)

Diagnostics on 8,688 games (in-sample fit, leak-free features): accuracy
**0.550**, log-loss **0.683**, Brier **0.245**, implied home advantage 52.8%
vs 52.75% observed. Two ablations are trained on the same rows and reported
alongside: **Elo-only** (log-loss 0.6844) and the **constant home-rate
baseline** (log-loss 0.6916). Walk-forward out-of-sample (see §5.9) the
ordering holds — hybrid < Elo-only < baseline in log-loss in both backtest
seasons — so the bottom-up roster feature adds real, if small, signal.

### 5.4 Vectorized Monte Carlo (`simulate.py`)

All trials are carried as `(n_sims, 30)` NumPy arrays. Two sources of
uncertainty are modeled separately:

- **Talent (parameter) uncertainty** — each trial draws every team's latent
  strength once from `N(rating, σ_talent)` and holds it fixed for the whole
  trial, regular season and postseason alike. σ_talent = 60 Elo points,
  sized empirically by the backtest (§5.9): with σ = 0 the simulated win SD
  is ~6.3 wins and the 80% bands cover only 70% of actual team-seasons;
  σ = 60 gives ~8.8-win SD and 81.7% coverage.
- **Game (aleatoric) noise** — for each remaining fixture the engine computes
  every trial's win probability from the classifier and draws Bernoulli
  outcomes; games are conditionally independent given the drawn talent.

The legacy in-trial Elo drift (updating ratings on the trial's own simulated
outcomes) is retained behind `ELO_IN_TRIAL_DRIFT` but off by default: a
simulated result carries no information about true talent, so drift adds an
arbitrary random walk rather than calibrated uncertainty.

The postseason follows the 2022+ format (seeds 1–2 byes; WC best-of-3: 3v6,
4v5; DS best-of-5; LCS/WS best-of-7), with series resolved from the
closed-form best-of-N binomial. Each trial's latent strengths carry
unchanged into its bracket, so October is consistent with that trial's
regular season by construction. Scenario comparisons share random seeds
(common random numbers), so deltas are causal, not noise.

### 5.5 General player impact (`impact.py` + injury lab)

Any player's loss is simulated by handing his PA and/or innings to
replacement level and re-running the Monte Carlo against the same-seed
baseline. Candidates are valued by a composite whose weights follow the
research literature (the WAR component framework — runs above replacement ÷
10 runs/win):

| Metric | Role & weight | Basis |
|---|---|---|
| wOBA | offensive driver: `(wOBA − lg)/scale × PA` | Tango et al., *The Book* |
| wRC+ | same signal, index-scaled — displayed, not double-counted | FanGraphs library |
| OAA / DRS | fielding runs: Statcast Fielding Runs Prevented enters directly; DRS unavailable (blocked source), OAA is its modern successor | MLB Statcast |
| SIERA | 60% of pitcher talent | predictive-validity studies rank SIERA ≥ xFIP > FIP |
| FIP | 40% of pitcher talent (keeps real HR info) | FanGraphs library |
| fWAR / bWAR | recomputed in-house from the components above, reported as **WAR\*** | FanGraphs WAR docs |
| WPA | **weight zero by design** — descriptive "story stat", no predictive validity | FanGraphs library; Hardball Times |

### 5.6 Award races (`awards.py`)

MVP and Cy Young are decided **inside every Monte Carlo trial**:

- **Ballot score** = current runs above replacement
  + talent-rate projection over the player's remaining workload
  + sampling noise
  + team-success bonus wired to *that trial's* simulated outcome
  (+6 runs-equivalent for a playoff berth, +3 for a division title)
  + Gumbel voter noise (scale 5).
  The literature shows fWAR-style value is by far the strongest correlate of
  MVP votes with team success secondary — the bonuses are sized accordingly.
- **Batter talent** blends **xwOBA (60%) / wOBA (40%)** — Statcast expected
  stats strip batted-ball luck; barrel%, hard-hit% and exit velocity are
  surfaced as evidence. Two-way players carry pitching runs (an Ohtani MVP
  case is bat + arm).
- **Pitcher talent** blends **SIERA 50% / FIP 25% / xERA 25%**, then a
  **Stuff composite** — z-scores of fastball velocity, usage-weighted spin
  rate, four-seam induced vertical break, and whiff% — nudges talent by
  0.12 RA9 per z, because stuff models predict future performance beyond
  results-based estimators and stabilize fastest in small samples (Stuff+ /
  PitchingBot research). Pitch mix, zone% and chase% appear on every
  candidate card.
- Award probability = share of trials in which the candidate is the league's
  ballot argmax (top 12 candidates per league per award).

### 5.7 Per-player projections & team metrics (`players.py`)

The focus team's lineup (top 9 by PA) and rotation (top 5 by IP) get
individual rest-of-season Monte Carlo projections: hitters via multinomial
draws over per-PA outcome rates, pitchers via Poisson event counts per out —
yielding full-season HR / OPS / ERA / WHIP with 5th–95th percentile bands.
Team OPS+ and ERA+ are league-relative (no park factors).

### 5.8 Fantasy draft (`fantasy.py`)

Players are projected with simplified **Marcel** (5/4/3 recency weights,
regression to the mean by playing time: 200 PA / 50 IP of league-average
ballast). A drafted club — 9 hitters sharing ~38 PA/G, 5 starters covering
65% of innings over a league-average bullpen — takes any franchise's schedule
slot; its starting Elo is imputed from the league run-diff → Elo regression,
and the full 162-game season replays against a same-seed baseline with the
real club.

### 5.9 Walk-forward calibration backtest (`backtest.py`)

The dashboard ships season-level distributions, not per-game probabilities,
so that is the quantity validated. For each backtest season B (2024, 2025)
the pipeline is rebuilt exactly as it would have existed on Opening Day of
B — Elo replayed over seasons < B then reverted, classifier trained on
seasons < B with leak-free features, talent taken from season B−1 stats —
and the full season is simulated blind, then scored against reality:

| Check | Result (σ_talent = 60, pooled 2024–25) | Nominal |
|---|---|---|
| Central 80% win-band coverage | **0.817** (0.700 at σ = 0) | 0.80 |
| Central 50% win-band coverage | 0.617 (0.450 at σ = 0) | 0.50 |
| Expected-wins MAE | ~7.0 wins | — |
| Playoff-odds Brier | 0.206 | 0.240 (constant 12/30 baseline) |
| Game log-loss, out-of-sample | hybrid 0.681–0.683 < Elo-only 0.682–0.684 < baseline 0.690–0.692 | — |

Also reported: randomized PIT per team-season, realized playoff rates by
predicted-probability bucket, and per-team detail CSVs under `outputs/`.
Actual playoff fields are derived from final standings with the simulator's
own seeding rule (head-to-head tiebreakers approximated by jitter).

`--sweep` re-scores the backtest across a σ_talent grid — this is how the
default of 60 was chosen, and how it should be re-fit as seasons accumulate.

## 6. Parameters reference

All tunables live in `mlb_sim/config.py` and at module tops.

| Parameter | Value | Where | Rationale |
|---|---|---|---|
| Elo K-factor | 20 | config | spec requirement |
| Elo home-field advantage | +24 pts | config | FiveThirtyEight MLB Elo |
| Off-season reversion | 25% → 1500 | config | roster turnover |
| Monte Carlo trials | 10,000 (500–20,000) | config | CLT-stable odds in <1 s |
| Random seed | 42 (+7 for awards) | config | reproducibility; common random numbers |
| Talent sigma | 60 Elo pts / trial | config | backtest-calibrated: 80% bands cover 81.7% of 2024–25 team-seasons (70% without it) |
| In-trial Elo drift | off | config | simulated outcomes carry no talent information; legacy switch |
| Backtest seasons / trials | 2024–2025 / 4,000 | config | walk-forward; earliest season needing only cached history |
| Batter qualification | ≥30 PA (≥100 injury lab) | config | noise floor |
| Pitcher qualification | ≥30 outs (≥90 injury lab) | config | noise floor |
| Replacement pitcher | league FIP + 1.00 | config | FanGraphs replacement level |
| Replacement batter | −20 runs / 600 PA | impact | FanGraphs replacement level |
| Runs per win | 10 | impact | Tango, *The Book* |
| Pitcher talent blend | SIERA .60 / FIP .40 (impact); SIERA .50 / FIP .25 / xERA .25 (awards) | impact / awards | predictive-validity ordering |
| Batter talent blend | xwOBA .60 / wOBA .40 | awards | expected-stats skill estimate |
| Stuff adjustment | 0.12 RA9 per composite z | awards | small nudge per Stuff+ findings |
| Award ballot bonuses | playoffs +6, division +3 (runs-eq.) | awards | value >> narrative in voting studies |
| Voter noise | Gumbel(0, 5 runs) | awards | ballot dispersion |
| Marcel weights / ballast | 5/4/3; 200 PA / 50 IP | fantasy | Tango's Marcel |
| Lineup PA share | 38 PA/G ÷ 9 slots | fantasy | league average |
| Rotation innings share | 65% | fantasy | modern usage |
| RA9 / ERA scale | 1.08 | features | unearned-run share |

## 7. Results snapshot (data through 2026-07-10, backtest-calibrated engine)

- **World Series favorite**: Dodgers — 21.0% title odds, 100.7 expected wins
  (the calibrated talent draw widens every distribution, deflating favorites:
  pre-calibration the same engine printed 25.0%).
- **Ohtani two-way loss** costs LAD ~0.9 rest-of-season wins and ~2.3 pp of
  WS odds; playoff odds barely move (the cushion absorbs it — star injuries
  cost championships, not berths). Smaller than the pre-fix figure because
  the leak-free classifier weights the roster feature honestly.
- **Award races**: NL MVP Ohtani 98%; AL MVP Y. Alvarez 88%; NL Cy
  Misiorowski 80% (100.5 mph, 2550 rpm, +1.62 stuff z); AL Cy a real race —
  Schlittler 47% vs Cease 16%.
- **Backtest**: as-of Opening Day 2024/2025 the engine's 80% win bands cover
  81.7% of actual team-seasons, playoff-odds Brier beats climatology
  (0.206 vs 0.240), and the hybrid classifier beats its Elo-only ablation
  out-of-sample in both seasons.

## 8. Documented simplifications

- Talent uncertainty is a team-level normal in Elo space, drawn once per
  trial — no player-level injury hazards, aging curves, or trade-deadline
  behavior inside the season (their aggregate effect is what σ_talent = 60
  absorbs, but shocks are not attributable to individual players).
- Run profiles carry no team defense or baserunning: Runs Created ignores
  fielding and FIP deliberately strips it, so teams at the defensive
  extremes carry a systematic bias.
- Higher seed hosts every playoff game (real series rotate 2-3-2 / 2-2-1);
  tie-breaks by random jitter, not head-to-head records; series use a
  constant per-game win probability (no starter sequencing).
- Traded players attribute to their most recent team.
- SIERA approximated with ground/air-out totals (no pop-up data); zone% et
  al. come from Savant qualified-pitcher boards (sub-threshold arms show —).
- wRC+ / OPS+ / xwOBA-derived values omit park factors; WAR\* has no
  positional, league, or baserunning adjustments.
- fWAR/bWAR/DRS/WPA are not ingested directly: blocked sources or
  play-by-play requirements. WAR is recomputed in-house, OAA substitutes for
  DRS, WPA is excluded deliberately (non-predictive).
- Award model has no historical ballot backtest — bonuses/noise are
  literature-informed priors, not fitted coefficients.

## 9. Citations

| Component | Source |
|---|---|
| Elo ratings | A. Elo, *The Rating of Chessplayers, Past and Present* (1978) |
| MLB Elo parameters | [FiveThirtyEight: How Our MLB Predictions Work](https://fivethirtyeight.com/methodology/how-our-mlb-predictions-work/) |
| Runs Created | Bill James, *The Bill James Baseball Abstract* (1985) |
| FIP / cFIP | T. Tango; C. Dreslough (DICE); [FanGraphs: FIP](https://library.fangraphs.com/pitching/fip/) |
| Replacement level | [FanGraphs: Replacement Level](https://library.fangraphs.com/misc/war/replacement-level/) |
| Win probability / paired comparison | Bradley & Terry (1952); James's Log5; scikit-learn |
| Monte Carlo | Metropolis & Ulam, *JASA* (1949); common random numbers (simulation variance reduction) |
| wOBA & linear weights | Tango, Lichtman & Dolphin, *The Book* (2007); [FanGraphs: wOBA](https://library.fangraphs.com/offense/woba/) |
| wRC+ | [FanGraphs: wRC and wRC+](https://library.fangraphs.com/offense/wrc/) |
| SIERA formula & predictive validity | M. Swartz — [FanGraphs: SIERA](https://www.fangraphs.com/library/pitching/siera/); [Pitcher List: Relative Value of FIP/xFIP/SIERA](https://pitcherlist.com/going-deep-the-relative-value-of-fip-xfip-and-siera/); [FanGraphs Community: Predictive Pitching Metrics](https://community.fangraphs.com/a-brief-analysis-of-predictive-pitching-metrics/) |
| OAA / Fielding Runs Prevented | [Baseball Savant: Outs Above Average](https://baseballsavant.mlb.com/leaderboard/outs_above_average) |
| Expected stats (xwOBA / xERA) | [Baseball Savant: Expected Statistics](https://baseballsavant.mlb.com/leaderboard/expected_statistics) |
| Stuff models (velocity/spin/movement) | [FanGraphs: Stuff+, Location+, Pitching+ primer](https://library.fangraphs.com/pitching/stuff-location-and-pitching-primer/); [FanGraphs: PitchingBot & Stuff+](https://blogs.fangraphs.com/pitchingbot-and-stuff-pitch-modeling-are-now-on-fangraphs/) |
| WPA non-predictive | [FanGraphs: WPA](https://library.fangraphs.com/misc/wpa/); [Hardball Times: 10 Lessons about WPA](https://tht.fangraphs.com/10-lessons-i-have-learned-about-win-probability-added/) |
| MVP voting predictors | [FanGraphs Community: Do MVP Voters Look at Some Stats Above Others?](https://community.fangraphs.com/do-mvp-voters-look-at-some-stats-above-others/) (fWAR strongest correlate); [The Award Index](https://mvpsportstalkcom.wordpress.com/2021/05/27/introducing-the-award-index-a-simple-way-to-forecast-mlb-mvp-cy-young-voting/); [SharpeStats: Thinking Like an MVP Voter](https://sharpestats.com/thinking-like-an-mlb-mvp-voter/) |
| WAR framework, 10 runs/win | [FanGraphs: WAR](https://library.fangraphs.com/misc/war/) |
| Marcel projections | [T. Tango: Marcel the Monkey Forecasting System](http://www.tangotiger.net/archives/stud0346.shtml) (2004) |
| Data | [MLB Stats API](https://statsapi.mlb.com); [Baseball Savant](https://baseballsavant.mlb.com); `pybaseball` (attempted first; FanGraphs/B-R 403-block automated clients as of mid-2026) |
