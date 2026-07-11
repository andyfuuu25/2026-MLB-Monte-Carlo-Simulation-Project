"""Application-facing pipeline: build once, simulate on demand.

The dashboard server constructs a :class:`SimContext` at startup (data load,
Elo replay, model training — ~0.2 s from cache) and then answers each UI
request with a fresh Monte Carlo run (~0.7 s for 10,000 trials). Results are
memoized by (sims, scenario, penalty) so toggling views is instant.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import (HISTORY_SEASONS, RANDOM_SEED, REPLACEMENT_FIP_PENALTY,
                     TARGET_SEASON)
from .awards import cy_pool, mvp_pool, simulate_awards
from .data import (fetch_arsenal, fetch_batting, fetch_fastball_movement,
                   fetch_oaa, fetch_pitch_discipline, fetch_pitching,
                   fetch_season_games, fetch_statcast_batters, fetch_teams,
                   fetch_xstats, load_all_games)
from .elo import EloEngine
from .fantasy import (build_hitter_pool, build_pitcher_pool, fantasy_profile,
                      implied_elo)
from .impact import composite_leaderboard, player_value_table
from .features import (MIN_OUTS, MIN_PA, apply_ohtani_injury,
                       apply_player_injury, build_team_profiles, fip,
                       league_fip_constant, ohtani_split_report, runs_created)
from .model import WinModel, build_training_frame, train_win_model
from .players import project_hitters, project_pitchers, team_rate_metrics
from .simulate import SimResult, run_simulation, summarize

log = logging.getLogger(__name__)

ALL_SEASONS = (*HISTORY_SEASONS, TARGET_SEASON)


def _games_played(finals: pd.DataFrame) -> pd.Series:
    return (finals["home_id"].value_counts()
            .add(finals["away_id"].value_counts(), fill_value=0))


@dataclass
class SimContext:
    """Everything the server needs, built once at startup."""
    teams: pd.DataFrame
    target_games: pd.DataFrame
    elo_now: dict[int, float]
    profiles: pd.DataFrame          # target-season, Ohtani healthy
    batting: pd.DataFrame           # target-season player lines
    pitching: pd.DataFrame
    model: WinModel
    ohtani: dict
    n_locked: int
    n_remaining: int
    batting_all: dict[int, pd.DataFrame] = field(default_factory=dict)
    pitching_all: dict[int, pd.DataFrame] = field(default_factory=dict)
    season_start_elo: dict[int, float] = field(default_factory=dict)
    hitter_pool: pd.DataFrame | None = None
    pitcher_pool: pd.DataFrame | None = None
    values: pd.DataFrame | None = None      # composite player-value table
    _cache: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def build(cls) -> "SimContext":
        teams = fetch_teams()
        games = load_all_games()
        engine = EloEngine(teams["team_id"].tolist())
        games_elo = engine.replay(games)

        # Roster stats: every simulated/trained season plus one season before
        # the earliest, because a season-S training game uses S-1 profiles
        # (leak-free features — see model.py).
        profile_seasons = sorted({*ALL_SEASONS, *(s - 1 for s in ALL_SEASONS)})
        batting = {s: fetch_batting(s) for s in profile_seasons}
        pitching = {s: fetch_pitching(s) for s in profile_seasons}

        finals = games_elo[games_elo["state"] == "final"]
        finals_by_season = {s: finals[finals["season"] == s]
                            for s in ALL_SEASONS}
        for s in profile_seasons:
            if s not in finals_by_season:  # pre-history season (GP proration)
                g = fetch_season_games(s)
                finals_by_season[s] = g[g["state"] == "final"]

        profiles = {
            s: build_team_profiles(
                batting[s], pitching[s],
                games_played=_games_played(finals_by_season[s]))
            for s in profile_seasons
        }
        train_profiles = {s: profiles[s - 1] for s in ALL_SEASONS}
        model = train_win_model(build_training_frame(games_elo, train_profiles))
        target = games_elo[games_elo["season"] == TARGET_SEASON]

        # Season-start Elo (for full-season fantasy replays): replay history
        # only, then apply the off-season reversion into the target season.
        start_engine = EloEngine(teams["team_id"].tolist())
        start_engine.replay(games[games["season"] < TARGET_SEASON])
        start_engine.revert_to_mean()

        return cls(
            teams=teams,
            target_games=target,
            elo_now=dict(engine.ratings),
            profiles=profiles[TARGET_SEASON],
            batting=batting[TARGET_SEASON],
            pitching=pitching[TARGET_SEASON],
            model=model,
            ohtani=ohtani_split_report(batting[TARGET_SEASON],
                                       pitching[TARGET_SEASON]),
            n_locked=int((target["state"] == "final").sum()),
            n_remaining=int((target["state"] == "future").sum()),
            batting_all=batting,
            pitching_all=pitching,
            season_start_elo=dict(start_engine.ratings),
            hitter_pool=build_hitter_pool(batting, TARGET_SEASON),
            pitcher_pool=build_pitcher_pool(pitching, TARGET_SEASON),
            values=player_value_table(batting[TARGET_SEASON],
                                      pitching[TARGET_SEASON],
                                      fetch_oaa(TARGET_SEASON)),
        )

    # ------------------------------------------------------------------
    def _run_profiles(self, key: tuple, profiles: pd.DataFrame,
                      sims: int) -> SimResult:
        """Memoized rest-of-season run for an arbitrary profile set.

        Every run shares the same seed, so any two cached results form a
        common-random-numbers pair and their differences are causal.
        """
        if key not in self._cache:
            self._cache[key] = run_simulation(
                self.target_games, self.elo_now, profiles, self.teams,
                self.model, n_sims=sims, seed=RANDOM_SEED)
        return self._cache[key]

    def _run(self, sims: int, scenario: str, penalty: float) -> SimResult:
        """Season-tab run: healthy league or the classic Ohtani stress case."""
        profiles = self.profiles
        if scenario == "injured":
            profiles = apply_ohtani_injury(self.profiles, self.batting,
                                           self.pitching, fip_penalty=penalty)
        # A healthy run is identical at every severity — one cache entry.
        key_penalty = round(penalty, 3) if scenario == "injured" else 0.0
        return self._run_profiles((sims, scenario, key_penalty),
                                  profiles, sims)

    def _distribution(self, result: SimResult, abbrev: str) -> dict:
        col = int(self.teams.index[self.teams["abbrev"] == abbrev][0])
        wins = result.wins[:, col]
        lo, hi = int(wins.min()), int(wins.max())
        counts, edges = np.histogram(
            wins, bins=np.arange(lo, hi + 2) - 0.5, density=True)
        return {
            "bins": [int(x) for x in (edges[:-1] + 0.5)],
            "pct": [round(float(c) * 100, 3) for c in counts],
            "mean": round(float(wins.mean()), 2),
            "p5": float(np.percentile(wins, 5)),
            "p95": float(np.percentile(wins, 95)),
        }

    # ------------------------------------------------------------------
    def simulate_payload(self, sims: int = 10_000, scenario: str = "healthy",
                         penalty: float = REPLACEMENT_FIP_PENALTY,
                         team: str = "LAD") -> dict:
        """One dashboard refresh: rankings, KPIs, distribution, deltas."""
        scenario = scenario if scenario in ("healthy", "injured") else "healthy"
        sims = int(np.clip(sims, 500, 20_000))
        penalty = float(np.clip(penalty, 0.0, 3.0))
        if team not in set(self.teams["abbrev"]):
            team = "LAD"

        result = self._run(sims, scenario, penalty)
        summary = summarize(result, self.teams)
        rankings = summary.round(
            {c: 1 for c in ("exp_wins", "wins_p5", "wins_p95", "div_pct",
                            "playoff_pct", "pennant_pct", "ws_pct",
                            "elo_final")}
        ).to_dict(orient="records")

        dist = {"team": team, "current": self._distribution(result, team)}
        deltas = None
        if scenario == "injured":
            base = self._run(sims, "healthy", penalty)
            dist["baseline"] = self._distribution(base, team)
            b, c = summarize(base, self.teams), summary
            b = b.set_index("abbrev").loc[team]
            c = c.set_index("abbrev").loc[team]
            deltas = {m: round(float(c[m] - b[m]), 2)
                      for m in ("exp_wins", "div_pct", "playoff_pct",
                                "pennant_pct", "ws_pct")}

        fav = rankings[0]
        sel = next(r for r in rankings if r["abbrev"] == team)
        return {
            "season": TARGET_SEASON,
            "sims": sims,
            "scenario": scenario,
            "penalty": penalty,
            "locked": self.n_locked,
            "remaining": self.n_remaining,
            "favorite": fav,
            "selected": sel,
            "rankings": rankings,
            "distribution": dist,
            "deltas": deltas,
            "legend": ({"current": "Pitching injury", "baseline": "Ohtani healthy"}
                       if scenario == "injured" else None),
            "model": self.model.metrics,
            "ohtani": self.ohtani,
        }

    # ------------------------------------------------------------------
    # Injury impact lab
    # ------------------------------------------------------------------
    def injury_targets(self) -> dict:
        """Every qualifying player per team, for the injury-lab selectors."""
        bat = self.batting[self.batting["PA"] >= max(MIN_PA, 100)].copy()
        bat["RC"] = runs_created(bat)
        pit = self.pitching[self.pitching["outs"] >= max(MIN_OUTS, 90)].copy()
        pit["FIP"] = fip(pit, league_fip_constant(
            self.pitching[self.pitching["outs"] >= MIN_OUTS]))

        abbrev_of = self.teams.set_index("team_id")["abbrev"]
        out: dict[str, dict] = {a: {"hitters": [], "pitchers": []}
                                for a in abbrev_of}
        for r in bat.sort_values("RC", ascending=False).itertuples():
            a = abbrev_of.get(r.team_id)
            if a:
                out[a]["hitters"].append({
                    "id": int(r.player_id), "name": r.name,
                    "PA": int(r.PA), "RC": round(float(r.RC), 1)})
        for r in pit.sort_values("outs", ascending=False).itertuples():
            a = abbrev_of.get(r.team_id)
            if a:
                out[a]["pitchers"].append({
                    "id": int(r.player_id), "name": r.name,
                    "IP": round(r.outs / 3.0, 1),
                    "FIP": round(float(r.FIP), 2)})
        return out

    def injury_payload(self, team: str, player_id: int, kind: str,
                       severity: float = 1.0, sims: int = 10_000) -> dict:
        """Rest-of-season odds with one player lost, vs the healthy baseline."""
        sims = int(np.clip(sims, 500, 20_000))
        severity = float(np.clip(severity, 0.0, 3.0))
        if team not in set(self.teams["abbrev"]):
            raise ValueError(f"unknown team {team!r}")
        team_row = self.teams[self.teams["abbrev"] == team].iloc[0]
        team_id = int(team_row["team_id"])

        injured_profiles = apply_player_injury(
            self.profiles, self.batting, self.pitching,
            player_id, team_id, kind=kind, severity=severity)
        result = self._run_profiles(
            ("injury", team, player_id, kind, round(severity, 3), sims),
            injured_profiles, sims)
        baseline = self._run(sims, "healthy", severity)

        summary = summarize(result, self.teams)
        round_cols = {c: 1 for c in ("exp_wins", "wins_p5", "wins_p95",
                                     "div_pct", "playoff_pct", "pennant_pct",
                                     "ws_pct", "elo_final")}
        rankings = summary.round(round_cols).to_dict(orient="records")
        sel = next(r for r in rankings if r["abbrev"] == team)
        b = summarize(baseline, self.teams).set_index("abbrev").loc[team]
        deltas = {m: round(float(sel[m] - b[m]), 2)
                  for m in ("exp_wins", "div_pct", "playoff_pct",
                            "pennant_pct", "ws_pct")}

        names = pd.concat([self.batting, self.pitching])
        player_name = names.loc[names["player_id"] == player_id, "name"]
        player_name = player_name.iloc[0] if len(player_name) else str(player_id)
        kind_label = {"batting": "bat", "pitching": "arm",
                      "both": "bat + arm"}[kind]

        prof = injured_profiles.set_index("team_id").loc[team_id]
        return {
            "season": TARGET_SEASON,
            "sims": sims,
            "mode": "injury",
            "team": team,
            "team_name": str(team_row["name"]),
            "player": player_name,
            "kind": kind,
            "severity": severity,
            "locked": self.n_locked,
            "remaining": self.n_remaining,
            "favorite": rankings[0],
            "selected": sel,
            "rankings": rankings,
            "distribution": {
                "team": team,
                "current": self._distribution(result, team),
                "baseline": self._distribution(baseline, team)},
            "deltas": deltas,
            "legend": {"current": f"{player_name} out",
                       "baseline": "Healthy roster"},
            "dist_note": (
                f"{player_name}'s {kind_label} replaced at severity "
                f"+{severity:.2f} for the rest of the season "
                f"(team now {prof['exp_rs']:.2f} RS/G, "
                f"{prof['exp_ra']:.2f} RA/G); identical seeds isolate "
                f"the effect."),
            "model": self.model.metrics,
        }

    def impact_leaderboard(self, team: str, severity: float = 1.0,
                           sims: int = 4_000, n_hitters: int = 6,
                           n_pitchers: int = 4) -> dict:
        """Rank a team's players by how much losing each one costs.

        Simulates a separate rest-of-season injury for every key player
        (top hitters by Runs Created, top pitchers by innings; a two-way
        player is knocked out entirely) and reports the deltas.
        """
        sims = int(np.clip(sims, 500, 10_000))
        severity = float(np.clip(severity, 0.0, 3.0))
        targets = self.injury_targets()
        if team not in targets:
            raise ValueError(f"unknown team {team!r}")
        team_id = int(self.teams.set_index("abbrev").loc[team, "team_id"])

        hitters = targets[team]["hitters"][:n_hitters]
        pitchers = targets[team]["pitchers"][:n_pitchers]
        pitcher_ids = {p["id"] for p in pitchers}
        plan: list[tuple[dict, str]] = []
        for h in hitters:
            plan.append((h, "both" if h["id"] in pitcher_ids else "batting"))
        two_way = {h["id"] for h, k in plan if k == "both"}
        plan += [(p, "pitching") for p in pitchers if p["id"] not in two_way]

        baseline = self._run(sims, "healthy", severity)
        b = summarize(baseline, self.teams).set_index("abbrev").loc[team]

        rows = []
        for player, kind in plan:
            injured = apply_player_injury(
                self.profiles, self.batting, self.pitching,
                player["id"], team_id, kind=kind, severity=severity)
            result = self._run_profiles(
                ("injury", team, player["id"], kind, round(severity, 3), sims),
                injured, sims)
            s = summarize(result, self.teams).set_index("abbrev").loc[team]
            rows.append({
                "id": player["id"], "name": player["name"], "kind": kind,
                "d_wins": round(float(s["exp_wins"] - b["exp_wins"]), 2),
                "d_playoff": round(float(s["playoff_pct"] - b["playoff_pct"]), 2),
                "d_ws": round(float(s["ws_pct"] - b["ws_pct"]), 2),
            })
        rows.sort(key=lambda r: r["d_ws"])   # most damaging first
        return {
            "team": team,
            "severity": severity,
            "sims": sims,
            "baseline": {"exp_wins": round(float(b["exp_wins"]), 1),
                         "playoff_pct": round(float(b["playoff_pct"]), 1),
                         "ws_pct": round(float(b["ws_pct"]), 1)},
            "players": rows,
        }

    # ------------------------------------------------------------------
    # Team breakdown: projected lineup / rotation + rate metrics
    # ------------------------------------------------------------------
    def team_breakdown_payload(self, team: str = "LAD",
                               n_sims: int = 10_000) -> dict:
        """Per-player rest-of-season Monte Carlo + team rate metrics."""
        if team not in set(self.teams["abbrev"]):
            team = "LAD"
        key = ("breakdown", team, n_sims)
        if key in self._cache:
            return self._cache[key]

        row = self.teams[self.teams["abbrev"] == team].iloc[0]
        team_id = int(row["team_id"])
        finals = self.target_games[self.target_games["state"] == "final"]
        future = self.target_games[self.target_games["state"] == "future"]
        team_gp = int(((finals["home_id"] == team_id)
                       | (finals["away_id"] == team_id)).sum())
        rem_games = int(((future["home_id"] == team_id)
                         | (future["away_id"] == team_id)).sum())

        rng = np.random.default_rng(RANDOM_SEED)
        payload = {
            "team": team,
            "name": str(row["name"]),
            "season": TARGET_SEASON,
            "games_played": team_gp,
            "games_remaining": rem_games,
            "sims": n_sims,
            "lineup": project_hitters(self.batting, team_id, team_gp,
                                      rem_games, n_sims, rng),
            "rotation": project_pitchers(self.pitching, team_id, team_gp,
                                         rem_games, n_sims, rng),
            "metrics": team_rate_metrics(self.batting, self.pitching, team_id),
        }
        self._cache[key] = payload
        return payload

    # ------------------------------------------------------------------
    # General player impact
    # ------------------------------------------------------------------
    def impact_rankings(self, top: int = 30) -> dict:
        """Composite-value leaderboard (WAR-estimate scale) for the UI."""
        key = ("impact-leaderboard", top)
        if key not in self._cache:
            lead = composite_leaderboard(self.values, self.teams, top=top)
            self._cache[key] = {
                "season": TARGET_SEASON,
                "players": [
                    {"player_id": int(r.player_id), "name": r.name,
                     "team": r.team, "war_est": round(r.war_est, 1),
                     "d_wins": round(r.d_wins, 1),
                     "two_way": int(r.roles) > 1, "metrics": r.metrics}
                    for r in lead.itertuples()],
            }
        return self._cache[key]

    # ------------------------------------------------------------------
    # Award races (MVP / Cy Young)
    # ------------------------------------------------------------------
    def awards_payload(self, sims: int = 10_000) -> dict:
        """MVP and Cy Young probabilities simulated across every trial."""
        sims = int(np.clip(sims, 500, 20_000))
        key = ("awards", sims)
        if key in self._cache:
            return self._cache[key]

        finals = self.target_games[self.target_games["state"] == "final"]
        future = self.target_games[self.target_games["state"] == "future"]

        def per_team(df: pd.DataFrame) -> pd.Series:
            return (df["home_id"].value_counts()
                    .add(df["away_id"].value_counts(), fill_value=0))

        team_gp, rem_games = per_team(finals), per_team(future)

        mvp = mvp_pool(self.values, fetch_xstats(TARGET_SEASON, "batter"),
                       fetch_statcast_batters(TARGET_SEASON), self.teams,
                       rem_games, team_gp)
        cy = cy_pool(self.values, fetch_xstats(TARGET_SEASON, "pitcher"),
                     fetch_pitch_discipline(TARGET_SEASON),
                     fetch_arsenal(TARGET_SEASON),
                     fetch_fastball_movement(TARGET_SEASON), self.teams,
                     rem_games, team_gp)

        base = self._run(sims, "healthy", 1.0)
        rng = np.random.default_rng(RANDOM_SEED + 7)
        races = simulate_awards(mvp, cy, base, rng)

        self._cache[key] = {"season": TARGET_SEASON, "sims": sims, **races}
        return self._cache[key]

    # ------------------------------------------------------------------
    # Fantasy draft mode
    # ------------------------------------------------------------------
    def player_pools(self) -> dict:
        """Draft pools for the UI, with league constants for live preview."""
        team_abbrev = self.teams.set_index("team_id")["abbrev"]
        hitters = self.hitter_pool.copy()
        pitchers = self.pitcher_pool.copy()
        hitters["team"] = hitters["team_id"].map(team_abbrev).fillna("—")
        pitchers["team"] = pitchers["team_id"].map(team_abbrev).fillna("—")
        return {
            "hitters": hitters.round({"rc_pa": 4, "rc650": 1})[
                ["player_id", "name", "team", "PA", "rc_pa", "rc650"]
            ].to_dict(orient="records"),
            "pitchers": pitchers.round({"fip": 2, "ip": 0})[
                ["player_id", "name", "team", "ip", "fip"]
            ].to_dict(orient="records"),
            "lg_fip": round(self.pitcher_pool.attrs["lg_fip"], 3),
            "lineup_slots": 9,
            "rotation_slots": 5,
        }

    def _full_season_games(self) -> pd.DataFrame:
        """Target-season schedule with every game unplayed (full replay)."""
        games = self.target_games.copy()
        games["state"] = "future"
        return games

    def fantasy_payload(self, hitter_ids: list[int], pitcher_ids: list[int],
                        slot: str = "ATH", sims: int = 10_000) -> dict:
        """Replay the full season with a drafted roster in ``slot``'s place."""
        sims = int(np.clip(sims, 500, 20_000))
        if slot not in set(self.teams["abbrev"]):
            slot = "ATH"
        slot_row = self.teams[self.teams["abbrev"] == slot].iloc[0]
        slot_id = int(slot_row["team_id"])

        prof = fantasy_profile(hitter_ids, pitcher_ids, self.hitter_pool,
                               self.pitcher_pool,
                               self.pitcher_pool.attrs["lg_fip"])
        elo0 = implied_elo(prof["run_diff"], self.profiles,
                           self.season_start_elo)

        games = self._full_season_games()

        # Baseline: full-season replay with the real league (memoized).
        base_key = ("replay-baseline", sims)
        if base_key not in self._cache:
            self._cache[base_key] = run_simulation(
                games, self.season_start_elo, self.profiles, self.teams,
                self.model, n_sims=sims, seed=RANDOM_SEED)
        baseline = self._cache[base_key]

        # Fantasy: swap the slot team's profile and starting Elo.
        profiles = self.profiles.copy()
        mask = profiles["team_id"] == slot_id
        profiles.loc[mask, ["exp_rs", "exp_ra", "run_diff"]] = (
            prof["exp_rs"], prof["exp_ra"], prof["run_diff"])
        elo_start = dict(self.season_start_elo)
        elo_start[slot_id] = elo0
        result = run_simulation(games, elo_start, profiles, self.teams,
                                self.model, n_sims=sims, seed=RANDOM_SEED)

        summary = summarize(result, self.teams)
        draft_name = f"Your Draft ({slot_row['name']})"
        summary.loc[summary["abbrev"] == slot, "name"] = draft_name
        round_cols = {c: 1 for c in ("exp_wins", "wins_p5", "wins_p95",
                                     "div_pct", "playoff_pct", "pennant_pct",
                                     "ws_pct", "elo_final")}
        rankings = summary.round(round_cols).to_dict(orient="records")

        base_summary = summarize(baseline, self.teams).set_index("abbrev")
        b = base_summary.loc[slot]
        sel = next(r for r in rankings if r["abbrev"] == slot)
        deltas = {m: round(float(sel[m] - b[m]), 2)
                  for m in ("exp_wins", "div_pct", "playoff_pct",
                            "pennant_pct", "ws_pct")}

        dist = {"team": slot,
                "current": self._distribution(result, slot),
                "baseline": self._distribution(baseline, slot)}

        return {
            "season": TARGET_SEASON,
            "sims": sims,
            "mode": "fantasy",
            "slot": slot,
            "slot_name": str(slot_row["name"]),
            "draft_name": draft_name,
            "profile": prof,
            "implied_elo": round(elo0, 0),
            "locked": 0,
            "remaining": len(games),
            "favorite": rankings[0],
            "selected": sel,
            "rankings": rankings,
            "distribution": dist,
            "deltas": deltas,
            "legend": {"current": "Your draft",
                       "baseline": f"Real {slot_row['name']}"},
            "model": self.model.metrics,
        }
