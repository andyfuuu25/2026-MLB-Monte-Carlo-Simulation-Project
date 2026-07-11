"""MLB Hybrid Simulator — local dashboard server.

Run:  python app.py            (serves http://127.0.0.1:8123 and opens it)
      python app.py --no-browser   (server only — used by dev tooling)

The simulation context (data ingestion, Elo replay, model training) is built
once at startup; every dashboard interaction triggers a fresh memoized
Monte Carlo run server-side and re-renders client-side.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from mlb_sim.pipeline import SimContext

HOST, PORT = "127.0.0.1", 8123

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("dashboard")

STATIC_DIR = Path(__file__).resolve().parent / "static"
app = Flask(__name__, static_folder=None)

log.info("building simulation context (data + Elo + model)...")
CTX = SimContext.build()
log.info("context ready: %d locked results, %d remaining fixtures",
         CTX.n_locked, CTX.n_remaining)


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/api/teams")
def teams():
    cols = ["team_id", "abbrev", "name", "league", "division"]
    return jsonify(CTX.teams[cols].to_dict(orient="records"))


@app.get("/api/team/<abbrev>")
def team_breakdown(abbrev: str):
    try:
        return jsonify(CTX.team_breakdown_payload(team=abbrev.upper()))
    except Exception:
        log.exception("team breakdown failed")
        return jsonify({"error": "breakdown failed; see server log"}), 500


@app.get("/api/players")
def players():
    return jsonify(CTX.player_pools())


@app.get("/api/impact/leaderboard")
def impact_lead():
    return jsonify(CTX.impact_rankings())


@app.get("/api/awards")
def awards():
    try:
        return jsonify(CTX.awards_payload(
            sims=int(request.args.get("sims", 10_000))))
    except Exception:
        log.exception("awards simulation failed")
        return jsonify({"error": "awards simulation failed; see log"}), 500


@app.get("/api/injury/targets")
def injury_targets():
    return jsonify(CTX.injury_targets())


@app.post("/api/injury/simulate")
def injury_simulate():
    body = request.get_json(silent=True) or {}
    try:
        payload = CTX.injury_payload(
            team=str(body.get("team", "LAD")),
            player_id=int(body.get("player_id", 0)),
            kind=str(body.get("kind", "batting")),
            severity=float(body.get("severity", 1.0)),
            sims=int(body.get("sims", 10_000)),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        log.exception("injury simulation failed")
        return jsonify({"error": "simulation failed; see server log"}), 500
    return jsonify(payload)


@app.post("/api/injury/leaderboard")
def injury_leaderboard():
    body = request.get_json(silent=True) or {}
    try:
        payload = CTX.impact_leaderboard(
            team=str(body.get("team", "LAD")),
            severity=float(body.get("severity", 1.0)),
            sims=int(body.get("sims", 4_000)),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        log.exception("impact leaderboard failed")
        return jsonify({"error": "simulation failed; see server log"}), 500
    return jsonify(payload)


@app.post("/api/fantasy")
def fantasy():
    body = request.get_json(silent=True) or {}
    try:
        hitters = list(dict.fromkeys(int(x) for x in body.get("hitters", [])))
        pitchers = list(dict.fromkeys(int(x) for x in body.get("pitchers", [])))
        payload = CTX.fantasy_payload(
            hitter_ids=hitters, pitcher_ids=pitchers,
            slot=str(body.get("slot", "ATH")),
            sims=int(body.get("sims", 10_000)),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        log.exception("fantasy simulation failed")
        return jsonify({"error": "simulation failed; see server log"}), 500
    return jsonify(payload)


@app.post("/api/simulate")
def simulate():
    body = request.get_json(silent=True) or {}
    try:
        payload = CTX.simulate_payload(
            sims=int(body.get("sims", 10_000)),
            scenario=str(body.get("scenario", "healthy")),
            penalty=float(body.get("penalty", 1.0)),
            team=str(body.get("team", "LAD")),
        )
    except Exception:  # surface a clean error to the UI
        log.exception("simulation request failed")
        return jsonify({"error": "simulation failed; see server log"}), 500
    return jsonify(payload)


if __name__ == "__main__":
    open_browser = ("--no-browser" not in sys.argv
                    and not os.environ.get("MLB_SIM_NO_BROWSER"))
    if open_browser:
        # Give the server a beat to bind, then pop the dashboard like an app.
        threading.Timer(
            1.0, lambda: webbrowser.open(f"http://{HOST}:{PORT}")).start()
    log.info("dashboard at http://%s:%d  (Ctrl+C to quit)", HOST, PORT)
    app.run(host=HOST, port=PORT, debug=False)
