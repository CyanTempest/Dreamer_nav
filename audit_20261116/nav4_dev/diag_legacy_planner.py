"""Diagnostic: drive nav4 with the LEGACY planner (nav3.lwy_control_v2.planning_result + configs/lwy_v2_planner.json)
instead of nav4.planner (env, sensing, scoring unchanged). Usage: diag_legacy_planner.py <nav4.evaluate args>."""
import json
import sys
import jax
import nav4.planner as planner
from nav3.lwy_control_v2 import planning_result

params = json.load(open('configs/lwy_v2_planner.json'))
legacy = jax.jit(lambda o: planning_result(o, **params)[0])
planner.Planner.__call__ = lambda self, obs: legacy(obs)
from nav4.evaluate import main  # noqa: E402
main(sys.argv[1:])
