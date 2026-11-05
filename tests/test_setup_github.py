import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("setup_github", ROOT / "scripts" / "setup_github.py")
setup_github = importlib.util.module_from_spec(spec)
sys.modules["setup_github"] = setup_github
spec.loader.exec_module(setup_github)


def test_parse_plan_milestones():
    ms = setup_github.parse_plan((ROOT / "PLAN.md").read_text())
    assert [m.key for m in ms] == [f"M{i}" for i in range(11)]
    assert ms[0].due.strftime("%Y-%m-%d") == "2026-11-08"
    assert ms[4].due.strftime("%Y-%m-%d") == "2026-12-04"
    assert ms[10].due.strftime("%Y-%m-%d") == "2027-03-12"
    assert all(m.tasks for m in ms)
    assert ms[0].done_when.startswith("everyone can run")
    assert {m.phase for m in ms[:5]} == {"phase-1"} and {m.phase for m in ms[5:]} == {"phase-2"}
