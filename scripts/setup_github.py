"""Create GitHub milestones M0-M10 and one issue per task, parsed from PLAN.md.

    uv run python scripts/setup_github.py            # dry run: print what would be created
    gh auth login                                    # once
    uv run python scripts/setup_github.py --apply    # create (skips anything that already exists)

Milestone due dates come from the "### Mn — Name · Day Mon D" headings; Nov/Dec are 2026 and
Jan-Mar are 2027. Each "- [ ] task" line under a milestone becomes an issue.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

PLAN = Path(__file__).resolve().parents[1] / "PLAN.md"
HEADING = re.compile(r"^###\s+(M\d+)\s+—\s+(.+?)\s+·\s+\w{3}\s+(\w{3})\s+(\d{1,2})")
TASK = re.compile(r"^- \[[ x]\]\s+(.+)$")
DONE_WHEN = re.compile(r"^- \*\*Done when:\*\*\s+(.+)$")
YEAR = {"Nov": 2026, "Dec": 2026, "Jan": 2027, "Feb": 2027, "Mar": 2027}


@dataclass
class Milestone:
    key: str
    name: str
    due: datetime
    done_when: str = ""
    tasks: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        return f"{self.key} — {self.name}"

    @property
    def phase(self) -> str:
        return "phase-1" if int(self.key[1:]) <= 4 else "phase-2"


def parse_plan(text: str) -> list[Milestone]:
    out: list[Milestone] = []
    cur: Milestone | None = None
    for line in text.splitlines():
        if line.startswith("#"):
            m = HEADING.match(line)
            cur = None
            if m:
                key, name, mon, day = m.groups()
                due = datetime.strptime(f"{YEAR[mon]} {mon} {day} 23:59:59", "%Y %b %d %H:%M:%S")
                cur = Milestone(key, name, due)
                out.append(cur)
            continue
        if cur is None:
            continue
        if t := TASK.match(line):
            cur.tasks.append(t.group(1).strip())
        elif d := DONE_WHEN.match(line):
            cur.done_when = d.group(1).strip()
    return out


def issue_title(task: str, key: str) -> str:
    plain = re.sub(r"\*\*(.+?)\*\*", r"\1", task)
    plain = re.sub(r"\$([^$]+)\$", r"\1", plain)
    title = f"[{key}] {plain}"
    return title if len(title) <= 120 else title[:117] + "..."


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--apply", action="store_true", help="actually create milestones and issues")
    ap.add_argument("--repo", help="owner/name (default: this checkout's GitHub remote)")
    args = ap.parse_args()

    milestones = parse_plan(PLAN.read_text())
    n_issues = sum(len(m.tasks) for m in milestones)
    print(f"Parsed {len(milestones)} milestones, {n_issues} tasks from {PLAN.name}\n")
    for m in milestones:
        print(f"{m.title}  (due {m.due:%a %Y-%m-%d}, {m.phase})")
        for t in m.tasks:
            print(f"    {issue_title(t, m.key)}")
    if not args.apply:
        print("\nDry run. Re-run with --apply to create these on GitHub.")
        return

    repo = args.repo or json.loads(gh("repo", "view", "--json", "nameWithOwner"))["nameWithOwner"]
    print(f"\nApplying to {repo}")
    for label, color in (("phase-1", "1f77b4"), ("phase-2", "9467bd")):
        gh("label", "create", label, "--repo", repo, "--color", color, "--force")

    existing_ms = {
        m["title"] for m in json.loads(gh("api", f"repos/{repo}/milestones?state=all&per_page=100"))
    }
    existing_issues = {
        i["title"]
        for i in json.loads(
            gh(
                "issue",
                "list",
                "--repo",
                repo,
                "--state",
                "all",
                "--limit",
                "1000",
                "--json",
                "title",
            )
        )
    }

    for m in milestones:
        if m.title not in existing_ms:
            gh(
                "api",
                f"repos/{repo}/milestones",
                "-f",
                f"title={m.title}",
                "-f",
                f"due_on={m.due:%Y-%m-%dT%H:%M:%SZ}",
                "-f",
                f"description=Done when: {m.done_when}" if m.done_when else "description=",
            )
            print(f"  + milestone {m.title}")
        for t in m.tasks:
            title = issue_title(t, m.key)
            if title in existing_issues:
                continue
            body = f"{t}\n\nFrom PLAN.md, **{m.title}** (due {m.due:%a %b %-d, %Y})."
            gh(
                "issue",
                "create",
                "--repo",
                repo,
                "--title",
                title,
                "--body",
                body,
                "--milestone",
                m.title,
                "--label",
                m.phase,
            )
            print(f"  + issue {title}")
    print("Done.")


if __name__ == "__main__":
    main()
