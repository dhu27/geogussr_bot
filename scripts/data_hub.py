"""Share datasets via the team's private Hugging Face dataset repo (see docs/DATA.md).

uv run python scripts/data_hub.py create          # once, by whoever sets up the repo
uv run python scripts/data_hub.py list            # what's on the hub
uv run python scripts/data_hub.py push pilot_v1   # upload datasets/pilot_v1/ (resumable)
uv run python scripts/data_hub.py pull pilot_v1   # download into datasets/pilot_v1/
"""

from __future__ import annotations

import argparse
import sys

from data import hub


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("create", help="create the private repo + dataset card")
    sub.add_parser("list", help="list datasets on the hub")
    p = sub.add_parser("push", help="upload <data_root>/<name>/")
    p.add_argument("name")
    p = sub.add_parser("pull", help="download into <data_root>/<name>/")
    p.add_argument("name")
    p.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    try:
        if args.cmd == "create":
            print(f"repo ready: {hub.create_repo()}")
        elif args.cmd == "list":
            datasets = hub.list_datasets()
            if not datasets:
                print(f"no datasets on {hub.hub_repo()} yet")
            for d in datasets:
                if d.get("incomplete"):
                    print(f"{d['name']:<20} INCOMPLETE (push interrupted; re-run push to resume)")
                else:
                    print(
                        f"{d['name']:<20} {d['n_images']:>7} images  {d['n_regions']:>4} regions"
                        f"  splits={d['splits']}  files={d['files']}  {d['created_at']}"
                    )
        elif args.cmd == "push":
            hub.push(args.name)
        elif args.cmd == "pull":
            hub.pull(args.name, force=args.force)
    except hub.HubError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
