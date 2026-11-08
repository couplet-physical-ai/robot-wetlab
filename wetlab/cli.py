"""wetlab CLI: serve | user add | plan | twin-usd."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .spec import DEFAULT_LAB, load_lab


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="wetlab")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the service (virtual cell by default)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--lab", default=str(DEFAULT_LAB))
    s.add_argument("--db", default="var/wetlab.db")
    s.add_argument("--artifacts", default="var/artifacts")
    s.add_argument("--speed", type=float, default=1.0, help="sim time multiplier for the runtime loop")

    u = sub.add_parser("user", help="manage users")
    usub = u.add_subparsers(dest="ucmd", required=True)
    ua = usub.add_parser("add", help="create a user and print its API key once")
    ua.add_argument("--org", required=True)
    ua.add_argument("--name", required=True)
    ua.add_argument("--role", choices=["renter", "operator", "admin"], default="renter")
    ua.add_argument("--stripe-customer")
    ua.add_argument("--db", default="var/wetlab.db")

    p = sub.add_parser("plan", help="dry-run a program (IR .json or SDK .py) in the CPU twin")
    p.add_argument("program")
    p.add_argument("--lab", default=str(DEFAULT_LAB))
    p.add_argument("--out", default="out/dry_run.mp4")

    t = sub.add_parser("twin-usd", help="write the Isaac Sim stage for a lab spec")
    t.add_argument("--lab", default=str(DEFAULT_LAB))
    t.add_argument("--out", default="out/cell-a.usda")

    a = ap.parse_args(argv)
    if a.cmd == "serve":
        import uvicorn

        from .service.app import create_app
        app = create_app({"db_path": a.db, "artifacts_root": a.artifacts, "speed": a.speed,
                          "cells": [{"lab_path": a.lab, "driver": "sim"}]})
        uvicorn.run(app, host=a.host, port=a.port)
    elif a.cmd == "user":
        from .service.db import DB
        user, key = DB(a.db).add_user(a.org, a.name, a.role, a.stripe_customer)
        print(json.dumps({**user, "api_key": key}))
    elif a.cmd == "plan":
        from .ir import Program
        from .twin.planner import plan_program
        from .twin.render import render_plan_video
        src = Path(a.program).read_text()
        if a.program.endswith(".py"):
            from .sandbox import compile_python
            program, logs = compile_python(src)
            if logs:
                print(logs, file=sys.stderr)
        else:
            program = Program.model_validate_json(src)
        lab = load_lab(a.lab)
        plan = plan_program(lab, program)
        print(json.dumps({"ok": plan.ok, "duration_s": plan.duration_s, "stats": plan.stats,
                          "violations": [v.model_dump() for v in plan.violations]}, indent=2, default=str))
        print(f"video: {render_plan_video(lab, plan, a.out)}", file=sys.stderr)
        sys.exit(0 if plan.ok else 1)
    elif a.cmd == "twin-usd":
        from .twin.usd import write_stage
        print(write_stage(load_lab(a.lab), a.out))


if __name__ == "__main__":
    main()
