"""Command line: python -m leaklens <command>

  make-sample   write a small made-up dataset in KKBox format, then prepare it
  prepare       turn raw KKBox CSVs into compact tables
  audit         steps 1-6: re-test the public notebooks, train honest models, euros
  report        write the manager summary and README results from the audit
  replay-step   one run of the weekly job (one month of simulated time)
  replay-run    several replay steps in a row (to fill the monitor quickly)
  replay-reset  delete the replay state and start fresh
"""

from __future__ import annotations

import argparse
import shutil
import tomllib
from pathlib import Path

from . import money


def load_config(path: str = "config.toml") -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _assumptions(cfg) -> money.Assumptions:
    m = cfg["money"]
    return money.Assumptions(m["offer_cost"], m["customer_value"], m["save_rate"])


def _job(cfg):
    from .data import load_prepared
    from .replay import ReplayConfig, ReplayJob

    ds = load_prepared(cfg["data"]["prepared_dir"])
    return ReplayJob(ds, ReplayConfig(**cfg["replay"]), _assumptions(cfg), cfg["audit"]["seed"], verbose=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="leaklens")
    ap.add_argument("command")
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--raw", default=None, help="folder with raw KKBox CSVs (prepare)")
    ap.add_argument("--out", default=None, help="output folder (prepare / make-sample)")
    ap.add_argument("--users", type=int, default=8000, help="make-sample: number of users")
    ap.add_argument("--steps", type=int, default=12, help="replay-run: number of months")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "make-sample":
        from .data import prepare
        from .sample import write_sample

        out = Path(args.out or "data/sample")
        write_sample(out / "raw", args.users)
        prepare(out / "raw", out / "prepared")

    elif args.command == "prepare":
        from .data import prepare

        prepare(args.raw or "data/kkbox/raw", args.out or cfg["data"]["prepared_dir"])

    elif args.command == "audit":
        from .audit import run_audit, save
        from .data import load_prepared

        ds = load_prepared(cfg["data"]["prepared_dir"])
        a = cfg["audit"]
        result = run_audit(ds, _assumptions(cfg), cfg["data"]["is_sample"], a["n_train_months"],
                           a["seed"], a["max_train_rows"])
        save(result, "reports/audit.json", Path(cfg["replay"]["site_data_dir"]) / "audit.json")
        from .report import write_all

        write_all(result)

    elif args.command == "report":
        import json

        from .report import write_all

        write_all(json.loads(Path("reports/audit.json").read_text(encoding="utf-8")))

    elif args.command == "replay-step":
        s = _job(cfg).step()
        print(f"loop {s['loop']}, today {s['today']}, model v{s['model_version']}, "
              f"pending {len(s['pending'])}, last event: {s['events'][-1]['type']}")

    elif args.command == "replay-run":
        job = _job(cfg)
        for i in range(args.steps):
            print(f"month {i + 1} of {args.steps} ...", flush=True)
            s = job.step()
            h = [x for x in s["history"] if x["auc"] is not None and x["loop"] == s["loop"]]
            last = f"checked AUC {h[-1]['auc']:.3f} ({h[-1]['cutoff']})" if h else "nothing checked yet"
            print(f"today {s['today']}: model v{s['model_version']}, {last}, "
                  f"drift {s['drift'][-1]['max_psi']:.2f}")

    elif args.command == "replay-reset":
        shutil.rmtree(cfg["replay"]["state_dir"], ignore_errors=True)
        shutil.rmtree(cfg["replay"].get("cache_dir", "data/replay_cache"), ignore_errors=True)
        print("replay state deleted")

    else:
        ap.error(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
