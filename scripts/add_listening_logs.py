"""LeakLens: add the KKBox listening logs and rerun everything.

Run from the project folder (C:\\Apps\\leaklens):

    python scripts\\add_listening_logs.py

Steps (each one is skipped if its result already exists, so you can run the
script again after a problem and it carries on where it stopped):
  1. checks: transactions data present, Kaggle login works, free disk space
  2. downloads user_logs.csv.7z and user_logs_v2.csv.7z from Kaggle (~7.8 GB)
  3. unpacks them into data/kkbox/raw (~30 GB)
  4. prepares the data (~400 million daily rows -> monthly totals per customer)
  5. deletes the big raw files again (frees ~37 GB)
  6. reruns the audit and the month-by-month replay
  7. asks before uploading the results to GitHub

Everything printed is also saved to logs_run.txt (kept out of GitHub).
"""

from __future__ import annotations

import datetime as dt
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "kkbox" / "raw"
DL = ROOT / "data" / "kkbox" / "download"
PREP = ROOT / "data" / "kkbox" / "prepared"
FILES = ["user_logs.csv", "user_logs_v2.csv"]
COMPETITION = "kkbox-churn-prediction-challenge"
LOG = ROOT / "logs_run.txt"


class Tee:
    """Print to the screen and to logs_run.txt at the same time."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def step(text: str) -> None:
    print(f"\n=== {dt.datetime.now():%H:%M:%S}  {text}", flush=True)


def run(args: list[str]) -> None:
    """Run a command, show its output live, stop if it fails."""
    p = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace",
                         env={**__import__("os").environ, "PYTHONWARNINGS": "ignore", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
    for line in p.stdout:
        if "Warning" in line or "warnings.warn(" in line:
            continue  # harmless library warnings
        print(line, end="", flush=True)
    if p.wait() != 0:
        raise SystemExit(f"This command failed: {' '.join(args)}")


def py(*args: str) -> None:
    run([sys.executable, *args])


def kaggle(*args: str) -> None:
    """Run the Kaggle tool straight on the screen, so its download progress bar shows."""
    code = "import sys; from kaggle.cli import main; sys.argv = ['kaggle'] + sys.argv[1:]; main()"
    sys.stdout.flush()
    if subprocess.call([sys.executable, "-c", code, *args], cwd=ROOT) != 0:
        raise SystemExit(f"This Kaggle command failed: kaggle {' '.join(args)}")


def main() -> None:
    LOG.parent.mkdir(exist_ok=True)
    log = open(LOG, "a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log)
    gi = ROOT / ".gitignore"
    if "logs_run.txt" not in gi.read_text(encoding="utf-8"):
        with open(gi, "a", encoding="utf-8") as f:
            f.write("\nlogs_run.txt\n")
    for d in (RAW, DL):
        d.mkdir(parents=True, exist_ok=True)

    # 1 ------------------------------------------------------------------ checks
    step("1/7  Checking the set-up")
    if not (RAW / "transactions.csv").exists() and not (PREP / "transactions.parquet").exists():
        raise SystemExit("The transactions data isn't in data/kkbox. Run this from C:\\Apps\\leaklens.")
    py("-m", "pip", "install", "-q", "kaggle", "py7zr")
    kaggle("competitions", "files", "-c", COMPETITION)
    logs_parquet = PREP / "logs_monthly.parquet"
    already = False
    if logs_parquet.exists():
        import pyarrow.parquet as pq
        already = pq.ParquetFile(logs_parquet).metadata.num_rows > 0
    free_gb = shutil.disk_usage(ROOT).free / 2**30
    print(f"Free disk space: {free_gb:.0f} GB")
    if not already and free_gb < 45:
        raise SystemExit(f"You need about 45 GB free, you have {free_gb:.0f} GB. Free some space and run again.")

    if not already:
        # 2 ------------------------------------------------------------ download
        step("2/7  Downloading the listening logs from Kaggle (about 7.8 GB)")
        for f in FILES:
            if (RAW / f).exists():
                print(f"{f} is already unpacked, skipping download")
                continue
            if (DL / f"{f}.7z").exists():
                print(f"{f}.7z already downloaded")
                continue
            kaggle("competitions", "download", "-c", COMPETITION, "-f", f"{f}.7z", "-p", str(DL))
            for z in list(DL.glob(f"{f}*.zip")):  # Kaggle sometimes wraps the file in a .zip
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(DL)
                z.unlink()
            if not (DL / f"{f}.7z").exists():
                raise SystemExit(f"The download of {f}.7z didn't produce the file.")

        # 3 -------------------------------------------------------------- unpack
        step("3/7  Unpacking (about 30 GB, can take 30-60 minutes with no output)")
        import py7zr

        for f in FILES:
            if (RAW / f).exists():
                print(f"{f} already unpacked")
                continue
            tmp = RAW.parent / "tmp_unpack"
            shutil.rmtree(tmp, ignore_errors=True)
            with py7zr.SevenZipFile(DL / f"{f}.7z") as z:
                z.extractall(tmp)
            csv = next(tmp.rglob("*.csv"))
            shutil.move(str(csv), str(RAW / f))
            shutil.rmtree(tmp)
            (DL / f"{f}.7z").unlink()  # free space as we go
            print(f"{f}: {(RAW / f).stat().st_size / 2**20:,.0f} MB")

        # 4 ------------------------------------------------------------- prepare
        step("4/7  Preparing the data (1-2 hours; prints one line per chunk)")
        py("-m", "leaklens", "prepare", "--raw", str(RAW), "--out", str(PREP))

        # 5 ------------------------------------------------------------- cleanup
        step("5/7  Deleting the big raw listening files")
        for f in FILES:
            (RAW / f).unlink(missing_ok=True)
    else:
        step("2-5/7  Listening logs are already prepared, skipping download and preparation")

    import pandas as pd

    n = len(pd.read_parquet(logs_parquet, columns=["msno"]))
    print(f"{n:,} customer-months of listening data")
    if n == 0:
        raise SystemExit("No listening data was prepared. Check the output above.")

    # 6 --------------------------------------------------------------------- rerun
    step("6/7  Rerunning the audit (30-60 minutes)")
    py("-m", "leaklens", "audit")
    step("6/7  Rerunning the month-by-month replay (about an hour)")
    py("-m", "leaklens", "replay-reset")
    py("-m", "leaklens", "replay-run", "--steps", "9")

    print("\nResults (first lines of reports/results.md):")
    print("\n".join((ROOT / "reports" / "results.md").read_text(encoding="utf-8").splitlines()[:20]))

    # 7 -------------------------------------------------------------------- upload
    step("7/7  Upload to GitHub")
    run(["git", "status", "--short"])
    if input("Upload these results to GitHub now? (y/n) ").strip().lower() == "y":
        run(["git", "add", "-A"])
        run(["git", "commit", "-m", "Add listening logs: rerun audit and replay with usage features"])
        run(["git", "push"])
        print("Uploaded. The page updates in about 2 minutes.")
    else:
        print("Not uploaded. When you're ready: git add -A ; git commit -m \"Add listening logs\" ; git push")
    print("\nDone. Send reports/results.md to Claude to update the README text and screenshots.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        if e.code not in (None, 0):
            print(f"\nSTOPPED: {e.code}\nFix the problem and run the script again; finished steps are skipped."
                  "\nThe full output is in logs_run.txt.")
        sys.exit(1 if e.code not in (None, 0) else 0)
    except KeyboardInterrupt:
        print("\nStopped by you. Run the script again to carry on where it stopped.")
        sys.exit(1)
