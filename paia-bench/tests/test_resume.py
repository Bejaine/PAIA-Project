"""Kill the runner mid-suite with SIGKILL, restart it, and confirm no clean cell is redone.
Also checks the retry path (contaminated -> clean, and contaminated -> unclean_exhausted).
GPU-free: uses the 'selftest' phase. Run: python -m tests.test_resume
"""
import collections
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MAX_RETRIES = 3


def cells(sess):
    out = {}
    for p in (sess / "cells").glob("*.json"):
        out[p.stem] = json.loads(p.read_text())  # must always parse: writes are atomic
    return out


def run_cmd(root):
    return [sys.executable, "-m", "bench.run", "--phase", "selftest", "--session", "t_resume",
            "--results-root", str(root), "--max-retries", str(MAX_RETRIES)]


def main():
    root = Path(tempfile.mkdtemp(prefix="paia_resume_"))
    sess = root / "selftest" / "t_resume"
    p = subprocess.Popen(run_cmd(root), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 60:
        if (sess / "cells").exists() and sum(c["status"] == "clean" for c in cells(sess).values()) >= 4:
            break
        time.sleep(0.05)
    time.sleep(0.15)  # land inside the next cell (each fake cell sleeps 0.3 s)
    os.kill(p.pid, signal.SIGKILL)
    p.wait()
    before = {cid: c for cid, c in cells(sess).items() if c["status"] == "clean"}
    n_exec_before = len((sess / "executions.log").read_text().split())
    assert 4 <= len(before) < 12, f"kill landed at an unhelpful point: {len(before)} clean"

    r = subprocess.run(run_cmd(root), capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    after = cells(sess)
    execs = collections.Counter((sess / "executions.log").read_text().split())

    # 1. no clean cell was redone
    for cid, c in before.items():
        assert after[cid]["t_start"] == c["t_start"], f"clean cell {cid} was rerun"
        assert execs[cid] == 1, f"clean cell {cid} executed {execs[cid]} times"
    # 2. everything finished
    assert len(after) == 12, len(after)
    st = collections.Counter(c["status"] for c in after.values())
    assert set(st) <= {"clean", "unclean_exhausted"}, st
    # 3. retry semantics
    by_n = {c["shape"]["N"]: c for c in after.values()}
    assert by_n[4]["status"] == "clean" and by_n[4]["retries"] == 2, by_n[4]
    assert by_n[8]["status"] == "unclean_exhausted" and by_n[8]["retries"] == MAX_RETRIES + 1, by_n[8]
    # 4. a third run does nothing
    r3 = subprocess.run(run_cmd(root), capture_output=True, text=True)
    assert "0 to run" in r3.stdout, r3.stdout[-500:]
    # 5. exactly one cell was interrupted mid-run by the kill: it has one more execution than its recorded attempts
    expected = {cid: c["retries"] + (1 if c["status"] == "clean" else 0) for cid, c in after.items()}
    interrupted = [cid for cid in after if execs[cid] == expected[cid] + 1]
    assert all(execs[cid] in (expected[cid], expected[cid] + 1) for cid in after), (dict(execs), expected)
    assert len(interrupted) == 1, f"expected exactly one mid-run cell to be redone, got {interrupted}"
    print(f"PASS: killed after {len(before)} clean cells ({n_exec_before} executions); resume redid none of them; "
          f"final {dict(st)}; retry counts ok; third run ran 0 cells; cell interrupted mid-run and redone: {interrupted}. dir={root}")


if __name__ == "__main__":
    main()
