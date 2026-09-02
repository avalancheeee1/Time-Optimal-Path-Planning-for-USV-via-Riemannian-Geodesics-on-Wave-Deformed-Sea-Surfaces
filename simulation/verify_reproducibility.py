"""Verify that ``python experiments.py`` reproduces ``results.json`` deterministically.

Run from the ``simulation/`` directory::

    python verify_reproducibility.py

It re-runs the full experiment suite and compares the regenerated ``results.json``
against the committed reference. Every scientific quantity (costs, path
reductions, convergence, multi-seed robustness) must reproduce **byte-for-byte**.
The only intentionally excluded fields are the wall-clock ``time_s`` measurements
under ``results.runtime``, which are machine/load-dependent and vary run-to-run.

Exit code 0 == reproducible; non-zero == mismatch or failure.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results.json")
BACKUP = os.path.join(HERE, ".results.json.repro_backup")


def strip_time(data):
    """Recursively drop wall-clock ``time_s`` fields (non-deterministic by nature)."""
    if isinstance(data, dict):
        return {k: strip_time(v) for k, v in data.items() if k != "time_s"}
    if isinstance(data, list):
        return [strip_time(v) for v in data]
    return data


def canonical_sha256(data) -> str:
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def main() -> int:
    if not os.path.exists(RESULTS):
        print("FAIL: results.json not found", file=sys.stderr)
        return 1

    with open(RESULTS, encoding="utf-8") as f:
        committed = json.load(f)
    before = canonical_sha256(strip_time(committed))
    shutil.copy2(RESULTS, BACKUP)

    try:
        print("Re-running experiments.py ... (this may take a few minutes)")
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, "experiments.py")],
            cwd=HERE, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print("FAIL: experiments.py exited with", proc.returncode, file=sys.stderr)
            print(proc.stderr[-4000:], file=sys.stderr)
            return proc.returncode

        with open(RESULTS, encoding="utf-8") as f:
            regenerated = json.load(f)
        after = canonical_sha256(strip_time(regenerated))

        if after == before:
            print("PASS: deterministic results reproduced byte-for-byte")
            print("      SHA-256 (time_s excluded)", after)
            return 0

        print("FAIL: deterministic results differ from the committed reference",
              file=sys.stderr)
        print("      committed  ", before, file=sys.stderr)
        print("      regenerated", after, file=sys.stderr)
        return 1
    finally:
        # Restore the committed results.json so the repo is left untouched.
        shutil.move(BACKUP, RESULTS)


if __name__ == "__main__":
    raise SystemExit(main())
