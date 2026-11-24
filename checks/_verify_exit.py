"""Turn printed FAILS lines into a nonzero exit status.

Many recounts only PRINT a verdict (`... HOLDS` / `... FAILS`) and never set an exit code,
so `run_all_recounts.sh` counted them as passing even when their own output said FAILS.
Import this module in such a checker (module level, before main runs); it tees stdout,
watches for failure markers, and exits 1 at interpreter shutdown if one was printed.

Deliberately narrow patterns (checker vocabulary), so descriptive prose does not trip it:
  FAILS, FAILED, MISMATCH, LABEL DRIFT, rows absent
"""
import atexit
import os
import re
import sys

_FAIL = re.compile(r"\bFAILS\b|\bFAILED\b|\bMISMATCH\b|LABEL DRIFT|\babsent\b|\bis empty\b")


class _Tee:
    def __init__(self, stream):
        self.stream = stream
        self.failed = False

    def write(self, s):
        if _FAIL.search(s):
            self.failed = True
        return self.stream.write(s)

    def flush(self):
        return self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


_tee = _Tee(sys.stdout)
sys.stdout = _tee


@atexit.register
def _exit_on_failure():
    try:
        sys.stdout.flush()
    except Exception:  # noqa: BLE001
        pass
    if _tee.failed:
        os._exit(1)
