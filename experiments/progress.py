"""progress.py — one-line progress reporting with an ETA, for the long unattended runs.

Collection, sweeps and training all run for hours. Dumping per-step output makes it impossible to
tell how far along a run is or when it will finish, and it buries the numbers that matter. This
prints a single updating line:

    [ 37/480]  32% | 1h04m elapsed | ETA 2h11m (18:42) | 1.4 min/it | clean 21 (57%)

Use as a context manager so the final summary always prints, even on an exception.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta


def _hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


class Progress:
    """Track N items, print one line per completed item with a running ETA.

    `note(**kw)` attaches running counters (e.g. clean demos kept) that appear on the line, so the
    thing you actually care about is visible without trawling a log.
    """

    def __init__(self, total: int, label: str = "", stream=sys.stdout, every: int = 1):
        self.total = max(1, int(total))
        self.label = label
        self.stream = stream
        self.every = max(1, int(every))
        self.done = 0
        self.counters: dict[str, int] = {}
        self.t0 = time.time()

    def __enter__(self):
        if self.label:
            print(f"{self.label}: {self.total} items", file=self.stream, flush=True)
        return self

    def __exit__(self, *exc):
        el = time.time() - self.t0
        extra = "  ".join(f"{k} {v}" for k, v in self.counters.items())
        print(f"  finished {self.done}/{self.total} in {_hms(el)}"
              + (f"   {extra}" if extra else ""), file=self.stream, flush=True)
        return False

    def note(self, **counters):
        """Add to named running counters, shown on subsequent progress lines."""
        for k, v in counters.items():
            self.counters[k] = self.counters.get(k, 0) + int(v)

    def step(self, suffix: str = ""):
        self.done += 1
        if self.done % self.every and self.done != self.total:
            return
        el = time.time() - self.t0
        per = el / self.done
        remaining = per * (self.total - self.done)
        eta_clock = (datetime.now() + timedelta(seconds=remaining)).strftime("%H:%M")
        pct = 100.0 * self.done / self.total
        counters = ""
        if self.counters:
            counters = " | " + " ".join(
                f"{k} {v}" + (f" ({100.0*v/self.done:.0f}%)" if v <= self.done else "")
                for k, v in self.counters.items())
        rate = f"{per/60:.1f} min/it" if per >= 30 else f"{per:.1f} s/it"
        print(f"  [{self.done:>4}/{self.total}] {pct:>3.0f}% | {_hms(el)} elapsed "
              f"| ETA {_hms(remaining)} ({eta_clock}) | {rate}{counters}"
              + (f" | {suffix}" if suffix else ""),
              file=self.stream, flush=True)
