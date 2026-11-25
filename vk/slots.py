"""Machine-wide concurrency limits shared by every agent on this Mac (flock-based semaphores).

Many video builds run at once on one 10-core / 16 GB machine. Heavy steps take a slot:
    render  (manim, ~1 GB, 1 core)        VK_RENDER_SLOTS, default 4
    tts     (Kokoro/torch, ~1.5 GB)       VK_TTS_SLOTS,    default 2
    asr     (parakeet, ~2 GB)             VK_ASR_SLOTS,    default 1
    net     (Semantic Scholar requests)   VK_NET_SLOTS,    default 1
"""
from __future__ import annotations

import fcntl
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

LOCKS = Path(__file__).resolve().parent.parent / ".locks"
DEFAULTS = {"render": 4, "tts": 2, "asr": 1, "net": 1}


_HELD: set[str] = set()


@contextmanager
def slot(kind: str, quiet: bool = False):
    """Hold one of the machine-wide `kind` slots for the duration of the block (re-entrant per process)."""
    if kind in _HELD:
        yield
        return
    LOCKS.mkdir(exist_ok=True)
    n = int(os.environ.get(f"VK_{kind.upper()}_SLOTS", DEFAULTS.get(kind, 1)))
    waited, t0 = False, time.time()
    while True:
        for i in range(n):
            fh = open(LOCKS / f"{kind}.{i}", "w")
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                fh.close()
                continue
            if waited and not quiet:
                print(f"[slots] got {kind} slot after {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
            _HELD.add(kind)
            try:
                yield
            finally:
                _HELD.discard(kind)
                fcntl.flock(fh, fcntl.LOCK_UN)
                fh.close()
            return
        if not waited and not quiet:
            print(f"[slots] all {n} {kind} slots busy (other agents); waiting…", file=sys.stderr, flush=True)
        waited = True
        time.sleep(2.0)
