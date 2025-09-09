from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class EpisodeLogger:
    path: Path
    print_every: int = 0

    def __post_init__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(["episode", "return", "length"])
        self._f.flush()

    def log(self, episode: int, ret: float, length: int):
        self._w.writerow([int(episode), float(ret), int(length)])
        if self.print_every and episode % self.print_every == 0:
            print(f"[log] ep={episode} return={ret:.3f} length={length}")
        self._f.flush()

    def close(self):  # pragma: no cover
        try:
            self._f.close()
        except Exception:
            pass

