from __future__ import annotations

import csv
from pathlib import Path
import statistics
from typing import List

import hydra
from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table

console = Console()


def moving_avg(xs: List[float], k: int) -> List[float]:
    if k <= 1:
        return xs
    out = []
    s = 0.0
    q = []
    for x in xs:
        q.append(x)
        s += x
        if len(q) > k:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    path = Path(getattr(cfg, "metrics_path", "metrics.csv"))
    if not path.exists():
        raise SystemExit(f"找不到指标文件: {path}")

    episodes = []
    returns = []
    lengths = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            returns.append(float(row["return"]))
            lengths.append(int(row["length"]))

    k = int(getattr(cfg, "ma_window", 50))
    ma = moving_avg(returns, k)
    table = Table(title=f"Metrics Summary ({path})")
    table.add_column("metric")
    table.add_column("value")
    table.add_row("episodes", str(len(episodes)))
    table.add_row("return_mean", f"{statistics.mean(returns):.3f}")
    table.add_row("return_last_ma", f"{ma[-1]:.3f}")
    table.add_row("length_mean", f"{statistics.mean(lengths):.1f}")
    console.print(table)


if __name__ == "__main__":
    main()

