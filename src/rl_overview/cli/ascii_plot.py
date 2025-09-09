from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig
from rich.console import Console

console = Console()


def ascii_plot(values: List[float], width: int = 60, height: int = 12) -> str:
    if not values:
        return "(no data)"
    vmin = min(values)
    vmax = max(values)
    if vmax - vmin < 1e-12:
        vmax = vmin + 1e-12
    # downsample to desired width
    n = len(values)
    if n > width:
        step = n / width
        xs = [values[int(i * step)] for i in range(width)]
    else:
        xs = values[:] + [values[-1]] * (width - n)
    # map to rows
    rows = []
    for r in range(height, -1, -1):
        y = vmin + (vmax - vmin) * r / height
        row = []
        for x in xs:
            row.append("█" if x >= y else " ")
        rows.append("".join(row))
    axis = f"min={vmin:.3f} max={vmax:.3f} last={values[-1]:.3f}"
    return "\n".join(rows + [axis])


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    path = Path(getattr(cfg, "metrics_path", "metrics.csv"))
    if not path.exists():
        raise SystemExit(f"找不到指标文件: {path}")
    returns = []
    with path.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            returns.append(float(row["return"]))
    console.rule(f"ASCII Plot ({path})")
    console.print(ascii_plot(returns, width=int(getattr(cfg, "width", 60)), height=int(getattr(cfg, "height", 12))))


if __name__ == "__main__":
    main()

