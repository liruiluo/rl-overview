from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean
import hydra
from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table

console = Console()


def last_value(path: Path) -> float:
    try:
        with path.open() as f:
            rdr = csv.DictReader(f)
            vals = [float(r["return"]) for r in rdr]
        return vals[-1] if vals else float("nan")
    except Exception:
        return float("nan")


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    root = Path(getattr(cfg, "sweep_dir", "sweep_gym_dqn"))
    if not root.exists():
        raise SystemExit(f"找不到 sweep 目录: {root}")

    rows = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        metrics = sub / "metrics.csv"
        val = last_value(metrics)
        rows.append((sub.name, val))

    table = Table(title=f"Sweep Summary ({root})")
    table.add_column("config")
    table.add_column("last_return")
    for name, val in rows:
        table.add_row(name, f"{val:.3f}")
    console.print(table)


if __name__ == "__main__":
    main()

