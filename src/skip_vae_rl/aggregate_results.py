from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_eval(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return {
        "run": path.parent.name,
        "mean_reward": payload["mean_reward"],
        "std_reward": payload["std_reward"],
        "mean_length": payload["mean_length"],
        "num_eval_seeds": payload.get("num_eval_seeds", 1),
        "episodes_per_seed": payload.get("episodes_per_seed", len(payload.get("rewards", []))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out", default="reports/results_summary.csv")
    args = parser.parse_args()

    eval_paths = sorted(Path(args.runs_dir).glob("*/eval.json"))
    if not eval_paths:
        raise FileNotFoundError(f"No eval.json files found under {args.runs_dir}")

    df = pd.DataFrame(load_eval(path) for path in eval_paths)
    df = df.sort_values(["mean_reward", "mean_length"], ascending=[False, True])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()

