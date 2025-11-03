import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from marl.train_ppo import train
import os

def aggregate_results(env_name, method, seeds):
    """Aggregate results from multiple seeds into one summary.csv."""
    base_dir = Path(f"results/marl/{env_name}/{method}")
    all_dfs = []

    for seed in range(seeds):
        seed_path = base_dir / f"seed_{seed}" / "train_metrics.csv"
        if not seed_path.exists():
            print(f" Skipping missing seed run: {seed}")
            continue
        df = pd.read_csv(seed_path)
        df["seed"] = seed
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError(" No results found — check your training runs.")

    combined = pd.concat(all_dfs)
    grouped = combined.groupby("step").agg({
        "mean_reward": ["mean", "std"],
        "steps_per_second": ["mean", "std"]
    }).reset_index()

    grouped.columns = [
        "step",
        "mean_reward_mean",
        "mean_reward_std",
        "steps_per_second_mean",
        "steps_per_second_std"
    ]

    out_path = base_dir / "summary.csv"
    grouped.to_csv(out_path, index=False)
    print(f"\n Saved aggregated summary to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="simple_spread_v3")
    parser.add_argument("--method", type=str, default="ippo")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=5000)
    args = parser.parse_args()

    env_name = args.env
    method = args.method
    seeds = args.seeds
    total_steps = args.steps

    print(f" Running {method.upper()} on {env_name} for {seeds} seeds ({total_steps} steps each)")

    for seed in range(seeds):
        print(f"\n Starting seed {seed} ...")
        os.environ["PYTHONHASHSEED"] = str(seed)
        result = train(env_name=env_name, total_steps=total_steps, seed=seed)
        print(f"Finished seed {seed} → {result['metrics_csv']}")

    # Aggregate all results
    aggregate_results(env_name, method, seeds)

if __name__ == "__main__":
    main()