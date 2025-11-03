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
    parser.add_argument("--method", type=str, default="ippo", 
                       help="Method: 'ippo' for baseline, 'myndra_mappo' for planner-aware")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=5000)
    # Planner-specific arguments
    parser.add_argument("--interval", type=int, default=32, 
                       help="Planner update interval (steps)")
    parser.add_argument("--context-dim", type=int, default=4,
                       help="Dimensionality of planner context vector")
    parser.add_argument("--planner-cache", choices=["on", "off"], default="on",
                       help="Whether to cache planner context between intervals")
    parser.add_argument("--actors", type=int, default=1,
                       help="Number of parallel actor environments for rollouts")
    parser.add_argument("--amp", choices=["on", "off"], default="off",
                       help="Enable automatic mixed precision (AMP)")
    parser.add_argument("--compile", choices=["on", "off"], default="off",
                       help="Enable torch.compile() for models")
    parser.add_argument("--target-return", type=float, default=None,
                       help="Target mean reward to track time-to-target (optional)")
    args = parser.parse_args()

    env_name = args.env
    method = args.method
    seeds = args.seeds
    total_steps = args.steps
    
    # Determine if planner should be used based on method name
    use_planner = (method == "myndra_mappo")
    planner_interval = args.interval
    context_dim = args.context_dim
    planner_cache = (args.planner_cache == "on")
    actors = args.actors
    use_amp = (args.amp == "on")
    use_compile = (args.compile == "on")
    target_return = args.target_return

    print(f" Running {method.upper()} on {env_name} for {seeds} seeds ({total_steps} steps each)")
    print(f"  Actors: {actors}, AMP: {use_amp}, Compile: {use_compile}")
    if target_return is not None:
        print(f"  Target return: {target_return}")
    if use_planner:
        print(f"  Planner: interval={planner_interval}, context_dim={context_dim}, cache={planner_cache}")

    for seed in range(seeds):
        print(f"\n Starting seed {seed} ...")
        os.environ["PYTHONHASHSEED"] = str(seed)
        result = train(
            env_name=env_name, 
            total_steps=total_steps, 
            seed=seed,
            use_planner=use_planner,
            planner_interval=planner_interval,
            context_dim=context_dim,
            planner_cache=planner_cache,
            method=method,
            actors=actors,
            use_amp=use_amp,
            use_compile=use_compile,
            target_return=target_return
        )
        print(f"Finished seed {seed} → {result['metrics_csv']}")

    # Aggregate all results
    aggregate_results(env_name, method, seeds)

if __name__ == "__main__":
    main()