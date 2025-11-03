# scripts/plot_scaling.py
"""Generate scaling plots showing throughput vs actors."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="simple_spread_v3")
    p.add_argument("--method", type=str, default="ippo")
    p.add_argument("--actors", type=int, nargs="+", default=[1, 2, 4, 8],
                   help="List of actor counts to plot")
    return p.parse_args()

def load_actor_results(env_name, method, actors_list):
    """Load results for different actor counts."""
    results = []
    base = Path(f"results/marl/{env_name}/{method}")
    
    for actors in actors_list:
        # Look for runs with this actor count
        seed_runs = []
        for seed_dir in sorted(base.glob("seed_*")):
            metrics_path = seed_dir / "train_metrics.csv"
            if not metrics_path.exists():
                continue
            df = pd.read_csv(metrics_path)
            # Check if this run used the target actor count
            if "actors" in df.columns and df["actors"].iloc[0] == actors:
                seed_runs.append(df)
        
        if seed_runs:
            # Aggregate across seeds
            steps_per_second = []
            for df in seed_runs:
                # Take last measurement as most stable
                steps_per_second.append(df["steps_per_second"].iloc[-1])
            
            results.append({
                "actors": actors,
                "mean_sps": np.mean(steps_per_second),
                "std_sps": np.std(steps_per_second, ddof=1) if len(steps_per_second) > 1 else 0,
                "n_seeds": len(steps_per_second)
            })
    
    return results

def plot_scaling(results, out_path, title):
    """Create bar plot of throughput vs actors."""
    if not results:
        print("No results found for scaling plot")
        return
    
    actors = [r["actors"] for r in results]
    mean_sps = [r["mean_sps"] for r in results]
    std_sps = [r["std_sps"] for r in results]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(range(len(actors)), mean_sps, yerr=std_sps, capsize=6, alpha=0.7)
    plt.xticks(range(len(actors)), actors)
    plt.xlabel("Number of Actors")
    plt.ylabel("Steps per Second")
    plt.title(title)
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, mean_sps)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_sps[i],
                f'{val:.0f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved scaling plot to {out_path}")

def main():
    args = parse_args()
    results = load_actor_results(args.env, args.method, args.actors)
    
    if not results:
        print(f"No results found for {args.method} on {args.env}")
        return
    
    out_dir = Path(f"results/marl/{args.env}/{args.method}")
    out_path = out_dir / "scaling.png"
    
    title = f"Scaling: {args.method.upper()} on {args.env}"
    plot_scaling(results, out_path, title)
    
    # Also save numerical data
    import json
    with open(out_dir / "scaling.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved scaling data to {out_dir / 'scaling.json'}")

if __name__ == "__main__":
    main()
