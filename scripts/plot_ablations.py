# scripts/plot_ablations.py
"""Generate ablation plots comparing different method configurations."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="simple_spread_v3")
    p.add_argument("--methods", type=str, nargs="+", default=["ippo", "myndra_mappo"],
                   help="Methods to compare")
    return p.parse_args()

def load_method_auc(env_name, method):
    """Load AUC data for a method if available."""
    auc_path = Path(f"results/marl/{env_name}/{method}/auc.json")
    if auc_path.exists():
        with open(auc_path, "r") as f:
            return json.load(f)
    return None

def load_method_throughput(env_name, method):
    """Load final throughput for a method."""
    summary_path = Path(f"results/marl/{env_name}/{method}/summary.csv")
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        # Get last step's throughput
        return {
            "mean_sps": df["steps_per_second_mean"].iloc[-1],
            "std_sps": df["steps_per_second_std"].iloc[-1]
        }
    return None

def plot_ablations(methods_data, out_path, title, ylabel="AUC"):
    """Create bar plot comparing methods."""
    if not methods_data:
        print("No data for ablation plot")
        return
    
    methods = list(methods_data.keys())
    means = [methods_data[m]["mean"] for m in methods]
    stds = [methods_data[m]["std"] for m in methods]
    
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(methods))
    bars = plt.bar(x_pos, means, yerr=stds, capsize=8, alpha=0.7)
    
    plt.xticks(x_pos, [m.upper().replace("_", "-") for m in methods], rotation=15, ha='right')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    
    # Add value labels
    for i, (bar, val, std) in enumerate(zip(bars, means, stds)):
        label = f'{val:.0f}' if ylabel.startswith('Steps') else f'{val:.1f}'
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std,
                label, ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved ablation plot to {out_path}")

def main():
    args = parse_args()
    env_name = args.env
    
    # Collect AUC data
    auc_data = {}
    for method in args.methods:
        auc = load_method_auc(env_name, method)
        if auc:
            auc_data[method] = {
                "mean": auc["auc_mean"],
                "std": (auc["auc_hi"] - auc["auc_lo"]) / 3.92  # Approximate std from 95% CI
            }
    
    # Collect throughput data
    throughput_data = {}
    for method in args.methods:
        tp = load_method_throughput(env_name, method)
        if tp:
            throughput_data[method] = {
                "mean": tp["mean_sps"],
                "std": tp["std_sps"]
            }
    
    # Create plots
    out_dir = Path(f"results/marl/{env_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if auc_data:
        plot_ablations(auc_data, out_dir / "ablations_auc.png",
                      f"Ablations: AUC Comparison on {env_name}",
                      ylabel="|AUC| of reward vs steps")
    
    if throughput_data:
        plot_ablations(throughput_data, out_dir / "ablations_throughput.png",
                      f"Ablations: Throughput Comparison on {env_name}",
                      ylabel="Steps per Second")
    
    # Save data
    with open(out_dir / "ablations.json", "w") as f:
        json.dump({"auc": auc_data, "throughput": throughput_data}, f, indent=2)
    print(f"Saved ablation data to {out_dir / 'ablations.json'}")

if __name__ == "__main__":
    main()
