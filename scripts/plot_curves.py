# scripts/plot_curves.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="simple_spread_v3")
    p.add_argument("--method", type=str, default="ippo")
    # AUC handling: raw (signed), abs (magnitude), normalized (divide by step range)
    p.add_argument(
        "--auc",
        type=str,
        choices=["raw", "abs", "normalized"],
        default="abs",
        help="How to report AUC: raw (signed), abs (magnitude), or normalized by step range.",
    )
    p.add_argument("--boot", type=int, default=1000, help="Bootstrap samples for AUC CI")
    p.add_argument("--ci", type=float, default=0.95, help="Confidence level for CI (e.g., 0.95)")
    return p.parse_args()

def load_seed_runs(env_name: str, method: str):
    base = Path(f"results/marl/{env_name}/{method}")
    seed_dirs = sorted([d for d in base.glob("seed_*") if d.is_dir()])
    dfs = []
    for d in seed_dirs:
        csv_path = d / "train_metrics.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["seed"] = int(d.name.split("_")[-1])
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No per-seed CSVs under {base}.")
    # Align by 'step'
    combined = pd.concat(dfs).sort_values(["step", "seed"])
    return combined

def summarize_and_bootstrap(combined: pd.DataFrame, auc_mode: str = "abs", n_boot: int = 1000, ci: float = 0.95):
    """Summarize learning curves and bootstrap the AUC across seeds.

    auc_mode:
      - "raw": signed area under reward-vs-steps curve
      - "abs": absolute AUC (magnitude), easier to compare across runs with negative rewards
      - "normalized": AUC divided by step range so values are comparable across different horizons
    """
    # Pivot to steps x seeds matrix
    wide = (
        combined
        .pivot(index="step", columns="seed", values="mean_reward")
        .dropna(axis=0, how="any")
    )
    steps = wide.index.values
    seeds = wide.columns.values
    mean_curve = wide.mean(axis=1).values
    std_curve  = wide.std(axis=1, ddof=1).values

    # Bootstrap AUC over seeds
    rng = np.random.default_rng(42)
    auc_samples = []
    seed_values = wide.values  # shape: [T, S]
    T, S = seed_values.shape

    step_range = float(steps.max() - steps.min()) if len(steps) > 1 else 1.0

    for _ in range(n_boot):
        # resample seeds with replacement
        idx = rng.integers(low=0, high=S, size=S)
        boot_curve = seed_values[:, idx].mean(axis=1)
        # trapezoidal AUC over steps (np.trapz deprecated → use trapezoid)
        auc = np.trapezoid(boot_curve, steps)
        if auc_mode == "abs":
            auc = abs(auc)
        elif auc_mode == "normalized":
            auc = auc / step_range
        auc_samples.append(auc)

    auc_samples = np.array(auc_samples)
    lo = np.percentile(auc_samples, (1 - ci) / 2 * 100)
    hi = np.percentile(auc_samples, (1 + ci) / 2 * 100)
    auc_mean = auc_samples.mean()

    return {
        "steps": steps,
        "mean_curve": mean_curve,
        "std_curve": std_curve,
        "auc_mean": auc_mean,
        "auc_lo": lo,
        "auc_hi": hi,
        "n_seeds": S,
        "auc_mode": auc_mode,
    }

def plot_learning_curve(steps, mean_curve, std_curve, out_path: Path, title: str):
    plt.figure()
    plt.plot(steps, mean_curve, label="Mean reward")
    plt.fill_between(steps, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2, label="±1 std")
    plt.xlabel("Steps")
    plt.ylabel("Episode mean reward")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()

def plot_auc_bar(auc_mean, auc_lo, auc_hi, out_path: Path, title: str, ylabel: str):
    plt.figure()
    plt.bar([0], [auc_mean], width=0.4)
    # error bar: asymmetric CI
    yerr = np.array([[max(0.0, auc_mean - auc_lo)], [max(0.0, auc_hi - auc_mean)]])
    plt.errorbar([0], [auc_mean], yerr=yerr, fmt="none", capsize=6)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xticks([0], ["AUC"])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()

def main():
    args = parse_args()
    env_name, method = args.env, args.method
    combined = load_seed_runs(env_name, method)
    stats = summarize_and_bootstrap(combined, auc_mode=args.auc, n_boot=args.boot, ci=args.ci)

    base = Path(f"results/marl/{env_name}/{method}")
    curve_png = base / "curves.png"
    auc_png   = base / "auc.png"

    title = f"{method.upper()} on {env_name}  (n={stats['n_seeds']} seeds)"
    plot_learning_curve(stats["steps"], stats["mean_curve"], stats["std_curve"], curve_png, title)

    if stats["auc_mode"] == "normalized":
        ylabel = "Normalized AUC of reward vs steps"
    elif stats["auc_mode"] == "abs":
        ylabel = "|AUC| of reward vs steps"
    else:
        ylabel = "AUC of reward vs steps"

    plot_auc_bar(stats["auc_mean"], stats["auc_lo"], stats["auc_hi"], auc_png, title, ylabel)

    # Also write a small JSON with AUC numbers for the paper appendix
    import json
    with open(base / "auc.json", "w") as f:
        json.dump({
            "auc_mode": stats["auc_mode"],
            "auc_mean": float(stats["auc_mean"]),
            "auc_lo": float(stats["auc_lo"]),
            "auc_hi": float(stats["auc_hi"]),
            "n_seeds": int(stats["n_seeds"])
        }, f, indent=2)

    print(
        " Saved plots:\n  - {}\n  - {}\n Saved AUC stats: {}".format(curve_png, auc_png, base / "auc.json")
    )

if __name__ == "__main__":
    main()