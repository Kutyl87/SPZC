"""
Plotting script for model stealing distillation results.

Supports both old flat metrics format and new nested {train, steal} format.
Looks for results_*.json recursively in output-dir subdirectories.

Usage:
    python plot.py                              # plot all experiments
    python plot.py --run img*                   # filter runs by pattern
    python plot.py --output-dir outputs --save-dir figures
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(output_dir):
    files = sorted(Path(output_dir).rglob("results_*.json"))
    if not files:
        print(f"No result files found under {output_dir}/")
        return []
    results = []
    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
        except json.JSONDecodeError as e:
            print(f"  Warning: skipping malformed JSON {f} — {e}")
            continue
        data["_file"] = str(f)
        results.append(data)
    return results


def filter_results(results, run_pattern):
    if not run_pattern:
        return results
    filtered = []
    for r in results:
        if any(p in r["_file"] for p in run_pattern):
            filtered.append(r)
    return filtered


def short_label(r):
    m = r["metadata"]
    return f"{m['dataset']}_{m['model']}"


def get_metric(entries, key):
    return [e[key] for e in entries]


def has_field(entries, key):
    return entries and key in entries[0]


def extract_phase(r, phase):
    if isinstance(r["metrics"], dict):
        return r["metrics"].get(phase, [])
    return None


def plot_steal_combined(results_list, save_dir, experiment_name, dpi):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    markers = ["o", "s", "^", "v"]

    for i, r in enumerate(results_list):
        label = short_label(r)
        steal = extract_phase(r, "steal")
        if not steal:
            continue
        epochs = get_metric(steal, "epoch")
        m = markers[i % len(markers)]

        axes[0].plot(epochs, get_metric(steal, "train_loss"), marker=m, label=label)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title(f"{experiment_name} — Loss")
        axes[0].grid(True, alpha=0.3)

        if has_field(steal, "teacher_match"):
            axes[1].plot(epochs, get_metric(steal, "teacher_match"), marker=m, label=label)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Teacher Match")
        axes[1].set_title(f"{experiment_name} — Val Teacher Match")
        axes[1].grid(True, alpha=0.3)

        if has_field(steal, "val_student_acc"):
            axes[2].plot(epochs, get_metric(steal, "val_student_acc"), marker=m, label=label)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Accuracy")
        axes[2].set_title(f"{experiment_name} — Val Accuracy")
        axes[2].grid(True, alpha=0.3)

    for ax in axes:
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)

    fig.suptitle(f"Experiment: {experiment_name}", fontsize=14)
    fig.tight_layout()
    safe_name = experiment_name.lower().replace(" ", "_")
    out_dir = save_dir / safe_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{safe_name}_combined.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


def plot_least_confident_run(r, save_dir, dpi):
    label = short_label(r)
    train_entries = extract_phase(r, "train")
    steal_entries = extract_phase(r, "steal")

    if not train_entries and not steal_entries:
        return

    epochs_t = get_metric(train_entries, "epoch") if train_entries else []
    offset = max(epochs_t) if epochs_t else 0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    if train_entries:
        ax.plot(epochs_t, get_metric(train_entries, "train_loss"), marker="s", label="Train Phase", color="C0")
    if steal_entries:
        epochs_s = get_metric(steal_entries, "epoch")
        ax.plot([e + offset for e in epochs_s], get_metric(steal_entries, "train_loss"), marker="o", label="Steal Phase (high entropy)", color="C1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{label} — Training + Stealing Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if train_entries:
        ax.plot(epochs_t, get_metric(train_entries, "teacher_match"), marker="o", label="Train Phase (val match)", color="C0")
    if steal_entries:
        epochs_s = get_metric(steal_entries, "epoch")
        ax.plot([e + offset for e in epochs_s], get_metric(steal_entries, "teacher_match"), marker="s", label="Steal Phase (val match)", color="C1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Teacher Match")
    ax.set_title(f"{label} — Val Teacher Match (both phases)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    if train_entries and has_field(train_entries, "val_student_acc"):
        ax.plot(epochs_t, get_metric(train_entries, "val_student_acc"), marker="o", label="Train Val Student", color="C2")
    if steal_entries and has_field(steal_entries, "val_student_acc"):
        epochs_s = get_metric(steal_entries, "epoch")
        ax.plot([e + offset for e in epochs_s], get_metric(steal_entries, "val_student_acc"), marker="s", label="Steal Val Student", color="C4")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{label} — Val Accuracy (both phases)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Least Confident (Entropy) — {label}", fontsize=14)
    fig.tight_layout()
    out_dir = save_dir / "least_confident"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{label}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot distillation results.")
    parser.add_argument("--output-dir", default="outputs", help="Directory with result JSONs.")
    parser.add_argument("--save-dir", default="figures", help="Directory to save plots.")
    parser.add_argument("--run", nargs="*", default=None, help="Filter runs by label pattern.")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def main():
    args = parse_args()
    all_results = load_results(args.output_dir)
    all_results = filter_results(all_results, args.run)

    if not all_results:
        print("No results to plot.")
        return

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    distillation = [r for r in all_results if "distillation" in r["_file"]]
    hard_label = [r for r in all_results if "hard_label" in r["_file"]]
    least_confident = [r for r in all_results if "least_confident" in r["_file"]]

    print(f"Found {len(all_results)} run(s):")
    for r in all_results:
        m = r["metadata"]
        folder = Path(r["_file"]).parent.name
        print(f"  [{folder}] {m['dataset']} / {m['model']}")

    if distillation:
        print("\n--- Experiment 1: Distillation Loss (all logits) ---")
        plot_steal_combined(distillation, save_dir, "Distillation", args.dpi)

    if hard_label:
        print("\n--- Experiment 2: Hard Label Loss (top-1 only) ---")
        plot_steal_combined(hard_label, save_dir, "Hard Label", args.dpi)

    if least_confident:
        print("\n--- Experiment 3: Least Confident (Entropy selection) ---")
        for r in least_confident:
            plot_least_confident_run(r, save_dir, args.dpi)

    print(f"\nAll plots saved to {save_dir}/")


if __name__ == "__main__":
    main()
