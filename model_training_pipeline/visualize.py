"""Visualization functions for training results."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
import pandas as pd
HARTREE_TO_EV = 27.2114


def plot_training_curves(history, fig_dir):
    """Plot training loss and learning rate curves."""
    has_lr = bool("lr" in history and history["lr"])
    fig, axes = plt.subplots(1, 1 + has_lr,
                             figsize=(7 + 5 * has_lr, 4.5))
    if not has_lr:
        axes = [axes]

    epochs = np.arange(1, len(history["train"]) + 1)

    # Loss subplot
    ax = axes[0]
    ax.plot(epochs, history["train"], label="Train MSE", linewidth=1.5)
    if "val" in history and history["val"]:
        ax.plot(epochs, history["val"], label="Val MSE", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (normalized)")
    ax.set_title("Training Curves (Frozen MACE)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # LR subplot
    if has_lr:
        ax2 = axes[1]
        ax2.plot(epochs, history["lr"], color="orange", linewidth=1.5)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("Learning Rate Schedule")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "training_curves.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {fig_dir / 'training_curves.png'}")


def plot_pred_vs_true(trues_ev, preds_ev, test_mae_ev, fig_dir):
    """Plot predicted vs true energy scatter plot (eV) with R² annotation."""
    r2 = r2_score(trues_ev, preds_ev)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(trues_ev, preds_ev, c="steelblue", edgecolors="k",
               linewidths=0.5, s=60, zorder=3)
    lo = min(trues_ev.min(), preds_ev.min()) - 0.05
    hi = max(trues_ev.max(), preds_ev.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True Energy (eV)")
    ax.set_ylabel("Predicted Energy (eV)")
    ax.set_title(f"Test Set: MAE = {test_mae_ev:.4f} eV")
    ax.text(0.05, 0.92, f"$R^2 = {r2:.4f}$",
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "pred_vs_true.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_dir / 'pred_vs_true.png'}  (R² = {r2:.4f})")


def plot_per_sample_error(names, errors_ev, test_mae_ev, fig_dir):
    """Plot per-sample prediction error bar chart (meV)."""
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.55), 5))
    colors = ["#d9534f" if abs(e) > test_mae_ev else "#5cb85c" for e in errors_ev]
    x_pos = np.arange(len(names))
    ax.bar(x_pos, errors_ev * 1000, color=colors, edgecolor="k", linewidth=0.4)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.axhline(test_mae_ev * 1000, color="grey", linestyle="--", linewidth=0.8,
               label=f"MAE = {test_mae_ev*1000:.2f} meV")
    ax.axhline(-test_mae_ev * 1000, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Error (meV)")
    ax.set_title("Per-sample Prediction Error (Test Set)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "per_sample_error.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_dir / 'per_sample_error.png'}")


def plot_pred_vs_true_with_uncertainty(trues_ev, preds_ev, stds_ev, test_mae_ev, fig_dir):
    """Scatter plot of pred vs true with GP uncertainty error bars."""
    r2 = r2_score(trues_ev, preds_ev)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(trues_ev, preds_ev, yerr=2 * stds_ev, fmt='o',
                capsize=3, alpha=0.7, markersize=5, color="steelblue",
                ecolor="salmon", label=r"Pred $\pm 2\sigma$")
    lo = min(trues_ev.min(), (preds_ev - 2 * stds_ev).min()) - 0.05
    hi = max(trues_ev.max(), (preds_ev + 2 * stds_ev).max()) + 0.05
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True Energy (eV)")
    ax.set_ylabel("Predicted Energy (eV)")
    ax.set_title(f"GP Predictions: MAE = {test_mae_ev:.4f} eV")
    ax.text(0.05, 0.92, f"$R^2 = {r2:.4f}$",
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "pred_vs_true_gp.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_dir / 'pred_vs_true_gp.png'}  (R² = {r2:.4f})")
    df = pd.DataFrame({
        "True Energy (eV)": trues_ev,
        "Predicted Energy (eV)": preds_ev,
        "Predicted Uncertainty (eV)": stds_ev
    })
    df.to_csv(fig_dir / "predictions_with_uncertainty.csv", index=False)


def plot_uncertainty_analysis(stds_ev, errors_ev, fig_dir):
    """Uncertainty calibration and distribution plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Calibration: predicted uncertainty vs actual error
    ax = axes[0]
    ax.scatter(stds_ev, np.abs(errors_ev), alpha=0.7, s=30, c="steelblue",
               edgecolors="k", linewidths=0.3)
    max_val = max(stds_ev.max(), np.abs(errors_ev).max()) * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="Calibrated")
    ax.set_xlabel(r"Predicted Uncertainty ($\sigma$, eV)")
    ax.set_ylabel("|Actual Error| (eV)")
    ax.set_title("Uncertainty Calibration")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Uncertainty histogram
    ax = axes[1]
    ax.hist(stds_ev, bins=20, alpha=0.7, edgecolor="black", color="steelblue")
    ax.set_xlabel(r"Predicted Uncertainty ($\sigma$, eV)")
    ax.set_ylabel("Count")
    ax.set_title("Uncertainty Distribution")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "uncertainty_analysis.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_dir / 'uncertainty_analysis.png'}")
