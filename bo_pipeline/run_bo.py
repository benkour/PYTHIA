"""Bayesian Optimization for adsorbate placement on defective Si clusters.

Usage:
    python run_bo.py                     # use config.json defaults
    python run_bo.py --sample-size 500   # override sample size
"""

import argparse
import json
import os
import re
import sys
import time
import warnings
from collections import Counter

# from sympy import re
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / 'bo_pipeline'))

from bo_functionalities import BO
# import bo_old_school
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'utils'))

from helpers import read_last_snapshot, atom_extraction, get_vdw_radius, save_points
from ase.data import vdw_radii, chemical_symbols


# bo_old_school / placement / helpers read their config at import time from
# the BO_CONFIG env var (default: config.json), so --config must be exported
# before those modules are imported below.
for _i, _a in enumerate(sys.argv):
    if _a == "--config" and _i + 1 < len(sys.argv):
        os.environ.setdefault("BO_CONFIG", sys.argv[_i + 1])
    elif _a.startswith("--config="):
        os.environ.setdefault("BO_CONFIG", _a.split("=", 1)[1])



ATOM_COLOR = {
    "Si": "#F0C8A0", "Al": "#BFA6A6", "Sc": "#E6E6E6", "B": "#FFB5B5",
    "P": "#FF8000", "C": "#909090", "H": "#FFFFFF", "O": "#FF0D0D",
    "N": "#3050F8", "S": "#FFFF30", "Ni": "#50D050", "Pt": "#D0D0E0",
    "Zn": "#7D80B0", "Fe": "#E06633", "Ti": "#BFC2C7",
    "Ge": "#668F8F", "As": "#BD80E3", "Ga": "#C28F8F", "Cu": "#C88033",
    "Nb": "#73C2C9", "Ta": "#4DA6FF", "Pd": "#006985",
}
ATOM_VDW = {
    "Si": 2.10, "Al": 1.84, "Sc": 2.15, "B": 1.92,
    "P": 1.80, "C": 1.70, "H": 1.20, "O": 1.52,
    "N": 1.55, "S": 1.80, "Ni": 1.63, "Pt": 1.75,
    "Zn": 1.39, "Fe": 1.56, "Ti": 1.76,
    "Ge": 2.11, "As": 1.85, "Ga": 1.87, "Cu": 1.40,
    "Nb": 2.18, "Ta": 2.22, "Pd": 1.63,
}





def make_3d_html(anchor_positions, mu, std, cluster_origin, defect_info,
                 fig_dir, title, n_best=5):
    """Interactive 3D HTML with cluster, energy surface, defect, and top sites."""
    fig = go.Figure()

    # Cluster atoms by element
    atom_groups = {}
    for atom, pos in cluster_origin:
        atom_groups.setdefault(atom, ([], [], []))
        atom_groups[atom][0].append(float(pos[0]))
        atom_groups[atom][1].append(float(pos[1]))
        atom_groups[atom][2].append(float(pos[2]))

    for atom, (xs, ys, zs) in atom_groups.items():
        color = ATOM_COLOR.get(atom, "#CCCCCC")
        size = max(4, ATOM_VDW.get(atom, 1.5) * 4)
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="markers",
            marker=dict(size=size, color=color, line=dict(width=0.5, color="#333")),
            name=f"{atom} (cluster)",
            hovertemplate=f"{atom}<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<br>z=%{{z:.2f}}<extra></extra>",
        ))

    # Highlight defect atoms
    cluster_pos = np.array([pos for _, pos in cluster_origin])
    for d in defect_info:
        pos = cluster_pos[d["index"]]
        fig.add_trace(go.Scatter3d(
            x=[float(pos[0])], y=[float(pos[1])], z=[float(pos[2])],
            mode="markers+text",
            marker=dict(size=12, color="#FF00FF", symbol="cross",
                        line=dict(width=2, color="black")),
            text=[f'{d["to"]} (defect)'], textposition="top center",
            textfont=dict(size=11, color="#FF00FF"),
            name=f'Defect: {d["from"]}->{d["to"]} (#{d["index"]})',
        ))

    # Energy surface samples
    fig.add_trace(go.Scatter3d(
        x=anchor_positions[:, 0], y=anchor_positions[:, 1], z=anchor_positions[:, 2],
        mode="markers",
        marker=dict(size=3, color=mu, colorscale="Viridis",
                    colorbar=dict(title="Energy (eV)", thickness=15), opacity=0.6),
        name="BO Samples", customdata=std,
        hovertemplate="E=%{marker.color:.3f} eV<br>std=%{customdata:.3f} eV<br>"
                      "x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>",
    ))

    # Top-N best sites
    best_idx = np.argsort(mu)[:n_best]
    colors_best = ["#FF0000", "#FF6600", "#FFCC00", "#66FF00", "#00CCFF"]
    for rank, idx in enumerate(best_idx):
        pos = anchor_positions[idx]
        e, s = mu[idx], std[idx]
        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]], mode="markers+text",
            marker=dict(size=10, color=colors_best[rank % len(colors_best)],
                        symbol="diamond", line=dict(width=2, color="black")),
            text=[f"#{rank+1}"], textposition="top center",
            textfont=dict(size=12, color=colors_best[rank % len(colors_best)]),
            name=f"Best #{rank+1} (E={e:.3f} eV)",
        ))

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="X [A]", yaxis_title="Y [A]", zaxis_title="Z [A]",
                   aspectmode="data"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(os.path.join(fig_dir, "bo_energy_surface_3d.html"), include_plotlyjs="cdn")


def make_3d_png(anchor_positions, mu, cluster_pos, defect_info, fig_dir, title):
    """Static 3D scatter plot."""
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    p = ax.scatter(anchor_positions[:, 0], anchor_positions[:, 1], anchor_positions[:, 2],
                   c=mu, alpha=0.8, s=12, cmap="viridis")
    for d in defect_info:
        dp = cluster_pos[d["index"]]
        ax.scatter(dp[0], dp[1], dp[2], c="magenta", s=200, marker="X", zorder=5)
    fig.colorbar(p, ax=ax, shrink=0.6, label="Energy (eV)")
    ax.set_title(title)
    fig.savefig(os.path.join(fig_dir, "bo_energy_surface_3d.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_analysis_plot(anchor_positions, mu, std, cluster_pos, defect_info,
                       fig_dir, title, n_samples):
    """4-panel analysis plot."""
    best_idx = np.argsort(mu)[:5]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    ax = axes[0, 0]
    ax.hist(mu, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
    ax.axvline(mu.min(), color="red", linestyle="--", lw=1.5, label=f"Best: {mu.min():.3f}")
    ax.axvline(mu.mean(), color="orange", linestyle=":", lw=1.5, label=f"Mean: {mu.mean():.3f}")
    ax.set_xlabel("Predicted Energy (eV)"); ax.set_ylabel("Count")
    ax.set_title(f"Energy Distribution ({n_samples} samples)"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.hist(std, bins=30, color="orange", edgecolor="black", alpha=0.7)
    ax.set_xlabel("Uncertainty (eV)"); ax.set_ylabel("Count")
    ax.set_title("Uncertainty Distribution"); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    sc = ax.scatter(mu, std, c=mu + 0.5 * std, cmap="RdYlGn_r", s=15, alpha=0.6,
                    edgecolors="k", linewidths=0.3)
    ax.scatter(mu[best_idx[:3]], std[best_idx[:3]], c="red", s=100, marker="*",
               zorder=5, label="Top-3")
    ax.set_xlabel("Energy (eV)"); ax.set_ylabel("Uncertainty (eV)")
    ax.set_title("Energy vs Uncertainty"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax, label="UCB (k=0.5)")

    ax = axes[1, 1]
    sc2 = ax.scatter(anchor_positions[:, 0], anchor_positions[:, 1],
                     c=mu, cmap="viridis", s=10, alpha=0.4)
    ax.scatter(cluster_pos[:, 0], cluster_pos[:, 1], c="gray", s=12, alpha=0.3,
               marker="s", label="Cluster")
    for d in defect_info:
        dp = cluster_pos[d["index"]]
        ax.scatter(dp[0], dp[1], c="magenta", s=150, marker="X", zorder=5,
                   label=f'Defect: {d["to"]}')
    ax.scatter(anchor_positions[best_idx, 0], anchor_positions[best_idx, 1],
               c="red", s=100, marker="*", zorder=5, label="Top-5")
    for rank, idx in enumerate(best_idx[:3]):
        ax.annotate(f"#{rank+1}", (anchor_positions[idx, 0], anchor_positions[idx, 1]),
                    fontsize=9, fontweight="bold", color="red",
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("X [A]"); ax.set_ylabel("Y [A]")
    ax.set_title("Top Sites + Defect (XY)"); ax.legend(fontsize=8)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    fig.colorbar(sc2, ax=ax, label="Energy (eV)")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "bo_analysis.png"), dpi=150)
    plt.close(fig)


def main():
    # parser = argparse.ArgumentParser(description="Run BO for adsorbate placement")
    # parser.add_argument("--sample-size", type=int, default=None,
    #                     help="Override sample_size from config")
    # parser.add_argument("--config", type=str, default="config.json",
    #                     help="Path to config file")
    # args = parser.parse_args()

    # ── Config ──
    current_script_path = Path(__file__).resolve()
    with open(current_script_path.parent.parent / 'configs' / 'config_bo.json') as f:
        config = json.load(f)
    num_defects = config["num_defects"]

    sample_size = config["sample_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Bayesian Optimization — Adsorbate Placement")
    print("=" * 60)
    print(f"Device:          {device}")
    print(f"Checkpoint dir:  {config['bo_checkpoint_dir']}")
    print(f"Sample size:     {sample_size}")
    print(f"Defects:         {config['num_defects']} from {config['possible_defects']}")
    print(f"Acquisition:     {config['bo_acquisition']}")
    print(f"Tradeoffs:       {config['bo_tradeoff']}")

    # ── Load data ──
    print(f"\n[1/5] Loading cluster data...")
    cluster_type = config["baseline_system"]
    parent_dir = Path(__file__).resolve().parent.parent
    data_dir = parent_dir / "data_creation_pipeline" / config["data_dir"]
    atoms = atom_extraction(cluster_type + "_", data_dir)
    # data_dir = os.path.join("..", config["data_dir"])
    import glob as _glob
    data_files = sorted(_glob.glob(os.path.join(data_dir, f"{cluster_type}__*_data_*.pt")))
    if not data_files:
        print(f"  ERROR: No data files found for {cluster_type} in {data_dir}")
        return
    data = torch.load(data_files[0], map_location="cpu", weights_only=False)
    # adsorbate_smiles = config["adsorbate_label"]
    # adsorbate_smiles = re.sub(r"([A-Za-z])(?!\d)", r"\g<1>1", adsorbate_smiles)


    # #sum the numbers appearing in the adsorbate smiles string
    # n_ads = sum(int(num) for num in re.findall(r'\d+', adsorbate_smiles))
    # print(adsorbate_smiles, n_ads)
    # exit()
    adsorbate_xyz_path = config.get('adsorbate_xyz', 'adsorbate.xyz')
    adsorbate_xyz_path = os.path.join(current_script_path.parent,  adsorbate_xyz_path)
    adsorbate_coordinates = read_last_snapshot(adsorbate_xyz_path)
    n_ads = adsorbate_coordinates.shape[0]

    if config["van_der_waals_distances"]:
        surface_elements = atoms[:-n_ads]
        adsorbate_elements = atoms[-n_ads:]

        # Count the frequency of each element
        surface_counts = Counter(surface_elements)
        adsorbate_counts = Counter(adsorbate_elements)

        # Get unique elements and sort them by frequency
        unique_surface_elements = sorted(
            surface_counts.keys(), key=lambda x: surface_counts[x], reverse=True)
        unique_surface_elements = unique_surface_elements + [elem for elem in config["possible_defects"] if elem not in unique_surface_elements]
        # Also include elements found in all data_raw xyz files
        for xyz_file in _glob.glob(os.path.join("..", "data_raw", "*.xyz")):
            with open(xyz_file, 'r') as f:
                lines = f.readlines()
                n = int(lines[0].strip())
                for line in lines[2:2+n]:
                    elem = line.split()[0]
                    if elem not in unique_surface_elements:
                        unique_surface_elements.append(elem)
        unique_adsorbate_elements = sorted(
            adsorbate_counts.keys(), key=lambda x: adsorbate_counts[x], reverse=True)

        # Create the min_distances dictionary
        min_distances = {}
        for adsorbate_element in unique_adsorbate_elements:
            min_distances[adsorbate_element] = {}
            for surface_element in unique_surface_elements:
                vdw_radius_adsorbate = 0.7*get_vdw_radius(adsorbate_element)
                vdw_radius_surface = 0.7*get_vdw_radius(surface_element)
                min_distances[adsorbate_element][surface_element] = vdw_radius_adsorbate + \
                    vdw_radius_surface
        # Save the min_distances dictionary to a JSON file
        with open('min_distances.json', 'w') as json_file:
            json.dump(min_distances, json_file, indent=4)





    cluster = data.pos[:-n_ads].numpy()
    adsorbate_xyz_path = config.get('adsorbate_xyz', 'adsorbate.xyz')
    adsorbate_xyz_path = os.path.join(current_script_path.parent,  adsorbate_xyz_path)
    adsorbate_coordinates = read_last_snapshot(adsorbate_xyz_path)
    atoms_cluster = atoms[:-n_ads]
    # print(f"  Cluster: {len(atoms_cluster)} atoms, Adsorbate: {n_ads} atoms ({config['adsorbate_label']})")

    # ── BO setup ──
    print(f"\n[2/5] Generating {sample_size} candidate configurations...")
    t0 = time.time()

    bo = BO(
        adsorbate_coordinates=adsorbate_coordinates,
        cluster=cluster, atoms=atoms, opt=0.0,
        sample_size=sample_size, device=device,
    )

    bo.create_samples()
    t_samples = time.time() - t0
    print(f"  Done in {t_samples:.1f}s")

    # ── Inference ──
    print(f"\n[3/5] Running MACETransformer + GP inference...")
    t0 = time.time()
    bo.use_model()
    t_infer = time.time() - t0

    # De-normalize predictions from normalized space to eV
    HARTREE_TO_EV = 27.2114
    parent_dir = Path(__file__).resolve().parent.parent
    norm_path = os.path.join(parent_dir, "data_creation_pipeline", config["data_dir"], "normalization_params.json")
    with open(norm_path) as f:
        norm_params = json.load(f)
    y_mean, y_std = norm_params["mean"], norm_params["std"]
    # bo.mu contains negated normalized predictions: bo.mu = -pred_norm
    mu_norm = bo.mu.flatten()
    std_norm = bo.std.flatten()
    mu = (-mu_norm * y_std + y_mean) * HARTREE_TO_EV
    std = std_norm * y_std * HARTREE_TO_EV

    print(f"  Done in {t_infer:.1f}s ({len(bo.model_filenames)} model(s))")
    print(f"  Energy (mu): [{mu.min():.4f}, {mu.max():.4f}] eV")
    print(f"  Uncertainty:  [{std.min():.4f}, {std.max():.4f}] eV")

    # ── Scoring with all acquisition + tradeoff combos ──
    print(f"\n[4/5] BO scoring...")
    defect_label = ", ".join([f'#{d["index"]}: {d["from"]}->{d["to"]}' for d in bo.defect_info])
    # Use closest-atom as anchor for visualization; fallback to last adsorbate atom
    if config.get("specific_atom_closest", False):
        anchor_idx = config["which_atom"] % n_ads
    else:
        anchor_idx = n_ads - 1
    anchor_positions = np.array([s.pos[-(n_ads - anchor_idx)].numpy() for s in bo.Xsamples])

    print(f"\n  Defect: {defect_label}")

    acquisitions = config["bo_acquisition"]
    tradeoffs = config["bo_tradeoff"]
    all_bo_results = []

    # Output dir is configurable so control-model runs don't overwrite the
    # main model's bo_data/ results.
    out_dir = config.get("bo_output_dir", "bo_data")
    current_script_path = Path(__file__).resolve()
    out_dir = os.path.join(current_script_path.parent, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    #count nr of folders in out_dir
    nr_of_folders = len([name for name in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, name))])
    #create a subfodler in the outdir with the name round_{nr_of_folders+1}
    out_dir = os.path.join(out_dir, f"round_{nr_of_folders+1}")
    os.makedirs(out_dir, exist_ok=True)

    for method in acquisitions:
        for kappa in tradeoffs:
            bo.method = method
            bo.tradeoff = kappa
            ix, best_pos = bo.get_score()

            all_bo_results.append({
                "method": method, "tradeoff": kappa,
                "best_index": int(ix),
                "energy": float(mu[ix]), "std": float(std[ix]),
                "anchor_position": anchor_positions[ix].tolist(),
                "cluster_permutations":bo.cluster_permutations[ix]
            })
            save_points(f"{out_dir}/{cluster_type}_{bo.method}_kappa{bo.tradeoff}_defects{num_defects}", bo.Xsamples[ix], float(mu[ix]))

    # Print results table
    print(f"\n  {'Method':<8s} {'kappa':>8s} {'Idx':>6s} {'Energy(eV)':>10s} "
          f"{'Std(eV)':>10s} {'Score':>10s} {'Anchor position':<30s}")
    print(f"  {'-'*86}")
    for r in all_bo_results:
        p = r["anchor_position"]
        if r["method"] == "ucb":
            score = r["energy"] + r["tradeoff"] * r["std"]
        else:
            score = float("nan")
        print(f"  {r['method']:<8s} {r['tradeoff']:>8.1f} {r['best_index']:>6d} "
              f"{r['energy']:>10.4f} {r['std']:>10.4f} {score:>10.4f} "
              f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")

    # Use the first acquisition+tradeoff combo as the primary recommendation
    primary = all_bo_results[0]
    primary_idx = primary["best_index"]

    # Best position full adsorbate
    print(f"\n  Primary recommendation ({primary['method']}, kappa={primary['tradeoff']}):")
    print(f"  Best sample index: {primary_idx}")
    print(f"  Full adsorbate atoms:")
    best_sample = bo.Xsamples[primary_idx]
    for i in range(n_ads):
        p = best_sample.pos[-(n_ads - i)].numpy()
        print(f"    atom {i}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")

    # ── Visualization (uses primary recommendation's scoring) ──
    print(f"\n[5/5] Generating visualizations...")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    cluster_origin = list(zip(bo.defective_atoms[:-n_ads], cluster))
    title_base = f"Defect: {defect_label}"

    make_3d_html(anchor_positions, mu, std, cluster_origin, bo.defect_info,
                 fig_dir, title=f"BO: {config['baseline_system']} | {title_base} | {sample_size} samples")
    print(f"  Saved: {fig_dir}/bo_energy_surface_3d.html")

    make_3d_png(anchor_positions, mu, cluster, bo.defect_info, fig_dir, title=title_base)
    print(f"  Saved: {fig_dir}/bo_energy_surface_3d.png")

    make_analysis_plot(anchor_positions, mu, std, cluster, bo.defect_info,
                       fig_dir, title=f"BO Results | {title_base}", n_samples=sample_size)
    print(f"  Saved: {fig_dir}/bo_analysis.png")

    # Save results JSON
    import pandas as pd
    results = {
        "defect_info": bo.defect_info,
        "checkpoint_dir": config["bo_checkpoint_dir"],
        "sample_size": sample_size,
        "scoring_results": all_bo_results,
        "time_samples_s": round(t_samples, 1),
        "time_inference_s": round(t_infer, 1),
    }
    results_path = os.path.join(out_dir, "bo_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {results_path}")

    # Save predictions CSV
    df = pd.DataFrame({
        "sample_idx": np.arange(sample_size),
        "anchor_x": anchor_positions[:, 0],
        "anchor_y": anchor_positions[:, 1],
        "anchor_z": anchor_positions[:, 2],
        "energy_mu": mu, "uncertainty_std": std,
    })
    csv_path = os.path.join(fig_dir, "bo_predictions.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    print(f"\n{'='*60}")
    print(f"Done! Total: samples={t_samples:.1f}s, inference={t_infer:.1f}s")
    print(f"Results: {out_dir}/")
    print(f"  bo_results.json            - scoring results + defect info")
    print(f"  figures/bo_analysis.png     - 4-panel analysis")
    print(f"  figures/bo_energy_surface_3d.html  - interactive 3D (open in browser)")
    print(f"  figures/bo_energy_surface_3d.png   - static 3D")
    print(f"  figures/bo_predictions.csv  - all predictions")


if __name__ == "__main__":
    main()
