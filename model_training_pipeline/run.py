import gc
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import gpytorch
from pathlib import Path


from model_training_pipeline.model import MaceFeatureExtractor, MACETransformer, ModularGPModel
from model_training_pipeline.data import load_chini_dataset, collate_fn
from model_training_pipeline.train import train_epoch_joint, evaluate_gp
from model_training_pipeline.visualize import (plot_training_curves, plot_pred_vs_true,
                       plot_per_sample_error, plot_pred_vs_true_with_uncertainty,
                       plot_uncertainty_analysis)

HARTREE_TO_EV = 27.2114
current_script_path = Path(__file__).resolve()

# Single pretrained backbone (no QM9 target loop), but the checkpoint folder
# keeps the target_7_energy_U0 name so BO scripts and cross-model comparisons
# find the same layout as the previous checkpoints_* dirs.


def load_config(path):
    """Read a JSON configuration file and return its contents as a dict."""
    with open(path) as f:
        return json.load(f)


def build_scheduler(optimizer, cfg):
    """Build a learning rate scheduler from config.  Returns None if disabled."""
    name = cfg.get("scheduler", None)
    if name is None or name == "none":
        return None
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.get("scheduler_factor", 0.5),
            patience=cfg.get("scheduler_patience", 10),
            min_lr=cfg.get("scheduler_min_lr", 1e-6),
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.get("training_epochs", 200),
            eta_min=cfg.get("scheduler_min_lr", 1e-6),
        )
    raise ValueError(f"Unknown scheduler: {name!r}")


# =============================================================================
# Feature precomputation (frozen MACE, run once, cached on disk)
# =============================================================================

def get_feature_dict(all_data, cfg, root, device):
    """Return dict filename -> [n_atoms, feature_dim] float32 CPU tensor.

    Features are cached per-structure on disk keyed by the MACE model size,
    so reruns (and different train/test splits) skip MACE entirely.
    """
    model_size = cfg.get("mace_model_size", "large")
    cache_dir = root / cfg.get("feature_cache_dir", "mace_transformer/feature_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"features_{model_size}.pt"

    feat_dict = {}
    if cache_path.exists():
        feat_dict = torch.load(cache_path, map_location="cpu", weights_only=True)
        print(f"Feature cache: loaded {len(feat_dict)} structures from {cache_path}")

    missing = [d for d in all_data if d.filename not in feat_dict]
    if missing:
        print(f"Extracting MACE features for {len(missing)} structures "
              f"(model={model_size})...")
        extractor = MaceFeatureExtractor(model_size=model_size, device=device)
        # assert extractor.feature_dim == cfg.get("feature_dim", 512), (
        #     f"MACE feature_dim {extractor.feature_dim} != config "
        #     f"{cfg.get('feature_dim', 512)}")

        # extract() chunks structures internally (chunk_size structures per
        # MACE forward pass), so one call over the collated set is safe.
        z, pos, batch_idx, _ = collate_fn(missing)
        feats = extractor.extract(z, pos, batch_idx).cpu()  # [N_atoms, D] float32
        batch_np = batch_idx.numpy()
        for i, d in enumerate(missing):
            feat_dict[d.filename] = feats[batch_np == i].clone()

        torch.save(feat_dict, cache_path)
        print(f"Feature cache: saved {len(feat_dict)} structures to {cache_path}")

        # Free the MACE backbone — it must never enter the training loop.
        del extractor.model
        del extractor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return feat_dict


def assemble_features(split_data, feat_dict, device):
    """Recompose per-structure cached features into full-batch tensors.

    Returns (h [N_atoms, D], batch_idx [N_atoms], y [B]) on device.
    """
    h_list, batch_list, y_list = [], [], []
    for i, d in enumerate(split_data):
        feats = feat_dict[d.filename]
        h_list.append(feats)
        batch_list.append(torch.full((feats.size(0),), i, dtype=torch.long))
        y_list.append(float(d.y))
    h = torch.cat(h_list).to(device)
    batch_idx = torch.cat(batch_list).to(device)
    y = torch.tensor(y_list, dtype=torch.float, device=device)
    return h, batch_idx, y


# =============================================================================
# Joint DKL training + evaluation for the single pretrained backbone
# =============================================================================

def run_single_target(train_data, test_data, feat_dict, target_save_dir,
                      y_mean, y_std, epochs, lr, device, cfg=None):
    """Train and evaluate the MACE-Transformer + GP model.

    Returns test_mae in eV.
    """
    target_save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 60}")
    print(f"# backbone: frozen MACE-MP ({(cfg or {}).get('mace_model_size', 'large')})")
    print(f"{'#' * 60}")

    # -- build model -----------------------------------------------------------
    bottleneck_dim = (cfg or {}).get("bottleneck_dim", None)
    # feature_dim = (cfg or {}).get("feature_dim", 512)
    feature_dim = MaceFeatureExtractor(model_size=(cfg or {}).get("mace_model_size", "large")).feature_dim
    model = MACETransformer(
        feature_dim=feature_dim,
        nhead=(cfg or {}).get("nhead", 8),
        dim_feedforward=(cfg or {}).get("dim_feedforward", 512),
        num_transformer_layers=(cfg or {}).get("num_transformer_layers", 2),
        dropout=(cfg or {}).get("dropout", 0.1),
        bottleneck_dim=bottleneck_dim,
    ).to(device)
    if bottleneck_dim:
        print(f"Bottleneck: {feature_dim} -> {bottleneck_dim}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # -- GP config -------------------------------------------------------------
    gp_mean_type = (cfg or {}).get("gp_mean_type", "linear")
    gp_kernel_type = (cfg or {}).get("gp_kernel_type", "matern_linear")
    hidden_channels = bottleneck_dim or feature_dim

    # Initialize GP with dummy data (replaced by set_train_data each step)
    # Constrain noise to [1e-6, 0.1] so the GP learns signal through the kernel
    # rather than attributing everything to noise (default init ≈ 0.693 is too high)
    noise_constraint = gpytorch.constraints.Interval(1e-6, 0.1)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=noise_constraint).to(device)
    gp_model = ModularGPModel(
        train_x=torch.zeros(2, hidden_channels, device=device),
        train_y=torch.zeros(2, device=device),
        likelihood=likelihood,
        mean_type=gp_mean_type,
        kernel_type=gp_kernel_type,
        input_dim=hidden_channels,
    ).to(device)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

    # -- joint optimizer: Transformer (+ bottleneck) + GP + likelihood ---------
    transformer_params = list(model.transformer.parameters())
    if model.bottleneck is not None:
        transformer_params += list(model.bottleneck.parameters())
    gp_params = list(gp_model.parameters()) + list(likelihood.parameters())
    all_params = transformer_params + gp_params
    print(f"Trainable parameters (Transformer + GP): "
          f"{sum(p.numel() for p in all_params):,}")

    optimizer = torch.optim.Adam(all_params, lr=lr)
    scheduler = build_scheduler(optimizer, cfg or {})

    # -- prepare full-batch training tensors (ExactGP requirement) -------------
    h_train, batch_train, y_train = assemble_features(train_data, feat_dict, device)
    h_test, batch_test, y_test = assemble_features(test_data, feat_dict, device)
    print(f"Training samples: {y_train.shape[0]} (full-batch, DKL)")

    # =========================================================================
    # Joint Training: Transformer + GP end-to-end
    # =========================================================================
    print(f"\n[Joint Training] Transformer + GP ({epochs} epochs)")
    print("=" * 60)

    best_loss = float("inf")
    history = {"train": [], "lr": []}

    for epoch in range(1, epochs + 1):
        loss = train_epoch_joint(
            model, gp_model, likelihood, mll,
            h_train, batch_train, y_train, optimizer)
        history["train"].append(loss)
        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)
        if scheduler is not None:
            scheduler.step(loss)
        if loss < best_loss:
            best_loss = loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "gp_state_dict": gp_model.state_dict(),
                "likelihood_state_dict": likelihood.state_dict(),
            }, target_save_dir / "best_joint.pt")

            tag = " *"
        else:
            tag = ""
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  neg_mll={loss:.6f}  "
                  f"noise={likelihood.noise.item():.6f}  lr={current_lr:.2e}{tag}")

    # Reload best checkpoint
    ckpt = torch.load(target_save_dir / "best_joint.pt",
                      map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    gp_model.load_state_dict(ckpt["gp_state_dict"])
    likelihood.load_state_dict(ckpt["likelihood_state_dict"])

    # Re-extract features with best model for GP posterior inference
    with torch.no_grad():
        model.eval()
        train_features = model.forward_features(h_train, batch_train)
    gp_model.set_train_data(inputs=train_features, targets=y_train, strict=False)

    print(f"Best neg MLL: {best_loss:.6f}")

    # Save models
    # IMPORTANT: gpytorch's state_dict() does NOT include the ExactGP training
    # data (train_inputs/train_targets). Without it, reloading the GP conditions
    # on dummy data and the predictive mean collapses to the prior mean
    # (constant for a ZeroMean). Persist the training data explicitly.
    torch.save({
        "gp_state_dict": gp_model.state_dict(),
        "likelihood_state_dict": likelihood.state_dict(),
        "train_x": train_features.detach().cpu(),
        "train_y": y_train.detach().cpu(),
        "gp_mean_type": gp_mean_type,
        "gp_kernel_type": gp_kernel_type,
        "gp_input_dim": hidden_channels,
    }, target_save_dir / "gp_model.pt")
    # Transformer only — the frozen MACE backbone is not an nn.Module, so its
    # weights can never leak into this state_dict. Assert anyway.
    model_sd = model.state_dict()
    assert not any("mace" in k.lower() for k in model_sd), \
        "MACE weights must not be in the transformer state_dict"
    torch.save(model_sd, target_save_dir / "best_model.pt")
    print("Models saved.")

    # =========================================================================
    # Evaluation
    # =========================================================================
    print("\n[Evaluation]")
    print("=" * 60)

    gp_mse, gp_mae, gp_means, gp_stds, gp_targets = evaluate_gp(
        model, gp_model, likelihood, h_test, batch_test, y_test)
    gp_mse_ev = gp_mse * y_std ** 2 * HARTREE_TO_EV ** 2
    gp_mae_ev = gp_mae * y_std * HARTREE_TO_EV
    print(f"GP — Test MSE: {gp_mse_ev:.6f} eV²  MAE: {gp_mae_ev:.6f} eV")

    # Per-sample GP predictions with uncertainty
    names, preds_ev, trues_ev, errors_ev, stds_ev = [], [], [], [], []
    print(f"\n{'Sample':<45s} {'Pred(eV)':>10s} {'True(eV)':>10s} "
          f"{'Err(eV)':>10s} {'σ(eV)':>10s}")
    print("-" * 98)

    model.eval()
    gp_model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for data in test_data:
            h = feat_dict[data.filename].to(device)
            batch_t = torch.zeros(h.size(0), dtype=torch.long, device=device)
            features = model.forward_features(h, batch_t)
            pred_dist = likelihood(gp_model(features))
            pred_norm = pred_dist.mean.item()
            std_norm = pred_dist.stddev.item()

            pred_ev = (pred_norm * y_std + y_mean) * HARTREE_TO_EV
            std_ev = std_norm * y_std * HARTREE_TO_EV
            true_ev = (float(data.y) * y_std + y_mean) * HARTREE_TO_EV
            err = pred_ev - true_ev

            name = getattr(data, "filename", "?").replace("_data_", " q=").replace(".pt", "")
            names.append(name)
            preds_ev.append(pred_ev)
            trues_ev.append(true_ev)
            errors_ev.append(err)
            stds_ev.append(std_ev)
            print(f"  {name:<43s} {pred_ev:>10.4f} {true_ev:>10.4f} "
                  f"{err:>+10.4f} {std_ev:>10.4f}")

    preds_ev = np.array(preds_ev)
    trues_ev = np.array(trues_ev)
    errors_ev = np.array(errors_ev)
    stds_ev = np.array(stds_ev)

    # =========================================================================
    # Visualization
    # =========================================================================
    fig_dir = target_save_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    plot_training_curves(history, fig_dir)
    plot_pred_vs_true(trues_ev, preds_ev, gp_mae_ev, fig_dir)
    plot_per_sample_error(names, errors_ev, gp_mae_ev, fig_dir)
    plot_pred_vs_true_with_uncertainty(trues_ev, preds_ev, stds_ev, gp_mae_ev, fig_dir)
    plot_uncertainty_analysis(stds_ev, errors_ev, fig_dir)

    return gp_mae_ev


def main(config_path=None, epochs_override=None, save_dir_override=None):
    root = Path(__file__).resolve().parent.parent

    # Resolve config path
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "configs" / "config_train.json"
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = root / config_path

    cfg = load_config(config_path)
    print(f"Config: {config_path}")

    data_dir = root / "data_creation_pipeline" / cfg["data_dir"]
    if save_dir_override is not None:
        save_dir = Path(save_dir_override)
        if not save_dir.is_absolute():
            save_dir = root / "checkpoints" / save_dir
    else:
        save_dir = root /  "checkpoints" / cfg["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)

    split_ratio = cfg["split_ratio"]
    epochs = epochs_override if epochs_override is not None \
        else cfg.get("training_epochs", cfg.get("epochs", 400))
    lr = cfg["lr"]
    seed = cfg["seed"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -- normalization params --------------------------------------------------
    with open(data_dir / "normalization_params.json") as f:
        norm = json.load(f)
    y_mean, y_std = norm["mean"], norm["std"]
    print(f"Normalization: mean={y_mean:.6f} Ha ({y_mean*HARTREE_TO_EV:.4f} eV), "
          f"std={y_std:.6f} Ha ({y_std*HARTREE_TO_EV:.4f} eV)")

    # -- load dataset ----------------------------------------------------------
    all_data = load_chini_dataset(str(data_dir))
    n_total = len(all_data)
    print(f"Total samples: {n_total}")

    # -- precompute frozen MACE features (cached on disk) ----------------------
    feat_dict = get_feature_dict(all_data, cfg, root, device)

    # Reproducible MOLECULE-TYPE-level split
    # (all variants of a molecule type go to the same split)
    sample_indices = OrderedDict()  # group_key -> [indices]
    for i, d in enumerate(all_data):
        key = d.filename.split("_data_")[0]  # e.g. "Pt6__0"
        sample_indices.setdefault(key, []).append(i)

    mol_types = OrderedDict()  # mol_type -> [group_keys]
    for gkey in sample_indices:
        mol_type = gkey.rsplit("__", 1)[0]  # "Pt6__0" -> "Pt6"
        mol_types.setdefault(mol_type, []).append(gkey)

    type_keys = list(mol_types.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(type_keys)
    n_types = len(type_keys)

    def collect_samples(types):
        return [all_data[i]
                for t in types
                for g in mol_types[t]
                for i in sample_indices[g]]

    # train / test only
    n_train_t = int(n_types * split_ratio[0])
    train_types = type_keys[:n_train_t]
    test_types = type_keys[n_train_t:]

    train_data = collect_samples(train_types)
    test_data = collect_samples(test_types)
    print(f"Molecule types: {n_types} total -> {len(train_types)} train / {len(test_types)} test")
    print(f"  Train: {train_types}")
    print(f"  Test:  {test_types}")
    print(f"Samples: {len(train_data)} train / {len(test_data)} test")

    # -- run the single target -------------------------------------------------
    torch.manual_seed(seed)
    target_save_dir = save_dir
    test_mae_ev = run_single_target(
        train_data, test_data, feat_dict, target_save_dir,
        y_mean, y_std, epochs, lr, device, cfg=cfg,
    )
    print(f"\nTest MAE: {test_mae_ev:.4f} eV")


def parse_args(argv):
    """Parse [--epochs N] [--save-dir PATH] [config.json]."""
    epochs = None
    save_dir = None
    cfg_path = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--epochs":
            epochs = int(argv[i + 1])
            i += 2
        elif arg == "--save-dir":
            save_dir = argv[i + 1]
            i += 2
        else:
            cfg_path = arg
            i += 1
    return cfg_path, epochs, save_dir


if __name__ == "__main__":
    cfg_arg, epochs_arg, save_dir_arg = parse_args(sys.argv[1:])
    main(config_path=cfg_arg, epochs_override=epochs_arg,
         save_dir_override=save_dir_arg)
