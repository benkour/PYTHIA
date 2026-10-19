"""Training and evaluation loops for MACETransformer and GP.

Mirrors schnet_transformer_gp_bochini/train.py, except the model operates on
PRECOMPUTED per-atom MACE features h instead of (z, pos): the frozen MACE
backbone never appears in the training loop.
"""

import torch
import gpytorch


# =========================================================================
# Joint DKL training (Transformer + GP end-to-end)
# =========================================================================

def train_epoch_joint(model, gp_model, likelihood, mll,
                      h, batch_idx, y, optimizer):
    """Joint DKL training step: MACETransformer + GP trained end-to-end.

    ExactGP requires all training data at once, so this expects the full
    training set (not mini-batches).

    Args:
        h:         [N_atoms, feature_dim] precomputed per-atom MACE features.
        batch_idx: [N_atoms] structure assignment of each atom.
        y:         [B] normalized targets.
    """
    model.train()
    gp_model.train()
    likelihood.train()

    optimizer.zero_grad()
    features = model.forward_features(h, batch_idx)
    gp_model.set_train_data(inputs=features, targets=y, strict=False)
    output = gp_model(features)
    loss = -mll(output, y)
    loss.backward()
    optimizer.step()

    return loss.item()


# =========================================================================
# Gaussian Process evaluation
# =========================================================================

@torch.no_grad()
def evaluate_gp(model, gp_model, likelihood, h, batch_idx, y):
    """Evaluate using GP predictions with uncertainty (full-batch on
    precomputed features).

    Returns (mse, mae, means_tensor, stds_tensor, targets_tensor).
    """
    model.eval()
    gp_model.eval()
    likelihood.eval()

    with gpytorch.settings.fast_pred_var():
        features = model.forward_features(h, batch_idx)
        pred_dist = likelihood(gp_model(features))
        means = pred_dist.mean
        stds = pred_dist.stddev

    mse = ((means - y) ** 2).mean().item()
    mae = (means - y).abs().mean().item()
    return mse, mae, means, stds, y
