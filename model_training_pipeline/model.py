import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import gpytorch
from gpytorch.models import ExactGP
from torch_geometric.utils import to_dense_batch
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, LinearKernel, SpectralMixtureKernel
from gpytorch.means import ConstantMean, LinearMean, ZeroMean
from gpytorch.distributions import MultivariateNormal

from ase import Atoms
from mace import data as mace_data
from mace.calculators import mace_mp
from mace.tools import AtomicNumberTable, torch_geometric as mace_tg


class MaceFeatureExtractor:

    def __init__(self, model_size="large", device="cuda", chunk_size=8):
        self.device = torch.device(device)
        self.chunk_size = chunk_size

        # Load raw frozen MACE exactly as validated in smoke_test.py.
        # The explicit .to(device) is REQUIRED, otherwise e3nn buffers stay
        # on CPU even when device= is passed to mace_mp.
        model = mace_mp(model=model_size, device=str(self.device),
                        return_raw_model=True, default_dtype="float64")
        model = model.to(self.device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model

        self._z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
        self._heads = model.heads if hasattr(model, "heads") else ["Default"]
        self._r_max = float(model.r_max)
        self._dtype = next(model.parameters()).dtype

        # Precompute per-layer (start, length) of the l=0 block inside the
        # concatenated node_feats (same logic as smoke_test.invariant_slices).
        num_interactions = int(model.num_interactions)
        irreps = model.products[0].linear.irreps_out
        n_inv = irreps.dim // (irreps.lmax + 1) ** 2
        layer_dims = [irreps.dim] * num_interactions
        layer_dims[-1] = n_inv  # last layer is invariant-only
        slices, start = [], 0
        for d in layer_dims:
            slices.append((start, n_inv))
            start += d
        self.invariant_slices = slices
        self.feature_dim = n_inv * num_interactions  # 512 for "large"

    def _atoms_list_to_batch(self, atoms_list):
        """Convert list of ASE Atoms into a MACE batch (smoke_test logic)."""
        dataset = [
            mace_data.AtomicData.from_config(
                mace_data.config_from_atoms(a, head_name="Default"),
                z_table=self._z_table, cutoff=self._r_max, heads=self._heads)
            for a in atoms_list
        ]
        loader = mace_tg.dataloader.DataLoader(
            dataset=dataset, batch_size=len(dataset), shuffle=False,
            drop_last=False)
        batch = next(iter(loader)).to(self.device)
        for key in batch.keys() if callable(getattr(batch, "keys", None)) else batch.keys:
            v = batch[key]
            if torch.is_tensor(v) and torch.is_floating_point(v):
                batch[key] = v.to(self._dtype)
        return batch

    @torch.no_grad()
    def extract(self, z, pos, batch):
        """Per-atom invariant MACE features.

        Args:
            z:     [N] long tensor of atomic numbers.
            pos:   [N, 3] float tensor of positions.
            batch: [N] long tensor assigning each atom to a structure.

        Returns:
            float32 tensor [N, feature_dim] on self.device, atoms in the
            same order as the input.
        """
        z_np = z.detach().cpu().numpy().astype(int)
        pos_np = pos.detach().cpu().numpy().astype(float)
        batch_np = batch.detach().cpu().numpy()

        n_structs = int(batch_np.max()) + 1 if len(batch_np) > 0 else 0
        atoms_list = [
            Atoms(numbers=z_np[batch_np == i], positions=pos_np[batch_np == i])
            for i in range(n_structs)
        ]

        chunks = []
        for start in range(0, n_structs, self.chunk_size):
            chunk_atoms = atoms_list[start:start + self.chunk_size]
            mace_batch = self._atoms_list_to_batch(chunk_atoms)
            out = self.model(mace_batch.to_dict(), training=False,
                             compute_force=False)
            nf = out["node_feats"]
            parts = [nf[:, s:s + n] for s, n in self.invariant_slices]
            chunks.append(torch.cat(parts, dim=-1))



        return torch.cat(chunks, dim=0).float()



class MACETransformer(nn.Module):
    """Transformer encoder over precomputed per-atom MACE features (atom-wise).

    Same math as the atom-wise SchNetTransformer of the reference, but the
    forward pass starts from precomputed features h instead of running a
    backbone on (z, pos).
    """

    def __init__(self, feature_dim=512, nhead=8, dim_feedforward=512,
                 num_transformer_layers=2, dropout=0.1, bottleneck_dim=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.bottleneck_dim = bottleneck_dim

        encoder_layer = TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = TransformerEncoder(encoder_layer,
                                              num_layers=num_transformer_layers)
        # Bottleneck: compress features for GP (optional)
        if bottleneck_dim is not None:
            self.bottleneck = nn.Sequential(
                nn.Linear(feature_dim, bottleneck_dim),
                nn.ReLU(),
            )
            head_in = bottleneck_dim
        else:
            self.bottleneck = None
            head_in = feature_dim
        # MLP head (input from bottleneck if present, otherwise from transformer)
        self.head = nn.Sequential(
            nn.Linear(head_in, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
        )

    def forward_features(self, h, batch):
        """h: [N_atoms, feature_dim] precomputed MACE features -> [B, D]."""
        # Convert to dense batch for Transformer: [B, MaxAtoms, D]
        # print("before to_dense_batch: h.shape, batch.shape: ", h.shape, batch.shape)
        h_dense, mask = to_dense_batch(h, batch)  # mask: [B, MaxAtoms] True=real

        # print("after to_dense_batch: h_dense.shape, mask.shape: ", h_dense.shape, mask.shape)
        # Transformer with padding mask (True = ignore)
        h_dense = self.transformer(h_dense, src_key_padding_mask=~mask)

        # Masked mean pooling
        mask_f = mask.unsqueeze(-1).float()  # [B, MaxAtoms, 1]
        out = (h_dense * mask_f).sum(dim=1) / mask_f.sum(dim=1)  # [B, D]

        if self.bottleneck is not None:
            out = self.bottleneck(out)
        return out

    def forward(self, h, batch):
        """Full forward pass through MLP head. Returns [B] energy predictions."""
        return self.head(self.forward_features(h, batch)).squeeze(-1)


class ModularGPModel(ExactGP):
    """
    Modular GP model that can use different mean and kernel functions
    """
    def __init__(self, train_x, train_y, likelihood, mean_type="linear", kernel_type="matern_linear", input_dim=None):
        super(ModularGPModel, self).__init__(train_x, train_y, likelihood)

        # Determine input dimension from training data if not provided
        if input_dim is None:
            if train_x is not None and len(train_x) > 0:
                input_dim = train_x.shape[-1]
            else:
                input_dim = 7  # Default fallback

        # Set up mean module based on type
        if mean_type == "constant":
            self.mean_module = ConstantMean()
        elif mean_type == "linear":
            self.mean_module = LinearMean(input_size=input_dim)
        elif mean_type == "zero":
            self.mean_module = ZeroMean()
        else:
            raise ValueError(f"Unknown mean type: {mean_type}")

        # Set up kernel based on type
        if kernel_type == "rbf":
            base_kernel = RBFKernel(ard_num_dims=input_dim)
            self.covar_module = ScaleKernel(base_kernel)
        elif kernel_type == "matern":
            base_kernel = MaternKernel(nu=2.5, ard_num_dims=input_dim)
            self.covar_module = ScaleKernel(base_kernel)
        elif kernel_type == "linear":
            base_kernel = LinearKernel(ard_num_dims=input_dim)
            self.covar_module = ScaleKernel(base_kernel)
        elif kernel_type == "matern_linear":
            matern_kernel = MaternKernel(
                nu=2.5,
                ard_num_dims=input_dim,
                lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 6.0)
            )
            linear_kernel = LinearKernel(ard_num_dims=input_dim)
            self.covar_module = ScaleKernel(
                matern_kernel + linear_kernel,
                outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.15)
            )
        elif kernel_type == "spectral_mixture":
            base_kernel = SpectralMixtureKernel(num_mixtures=2, ard_num_dims=input_dim)
            self.covar_module = ScaleKernel(base_kernel)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
