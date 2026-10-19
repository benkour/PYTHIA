#%%
import torch
from ase.build import molecule
from mace import data as mace_data
from mace.calculators import mace_mp
from mace.tools import AtomicNumberTable, torch_geometric

# Try to import mendeleev for atomic properties
try:
	from mendeleev import element
	MENDELEEV_AVAILABLE = True
except ImportError:
	MENDELEEV_AVAILABLE = False
	print("Warning: mendeleev not available. Using default VdW radii and valences.")


def atoms_to_mace_batch(atoms, model, device="cpu"):
	"""Build a single-structure MACE batch from an ASE Atoms object."""
	config = mace_data.config_from_atoms(atoms, head_name="Default")
	z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
	heads = model.heads if hasattr(model, "heads") else ["Default"]

	data_loader = torch_geometric.dataloader.DataLoader(
		dataset=[
			mace_data.AtomicData.from_config(
				config,
				z_table=z_table,
				cutoff=float(model.r_max),
				heads=heads,
			)
		],
		batch_size=1,
		shuffle=False,
		drop_last=False,
	)
	batch = next(iter(data_loader)).to(device)

	model_dtype = next(model.parameters()).dtype
	batch_keys = batch.keys() if callable(getattr(batch, "keys", None)) else batch.keys
	for key in batch_keys:
		value = batch[key]
		if torch.is_tensor(value) and torch.is_floating_point(value):
			batch[key] = value.to(dtype=model_dtype)
	return batch


def descriptor_dict_to_mace_batch(descriptors, model, device="cpu"):
	"""Build raw-model input directly from descriptor tensors (no ASE conversion)."""
	positions = torch.as_tensor(
		descriptors["positions"], dtype=torch.get_default_dtype()
	)
	atomic_numbers = torch.as_tensor(descriptors["atomic_numbers"], dtype=torch.long)
	num_atoms = int(positions.shape[0])

	if "edge_index" in descriptors:
		edge_index = torch.as_tensor(descriptors["edge_index"], dtype=torch.long)
	else:
		bond_pairs = descriptors.get("bond_pairs", [])
		if len(bond_pairs) == 0:
			raise ValueError("Provide 'edge_index' or 'bond_pairs' in descriptors.")
		senders, receivers = [], []
		for i, j in bond_pairs:
			senders.extend([int(i), int(j)])
			receivers.extend([int(j), int(i)])
		edge_index = torch.tensor([senders, receivers], dtype=torch.long)

	shifts = torch.as_tensor(
		descriptors.get("shifts", torch.zeros((edge_index.shape[1], 3))),
		dtype=torch.get_default_dtype(),
	)
	unit_shifts = torch.as_tensor(
		descriptors.get("unit_shifts", torch.zeros((edge_index.shape[1], 3))),
		dtype=torch.get_default_dtype(),
	)

	cell_default = torch.zeros((1, 3, 3), dtype=torch.get_default_dtype())
	cell = torch.as_tensor(descriptors.get("cell", cell_default), dtype=torch.get_default_dtype())
	if cell.dim() == 2:
		cell = cell.unsqueeze(0)

	batch = torch.as_tensor(
		descriptors.get("batch", torch.zeros(num_atoms, dtype=torch.long)),
		dtype=torch.long,
	)
	ptr = torch.as_tensor(
		descriptors.get("ptr", torch.tensor([0, num_atoms], dtype=torch.long)),
		dtype=torch.long,
	)

	z_to_index = {int(z): idx for idx, z in enumerate(model.atomic_numbers.tolist())}
	unknown_z = sorted(set(int(z.item()) for z in atomic_numbers) - set(z_to_index.keys()))
	if unknown_z:
		raise ValueError(f"Atomic numbers {unknown_z} are not in this model.")

	node_indices = torch.tensor(
		[z_to_index[int(z.item())] for z in atomic_numbers], dtype=torch.long
	)
	node_attrs = torch.nn.functional.one_hot(
		node_indices, num_classes=len(z_to_index)
	).to(torch.get_default_dtype())

	default_head = 0
	if hasattr(model, "heads") and len(model.heads) > 0:
		default_head = model.heads.index("Default") if "Default" in model.heads else 0
	head = torch.as_tensor(
		descriptors.get("head", torch.tensor([default_head], dtype=torch.long)),
		dtype=torch.long,
	)
	if head.dim() == 0:
		head = head.unsqueeze(0)

	data = {
		"positions": positions,
		"edge_index": edge_index,
		"shifts": shifts,
		"unit_shifts": unit_shifts,
		"node_attrs": node_attrs,
		"batch": batch,
		"ptr": ptr,
		"cell": cell,
		"head": head,
	}

	model_dtype = next(model.parameters()).dtype
	for key, value in data.items():
		if torch.is_tensor(value):
			value = value.to(device)
			if torch.is_floating_point(value):
				value = value.to(dtype=model_dtype)
			data[key] = value
	return data


def get_penultimate_layer_features(model, node_feats):
	"""Extract penultimate interaction-layer features from concatenated node feats."""
	num_interactions = int(model.num_interactions)
	if num_interactions < 2:
		raise ValueError("Model has fewer than 2 interaction layers.")

	irreps_out = model.products[0].linear.irreps_out
	l_max = irreps_out.lmax
	num_invariant_features = irreps_out.dim // (l_max + 1) ** 2

	per_layer_features = [irreps_out.dim for _ in range(num_interactions)]
	per_layer_features[-1] = num_invariant_features

	start = int(sum(per_layer_features[: num_interactions - 2]))
	end = int(sum(per_layer_features[: num_interactions - 1]))
	return node_feats[:, start:end]


def generate_bond_pairs_from_coordinates(positions, atomic_numbers):
	"""
	Generate bond pairs from coordinates using van der Waals radii and valence.
	Atoms are bonded if distance <= sum of VdW radii * scale_factor (1.3).
	Also respects maximum bonds per element (valence).
	Uses mendeleev library if available, otherwise falls back to defaults.
	"""
	# Default Van der Waals radii in Angstroms (Bondi)
	default_vdw_radii = {
		1: 1.20,   # H
		6: 1.70,   # C
		7: 1.55,   # N
		8: 1.52,   # O
		16: 1.80,  # S
	}

	# Default maximum bonds per element (valence)
	# get maximum bonds for every element

	default_max_bonds = {
		1: 1,      # H
		6: 4,      # C
		7: 3,      # N
		8: 2,      # O
		16: 2,     # S
	}

	# Load atomic properties from mendeleev if available and augment for metals
	vdw_radii = {}
	max_bonds = {}
	is_metal = {}

	if MENDELEEV_AVAILABLE:
		for z in set(atomic_numbers if isinstance(atomic_numbers, (list, tuple)) else atomic_numbers.tolist()):
			try:
				el = element(z)
				# Prefer metallic radius for metals where available (mendeleev reports pm)
				metal_r = getattr(el, "metallic_radius", None)
				vdw_r = getattr(el, "vdw_radius", None)
				if metal_r:
					radius = metal_r / 100.0
				elif vdw_r:
					radius = vdw_r / 100.0
				else:
					radius = default_vdw_radii.get(z, 1.70)
				vdw_radii[z] = radius

				# Determine if element is metallic; fall back to category string check
				metal_flag = False
				try:
					metal_flag = bool(getattr(el, "is_metal", False)) or (getattr(el, "category", "") and "metal" in el.category.lower())
				except Exception:
					metal_flag = False
				is_metal[z] = metal_flag

				# Use valence electrons for non-metals; give metals higher coordination
				if metal_flag:
					# Metals can have high coordination; allow up to 12 but clamp to reasonable default
					max_bonds[z] = max(default_max_bonds.get(z, 4), 8)
				else:
					valence = None
					try:
						# mendeleev stores electrons per shell; use last shell electron count
						electrons = el.electrons
						if electrons:
							valence = electrons[-1]
					except Exception:
						valence = None
					if valence is None:
						max_bonds[z] = default_max_bonds.get(z, 4)
					else:
						max_bonds[z] = min(valence, default_max_bonds.get(z, 4))
			except Exception:
				vdw_radii[z] = default_vdw_radii.get(z, 1.70)
				max_bonds[z] = default_max_bonds.get(z, 4)
				is_metal[z] = False
	else:
		# No mendeleev: fall back to defaults and assume non-metal except common metals
		vdw_radii = default_vdw_radii.copy()
		max_bonds = default_max_bonds.copy()
		# mark common metals conservatively (Fe, Cu, Zn, etc.) if present in atomic_numbers
		common_metal_z = {26, 29, 30, 13, 14}
		for z in set(atomic_numbers if isinstance(atomic_numbers, (list, tuple)) else atomic_numbers.tolist()):
			is_metal[z] = z in common_metal_z
			if is_metal[z] and z not in max_bonds:
				max_bonds[z] = max(default_max_bonds.get(z, 4), 8)
	positions = torch.as_tensor(positions, dtype=torch.float32)
	atomic_numbers = torch.as_tensor(atomic_numbers, dtype=torch.long)

	distances = torch.cdist(positions, positions)

	# First pass: collect all possible bonds
	possible_bonds = []
	for i in range(len(atomic_numbers)):
		for j in range(i + 1, len(atomic_numbers)):
			z_i = int(atomic_numbers[i].item())
			z_j = int(atomic_numbers[j].item())

			r_i = vdw_radii.get(z_i, 1.70)
			r_j = vdw_radii.get(z_j, 1.70)
			threshold = (r_i + r_j) * 1.3

			if distances[i, j] < threshold:
				possible_bonds.append((i, j, distances[i, j].item()))

	# Sort by distance (prefer closer atoms)
	possible_bonds.sort(key=lambda x: x[2])

	# Second pass: greedily select bonds respecting valence constraints
	bond_pairs = []
	bond_count = {i: 0 for i in range(len(atomic_numbers))}

	for i, j, dist in possible_bonds:
		z_i = int(atomic_numbers[i].item())
		z_j = int(atomic_numbers[j].item())
		max_i = max_bonds.get(z_i, 4)
		max_j = max_bonds.get(z_j, 4)

		if bond_count[i] < max_i and bond_count[j] < max_j:
			bond_pairs.append((i, j))
			bond_count[i] += 1
			bond_count[j] += 1

	return bond_pairs

name_a = "CH3OH"
mol_a = molecule(name_a)

# Load pretrained MACE-MP model (downloads weights on first run).
calc = mace_mp(model="large", device="cpu")  # use device="cuda" if available


mol_a.calc = calc

e_a = mol_a.get_potential_energy()


print(f"Molecule A: {name_a}")

print(f"E(A): {e_a:.6f} eV")


full_model = mace_mp(
	model="large", device="cpu", return_raw_model=True
)  # use device="cuda" if available
full_model.eval()

batch = atoms_to_mace_batch(mol_a, full_model, device="cpu")

raw_output = full_model(
	batch.to_dict(),
	training=False,
	compute_force=True,
	compute_virials=False,
	compute_stress=False,
)

raw_energy = raw_output["energy"].detach().cpu().reshape(-1)[0].item()
raw_forces = raw_output["forces"].detach().cpu().numpy()
pre_energy_tensor = raw_output["node_energy"]
penultimate_feats = get_penultimate_layer_features(full_model, raw_output["node_feats"])
penultimate_feats_row_summed = penultimate_feats.sum(dim=0).detach().cpu().numpy()

#print cooedinates of mol_a
coords = mol_a.get_positions()
print(f"Coordinates of {name_a}:")
for i, (symbol, pos) in enumerate(zip(mol_a.get_chemical_symbols(), coords)):
    print(f"Atom {i}: {symbol} at {pos}")

#%%
reconstructed_energy = pre_energy_tensor.sum().detach().cpu().item()
print(f"Raw model energy for {name_a}: {raw_energy:.6f} eV")
print(f"Reconstructed energy from node_energy sum: {reconstructed_energy:.6f} eV")
print(f"Energy difference: {abs(raw_energy - reconstructed_energy):.10f} eV")
print(f"Raw model forces shape for {name_a}: {raw_forces.shape}")
print(f"Pre-energy tensor (node energies) shape for {name_a}: {tuple(pre_energy_tensor.shape)}")
print(f"Penultimate layer feature shape for {name_a}: {tuple(penultimate_feats.shape)}")



#%%

# Compare: auto-generated bond pairs (from coordinates) vs hard-coded bond pairs

comparison_input = {
	"atomic_numbers": [6, 8, 1, 1, 1, 1],
	"positions": [
		[0.000, 0.000, 0.000],
		[1.430, 0.000, 0.000],
		[-0.540, 0.930, 0.000],
		[-0.540, -0.470, 0.820],
		[-0.540, -0.470, -0.820],
		[1.950, 0.750, 0.000],
	],
}

hardcoded_bond_pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5)]
autogenerated_bond_pairs = generate_bond_pairs_from_coordinates(
	comparison_input["positions"],
	comparison_input["atomic_numbers"]
)

print("\n" + "="*80)
print("BOND PAIR COMPARISON")
print("="*80)
print(f"Hard-coded bond pairs: {hardcoded_bond_pairs}")
print(f"Auto-generated bond pairs: {autogenerated_bond_pairs}")
print(f"Hard-coded count: {len(hardcoded_bond_pairs)}")
print(f"Auto-generated count: {len(autogenerated_bond_pairs)}")


# Run MACE with hard-coded bond pairs
comparison_batch_hardcoded = descriptor_dict_to_mace_batch(
	{**comparison_input, "bond_pairs": hardcoded_bond_pairs},
	full_model, device="cpu"
)
comparison_output_hardcoded = full_model(
	comparison_batch_hardcoded,
	training=False,
	compute_force=True,
	compute_virials=False,
	compute_stress=False,
)
energy_hardcoded = comparison_output_hardcoded["energy"].detach().cpu().reshape(-1)[0].item()
node_energy_hardcoded = comparison_output_hardcoded["node_energy"]


# Run MACE with auto-generated bond pairs
comparison_batch_autogen = descriptor_dict_to_mace_batch(
	{**comparison_input, "bond_pairs": autogenerated_bond_pairs},
	full_model, device="cpu"
)
comparison_output_autogen = full_model(
	comparison_batch_autogen,
	training=False,
	compute_force=True,
	compute_virials=False,
	compute_stress=False,
)
energy_autogen = comparison_output_autogen["energy"].detach().cpu().reshape(-1)[0].item()
node_energy_autogen = comparison_output_autogen["node_energy"]

print("\n" + "="*80)
print("ENERGY COMPARISON")
print("="*80)
print(f"Hard-coded bond pairs energy: {energy_hardcoded:.6f} eV")
print(f"Hard-coded reconstructed from node_energy: {node_energy_hardcoded.sum().detach().cpu().item():.6f} eV")
print(f"Auto-generated bond pairs energy: {energy_autogen:.6f} eV")
print(f"Auto-generated reconstructed from node_energy: {node_energy_autogen.sum().detach().cpu().item():.6f} eV")
print(f"Energy difference: {abs(energy_hardcoded - energy_autogen):.10f} eV")
print(f"Edge count difference: {comparison_batch_autogen['edge_index'].shape[1] - comparison_batch_hardcoded['edge_index'].shape[1]}")

#%%
# Metal-containing molecule comparison (Cu4 tetrahedron)
comparison_input_metal = {
	"atomic_numbers": [29, 29, 29, 29],
	"positions": [
		[0.0, 0.0, 0.0],
		[2.56, 0.0, 0.0],
		[1.28, 2.216, 0.0],
		[1.28, 0.738, 2.08],
	],
}

# Hard-coded: fully-connected tetrahedron edges
hardcoded_bond_pairs_metal = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
autogenerated_bond_pairs_metal = generate_bond_pairs_from_coordinates(
	comparison_input_metal["positions"], comparison_input_metal["atomic_numbers"]
)

print("\n" + "="*80)
print("METAL-BASED BOND PAIR COMPARISON (Cu4)")
print("="*80)
print(f"Hard-coded bond pairs: {hardcoded_bond_pairs_metal}")
print(f"Auto-generated bond pairs: {autogenerated_bond_pairs_metal}")
print(f"Hard-coded count: {len(hardcoded_bond_pairs_metal)}")
print(f"Auto-generated count: {len(autogenerated_bond_pairs_metal)}")

# Run MACE with hard-coded metal bonds
comparison_batch_hardcoded_metal = descriptor_dict_to_mace_batch(
	{**comparison_input_metal, "bond_pairs": hardcoded_bond_pairs_metal}, full_model, device="cpu"
)
comparison_output_hardcoded_metal = full_model(
	comparison_batch_hardcoded_metal,
	training=False,
	compute_force=True,
	compute_virials=False,
	compute_stress=False,
)
energy_hardcoded_metal = comparison_output_hardcoded_metal["energy"].detach().cpu().reshape(-1)[0].item()
node_energy_hardcoded_metal = comparison_output_hardcoded_metal["node_energy"]

# Run MACE with auto-generated metal bonds
comparison_batch_autogen_metal = descriptor_dict_to_mace_batch(
	{**comparison_input_metal, "bond_pairs": autogenerated_bond_pairs_metal}, full_model, device="cpu"
)
comparison_output_autogen_metal = full_model(
	comparison_batch_autogen_metal,
	training=False,
	compute_force=True,
	compute_virials=False,
	compute_stress=False,
)
energy_autogen_metal = comparison_output_autogen_metal["energy"].detach().cpu().reshape(-1)[0].item()
node_energy_autogen_metal = comparison_output_autogen_metal["node_energy"]

print("\n" + "="*80)
print("METAL ENERGY COMPARISON (Cu4)")
print("="*80)
print(f"Hard-coded bond pairs energy: {energy_hardcoded_metal:.6f} eV")
print(f"Hard-coded reconstructed from node_energy: {node_energy_hardcoded_metal.sum().detach().cpu().item():.6f} eV")
print(f"Auto-generated bond pairs energy: {energy_autogen_metal:.6f} eV")
print(f"Auto-generated reconstructed from node_energy: {node_energy_autogen_metal.sum().detach().cpu().item():.6f} eV")
print(f"Energy difference: {abs(energy_hardcoded_metal - energy_autogen_metal):.10f} eV")
print(f"Edge count difference: {comparison_batch_autogen_metal['edge_index'].shape[1] - comparison_batch_hardcoded_metal['edge_index'].shape[1]}")

# %%


# %%
# max_features_to_print = min(32, penultimate_feats.shape[1])
# for i in range(max_features_to_print):
# 	print(penultimate_feats[:, i])



#TODO
#understand what happened
# look if penultimate layer is correct
# look at what other things MACE can output
#look at their penultimate layers
# maybe let s look at node level outputs. There are layers that output node nr of features. Maybe we can use them to construct a new graph to put into a simple GNN
# look at what other MACE models we can call

# %%
