from scipy.optimize import linprog
import matplotlib.pyplot as plt
import random
import os
import numpy as np
import warnings
from scipy.stats import norm
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay, ConvexHull
from sklearn.metrics import pairwise_distances
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as GeoLoader
import gpytorch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / 'utils'))

from helpers import get_graph
sys.path.append(str(Path(__file__).resolve().parent.parent / 'model_training_pipeline'))

from model import MACETransformer, MaceFeatureExtractor, ModularGPModel
from placement import *
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch.multiprocessing as mp
import glob

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Config file can be overridden via BO_CONFIG env var (defaults to config.json,
# preserving the original behavior). Used to run BO with the no-transformer
# control model without touching the main config.
sys.path.append(str(Path(__file__).resolve().parent.parent / 'bo_pipeline'))
current_script_path = Path(__file__).resolve()
with open(os.environ.get('BO_CONFIG', current_script_path.parent.parent / 'configs' / 'config_bo.json'), 'r') as json_file:
    config = json.load(json_file)
with open(os.environ.get('BO_CONFIG', current_script_path.parent.parent / 'configs' / 'config_train.json'), 'r') as json_file:
    config_train = json.load(json_file)
possible_defects = config["possible_defects"]
nr_of_defects = config["num_defects"]
visualization_optimization = config["visualization_optimization"]
adsorbate_xyz_path = config.get('adsorbate_xyz', 'adsorbate.xyz')
adsorbate_xyz_path = os.path.join(current_script_path.parent,  adsorbate_xyz_path)
adsorbate_coordinates = read_last_snapshot(adsorbate_xyz_path)
NR_OF_ABSORBANTS = adsorbate_coordinates.shape[0]
visualization_optimization = config["visualization_optimization"]



def resolve_bo_checkpoint_dirs():
    """Build checkpoint directory list from config fields bo_checkpoint_dir and bo_targets."""
    current_script_path = Path(__file__).resolve().parent.parent
    ckpt_base = os.path.join(current_script_path, "checkpoints", config["bo_checkpoint_dir"])
    print(f"Checkpoint base: {ckpt_base}")
    return ckpt_base

def generate_samples_in_batches(defective_atoms, cluster, adsorbate_coordinates, min_distances, min_dist, device, indices, xsamples):
    results = []
    for i in indices:
        defective_atom = defective_atoms[i]
        print(defective_atom)
        graph = generate_sample(
            defective_atom, cluster, adsorbate_coordinates, min_distances, min_dist, device, i)
        results.append((i, graph))
        output_dir = config.get('bo_samples_dir', 'evaluating_positions_for_bo')
        current_script_path = Path(__file__).resolve()
        output_dir = os.path.join(current_script_path.parent, output_dir)
        file_path = os.path.join(output_dir, f'graph_{i}.pt')
        torch.save(graph, file_path)
        print("created sample: ", i)
    return results


def generate_sample(defective_atoms, cluster, adsorbate_coordinates, min_distances, min_dist, device, index):
    Form_sample = place_adsorbate(
        defective_atoms, cluster, adsorbate_coordinates, min_distances, min_dist)
    new_X = np.concatenate((cluster, Form_sample), axis=0)

    graph = get_graph(defective_atoms, new_X)

    return index, graph.to(device)


def optimum_energy(model, extractor, gp_model, likelihood, data_list, device):
    model.eval()
    gp_model.eval()
    likelihood.eval()
    batch = Batch.from_data_list(data_list).to(device)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        h = extractor.extract(batch.z, batch.pos, batch.batch)
        features = model.forward_features(h, batch.batch)
        means = -likelihood(gp_model(features)).mean.cpu().detach().numpy()
    best_so_far = means.max()
    return best_so_far


def points_on_sphere(center=None, N=100000, radius=1):
    '''generate N points on the surface of a sphere with given radius and center'''
    # spherical coordinates
    theta = np.random.uniform(0, 2*np.pi, (N,))
    phi = np.random.uniform(0, np.pi, (N,))
    # cartesian coordinates
    x = radius*np.sin(phi)*np.cos(theta)
    y = radius*np.sin(phi)*np.sin(theta)
    z = radius*np.cos(phi)
    points = center + np.vstack((x, y, z)).T
    return points
    # optimizer.zero_grad()
    # output from model


def is_feasible(points, hull):
    '''returns an array of boolean values that indicate which points are feasible and which are not.'''
    # if point is not in simplex
    return hull.find_simplex(points) < 0


def distance_to_hull(point, points):
    c = np.hstack((np.zeros(points.shape[1]), [-1]))
    A = np.vstack((np.hstack((points, np.ones((points.shape[0], 1)))), np.hstack(
        (point.reshape(1, -1), [[1]]))))
    b = np.ones(points.shape[0] + 1)
    b[-1] = -1
    lp = linprog(c, A_ub=A, b_ub=b)
    projection = lp.x[:-1]

    # Calculate the distance from the point to its projection
    distance = np.linalg.norm(point - projection)
    return distance


class BO:
    def __init__(self, adsorbate_coordinates, cluster, atoms, opt, model_filenames=None,
                 method="ucb", tradeoff=0, sample_size=1, device='cpu'):
        self.cluster = cluster
        self.atoms = atoms
        self.opt = opt
        # Resolve checkpoint dirs: explicit list > config (bo_checkpoint_dir + bo_targets)
        if model_filenames is not None:
            self.model_filenames = model_filenames
        else:
            self.model_filenames = resolve_bo_checkpoint_dirs()
        print(f"  Using checkpoints: {[os.path.basename(d) for d in self.model_filenames]}")
        self.method = method
        self.tradeoff = tradeoff
        self.sample_size = sample_size
        self.device = device
        self.adsorbate_coordinates = adsorbate_coordinates
        self.std_flag = False
        # MACE feature extractor is heavy to construct: built lazily ONCE and cached
        self._extractor = None

    def create_samples(self):
        current_script_path = Path(__file__).resolve()
        output_dir = config.get('bo_samples_dir', 'evaluating_positions_for_bo')
        output_dir = os.path.join(current_script_path.parent, output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # list of Data input objects to feed to GNNGP
        if config["create_new_samples"]:
            if config["fixed_or_random_number_defects"] == "random":
                defect_nr_flag = True
            else:
                defect_nr_flag = False

            self.Xsamples = [None] * self.sample_size
            self.cluster_permutations  = [None] * self.sample_size
            # Generate ONE defect configuration shared by all samples
            for cluster_nr in range(self.sample_size):
                self.defective_atoms = self.atoms.copy()
                if defect_nr_flag:
                    actual_nr_defects = random.randint(1, nr_of_defects)
                else:
                    actual_nr_defects = config["num_defects"]
                host_element = config.get("defect_host_element", "Pt")
                si_indices = [i for i, atom in enumerate(self.defective_atoms) if atom == host_element]
                # choose actual_nr_defects random indices from si_indices without replacement
                defect_indices = random.sample(si_indices, min(actual_nr_defects, len(si_indices)))
                for k in range(actual_nr_defects):
                    if visualization_optimization == "optimization":
                        defect_index = random.randint(0, len(possible_defects)-1)
                        defect = possible_defects[defect_index]
                        substitute_index = defect_indices[k]
                        self.defective_atoms[substitute_index] = defect
                    elif visualization_optimization == "visualization":
                        # For visualization, load atoms from a test system xyz file
                        xyz_prefix = config.get("visualization_xyz_prefix", "Pt3_")
                        parent_dir = Path(__file__).resolve().parent.parent
                        xyz_dir = config.get("visualization_xyz_dir")
                        xyz_dir = os.path.join(parent_dir,"data_creation_pipeline", xyz_dir)
                        xyz_matches = glob.glob(os.path.join(xyz_dir, f"{xyz_prefix}*.xyz"))
                        if not xyz_matches:
                            raise FileNotFoundError(
                                f"No xyz file matching '{xyz_prefix}*.xyz' in {xyz_dir} "
                                f"(check visualization_xyz_prefix / visualization_xyz_dir in the config)")
                        xyz_path = xyz_matches[0]
                        with open(xyz_path, 'r') as f:
                            lines = f.readlines()
                            n_atoms = int(lines[0].strip())
                            self.defective_atoms = [line.split()[0] for line in lines[2:2+n_atoms]]
                        break
                self.cluster_permutations[cluster_nr] = self.defective_atoms
            # Record which defects were applied
            self.defect_info = []
            for i, (orig, new) in enumerate(zip(self.atoms, self.defective_atoms)):
                if orig != new:
                    self.defect_info.append({'index': i, 'from': orig, 'to': new})
            print(f"  Defect configuration: {self.defect_info}")

            # create dictionary of minimum allowed distances between atoms
            with open('min_distances.json', 'r') as json_file:
                min_distances = json.load(json_file)

            min_dist = np.zeros(
                (NR_OF_ABSORBANTS, len(self.atoms)-NR_OF_ABSORBANTS))

            # for i in range(len(self.atoms) - NR_OF_ABSORBANTS):
            #     atom = self.defective_atoms[i]
            #     for j in range(NR_OF_ABSORBANTS):
            #         min_dist[j, i] = min_distances[self.atoms[len(
            #             self.atoms) - NR_OF_ABSORBANTS + j]][atom]
            if config["parallel_sample_creation"]:
                # Specify the number of workers (cores) to use
                num_workers = config['number_of_cores']
                chunk_size = (self.sample_size + num_workers -
                              1) // num_workers  # Calculate chunk size

                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = [
                        executor.submit(generate_samples_in_batches, self.cluster_permutations, self.cluster, self.adsorbate_coordinates,
                                        min_distances, min_dist, self.device, range(i * chunk_size, min((i + 1) * chunk_size, self.sample_size)), self.Xsamples)
                        for i in range(num_workers)
                    ]

                    for future in as_completed(futures):
                        batch_results = future.result()
                        for index, graph in batch_results:
                            self.Xsamples[index] = graph[1]
            else:
                for counter in range(self.sample_size):
                    print(self.cluster_permutations[counter])
                    Form_sample = place_adsorbate(
                        self.cluster_permutations[counter], self.cluster, self.adsorbate_coordinates, min_distances, min_dist)
                    new_X = np.concatenate(
                        (self.cluster, Form_sample), axis=0)
                    graph = get_graph(self.cluster_permutations[counter], new_X)
                    file_path = os.path.join(output_dir, f'graph_{counter}.pt')
                    torch.save(graph, file_path)
                    self.Xsamples[counter] = graph
                    print("number of samples created: ", counter)

            print("samples created")
        else:

            # REad the samples from the directory.
            # read how many samples are in the directory
            directory = config.get('bo_samples_dir', 'evaluating_positions_for_bo')
            #if it doesnt exist, create the directory
            # os.makedirs(directory, exist_ok=True)
            file_ending = '*.pt'
            current_script_path = Path(__file__).resolve()
            directory = os.path.join(current_script_path.parent, directory)
            file_list = glob.glob(os.path.join(directory, file_ending))
            # Sort numerically by graph index so sample i corresponds to graph_i.pt
            file_list = sorted(
                file_list,
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split('_')[-1]))
            num_files = len(file_list)
            self.Xsamples = [None] * num_files
            for i, file_path in enumerate(file_list):
                graph = torch.load(file_path, weights_only=False)
                # Cached files may hold either a plain Data object or an
                # (index, Data) tuple from parallel sample creation.
                if isinstance(graph, (tuple, list)):
                    graph = graph[1]
                self.Xsamples[i] = graph

            # The create-branch also sets defect_info / defective_atoms /
            # cluster_permutations, which run_bo.py reads downstream.
            # Reconstruct them from the loaded graphs (element symbols come
            # straight from z); the defect assignment itself is not stored in
            # the cache, so defect_info stays empty.
            from ase.data import chemical_symbols
            self.defect_info = []
            per_sample_symbols = [
                [chemical_symbols[int(z)] for z in g.z] for g in self.Xsamples]
            self.cluster_permutations = per_sample_symbols
            self.defective_atoms = per_sample_symbols[0]

    def use_model(self):
        self.mu = 0
        self.std = 0
        bottleneck_dim = config_train.get("bottleneck_dim", None)
        feature_dim = bottleneck_dim or 128

        # Build the MACE feature extractor ONCE per process (it is heavy to construct)
        if self._extractor is None:
            self._extractor = MaceFeatureExtractor(
                model_size=config_train["mace_model_size"], device=self.device)
        extractor = self._extractor

        # Extract MACE features for ALL samples first. The pretrained MACE
        # backbone is frozen and checkpoint-independent, so features are
        # computed once and reused for every checkpoint in the ensemble.
        feature_loader = GeoLoader(self.Xsamples, batch_size=64)
        sample_features = []  # one [n_atoms_i, feature_dim] tensor per structure
        with torch.no_grad():
            for test_data in feature_loader:
                test_data = test_data.to(self.device)
                h = extractor.extract(test_data.z, test_data.pos, test_data.batch)
                n_structs = int(test_data.batch.max().item()) + 1
                for s in range(n_structs):
                    sample_features.append(h[test_data.batch == s].cpu())

        model_filenames = self.model_filenames
        if isinstance(model_filenames, str):
            model_filenames = [model_filenames]

        for ckpt_dir in model_filenames:
            print("self.model_filenames: ", self.model_filenames)
            print("ckpt_dir: ", ckpt_dir)
            self.mu_temp = []
            self.std_temp = []

            # Load the feature encoder. Default is the MACETransformer of the
            # main pipeline; bo_model_type="mean_pool_gp" selects the
            # no-transformer control model (frozen MACE + mean pool + GP).
            feature_dim = MaceFeatureExtractor(model_size=(config_train or {}).get("mace_model_size", "large")).feature_dim

            if config.get("bo_model_type", "transformer") == "mean_pool_gp":
                from run_mace_gp import MeanPoolEncoder

                model = MeanPoolEncoder(config_train["feature_dim"]).to(self.device)
            else:
                model = MACETransformer(
                    feature_dim=feature_dim,
                    nhead=config_train["nhead"],
                    dim_feedforward=config_train["dim_feedforward"],
                    num_transformer_layers=config_train["num_transformer_layers"],
                    dropout=config_train["dropout"],
                    bottleneck_dim=config_train.get("bottleneck_dim"),
                ).to(self.device)

            model.load_state_dict(torch.load(
                os.path.join(ckpt_dir, 'best_model.pt'),
                map_location=self.device, weights_only=True))
            model.eval()

            # Load GP model and likelihood from checkpoint
            gp_ckpt = torch.load(
                os.path.join(ckpt_dir, 'gp_model.pt'),
                map_location=self.device, weights_only=False)

            # The ExactGP predictive mean conditions on the training data
            # (train_x/train_y). These are NOT part of gpytorch's state_dict, so
            # they must be stored explicitly in the checkpoint. Without them the
            # GP conditions on dummy data and the predictive mean collapses to
            # the prior mean (a constant energy for every candidate).
            if 'train_x' not in gp_ckpt or 'train_y' not in gp_ckpt:
                raise KeyError(
                    f"Checkpoint '{os.path.join(ckpt_dir, 'gp_model.pt')}' is missing "
                    "'train_x'/'train_y'. It was saved by an old run.py that dropped the "
                    "GP training data, so BO predictions would collapse to a constant "
                    "energy. Please re-train with the updated run.py.")
            train_x = gp_ckpt['train_x'].to(self.device)
            train_y = gp_ckpt['train_y'].to(self.device)

            noise_constraint = gpytorch.constraints.Interval(1e-6, 0.1)
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=noise_constraint).to(self.device)
            gp_model = ModularGPModel(
                train_x=train_x,
                train_y=train_y,
                likelihood=likelihood,
                mean_type=gp_ckpt.get('gp_mean_type', 'linear'),
                kernel_type=gp_ckpt.get('gp_kernel_type', 'matern_linear'),
                input_dim=gp_ckpt.get('gp_input_dim', feature_dim),
            ).to(self.device)
            gp_model.load_state_dict(gp_ckpt['gp_state_dict'])
            # Re-attach the real training data as the GP conditioning set.
            gp_model.set_train_data(inputs=train_x, targets=train_y, strict=False)
            likelihood.load_state_dict(gp_ckpt['likelihood_state_dict'])
            gp_model.eval()
            likelihood.eval()
            safe_mode_for_positive_definite = config.get("safe_mode_for_positive_definite", False)
            # if safe_mode_for_positive_definite:
                # Add a small jitter to the GP's noise level to help ensure positive definiteness
            if safe_mode_for_positive_definite:
                gp_batch_size = 1
            else:
                gp_batch_size = 100
            with torch.no_grad():
                # make a loading bar for the test loader

                for start in range(0, len(sample_features), gp_batch_size):
                    try:
                        chunk = sample_features[start:start + gp_batch_size]
                        h_chunk = torch.cat(chunk, dim=0).to(self.device)
                        batch_chunk = torch.cat([
                            torch.full((f.shape[0],), i, dtype=torch.long)
                            for i, f in enumerate(chunk)
                        ]).to(self.device)
                        features = model.forward_features(h_chunk, batch_chunk)
                        test_predictions = likelihood(gp_model(features))
                    except Exception as e:
                        print(f"Error occurred during model prediction: {e}")
                        continue
                    try:
                        self.mu_temp = np.hstack(
                            (self.mu_temp, test_predictions.mean.detach().cpu().numpy()))
                        self.std_temp = np.hstack(
                            (self.std_temp, test_predictions.stddev.cpu().detach().numpy()))
                    except Exception as e:
                        print(f"Error occurred while processing test data: {e}")
                        self.mu_temp = test_predictions.mean.cpu().detach().numpy()
                        self.std_temp = test_predictions.stddev.cpu().detach().numpy()


            self.mu_temp, self.std_temp = np.asarray(
                self.mu_temp), np.asarray(self.std_temp)
            self.std_temp = np.reshape(
                self.std_temp, (self.std_temp.shape[0], 1))
            self.mu_temp = np.reshape(self.mu_temp, (self.mu_temp.shape[0], 1))
            self.mu_temp = -self.mu_temp
            self.std_temp = np.sqrt(self.std_temp)

            if self.std_flag:
                try:
                    self.mu = np.hstack((self.mu, self.mu_temp))
                    self.std = np.hstack((self.std, self.std_temp))
                except:
                    self.mu = self.mu_temp
                    self.std = self.std_temp
            else:
                self.mu += self.mu_temp / len(self.model_filenames)
                self.std += self.std_temp / len(self.model_filenames)

        if self.std_flag:
            indices = np.argmin(self.std, axis=1)
            self.mu = self.mu[np.arange(self.mu.shape[0]), indices]
            self.std = self.std[np.arange(self.std.shape[0]), indices]

    def create_predictions(self):
        try:
            self.use_model()
        except:
            self.create_samples()
            self.use_model()

    def expected_improvement(self):
        with np.errstate(divide='warn'):
            imp = self.mu - self.opt - self.tradeoff
            Z = imp / self.std
            ei = imp * norm.cdf(Z) + self.std * norm.pdf(Z)
            ei[self.std == 0.0] = 0.0
        return ei

    def get_score(self):
        print("method: ", self.method, " tradeoff: ", self.tradeoff)
        if self.method == "pi":

            scores = norm.cdf((self.mu - self.opt) / (self.std+1E-9))

        elif self.method == "pe":
            scores = self.expected_improvement()

        elif self.method == "ucb":
            scores = self.mu + self.tradeoff*self.std

        # ix = np.argsort(scores, axis=0)[::-1][:1].reshape(-1)
        ix = np.argmax(scores)
        # print(ix)
        best_position = (
            self.Xsamples[ix].pos[-NR_OF_ABSORBANTS:])
        return ix, best_position

# %%
