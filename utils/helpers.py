import os
import json
from pathlib import Path
import re
import torch
import numpy as np
from sklearn.metrics import pairwise_distances
import torch
from torch_geometric.data import Data, Batch
import random
from torch_geometric.data import DataLoader as GeoLoader
from scipy.spatial import Delaunay
import torch.nn.functional as F
from sklearn.model_selection import KFold
from ase.io import read
from ase.geometry import Cell, find_mic
from ase.data import vdw_radii, covalent_radii, chemical_symbols
import periodictable as pt

def get_vdw_radius(element):
    atomic_number = chemical_symbols.index(element)
    r = vdw_radii[atomic_number]
    if np.isnan(r):
        r = covalent_radii[atomic_number]
    return r

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_graph(atoms, pos,target=None, distance = 8, domain=None):
    '''
    Get a Data graph from the numpy coordinates, the type of atom and the target.
    '''
        # edge index
    current_script_path = Path(__file__).resolve()

    a = np.arange(len(atoms))
    edges = np.array(np.meshgrid(a,a)).T.reshape(-1,2).T
    edges = torch.tensor(edges, dtype=torch.int64)
    with open(current_script_path.parent / 'element_properties.json', 'r') as json_file:
        data = json.load(json_file)
        atom_to_num = data['atom_to_num']
        atom_to_en = data['atom_to_en']
        atom_to_r = data['atom_to_r']

    atomic_nums = np.asarray([atom_to_num[atom] for atom in atoms])[:, np.newaxis] # keep as numpy for later use
    z = torch.tensor(np.asarray([atom_to_num[atom] for atom in atoms]))
    electroneg = torch.tensor(np.asarray([atom_to_en[atom] for atom in atoms])[:, np.newaxis], dtype=torch.float)
    atomic_radius = torch.tensor(np.asarray([atom_to_r[atom] for atom in atoms])[:, np.newaxis], dtype=torch.float)

    # In the loop we extract the nodes' embeddings, edges connectivity
    # and label for a graph, process the information and put it in a Data
    # object, then we add the object to a list

    # Node features
    # atomic number abd electronegativity # TODO: add atomic radius

    # Edge features
        # shape [N', D'] N': number of edges, D': number of edge features
        # cm matrix and bond matrix
    pair_dist = pairwise_distances(pos)
    cm = (atomic_nums*atomic_nums.T) / pair_dist
    np.fill_diagonal(cm, 0.5*atomic_nums**2.4)
    edge_1 = []
    edge_2 = []
    cm_sparse = []
    dist_sparse = []
    for i in range(pair_dist.shape[0]):
        for j in range(pair_dist.shape[1]):
            if pair_dist[i,j] < 6:
                edge_1.append(i)
                edge_2.append(j)
                cm_sparse.append(cm[i,j])
                dist_sparse.append(pair_dist[i,j])
    cm_sparse = np.array(cm_sparse)
    dist_sparse = np.array(dist_sparse)
    cm_sparse = torch.tensor(cm_sparse.flatten()[:, np.newaxis], dtype = torch.float)
    dist_sparse = torch.tensor(dist_sparse.flatten()[:, np.newaxis], dtype = torch.float)
    cm = cm.flatten()[:, np.newaxis]
    # edge_attr = torch.tensor(cm, dtype=torch.float)
    # print(cm.shape, pair_dist.shape)

    # edge_attr = torch.cat([torch.tensor(cm, dtype=torch.float), torch.tensor(pair_dist.flatten()[:, np.newaxis], dtype = torch.float)], dim = 1)

    edge_attr = torch.cat([cm_sparse, dist_sparse], dim = 1)
    edges = torch.tensor(np.array([edge_1, edge_2]), dtype=torch.int64)

    if target:
        target = torch.tensor(target, dtype=torch.float)
    # belonging = torch.zeros(pos.shape[0],1)
    # belonging[:-2] = 1
    # belonging[:-2,0] = 1
    # belonging[-2:,1] = 1


    node_attrs = torch.cat([torch.tensor(atomic_nums, dtype=torch.float), electroneg,atomic_radius], dim=1)
    distances_co = pair_dist[-1:,:]
    nearby_nodes = np.argwhere(distances_co < distance)[:,1] #
    nearby_nodes = np.array(list(set(nearby_nodes)))
    mask = torch.zeros((pair_dist.shape[0]), dtype=torch.bool)
    mask[nearby_nodes] = 1
    pos = torch.tensor(pos, dtype=torch.float)

    graph = Data(x=node_attrs,
                 z=z,
                 pos=pos,
            edge_index=edges,
            edge_attr=edge_attr,
            y=target,
            mask = mask,
            domain = domain)

    return graph


def save_points(name, graph, energy):
    coordinates = graph.pos.numpy()
    atomic_nr = graph.z.numpy()
    elements = [pt.elements[n].symbol for n in atomic_nr]
    with open(name+'.xyz', "w") as file:
        file.write(f" {coordinates.shape[0]}\n")
        file.write(f"Energy {energy}\n")
        for i in range(coordinates.shape[0]):
            coordinate = coordinates[i]
            line = f"{elements[i]} {coordinate[0]} {coordinate[1]} {coordinate[2]}\n"
            file.write(line)

def save_atoms(root_path, txt_name, atoms):
    file_path = os.path.join(root_path, txt_name)
    if not os.path.isfile(file_path):
        np.savetxt(file_path, atoms, fmt='%s')
        print(f'File {txt_name} saved.')
    else:
        print(f'File {txt_name} exists')


def array_reshape(a):
    a = np.array(a)
    a = np.reshape(a, (a.shape[0]*a.shape[1],1))
    return a


def data_preparation(folder='/data_processed/', device=device, types_to_include=['Pt3', 'Pt6']):
    data_list = []
    data_dict = {t: {} for t in types_to_include}
    root_path = os.getcwd() + folder
    filenames = os.listdir(root_path)
    filenames.sort()
    pattern = re.compile(r'(.*)__(\d+)_data_(-?\d+)\.pt')

    for filename in filenames:
        if filename.endswith('.pt'):
            match = pattern.match(filename)
            if match:
                type_key, file_index, data_index = match.groups()
                type_key = type_key + '_'
                print('type_key', type_key)
                file_index = int(file_index)
                data_index = int(data_index)

                if type_key in types_to_include:
                    full_path = os.path.join(root_path, filename)
                    graph = torch.load(full_path,weights_only= False)
                    graph.x = graph.x.to(device)
                    graph.edge_attr = graph.edge_attr.to(device)
                    graph.edge_index = graph.edge_index.to(device)
                    graph.y = torch.tensor(graph.y, dtype=torch.float).to(device)
                    data_list.append(graph)

                    if file_index not in data_dict[type_key]:
                        data_dict[type_key][file_index] = {}
                    if data_index not in data_dict[type_key][file_index]:
                        data_dict[type_key][file_index][data_index] = []
                    data_dict[type_key][file_index][data_index].append(graph)

    return data_list, data_dict

def simple_split_train_val_test(data_dict, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    file_index_list = []
    for type_key in data_dict:
        for file_index in data_dict[type_key]:
            file_index_list.append((type_key, file_index))

    random.shuffle(file_index_list)
    total = len(file_index_list)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_keys = file_index_list[:train_end]
    val_keys   = file_index_list[train_end:val_end]
    test_keys  = file_index_list[val_end:]

    train_data = []
    for type_key, file_index in train_keys:
        for data_idx in data_dict[type_key][file_index]:
            train_data.extend(data_dict[type_key][file_index][data_idx])

    val_data = []
    for type_key, file_index in val_keys:
        for data_idx in data_dict[type_key][file_index]:
            val_data.extend(data_dict[type_key][file_index][data_idx])

    test_data = []
    for type_key, file_index in test_keys:
        for data_idx in data_dict[type_key][file_index]:
            test_data.extend(data_dict[type_key][file_index][data_idx])

    #  DataLoader
    from torch_geometric.data import DataLoader as GeoLoader
    train_loader = GeoLoader(train_data, batch_size=len(train_data), shuffle=True)
    val_loader   = GeoLoader(val_data,   batch_size=len(val_data), shuffle=False)
    test_loader  = GeoLoader(test_data,  batch_size=len(test_data), shuffle=False)


    return train_loader, val_loader, test_loader,

def cross_validation_split(remaining_keys, k=5, random_state=42):
    random.seed(random_state)
    remaining_keys = remaining_keys.copy()
    random.shuffle(remaining_keys)
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    folds = []
    remaining_keys = np.array(remaining_keys)
    for train_index, val_index in kf.split(remaining_keys):
        train_keys = remaining_keys[train_index].tolist()
        val_keys   = remaining_keys[val_index].tolist()
        folds.append((train_keys, val_keys))
    return folds

def build_data_loaders(data_dict, keys, batch_size):
    data_list = []
    for type_key, file_index in keys:
        file_index = int(file_index)
        for data_idx in data_dict[type_key][file_index]:
            data_list.extend(data_dict[type_key][file_index][data_idx])
    from torch_geometric.data import DataLoader as GeoLoader
    loader = GeoLoader(data_list, batch_size=batch_size, shuffle=True)
    return loader

def split_train_test(data_dict, train_ratio=0.9):
    with open('parameters.json', 'r') as f:
        config = json.load(f)
    trajectory_positions = config.get("simulation_timesteps", config.get("trajectory_positions", [0, 5, 10, -1]))
    # Separate type_keys into train and test sets
    type_keys = list(data_dict.keys())
    random.shuffle(type_keys)

    train_key_count = int(len(type_keys) * train_ratio)
    train_keys = type_keys[:train_key_count]
    test_keys = type_keys[train_key_count:]

    # Collect train and test data based on the split type_keys
    train_data = []
    for type_key in train_keys:
        for file_index in data_dict[type_key]:
            if trajectory_positions == "all":
                for data_idx in data_dict[type_key][file_index]:
                    train_data.extend(data_dict[type_key][file_index][data_idx])
            else:
                for data_idx in trajectory_positions:
                    train_data.extend(data_dict[type_key][file_index][data_idx])

    test_data = []
    for type_key in test_keys:
        for file_index in data_dict[type_key]:
            if trajectory_positions == "all":
                for data_idx in data_dict[type_key][file_index]:
                    test_data.extend(data_dict[type_key][file_index][data_idx])
            else:
                for data_idx in trajectory_positions:
                    test_data.extend(data_dict[type_key][file_index][data_idx])

    # Create loaders
    train_loader = GeoLoader(train_data, batch_size=len(train_data), shuffle=True)
    test_loader = GeoLoader(test_data, batch_size=len(test_data), shuffle=False)

    return train_loader, test_loader, test_keys
def read_last_snapshot(filename):
    atoms = read(filename, index='-1')
    coordinates = atoms.get_positions()
    return coordinates

def atom_extraction(cluster_name='Pt3', data_dir=None):
    print(cluster_name)
    if data_dir is None:
        data_dir = 'data_processed'
    with open(os.path.join(data_dir, cluster_name + '_atoms.txt'), 'r') as f:
        lines = f.readlines()
        atoms = [line.strip() for line in lines]

    return atoms

def create_samples(atoms, cluster, sample_size, NR_OF_ABSORBANTS, device):
    Xsamples = [] # list of Data input objects to feed to GNNGP

    # NOTE: we could generate C depending on the convex hull of the adsorpant. Have to modify the is_feasible function and the generation of C points.
    # Convert cluster to numpy if it's a tensor
    if torch.is_tensor(cluster):
        cluster_np = cluster.cpu().numpy()
    else:
        cluster_np = np.array(cluster)
    hull = Delaunay(cluster_np)
    # print(cluster)
    counter = 0
    Zn_samples = []

    while counter < sample_size:
        # pick an atom at random from the cluster
        n = np.random.randint(0, len(cluster)-NR_OF_ABSORBANTS) # high is exclusive
        atom_cluster = atoms[n]
        Flag = True
        # Get the position of the selected atom
        if torch.is_tensor(cluster):
            atom_pos = cluster[n:n+1].cpu().numpy()
        else:
            atom_pos = cluster[[n]]

        if atom_cluster == 'Pt':
            Zn_sample = points_on_sphere(center=atom_pos, N=1, radius=2.14)
        elif atom_cluster == 'O':
            Zn_sample = points_on_sphere(center=atom_pos, N=1, radius=2.54)
        elif atom_cluster == 'Ni':
            Zn_sample = points_on_sphere(center=atom_pos, N=1, radius=2)
        elif atom_cluster == 'C':   #C only applied weak interactions. So we don't want suggestions near C.
            continue
        # Check minimum distance to all atoms (including adsorbate)
        for i in range(0,len(cluster)):
            if torch.is_tensor(cluster):
                cluster_i = cluster[i].cpu().numpy()
            else:
                cluster_i = cluster[i]
            dis = np.linalg.norm(Zn_sample - cluster_i)
            if dis < 1.8:  # Increased from 1.5 to 1.8 Angstrom for better separation
                Flag = False
                break
        if Flag == True:
            if is_feasible(Zn_sample, hull):
                # Additional check: ensure point is not too close to cluster center
                # Convert cluster to numpy if it's a tensor
                if torch.is_tensor(cluster):
                    cluster_np = cluster.cpu().numpy()
                else:
                    cluster_np = cluster

                cluster_center = np.mean(cluster_np, axis=0)
                dist_to_center = np.linalg.norm(Zn_sample - cluster_center)
                min_dist_to_center = np.min([np.linalg.norm(c - cluster_center) for c in cluster_np]) * 0.8

                if dist_to_center > min_dist_to_center:  # Point should be outside 80% of minimum radius
                    counter += 1
                    # Zn_sample is (1, 3), so we just need the first element
                    Zn_samples.append(Zn_sample[0])
    # Convert list to numpy array
    Zn_samples_array = np.array(Zn_samples)  # (N,3)
    # Zn_samples = np.asarray(Zn_samples).reshape(sample_size,3)

    # create new cluster
    # Convert cluster to numpy if needed for concatenation
    if torch.is_tensor(cluster):
        cluster_for_concat = cluster.cpu().numpy()
    else:
        cluster_for_concat = np.array(cluster)

    for i in range(sample_size):
        new_X = np.concatenate((cluster_for_concat, Zn_samples_array[i:i+NR_OF_ABSORBANTS]), axis=0)
        graph = get_graph(atoms, new_X,distance = 6)
        graph.x = graph.x.to(device)
        graph.edge_s = graph.edge_attr.to(device)
        graph.edge_index = graph.edge_index.to(device)
        graph.mask = graph.mask.to(device)
        # print(graph.mask)
        Xsamples.append(graph.to(device))
    return Xsamples

def points_on_sphere(center=None, N=100000, radius=1):
    '''generate N points on the surface of a sphere with given radius and center'''
    # spherical coordinates
    theta = np.random.uniform(0, 2*np.pi, (N,))
    phi = np.random.uniform(0, np.pi, (N,))
    # cartesian coordinates
    x = radius*np.sin(phi)*np.cos(theta)
    y = radius*np.sin(phi)*np.sin(theta)
    z = radius*np.cos(phi)
    points = center + np.vstack((x,y,z)).T
    return points
        # optimizer.zero_grad()
        # output from model

def is_feasible(points, hull):
    '''returns an array of boolean values that indicate which points are feasible and which are not.'''
    # Point should be outside the convex hull of the cluster
    # find_simplex returns -1 if point is outside convex hull
    return hull.find_simplex(points) < 0

def use_model(test_loader, model_dimenet, transformer_model, device, gp_model, likelihood, fixed_length=31):
    model_dimenet.eval()
    transformer_model.eval()
    gp_model.eval()
    likelihood.eval()

    all_means = []
    all_stds = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            # DimeNet  (B, L, 7)
            P = model_dimenet(data.z, data.pos, data.batch)

            #  (L, B, d_model)
            # P = P.permute(1, 0, 2)

            # current_seq_len = P.size(-1)
            # if current_seq_len < fixed_length:
            #     # F.pad (pad_last_dim_left, pad_last_dim_right, pad_second_last_dim_left, pad_second_last_dim_right, …)
            #     P = F.pad(P, (0, fixed_length - current_seq_len))
            # elif current_seq_len > fixed_length:
            #     P = P[..., :fixed_length]
            # transformer_output = transformer_model(P).squeeze(-1)
            transformer_output = transformer_model(P, data.z, data.batch).squeeze(-1)
            # transformer_output = transformer_output.sum(dim=0)
            observed_pred = likelihood(gp_model(transformer_output))
            pred_mean = observed_pred.mean.cpu().numpy()
            all_means.append(pred_mean)
            pred_std = observed_pred.stddev.cpu().numpy()
            all_stds.append(pred_std)

    all_means = np.concatenate(all_means, axis=0)
    all_stds = np.concatenate(all_stds, axis=0)
    mean_value = np.mean(all_means)
    std_value = np.mean(all_stds)
    mu = -mean_value
    stdq = np.sqrt(std_value)
    return all_means, stdq

def read_last_snapshot(filename):
    atoms = read(filename, index='-1')
    coordinates = atoms.get_positions()
    return coordinates

def get_periodic_distance(coordinates, config):

    # # get cell vectors from config.json
    # with open('config.json', 'r') as json_file:
    #     config = json.load(json_file)

    # Extract the cell vectors
    cell_vectors = config['cell_vectors']
    a, b, c = np.array(cell_vectors[0]), np.array(
        cell_vectors[1]), np.array(cell_vectors[2])

    # get box size from coordinates
    num_points = len(coordinates)
    # Calculate the pairwise distance of every point to every point using vectorized operations
    cell = Cell([a, b, c])
    diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    dist_vec, pairwise_distances = find_mic(diff.reshape(-1, 3), cell)
    pairwise_distances = pairwise_distances.reshape(num_points, num_points)
    return pairwise_distances

def cluster_convex_hull(points, cluster):
    hull = Delaunay(cluster)
    '''returns an array of boolean values that indicate which points are feasible and which are not.'''
    # if point is not in simplex
    return np.all(hull.find_simplex(points) < 0)