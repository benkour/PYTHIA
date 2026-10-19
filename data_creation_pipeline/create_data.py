#%%
import re
import json
import numpy as np
import torch
#import helpers from the utils folder one level up from the current script
import sys
from pathlib import Path
import re
sys.path.append(str(Path(__file__).resolve().parent.parent / 'utils'))
from helpers import *
import pandas as pd
#append current working directory to sys.path
import os
from data_creation_pipeline.energy_adsorption import energy_adsorp_dict
#%%
#get path of the current script
current_script_path = Path(__file__).resolve()

with open(current_script_path.parent.parent / "configs" / 'data_processing_parameters.json', 'r') as f:
        config = json.load(f)
def normalization(a):
    return (a - a.min(axis = 0))/(a.max(axis = 0) - a.min(axis = 0)), a.max(axis = 0), a.min(axis = 0)


#define a regular expression that takes a filename and extracts the string from the beginning to the last _

if config["interaction_full"] == "interaction":
    types_to_process = config["types_to_process"]
    if types_to_process is None or len(types_to_process) == 0:
        # If types_to_process is None or empty, use all keys from energy_adsorp_dict
        types_to_process = list(energy_adsorp_dict.keys())
elif config["interaction_full"] == "full":
    #types_to_process should be the list of strings from all the files in the data_raw folder
    raw_data_path = current_script_path.parent / 'data_raw'
    types_to_process = []
    for file in raw_data_path.glob('*.xyz'):
        #type_should be the string from the beginning of the filename to the last _ in the filename
        type_key = re.match(r'^(.*_).*', file.stem).group(1) if re.match(r'^(.*_).*', file.stem) else None
        if type_key and type_key not in types_to_process:
            types_to_process.append(type_key)



# excluded 'Ti_Si_CH2O_' as it wasnt part of the possible defects
raw_data_path = current_script_path.parent / 'data_raw'
processed_data_path = current_script_path.parent / config.get("data_output_dir", "data_processed")
# create processed_data_path if it doesn't exist
processed_data_path.mkdir(parents=True, exist_ok=True)
# delete all files in processed_data_path folder
for file in processed_data_path.glob('*'):
    file.unlink()

# default
np.seterr(divide='ignore', invalid='ignore')

# ensure processed data folder exists and clear it
processed_data_path.mkdir(parents=True, exist_ok=True)
for file in processed_data_path.glob('*'):
    file.unlink()

# get all files with extension in directory
def files_with_extension(directory, extension):
    return sorted(directory.rglob(f'*{extension}'))

# read files and process them
def process_files(files, types_to_process):

    graph_list = []
    energies = []

    simulation_timesteps = config["simulation_timesteps"]
    index = simulation_timesteps # which iterations of the files to pick: we pick the last ones, since they have the lowest energy
    file_indices = {t: 0 for t in types_to_process}




    domain_dict ={}
    count = 0
    lowest_energy = float('inf')
    file_with_lowest_energy = None
    key_ind = []
    for filename in files:
        # check filename for type
        # type_key = next((key for key in types_to_process if filename.stem.startswith(key)), None)
        #type_should be the string from the beginning of the filename to the last _ in the filename
        type_key = re.match(r'^(.*_).*', filename.stem).group(1) if re.match(r'^(.*_).*', filename.stem) else None
        if not type_key:
            continue
        file_index = file_indices[type_key]
        if config["interaction_full"] == "full":
            isolated_en_cluster, isolated_en_mol = 0,0
        elif config["interaction_full"] == "interaction":
            isolated_en_cluster, isolated_en_mol = energy_adsorp_dict[type_key]
        e_adsorp = isolated_en_cluster + isolated_en_mol
        i = 0

        # Use type_key as domain_key for simplicity
        domain_key = type_key.rstrip('_')

        if domain_key in domain_dict.keys():
            domain = domain_dict[domain_key]
        else:
            domain_dict[domain_key] = f'domain{count}'
            domain = f'domain{count}'
            count += 1

        with open(filename, 'r') as file:
            energy_temp, graph_temp = [], []
            for line in file:
                if line.strip().isdigit():
                    atom_num = int(line.strip())
                    atom_count = 0
                    atoms, coordinates = [], []
                    i+=1
                    continue

                energy_match = re.search(r'E\s*=?\s*([+-]?\d+\.\d+)', line)
                if energy_match:
                    e_value = float(energy_match.group(1))
                    energy_temp.append(e_value)
                    continue

                if re.match(r'^\s*[A-Z][a-z]?', line):
                    atom, x1, x2, x3 = line.split()
                    atoms.append(atom)
                    coordinates.append([float(x1), float(x2), float(x3)])
                    atom_count += 1
                    if atom_count == atom_num:
                        graph_temp.append(get_graph(np.array(atoms), np.array(coordinates), e_value,distance = 8, domain=domain))
            if i > 1: # some files contain many iterations
                # If simulation_timesteps is None or "all", extract all timesteps
                if simulation_timesteps is None or simulation_timesteps == "all":
                    # print(f"Extracting all {len(graph_temp)} timesteps from {filename}")
                    selected_graphs = graph_temp
                    selected_energies = energy_temp
                else:
                    # Otherwise, select specific timesteps as before
                    #if index is larger than len(graph_temp) take only the indices that are available

                    selected_graphs = [graph_temp[ind] for ind in index if ind < len(graph_temp)]
                    selected_energies = [energy_temp[ind] for ind in index if ind < len(energy_temp)]

            else:
                selected_graphs = graph_temp
                selected_energies = energy_temp

            for i, graph in enumerate(selected_graphs):
                graph_index = i if (simulation_timesteps is None or simulation_timesteps == "all") else (index[i] if i < len(index) else i)
                graph_list.append((type_key, graph, file_index, graph_index))
            for energy in selected_energies:
                    energies.append(energy - e_adsorp)
                    key_ind.append(type_key)

                    if abs(energy - e_adsorp) < 100 and config["interaction_full"] == "interaction":
                        print("Anomalous energy detected:", energy , e_adsorp,isolated_en_cluster , isolated_en_mol, "in file:", filename)
                    if energy - e_adsorp < lowest_energy:
                        lowest_energy = energy - e_adsorp
                        file_with_lowest_energy = filename
                    # energies.append(energy)
            save_atoms(processed_data_path, type_key + '_atoms.txt', atoms)
            file_indices[type_key] += 1

            # print(f"Processed {filename}", energies[-len(selected_energies):])

    print(f"Lowest energy: {lowest_energy} in file: {file_with_lowest_energy}")
    # energy in Hartree to eV conversion factor
    hartree_to_ev = 27.2114
    print(f"Lowest energy in eV: {lowest_energy * hartree_to_ev} in file: {file_with_lowest_energy}")
    energy_in_ev = [energy * hartree_to_ev for energy in energies]
    df = pd.DataFrame({'type_key': key_ind, 'energy': energy_in_ev})
    df = df.sort_values('energy', ascending=True).reset_index(drop=True)
    df.to_csv(processed_data_path / 'energies.csv', index=False)
    return graph_list, energies

# extract all .xyz files
file_list = files_with_extension(raw_data_path, '.xyz')
# save energies and key_ind





graph_list, energies = process_files(file_list, types_to_process)
#print min energy and the file it belongs to

# normalize energies
energies = np.array(energies)


# print(graph_list[:3])
# print(energies)


# print("energies = ", energies)
mean, std = np.mean(energies), np.std(energies)
energies = (energies - mean) / std  #
# print(mean, std)

# Save standardization parameters to JSON
normalization_params = {
    "mean": float(mean),
    "std": float(std),
    "unit": "Hartree",
    "hartree_to_ev": 27.2114,
    "description": "Normalization parameters for energy values in Hartree units"
}

with open(processed_data_path / 'normalization_params.json', 'w') as f:
    json.dump(normalization_params, f, indent=4)
# print(f"Saved normalization parameters: mean={mean}, std={std}")
# energies ,energie_max, energie_min = normalization(energies)
# print(energie_max, energie_min)
# shuffle data
#saved normalization parameters to json file in the processed_data_path folder
print(f"Saved normalization parameters to {processed_data_path / 'normalization_params.json'}")
for i, (type_key, graph, _, _) in enumerate(graph_list):
    graph.y = energies[i]

# plot histogram of energies

# print(graph_list[:3])
# print(energies)

# save data
processed_data_path.mkdir(parents=True, exist_ok=True)
for type_key, graph, file_index, data_point_index in graph_list:
    filename = f'{type_key}_{file_index}_data_{data_point_index}.pt'
    torch.save(graph, processed_data_path / filename)
# %%
