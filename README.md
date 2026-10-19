# PYTHIA, a transferable machine learning framework for simultaneous defect and adsorption site optimization.
<p align="center">
  <img src="imgs/pythia.jpg"  width="200" />
    <figcaption style="font-size: 0.8em;">King Aigeus in front of the Pythia. Public Domain.</figcaption>
</p>



This repository includes the code base of the paper **"PYTHIA, a transferable machine learning framework for simultaneous defect and adsorption site optimization."**. In here, we present the pipeline for the silicene surface to allow for a flexible template for ease of use.

## Table of Contents

- [PYTHIA, a transferable machine learning framework for simultaneous defect and adsorption site optimization.](#pythia-a-transferable-machine-learning-framework-for-simultaneous-defect-and-adsorption-site-optimization)
  - [Table of Contents](#table-of-contents)
  - [Key Features](#key-features)
  - [System Requirements](#system-requirements)
  - [Expected Output](#expected-output)
  - [Parameter and Input Specification](#parameter-and-input-specification)
    - [General Pipeline settings](#general-pipeline-settings)
    - [data processing settings](#data-processing-settings)
    - [training settings](#training-settings)
    - [Bayesian Optimization settings](#bayesian-optimization-settings)
  - [Command-Line-Usage ](#command-line-usage-)
  - [Tutorials ](#tutorials-)
    - [Tutorial 1 - First End-to-End Run (create data + train + BO)](#tutorial-1---first-end-to-end-run-create-data--train--bo)
    - [Tutorial 2 - Retrain Only (reuse processed data)](#tutorial-2---retrain-only-reuse-processed-data)
    - [Tutorial 3 - BO Only (reuse existing model)](#tutorial-3---bo-only-reuse-existing-model)
    - [Notes](#notes)
  - [Advanced Usage and troubleshooting ](#advanced-usage-and-troubleshooting-)
  - [How to cite PYTHIA ](#how-to-cite-pythia-)




## Key Features

This section can include what makes PYTHIA special in comparision to the existing methods.
- Stochastic GNN
- Flexible
- Symmetry invariant
- No hand crafted features needed
- Easy-to-Use

## System Requirements
**Tested Configuration**:

- Operating System: Ubuntu 24.04 (noble)
- IDE: Visual Studio Code

**Note**: As a rule, everything should work on other systems as well but we cant guarantee it.

**Tested quantum chemistry packages**:
- ORCA 5.04 (Trajectory file compatibility for systems: chini clusters)
- CP2k (systems - Silicene)

*For using PYTHIA with any other simulation softwares, adjust the create_data.py to read the respective trajectory formats or contact us at Johnkouroudis[at]gmail[dot]com. (change the things in [ ] with their symbols. They are only there to confuse the vulture bots prowling the internet *

## Expected Output
If everything is done correctly (and you have set the correct flags in the pipeline_components.json),

- the bo suggestions should appear in the **bo_pipeline/name_you_specified/round_latest_round/system_elements.xyz** folder.
- the model pickles should appear in the **checkpoints/name_you_specified** folder
- the energy surface, std surface and their corresponding csv containing the data should appear in the **bo_pipeline/folder_you_specified/round_latest_round/figures**


## Parameter and Input Specification<a name="parameter-specification"></a>


  - Place all the optimized .xyz trajectory files obtained from the converged simulations in the **data_creation_pipeline/data_raw** folder. (For example, for the current software in use (Orca 5.04), the optimized trajectory files look as in the data_raw folder of the current repository). Note that a minimum of 3 converged trajectory files are required before PYTHIA can run.
  - save the coordinates of your adsorbate as an xyz in the **bo_pipeline** folder.
  - Should you wish to place the adsorbate atoms at specific distances (for example O and Zn should have a minimum 3.54 distance or something like this), complete the min_distances.json (you can see an example in the repo).



In the configs folder you will see all the parameters you can adjust off the box. For ease of use we have separated them into functionalities.



### General Pipeline settings

This file controls which parts of PYTHIA will be run.


| Parameter                     | Type    | Description                                                                                                                                                                                                                                                                                |
| ----------------------------- | ------  | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `create_data`              | Bool    |          If the pipeline should process the data in the raw folder or use the already processed ones in the data_processed folder                                                                   |
| `train_model`            | Bool    |     If the model should be retrained (for example using different data or parameters)                                                                                                                                                                                                     |
| `run_bo`           | Bool  | If the optimization pipeline should be run or not (for example set to false if you just want to tune the model)
|



### data processing settings



This file controls how the `.xyz` trajectories in data_creation_pipeline/data_raw are converted into training graphs.

**data_processing_parameters.json parameters**
| Parameter                | Type                               | Description                                                                                                                                                                           |
| ------------------------ | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `interaction_full`       | String (`"interaction"` or `"full"`) | `"interaction"`: subtracts isolated adsorbate + isolated surface/cluster energies from each frame energy. Recommended for highest accuracy `"full"`: uses raw total energies (for this mode prediction accuracy will suffer but you wont need to simulate the cluster and adsorbate separately and together.).                  |
| `types_to_process`       | List of Strings                    | Used when `interaction_full` is `"interaction"`. Provide system prefixes (for example `"Al_Si_CH2O_"`). If empty (`[]`) or null, all keys from `energy_adsorp_dict` are used if `interaction_full`  is "interaction" and all files from the data_processed if "full". |
| `simulation_timesteps`   | List of Int, `"all"`, or `null`  | Selects which trajectory steps are extracted per file. Example `[-1, 0, 5]` means last, first, and sixth snapshot. `"all"` or `null` extracts all available snapshots. It is a balance between extracting too few points (thus accuracy suffers) and extracting too many and similar (thus causing the covariance matrix to be non positive definite)        |
| `data_output_dir`   | String  | Name of the repository you want your processed graphs to be saved. Useful when you have multiple cases to process.       |


### training settings

**config_train.json**

| Parameter                | Type                  | Description                                                                                                                                                                                                 |
| ------------------------ | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data_dir`               | String                | Name of the folder inside `data_creation_pipeline` containing processed `.pt` graphs and `normalization_params.json` (for example `"data_processed"`).                                                   |
| `save_dir`               | String                | Output folder name inside `checkpoints` where model weights, GP state, and training diagnostic figures are saved.                                                                                                            |
| `mace_model_size`        | String                | Size of the frozen pretrained MACE backbone used for feature extraction (for example `"small"`, `"medium"`, `"large"`).                                                                             |
| `feature_cache_dir`      | String                | Folder where precomputed MACE features are cached to speed up reruns.                                                                                                                                      |
| `split_ratio`            | List of Float         | Train/test split ratio at molecule-type level. Example `[0.85, 0.15]` means 85% of molecule types for training and 15% for testing.  If you have multiple simulations of the same underlying system, they will all end in the same set                                                                    |
| `training_epochs`        | Int                   | Number training epochs.                                                                                                                                                           |
| `lr`                     | Float                 | Learning rate for the Adam optimizer.                                                                                                                                                                       |
| `scheduler`              | String                | Learning rate scheduler. Supported values are `"plateau"`, `"cosine"`, or `"none"` (or `null`).                                                                                                      |
| `scheduler_patience`     | Int                   | Used with `"plateau"`: number of epochs to wait before reducing learning rate.                                                                                                                           |
| `scheduler_factor`       | Float                 | Used with `"plateau"`: multiplicative factor for LR reduction.                                                                                                                                             |
| `scheduler_min_lr`       | Float                 | Lower bound for the learning rate for both `"plateau"` and `"cosine"`.                                                                                                                                  |
| `seed`                   | Int                   | Random seed for reproducible molecule-type train/test splitting and model training.                                                                                                                        |
| `nhead`                  | Int                   | Number of attention heads in the Transformer encoder.                                                                                                                                                       |
| `num_transformer_layers` | Int                   | Number of Transformer encoder layers.                                                                                                                                                                       |
| `dim_feedforward`        | Int                   | Feed-forward hidden size inside Transformer encoder layers.                                                                                                                                                 |
| `dropout`                | Float                 | Dropout probability in Transformer layers.                                                                                                                                                                  |
| `bottleneck_dim`         | Int / null            | Optional projection dimension applied before GP. If null, GP uses full MACE feature dimension.                                                                                                            |
| `gp_mean_type`           | String                | GP mean function type. Supported options are `"linear"`, `"constant"`, and `"zero"`.                                                                                                                 |
| `gp_kernel_type`         | String                | GP covariance kernel type. Supported options are `"matern"`, `"rbf"`, `"linear"`, `"matern_linear"`, and `"spectral_mixture"`.                                                                  |



### Bayesian Optimization settings

**config_bo.json**

| Parameter                         | Type                              | Description                                                                                                                                                                                                                          |
| --------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `data_dir`                        | String                            | Name of the processed-data folder inside data_creation_pipeline used by BO for baseline graphs and normalization parameters (for example "data_processed").                                                                        |
| `baseline_system`                 | String                            | System prefix used to load the baseline structure for BO (for example "Si_CH2O").                                                                                                                                              |
| `possible_defects`                | List of Strings                   | Allowed element substitutions for generating defected surfaces/clusters during optimization mode.                                                                                                                                   |
| `num_defects`                     | Int                               | Number of defect substitutions applied per sampled structure.                                                                                                                                                                        |
| `fixed_or_random_number_defects`                     | String                               | "fixed" or "random" for fixed you always get num_defects number of defects, for random you get from 1 up to num_defects number of defects                                                                                                                                       |
| `defect_host_element`             | String                            | Element in the baseline structure that can be replaced by entries from possible_defects.                                                                                                                                            |
| `adsorbate_xyz`                   | String                            | Adsorbate xyz file name in the folder bo_pipeline (last snapshot is used).                                                                                                                                                                     |
| `visualization_optimization`      | String                            | Controls defect-source mode. Use `"optimization"` to generate random defect substitutions from `possible_defects`; use `"visualization"` to load a fixed defect pattern from an xyz file.                                          |
| `visualization_xyz_prefix`        | String                            | Prefix used to find the xyz file when `visualization_optimization` is set to `"visualization"` (for example `"Si_Ga_CH2O_"`).                                                                                                        |
| `visualization_xyz_dir`           | String                            | Directory under `data_creation_pipeline` that contains the xyz files used in visualization mode (for example `"data_raw"`).                                                                                                                |
| `cell_vectors`                    | List of 3 vectors                 | Unit-cell vectors used when periodicity is true.                                                                                                                                                                                     |
| `surface_cluster`                 | String                            | Geometry mode: "surface" or "cluster". Controls feasibility checks for adsorbate placement.                                                                                                                                     |
| `van_der_waals_distances`         | Bool                              | If true, auto-generates min_distances.json from van der Waals radii before placement.                                                                                                                                              |
| `specific_atom_closest`           | Bool                              | If true, enforces that a specific adsorbate atom is preferentially closest to the surface.                                                                                                                                         |
| `which_atom`                      | Int                               | Index of the adsorbate atom used when specific_atom_closest is true (supports negative indexing, for example -1 for last atom).                                                                                                   |
| `probability_of_closest`          | Float (0 to 1)                    | Probability of enforcing the closest-atom condition when specific_atom_closest is true.                                                                                                                                            |
| `non_viable_surface_atoms`        | List of Strings                   | Surface atom symbols excluded from candidate anchoring during adsorbate placement (for example when there is prior chemical knowledge ).                                                                                                                                                  |
| `bo_checkpoint_dir`               | String                            | Folder inside checkpoints containing trained model files used for BO inference (best_model.pt and gp_model.pt).                                                                                                                    |
| `bo_output_dir`                   | String                            | Output folder in bo_pipeline where BO rounds are written (round_N subfolders are created automatically).                                                                                                                           |
| `bo_samples_dir`                  | String                            | Folder in bo_pipeline used to cache candidate sample graphs (graph_*.pt).                                                                                                                                                          |
| `sample_size`                     | Int                               | Number of candidate placements to generate/evaluate per BO round.                                                                                                                                                                  |
| `create_new_samples`              | Bool                              | If true, generate new candidate graphs; if false, reuse cached graphs from bo_samples_dir.                                                                                                                                        |
| `parallel_sample_creation`        | Bool                              | If true, generate samples with multiprocessing.                                                                                                                                                                                      |
| `number_of_cores`                 | Int                               | Number of worker processes used when parallel_sample_creation is true.                                                                                                                                                              |
| `periodicity`                     | Bool                              | Ensures the placement does not end up in a periodic adjacent cell                                                                                                                      |
| `bo_acquisition`                  | List of Strings                   | Acquisition methods to evaluate. Supported in code: ucb, pi (probability of improvement) and pe (expected improvement).                                                                                                                                            |
| `bo_tradeoff`                     | List of Float                     | Tradeoff values paired with each acquisition method for scoring(available for pe and ucb).                                                                                                                                                      |
| `safe_mode_for_positive_definite` | Bool                              | If true, evaluates GP predictions in very small batches to reduce numerical issues related to positive-definite covariance handling.                                                                                               |





## Command-Line-Usage <a name="installation"></a>


- The specific library versions we tested PYTHIA with can be found in the requirements.txt. They can be installed directly using `pip install -r requirements.txt` from the folder level the requirements.txt resides.

  ```
  # Clone the Repository
  git clone https://github.com/benkour/PYTHIA.git
  # Make a virtual env
  python -m venv PYTHIA_env
  source ./PYTHIA_env/bin/activate
  pip install -r requirements.txt
  cd PYTHIA
  ```
The code will also benefit from CUDA should you have an NVIDIA graphics card

**What to run** - once you configure everything run main.py


## Tutorials <a name="tutorials"></a>


### Tutorial 1 - First End-to-End Run (create data + train + BO)

This tutorial is the recommended first run for a new system.

1. **Prepare your environment**

```bash
# from repository root
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. **Prepare inputs**
- Put converged trajectory `.xyz` files in `data_creation_pipeline/data_raw`.
- Put your adsorbate xyz file in `bo_pipeline` (for example `adsorbate_ch2o.xyz`).
- If you use custom distance constraints, update `min_distances.json`.

3. **Configure preprocessing** (`configs/data_processing_parameters.json`)

4. **Configure training** (`configs/config_train.json`)
- Set `data_dir` equal to `data_output_dir` from step 3.

1. **Configure BO** (`configs/config_bo.json`)
- Set `data_dir` equal to preprocessing output.
- Set `bo_checkpoint_dir` to the same folder name as `save_dir` in training config.
- Set `adsorbate_xyz` to your adsorbate file name in `bo_pipeline`.

1. **Enable all pipeline stages** (`configs/pipeline_components.json`)

```json
{
  "create_data": true,
  "train_model": true,
  "run_bo": true
}
```

7. **Run**

```bash
python main.py
```

8. **Check outputs**
- Processed graphs: `data_creation_pipeline/<data_output_dir>`
- Trained models: `checkpoints/<save_dir>`
- BO results: `bo_pipeline/<bo_output_dir>/round_N`


### Tutorial 2 - Retrain Only (reuse processed data)

Use this when you want to tune model hyperparameters without regenerating graphs.

1. Set pipeline switches:

```json
{
  "create_data": false,
  "train_model": true,
  "run_bo": false
}
```

2. Ensure `configs/config_train.json -> data_dir` points to an existing processed folder.

3. Run:

```bash
python main.py
```


### Tutorial 3 - BO Only (reuse existing model)

Use this when you already have a trained model and want new BO suggestions.

1. Set pipeline switches:

```json
{
  "create_data": false,
  "train_model": false,
  "run_bo": true
}
```

2. In `configs/config_bo.json`, verify:
- `bo_checkpoint_dir` points to the trained checkpoint folder in `checkpoints`.
- `data_dir` matches the processed data used to train that model.
- `create_new_samples`: `true` for fresh candidates, `false` to reuse cached `graph_*.pt` samples.

3. Run:

```bash
python main.py
```

4. Read BO outputs in `bo_pipeline/<bo_output_dir>/round_N/figures`:
- `bo_energy_surface_3d.html`
- `bo_analysis.png`
- `bo_predictions.csv`


### Notes

- If BO sample generation hangs or runs out of memory, set `parallel_sample_creation` to `false`.
- If GP prediction raises positive-definite warnings, keep `safe_mode_for_positive_definite` as `true`.
- For reproducibility, keep `seed` fixed in training config.

## Advanced Usage and troubleshooting <a name="advanced"></a>

- You can go into the model.py and change the model architecture at will. This is where both the transformer model and GP are located.
- You can choose to include more or fewer points of the simulation trajectory by going into the data_processing_parameters.json and adding more values in the parameter `simulation_timesteps`
be wary! If you add too many points that are too similar to each other you will end up with a non invertible covariance matrix. So change these values keeping them sparse enough to be distinct but not too sparse so that information is not included. Trial and error might be required. (Also, dont put more points than the simulation has)
- Sometimes the parallel sample creation runs out of memory and hangs. In this case, you are adviced to run the data preprocessing first (changing the  reprocess_data_for_training variable to true and create_new_samples false in the configjson), running the code and then, run it again by setting the reprocess_data_for_training to false and create_new_samples to true. Sequential placement should always work.

## How to cite PYTHIA <a name="citation"></a>
if you found PYTHIA useful you can cite it as follows
