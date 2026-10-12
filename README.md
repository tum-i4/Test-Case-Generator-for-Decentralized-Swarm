# NOTE:
This release reflects a piece of the replication package accompanying the publication "How to Break a Swarm: Search-based Robustness Testing of Decentralized UAV Swarms," accepted for publication at the 2026 International Symposium on Software Reliability Engineering (ISSRE). Please consider citing us: https://doi.org/10.6084/m9.figshare.30934004

# INFO
- **Title:** How to Break a Swarm: Search-based Robustness Testing of Decentralized UAV Swarms
- **Submission ID:** 217
- **Authors:** 
   - David Marson
      - Technical University of Munich
      - david.marson@tum.de
   - Alexander Pretschner
      - Technical University of Munich
      - alexander.pretschner@tum.de

# EXPECTED BEHAVIOUR
This artifact is a collection of scripts and functions that will execute the experiments described in the paper. Results will be generated in the form of CSV files, figures, and text logs. The process is split into the text case generator, which is written in Python and saved in the `experiment-targeted-failure-modes` directory, and the swarm simulator, which is written in MATLAB and saved in the `swarmlab-augmented` directory. The Python scripts will call the MATLAB simulation using the `matlabengine` package.

# ARTIFACT DESCRIPTION

## Contents
This replication package includes:

1. **Experiment Results** – Data generated from the evaluations of the 16 campaign configurations, saved to the `results_to_save/` directory.
2. **Code to reproduce figures and statistical tests** – Uses provided experiment results from the `results_to_save` directory.
3. **Code to rerun all experiments** – Re-generates results from scratch (WARNING: this may take hours to days to re-run, depending on your hardware).

## Directory Structure
Below are a few key contents of the replication package (not all files are listed):
```
replication-package
├── experiment-targeted-failure-modes/
│   ├── results/
│   ├── results_to_save/
│   ├── main_failure_modes.py
│   ├── main_failure_modes_figures.py
│   ├── LICENSE
│   ├── ...
├── swarmlab-augmented/
│   ├── LICENSE
│   ├── ...
└── README.txt
```

# ENVIRONMENT SETUP
Our experiments were run on a machine with the following hardware specifications:
- CPU: 4.3GHz AMD Ryzen 9 9950X3D 16-cores
- RAM: 128 GB

## Operating System
- Windows 11 (developed and tested on this platform; other operating systems untested).

## Software Requirements
| Software | Version | Notes |
|----------|---------|-------|
| Python | 3.11.9 | Recommended; other 3.11.x versions may work |
| MATLAB | 2024b | Recommended; if you use a different version of MATLAB, the `matlabengine` version in `pyproject.toml` must be adjusted (check the [matlabengine PyPi release history](https://pypi.org/project/matlabengine/#history) for more information.) |

### MATLAB Toolboxes
- Statistics & Machine Learning
- Parallel Processing

### Python Dependencies
All Python dependencies are listed in `pyproject.toml`. We recommend installing by using Poetry (see Installation instructions below).

## Installation
This project is configured to use [Poetry](https://python-poetry.org/) for dependency and virtual environment management.

1. Install `poetry` if you do not have it already:
   ```bash
   pip install poetry
   ```
   or (we recommend):
   ```bash
   pipx install poetry
   ```
2. Navigate to the `experiment-targeted-failure-modes` directory:
   ```bash
   cd experiment-targeted-failure-modes
   ```
3. Install dependencies (creates virtual environment automatically):
   ```bash
   poetry env use < path to python 3.11 >
   poetry install
   ```
   If you do not have an existing virtual environment, Poetry should create one. For help with specifying the virtual environment, we encourage you to consult the [Poetry documentation on managing environments](https://python-poetry.org/docs/managing-environments/#switching-between-environments).
4. All subsequent Python executions should use Poetry:
   ```bash
   poetry run python < file.py >
   ```

# GETTING STARTED

## Option 1: Reproduce Figures and Statistical Tests
Run the figure generation script using existing results (existing results are provided in the `results_to_save/` directory):
```bash
poetry run python main_failure_modes_figures.py
```
Results are automatically saved to the `results/` directory.

## Option 2: Rerun Test Case Generation Experiments
To re-run the nominal performance tests and the robustness test case generation process:
```bash
poetry run python main_failure_modes.py
```
Results are automatically saved to the `results/` directory. 

You can choose a subset of experiments to run by changing `'run': True` to `'run': False` in the `FUNCTIONAL_CONFIGS` and `EXPERIMENT_CONFIGS` dictionaries in `main_failure_modes.py`. To further reduce the run time of the robustness test case generation, you can set an earlier termination criterion in the `DEFAULT_OPTIM_PARAMS` dictionary (e.g., set `num_gen` to a smaller value) and/or reduce the `population_size` parameter. This will help to demonstrate the functionality more quickly but will affect the results.

# REPRODUCIBILITY
To rerun the entire evaluation pipeline, follow the steps below. Note that, using the hardware described above, our cumulative execution time was approximately 265 hours.

1. **Run nominal performance tests first:**
   In `main_failure_modes.py`:
   1. Set `'run': True` in both entries of the `FUNCTIONAL_CONFIGS` dictionary.
   2. Set `'run': False` in all entries of the `EXPERIMENT_CONFIGS` dictionary (to skip the robustness test case generation for now).
   3. Execute: `poetry run python main_failure_modes.py`.
2. **Update data paths to point to nominal performance test results:**
   - After nominal tests are completed, update the `data_path` fields in `FUNCTIONAL_CONFIGS` dictionary to store the relative paths to the recently generated `dataframe.csv` files (automatically saved in `results/` when nominal performance tests were completed).
3. **Generate robustness tests:**
   1. Set `'run': True` in all entries of the `EXPERIMENT_CONFIGS` dictionary to generate robustness tests.
   2. Set `'run': False` in both entries of the `FUNCTIONAL_CONFIGS` dictionary to avoid re-running nominal tests.
   3. Execute: `poetry run python main_failure_modes.py`.
4. **Generate figures and statistics:**
   1.  In the file `main_failure_modes_figures.py`, update the paths in the `RESULT_PATHS_FUNCTIONAL` and `RESULT_PATHS_EXPERIMENT_v2` dictionaries (i.e., the values corresponding to `"path"` keys) to point to the **directories** containing the `dataframe.csv` files generated in the nominal tests and robustness test generation, respectively.
   2. Execute: `poetry run python main_failure_modes_figures.py`.
   3. As part of post-processing, the best search solutions will be re-run to extract detailed state information. After the re-runs are completed, you can add the paths to the **directories** containing the re-run results to the `RESULT_PATHS_EXPERIMENT_v2` dictionary (i.e., the values corresponding to `"path_sim"` keys). Re-executing this script will then generate the figures and statistical tests without needing to re-run the best search solutions again.