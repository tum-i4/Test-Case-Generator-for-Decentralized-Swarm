"""
Data loading utilities.
"""

import pickle
import dill
from pathlib import Path
import pandas as pd

# Load summary data from dataframes
def load_pickled_dataframe(results_dict):
    """
    Load results from functional or experiment tests.

    Parameters:
        results_dict: Dictionary with at minimum a "path" key, and optionally
                     other metadata keys (e.g., "type", "obs", "algo", "r")

    Returns:
        DataFrame with loaded data and additional metadata columns
    """
    # Construct path to dataframe.pkl
    functional_path = Path(results_dict["path"]) / "dataframe.pkl"

    # Check if valid
    if not functional_path.exists():
        raise FileNotFoundError(f"Data file not found: {functional_path}")

    # Results from functional test should be dataframe
    df = pd.read_pickle(functional_path)

    # If additional keys in dictionary, iterate through and add to dataframe
    for key, value in results_dict.items():
        if key != "path":  # Skip the path key itself
            df[key] = value

    return df

# Iterate through all in config and load into one combined dataframe
def load_all_pickled_dataframes(config_dict):

    df_combined = None
    for key, value in config_dict.items():
        df_temp = load_pickled_dataframe(value)

        # Add extra identifier column for tracing
        df_temp["config_key"] = key

        # Append to existing
        if df_combined is None:
            df_combined = df_temp
        else:
            df_combined = pd.concat([df_combined, df_temp], ignore_index=True)

    return df_combined

# Load pickled results object: loads everything in directory with [keyword_include] in name
def load_pickled_trial_results(results_path, keyword_include="", keyword_exclude=""):
    """
    Load pickled results from files matching specified criteria.
    Supports both .pkl (pickle) and .dpkl (dill) file formats.

    Parameters:
        results_path: Directory path containing the result files
        keyword_include: Keyword that must be in filename (default "" matches all)
        keyword_exclude: Keyword to exclude from results (default "" excludes none)

    Returns:
        List of unpickled objects, or None if no files found
    """
    # Initialize
    res_combined = []

    # Convert directory to Path and make sure it exists
    results_path = Path(results_path)
    if not results_path.exists():
        print(f"Directory not found: {results_path}")
        return None

    # Identify all .pkl and .dpkl files in the given directory that match the include pattern
    pkl_files = sorted(results_path.glob(f"*{keyword_include}*.pkl"))
    dpkl_files = sorted(results_path.glob(f"*{keyword_include}*.dpkl"))
    all_files = pkl_files + dpkl_files

    # Filter out files that contain the exclude keyword (case-insensitive)
    if keyword_exclude:
        filtered_files = [f for f in all_files if keyword_exclude not in f.name]
    else:
        filtered_files = all_files

    if not filtered_files:
        if keyword_include or keyword_exclude:
            print(f"No files matching criteria (include: '{keyword_include}', exclude: '{keyword_exclude}') found in directory {results_path}.")
        else:
            print(f"No .pkl or .dpkl files found in directory {results_path}.")
        return None

    # Iterate through the files and unpickle them using the appropriate loader
    for file_path in filtered_files:
        with open(file_path, 'rb') as f:
            if file_path.suffix.lower() == '.dpkl':
                res_combined.append(dill.load(f))
            elif file_path.suffix.lower() == '.pkl':
                res_combined.append(pickle.load(f))
            else:
                print(f"Skipping file {file_path}, file type not recognized.")

    return res_combined

