"""
Functions to rerun best results, verify fitness, and save sim outputs.
"""
from src.postprocessing.data_loader import load_pickled_trial_results
from src.multifail.failure_modes_problem import setup_configs
from src.simulator.simulator_swarmlab import SimulatorSwarmlab
from src.constants import SIM_DIR
from src.utils import create_results_dir
import numpy as np
import pickle

def rerun_best(res_combined, simulator):
    # Iterate through each trial and re-run
    sim_results_all_trials = []
    for idx, res_trial in enumerate(res_combined):
        # Due to trials being different, need to run consecutively
        n_ind = 1

        # Extract
        agent_ids = res_trial['best_agent_ids']
        fail_times = res_trial['best_fail_times']
        obj_problem = res_trial['result'].problem
        swarm_config = obj_problem.swarm_config.copy()
        flags = obj_problem.flags.copy()

        # Update flag for our use case
        flags['output_all_states'] = True
        flags['run_parallel'] = False

        # Extract either fail_power OR velocity vector
        if 'best_fail_powers' in res_trial:
            fail_powers = res_trial['best_fail_powers']
            fail_v_n = None
            fail_v_e = None
            fail_v_d = None
        elif all(k in res_trial for k in ('best_fail_v_n', 'best_fail_v_e', 'best_fail_v_d')):
            fail_powers = None
            fail_v_n = res_trial['best_fail_v_n']
            fail_v_e = res_trial['best_fail_v_e']
            fail_v_d = res_trial['best_fail_v_d']
        else:
            raise ValueError("Trial result does not contain a known failure property.")

        # Run setup; expected output is:
        # isvalid [ndarray], sim_config [dict containing lists with floats or ints], sim_fail_config [dict containing lists with int, bool, or ndarray]
        is_valid, n_valid, sim_config, sim_fail_config = setup_configs(
            n_ind, obj_problem, agent_ids, fail_times, fail_powers, fail_v_n, fail_v_e, fail_v_d)

        # Rerun best results
        result = simulator.failure_modes_simulate(
            sim_config=sim_config,
            swarm_config=swarm_config,
            sim_fail_config=sim_fail_config,
            results_dir="",
            flags=flags
        )

        # Calculate fitness
        times_end = np.array(result['times_end'][0]).flatten()
        result['rerun_f'] = [obj_problem._calculate_fitness(times_end)]

        # Save
        sim_results_all_trials.append(result)
        print(f"\tRerun trial {idx+1} of {len(res_combined)} completed.")

    return sim_results_all_trials


def verify_with_result(sim_out, df, round_decimals=3):
    # First, check that lengths of inputs match
    if len(sim_out) != len(df):
        raise ValueError("Number of trials in rerun does not match number of trials in dataframe of results.")

    # Iterate through sim_out list and save: trial number, verification of match (T/F)
    trial_nums = []
    matches = []
    for out in sim_out:

        # Check that sim_config.env_maps and .env_starts match each other
        if out['sim_config']['env_maps'] != out['sim_config']['env_starts']:
            raise ValueError(f"Rerun with env_maps {out['sim_config']['env_maps']} and env_starts {out['sim_config']['env_starts']} that did not match.")
        elif (len(out['sim_config']['env_maps']) != 1) or (len(out['sim_config']['env_maps']) != 1):
            raise ValueError(f"Rerun has multiple values for env_maps {out['sim_config']['env_maps']} or env_starts {out['sim_config']['env_starts']}.")
        else:
            trial_rerun = out['sim_config']['env_maps'][0]
            out['trial'] = trial_rerun

        # Compare fitness to corresponding trial (same number as env_maps) in df
        trial_nums.append(trial_rerun)
        matches.append(round(out['rerun_f'][0], round_decimals) ==
                       round(df[df['trial']==trial_rerun]['fitness'].iloc[0][0], round_decimals))

    # Sort by trial number before returning
    sorted_items = sorted(zip(trial_nums, matches, sim_out))
    trial_nums, matches, sim_out_sorted = zip(*sorted_items)

    return np.array(trial_nums, dtype=int), np.array(matches, dtype=bool), list(sim_out_sorted)


def run_rerun_best_and_verify(config_dict, df_exp):
    # Start simulator
    simulator = SimulatorSwarmlab(str(SIM_DIR))

    # Iterate through each results config
    for key, value in config_dict.items():
        # Check if detailed sim results already exist
        if value["path_sim"] is not None:
            # Assume detailed sim results are already saved, and skip
            print(f"\nFor {key}: path to sim results has been specified, skipping rerun.")

        else:
            print(f"\nStarting rerun and verification for {key}.")

            # Load pymoo results objs from pickles; load all trials for this config
            res_combined = load_pickled_trial_results(value["path"], "result", "checkpoint")

            # Rerun based on configs of best results from all trials
            out_combined = rerun_best(res_combined, simulator)

            # Verify fitness with results for this set of trials
            temp_df = df_exp[df_exp["config_key"]==key]
            trial_num, match, out_combined = verify_with_result(out_combined, temp_df)

            # If fitness matches, save sim outputs (dill? maybe need to iterate through each trial, sort by trial)
            if len(match) == sum(match):
                print(f"\tOutputs from reruns have been verified as matching original results. Saving sim outputs...")
                folder_path = create_results_dir(key, prefix='rerun_verify')
                out_combined_filename = f"{key}_out_sim_all_trials.pkl"
                out_combined_path = folder_path / out_combined_filename
                with open(out_combined_path, "wb") as f:
                    pickle.dump(out_combined, f)
                print(f"\tSaved to: {out_combined_path}")
            else:
                print(f"\tOutputs from reruns did not match original results for: {key}")
                print(f"\tContinuing to next in config...")

    # Shut down simulator
    simulator.shutdown_engine()

    return None