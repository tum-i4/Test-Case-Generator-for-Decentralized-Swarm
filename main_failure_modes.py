import numpy as np
import src.functional.run_functional as run_functional
import src.multifail.run_optim_failure_modes as optim_failure_modes

# ========================================
# EXPERIMENT CONFIGURATIONS
# ========================================

# Functional baseline configurations; must update data_path and vect_maps manually after inspection
FUNCTIONAL_CONFIGS = {
    'rInf': {
        'run': True,
        'run_type': 'functional_rInf',
        'swarm_r': float('inf'),
        'env_config_type': 0,  # with obstacles
        'data_path': 'results_to_save/exp_failure_modes_v202603/functional/result_20251009_163009_functional_rInf/dataframe.csv',
        'vect_maps': np.arange(1, 30+1, 1)  # 30 collision-free trials
    },
    'r20': {
        'run': True,
        'swarm_r': 20.0,
        'env_config_type': 0,  # with obstacles
        'run_type': 'functional_r20',
        'data_path': 'results_to_save/exp_failure_modes_v202603/functional/result_20251024_175212_functional_r20/dataframe.csv',
        'vect_maps': np.arange(1, 30+1, 1)  # 30 collision-free trials
    }
}

# Common functional test parameters
DEFAULT_FUNCTIONAL_PARAMS = {
    'vect_agents': np.array([12]),
    'num_trials': 50,  # At least 30; can generate extras in case of collisions
    'cooperating': True,
    'flag_parallel': False,
    'end_time': 300.0,
    'start_x': 0.0,
    'end_x': 100.0
}

# Experiment definitions: each entry specifies failure_option, failure_type, and num_failures
EXPERIMENT_CONFIGS = {
    'exp2_power_cutoff': {
        'run': True,
        'functional_path': FUNCTIONAL_CONFIGS['rInf']['data_path'],
        'env_config_type': FUNCTIONAL_CONFIGS['rInf']['env_config_type'],
        'vect_trials': FUNCTIONAL_CONFIGS['rInf']['vect_maps'],
        'r_aware': FUNCTIONAL_CONFIGS['rInf']['swarm_r'],
        'failure_option': 1.1,
        'num_failures': 1,
        'description_short': 'power_cutoff',
        'description': 'Power Cutoff, rInf'
    },
    'exp3_velocity_single': {
        'run': True,
        'functional_path': FUNCTIONAL_CONFIGS['rInf']['data_path'],
        'env_config_type': FUNCTIONAL_CONFIGS['rInf']['env_config_type'],
        'vect_trials': FUNCTIONAL_CONFIGS['rInf']['vect_maps'],
        'r_aware': FUNCTIONAL_CONFIGS['rInf']['swarm_r'],
        'failure_option': 2.1,
        'num_failures': 1,
        'description_short': 'velocity_single',
        'description': 'Single Velocity, rInf'
    },
    'exp4_velocity_double': {
        'run': True,
        'functional_path': FUNCTIONAL_CONFIGS['rInf']['data_path'],
        'env_config_type': FUNCTIONAL_CONFIGS['rInf']['env_config_type'],
        'vect_trials': FUNCTIONAL_CONFIGS['rInf']['vect_maps'],
        'r_aware': FUNCTIONAL_CONFIGS['rInf']['swarm_r'],
        'failure_option': 2.1,
        'num_failures': 2,
        'description_short': 'velocity_double',
        'description': 'Double Velocity, rInf'
    },
    'exp5_velocity_triple': {
        'run': True,
        'functional_path': FUNCTIONAL_CONFIGS['rInf']['data_path'],
        'env_config_type': FUNCTIONAL_CONFIGS['rInf']['env_config_type'],
        'vect_trials': FUNCTIONAL_CONFIGS['rInf']['vect_maps'],
        'r_aware': FUNCTIONAL_CONFIGS['rInf']['swarm_r'],
        'failure_option': 2.1,
        'num_failures': 3,
        'description_short': 'velocity_triple',
        'description': 'Triple Velocity, rInf'
    },
    'exp7_power_cutoff_r20': {
        'run': True,
        'functional_path': FUNCTIONAL_CONFIGS['r20']['data_path'],
        'env_config_type': FUNCTIONAL_CONFIGS['r20']['env_config_type'],
        'vect_trials': FUNCTIONAL_CONFIGS['r20']['vect_maps'],
        'r_aware': FUNCTIONAL_CONFIGS['r20']['swarm_r'],
        'failure_option': 1.1,
        'num_failures': 1,
        'description_short': 'power_cutoff_r20',
        'description': 'Power Cutoff, r20'
    },
    'exp8_velocity_single_r20': {
        'run': True,
        'functional_path': FUNCTIONAL_CONFIGS['r20']['data_path'],
        'env_config_type': FUNCTIONAL_CONFIGS['r20']['env_config_type'],
        'vect_trials': FUNCTIONAL_CONFIGS['r20']['vect_maps'],
        'r_aware': FUNCTIONAL_CONFIGS['r20']['swarm_r'],
        'failure_option': 2.1,
        'num_failures': 1,
        'description_short': 'velocity_single_r20',
        'description': 'Single Velocity, r20'
    },
    'exp9_velocity_double_r20': {
        'run': True,
        'functional_path': FUNCTIONAL_CONFIGS['r20']['data_path'],
        'env_config_type': FUNCTIONAL_CONFIGS['r20']['env_config_type'],
        'vect_trials': FUNCTIONAL_CONFIGS['r20']['vect_maps'],
        'r_aware': FUNCTIONAL_CONFIGS['r20']['swarm_r'],
        'failure_option': 2.1,
        'num_failures': 2,
        'description_short': 'velocity_double_r20',
        'description': 'Double Velocity, r20'
    },
    'exp10_velocity_triple_r20': {
        'run': True,
        'functional_path': FUNCTIONAL_CONFIGS['r20']['data_path'],
        'env_config_type': FUNCTIONAL_CONFIGS['r20']['env_config_type'],
        'vect_trials': FUNCTIONAL_CONFIGS['r20']['vect_maps'],
        'r_aware': FUNCTIONAL_CONFIGS['r20']['swarm_r'],
        'failure_option': 2.1,
        'num_failures': 3,
        'description_short': 'velocity_triple_r20',
        'description': 'Triple Velocity, r20'
    },
    'exp12_power_cutoff_rInf': {
        'run': True,
        'algo': 'random',
        'functional_path': FUNCTIONAL_CONFIGS['rInf']['data_path'],
        'env_config_type': FUNCTIONAL_CONFIGS['rInf']['env_config_type'],
        'vect_trials': FUNCTIONAL_CONFIGS['rInf']['vect_maps'],
        'r_aware': FUNCTIONAL_CONFIGS['rInf']['swarm_r'],
        'failure_option': 1.1,
        'num_failures': 1,
        'description_short': 'power_cutoff_rInf',
        'description': 'Power Cutoff, rInf w random'
    },
    'exp13_velocity_single_rInf': {
        'run': True,
        'algo': 'random',
        'functional_path': FUNCTIONAL_CONFIGS['rInf']['data_path'],
        'env_config_type': FUNCTIONAL_CONFIGS['rInf']['env_config_type'],
        'vect_trials': FUNCTIONAL_CONFIGS['rInf']['vect_maps'],
        'r_aware': FUNCTIONAL_CONFIGS['rInf']['swarm_r'],
        'failure_option': 2.1,
        'num_failures': 1,
        'description_short': 'velocity_single_rInf',
        'description': 'Single Velocity, rInf w random'
    },
    'exp14_velocity_double_rInf': {
        'run': True,
        'algo': 'random',
        'functional_path': FUNCTIONAL_CONFIGS['rInf']['data_path'],
        'env_config_type': FUNCTIONAL_CONFIGS['rInf']['env_config_type'],
        'vect_trials': FUNCTIONAL_CONFIGS['rInf']['vect_maps'],
        'r_aware': FUNCTIONAL_CONFIGS['rInf']['swarm_r'],
        'failure_option': 2.1,
        'num_failures': 2,
        'description_short': 'velocity_double_rInf',
        'description': 'Double Velocity, rInf w random'
    },
    'exp15_velocity_triple_rInf': {
        'run': True,
        'algo': 'random',
        'functional_path': FUNCTIONAL_CONFIGS['rInf']['data_path'],
        'env_config_type': FUNCTIONAL_CONFIGS['rInf']['env_config_type'],
        'vect_trials': FUNCTIONAL_CONFIGS['rInf']['vect_maps'],
        'r_aware': FUNCTIONAL_CONFIGS['rInf']['swarm_r'],
        'failure_option': 2.1,
        'num_failures': 3,
        'description_short': 'velocity_triple_rInf',
        'description': 'Triple Velocity, rInf w random'
    },
    'exp17_power_cutoff_r20': {
        'run': True,
        'algo': 'random',
        'functional_path': FUNCTIONAL_CONFIGS['r20']['data_path'],
        'env_config_type': FUNCTIONAL_CONFIGS['r20']['env_config_type'],
        'vect_trials': FUNCTIONAL_CONFIGS['r20']['vect_maps'],
        'r_aware': FUNCTIONAL_CONFIGS['r20']['swarm_r'],
        'failure_option': 1.1,
        'num_failures': 1,
        'description_short': 'power_cutoff_r20',
        'description': 'Power Cutoff, r20 w random'
    },
    'exp18_velocity_single_r20': {
        'run': True,
        'algo': 'random',
        'functional_path': FUNCTIONAL_CONFIGS['r20']['data_path'],
        'env_config_type': FUNCTIONAL_CONFIGS['r20']['env_config_type'],
        'vect_trials': FUNCTIONAL_CONFIGS['r20']['vect_maps'],
        'r_aware': FUNCTIONAL_CONFIGS['r20']['swarm_r'],
        'failure_option': 2.1,
        'num_failures': 1,
        'description_short': 'velocity_single_r20',
        'description': 'Single Velocity, r20 w random'
    },
    'exp19_velocity_double_r20': {
        'run': True,
        'algo': 'random',
        'functional_path': FUNCTIONAL_CONFIGS['r20']['data_path'],
        'env_config_type': FUNCTIONAL_CONFIGS['r20']['env_config_type'],
        'vect_trials': FUNCTIONAL_CONFIGS['r20']['vect_maps'],
        'r_aware': FUNCTIONAL_CONFIGS['r20']['swarm_r'],
        'failure_option': 2.1,
        'num_failures': 2,
        'description_short': 'velocity_double_r20',
        'description': 'Double Velocity, r20 w random'
    },
    'exp20_velocity_triple_r20': {
        'run': True,
        'algo': 'random',
        'functional_path': FUNCTIONAL_CONFIGS['r20']['data_path'],
        'env_config_type': FUNCTIONAL_CONFIGS['r20']['env_config_type'],
        'vect_trials': FUNCTIONAL_CONFIGS['r20']['vect_maps'],
        'r_aware': FUNCTIONAL_CONFIGS['r20']['swarm_r'],
        'failure_option': 2.1,
        'num_failures': 3,
        'description_short': 'velocity_triple_r20',
        'description': 'Triple Velocity, r20 w random'
    }
}

# Common GA parameters (can be overridden per experiment if needed)
DEFAULT_OPTIM_PARAMS = {
    'vect_agents': np.array([12]),
    'algo': 'ga',
    'population_size': 100,
    'end_time_mult': 2.0,
    'f_min_points': 0.5,
    'v_ref': 6.0,
    'flag_parallel': True,
    'flag_eng_persist': True,
    'terminate': {
        'num_gen': 25,
        'max_time': None,  # maximum time allowed PER TRIAL/SEARCH [sec]
        'gen_stagnation': None,
        'gen_stagnation_tol': None,
        'min_f': 1e-6  # stop if fitness reaches 0 (some tolerance for machine error)
    }
}
# DEFAULT_OPTIM_PARAMS = {  # FOR DEBUGGING
#     'vect_agents': np.array([12]),
#     'algo': 'ga',
#     'population_size': 5,
#     'end_time_mult': 2.0,
#     'f_min_points': 0.5,
#     'v_ref': 6.0,
#     'flag_parallel': True,
#     'flag_eng_persist': True,
#     'terminate': {
#         'num_gen': 3,
#         'max_time': None,  # maximum time allowed PER TRIAL/SEARCH [sec]
#         'gen_stagnation': None,
#         'gen_stagnation_tol': None,
#         'min_f': 1e-6  # stop if fitness reaches 0 (some tolerance for machine error)
#     }
# }


# ========================================
# HELPER FUNCTIONS
# ========================================

def load_and_display_results(results_path, experiment_name=""):
    """
    Load GA results and display summary information.

    Parameters:
        results_path (Path): Path to results directory
        experiment_name (str): Name of experiment for display
    """
    try:
        results_data = optim_failure_modes.load_ga_results(results_path)
        df = results_data['dataframe']

        print(f"\n{experiment_name} Results Summary:")
        print(f"Number of runs: {len(df)}")

        # Extract scalar fitness values (handle both scalar and array formats)
        fitness_values = []
        for f in df['fitness']:
            if isinstance(f, (list, np.ndarray)):
                fitness_values.append(f[0] if len(f) > 0 else np.nan)
            else:
                fitness_values.append(f)
        print(f"Best fitness values: {fitness_values}")

        # Display dynamic columns based on what's available in the dataframe
        failure_cols = ['num_agents', 'num_failures']
        failure_cols.extend([col for col in df.columns if col.startswith('fail_agent_num_')])
        failure_cols.extend([col for col in df.columns if col.startswith('fail_time_')])
        print(f"Agent failure patterns:\n{df[failure_cols].to_string()}")

    except Exception as e:
        print(f"Could not load {experiment_name} results summary: {e}")

def run_functional_baseline(functional_key, **override_params):
    """
    Run a functional baseline test if enabled.

    Parameters:
        functional_key (str): Key from FUNCTIONAL_CONFIGS
        **override_params: Any parameters to override from defaults
    """
    func_config = FUNCTIONAL_CONFIGS[functional_key]

    # Check if this functional test should run
    if not func_config.get('run', False):
        print(f"Skipping functional baseline: {func_config['run_type']} (run=False)")
        return

    params = {**DEFAULT_FUNCTIONAL_PARAMS, **override_params}

    print(f"\n{'='*60}")
    print(f"Running Functional Baseline: {func_config['run_type']}")
    print(f"  Swarm awareness range: {func_config['swarm_r']}")
    print(f"  Environment config: {func_config['env_config_type']}")
    print(f"  Number of trials: {params['num_trials']}")
    print(f"{'='*60}")

    run_functional.run_trial_new_wrapper(
        vect_agents=params['vect_agents'],
        run_type=func_config['run_type'],
        swarm_r=func_config['swarm_r'],
        env_config_type=func_config['env_config_type'],
        num_trials=params['num_trials'],
        cooperating=params['cooperating'],
        flag_parallel=params['flag_parallel'],
        end_time=params['end_time'],
        start_x=params['start_x'],
        end_x=params['end_x']
    )

    print(f"Completed functional baseline: {func_config['run_type']}")
    print(f"Update FUNCTIONAL_CONFIG['{functional_key}'] with the new data_path and vect_maps after reviewing results.\n")


def run_optim_experiment(exp_key):
    """
    Run a GA experiment if enabled.

    Parameters:
        exp_key (str): Key from EXPERIMENT_CONFIGS
    """
    exp_config = EXPERIMENT_CONFIGS[exp_key]

    # Check if this experiment should run
    if not exp_config.get('run', False):
        print(f"Skipping experiment: {exp_config['description']} (run=False)")
        return None

    # Merge default params with experiment-specific config and overrides
    # Priority: override_params > exp_config > DEFAULT_OPTIM_PARAMS
    params = {**DEFAULT_OPTIM_PARAMS, **exp_config}

    # Get some experiment parameters - will use in printout
    vect_trials = params['vect_trials']
    vect_agents = params['vect_agents']
    env_config_type = params['env_config_type']
    terminate_config = params['terminate']

    # Build run_type string from description_short
    run_type = f"{params['algo'].upper()}_fail{params['num_failures']}_{params['description_short']}"

    # Environment label for display
    env_label = 'No obstacles' if env_config_type == 2 else 'With obstacles'

    print(f"\n{'='*60}")
    print(f"Running Experiment: {params['description']}")
    print(f"  Failure Option: {params['failure_option']}")
    print(f"  Num Failures: {params['num_failures']}")
    print(f"  Awareness Range: {params['r_aware']}")
    print(f"  Environment: {env_label} (config_type={env_config_type})")
    print(f"  Num Agents: {vect_agents}")
    print(f"  Num Trials: {len(vect_trials)}")
    print(f"  Population Size: {params['population_size']}")
    print(f"  Max Time/Trial: {terminate_config.get('max_time')} sec")
    print(f"{'='*60}")

    # Run optimization / metaheuristic search
    results_path = optim_failure_modes.run_optim_failure_modes(
        vect_agents=vect_agents,
        vect_trials=vect_trials,
        algo=params['algo'],
        failure_option=params['failure_option'],
        num_failures=params['num_failures'],
        functional_data_path=params['functional_path'],
        run_type=run_type,
        population_size=params['population_size'],
        range_aware=params['r_aware'],
        end_time_mult=params['end_time_mult'],
        f_min_points=params['f_min_points'],
        v_ref=params['v_ref'],
        env_config_type=env_config_type,
        terminate_config=terminate_config,
        flag_parallel=params['flag_parallel'],
        flag_eng_persist=params['flag_eng_persist']
    )

    load_and_display_results(results_path, params['description'])
    return results_path


# ========================================
# MAIN FUNCTION
# ========================================

def main():
    # ========================================
    # RUN FUNCTIONAL BASELINES
    # ========================================
    # Control which functional baselines to run via 'run' flag in FUNCTIONAL_CONFIGS

    for func_key in FUNCTIONAL_CONFIGS.keys():
        run_functional_baseline(func_key)


    # ========================================
    # RUN OPTIMIZATION EXPERIMENTS
    # ========================================
    # Control which experiments to run via 'run' flag in EXPERIMENT_CONFIGS

    for exp_key in EXPERIMENT_CONFIGS.keys():
        run_optim_experiment(exp_key)


if __name__ == "__main__":
    main()
