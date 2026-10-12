# Run a "functional" test to:
# 1. Check if safety violation occurs, and if not...
# 2. Check how long it takes the group to reach goal

import numpy as np
import pandas as pd
from src.simulator.simulator_swarmlab import SimulatorSwarmlab
from src.constants import SIM_DIR
from src.utils import create_results_dir


def run_trial_new_wrapper(vect_agents, run_type="functional", swarm_r=float('inf'),
                          env_config_type=0, num_trials=1,
                         cooperating=True, flag_parallel=False, end_time=300.0, start_x=0.0, end_x=100.0):
    """
    Run a functional trial using the new MATLAB wrapper_run_generation_exp_failure_modes function.
    
    This function configures and executes simulation runs with no failures to assess baseline system performance 
    under different swarm sizes using the new wrapper infrastructure.
    
    Parameters:
        vect_agents (np.array): Array of swarm sizes to test
        run_type (str): Name identifier for the run type
        swarm_r (float): Awareness radius of UAVs
        env_config_type (int): Flag to indicate environment config type (see swarmlab wrapper for details)
        num_trials (int): Number of trials to be executed
        cooperating (bool): Whether agents cooperate
        flag_parallel (bool): Whether to run simulations in parallel
        end_time (float): Maximum simulation time
        start_x (float): Start plane x-coordinate
        end_x (float): End plane x-coordinate
        
    Returns:
        Path: Path to the results directory
    """
    
    # Initialize the simulator object
    sim = SimulatorSwarmlab(str(SIM_DIR))
    
    # Number of agents - functional test will be run once per agent size
    num_agents = np.flip(vect_agents)  # Flip to descending order for potential runtime benefits
    
    # Configure simulation parameters for each run (must be lists within the dictionary for MATLAB
    sim_config = {
        'end_time': [end_time] * num_trials,
        'start_x': [start_x] * num_trials,
        'end_x': [end_x] * num_trials,
        'env_config_type': [env_config_type] * num_trials,
        'env_maps': np.arange(1, num_trials+1).tolist(),  # MATLAB indexing starts at 1
        'env_starts': np.arange(1, num_trials+1).tolist()
    }
    
    # Configure swarm parameters (same for all runs, will be set per individual run)
    # Note: num_agents will be set individually per run since it varies
    swarm_config = {
        'num_agents': int(num_agents[0]),  # Will be overridden per run
        'd_ref': 10.0,
        'v_ref': 6.0, 
        'r': swarm_r
    }
    
    # Configure failure parameters (no failures for functional test)
    sim_fail_config = {
        'failure_option': [0] * num_trials,  # No failures
        'fail_agent_num': [float('nan')] * num_trials,
        'fail_time': [float('nan')] * num_trials,
        'fail_power': [float('nan')] * num_trials,
        'fail_in_swarm': [True] * num_trials  # Not used when no failures
    }
    
    # Configure execution flags
    flags = {
        'output_all_states': True,
        'save': False,  # Don't save individual MATLAB results
        'viz': False,
        'viz_video': False,
        'dynamic_end': True,
        'run_parallel': flag_parallel,
        'cooperating': cooperating,
        'ACTIVE_ARENA_WALLS': True
    }
    
    # Create results directory
    folder_path = create_results_dir(run_type)
    results_dir = str(folder_path)
    
    # Collect results for each agent count
    all_results = []
    
    for i, n_agents in enumerate(num_agents):
        print(f"Running functional test(s): {num_trials} trials for {n_agents} agents")
        
        # Update swarm config for current agent count
        current_swarm_config = swarm_config.copy()
        current_swarm_config['num_agents'] = int(n_agents)
        
        # Run simulation
        result = sim.failure_modes_simulate(
            sim_config=sim_config,
            swarm_config=current_swarm_config,
            sim_fail_config=sim_fail_config,
            results_dir=results_dir,
            flags=flags
        )
        
        all_results.append({
            'num_agents': n_agents,
            'result': result,
            'sim_params': {
                'end_time': end_time,
                'start_x': start_x, 
                'end_x': end_x,
                'env_config_type': env_config_type
            }
        })
    
    # Process results into dataframe format
    dataframe_rows = []
    
    for i, res_data in enumerate(all_results):
        n_agents = res_data['num_agents']
        result = res_data['result']
        sim_params = res_data['sim_params']
        
        # Process each trial within this agent count result
        for trial_idx in range(num_trials):
            
            # Extract timing data from MATLAB result for this trial
            if ('times_start' in result) and ('times_end' in result):
                times_start = np.array(result['times_start'][trial_idx]).flatten()
                times_end = np.array(result['times_end'][trial_idx]).flatten()
                
                # Calculate flight time as difference between first over the start and last over the end
                flight_start_val = np.min(times_start) if len(times_start) > 0 else np.nan
                flight_end_val = np.max(times_end) if len(times_end) > 0 else np.nan
                flight_time_val = flight_end_val - flight_start_val if not (np.isnan(flight_start_val) or np.isnan(flight_end_val)) else np.nan
            else:
                flight_time_val = np.nan
                flight_start_val = np.nan 
                flight_end_val = np.nan
            
            # Extract collision data from MATLAB result for this trial
            collision_occurred = False
            collision_type = 'not_specified'
            
            if 'collision' in result and trial_idx < len(result['collision']):
                collision_array = result['collision'][trial_idx]
                # Convert MATLAB logical array to numpy array and check for any True values
                collision_data = np.array(collision_array)
                if collision_data.size > 0 and np.any(collision_data):
                    collision_occurred = True
                    collision_type = 'detected'
            
            # Create row for dataframe
            row = {
                'trial': sim_config['env_maps'][trial_idx],
                'seed_num': np.nan,  # No randomization
                'pop_size': 0,  # Not applicable
                'num_gen': 0,  # Not applicable
                'end_time': sim_params['end_time'],
                'start_x': sim_params['start_x'],
                'end_x': sim_params['end_x'],
                'num_agents': int(n_agents),
                'fail_agent_num': 0,  # No failures
                'fail_time': np.nan,  # No failures
                'fail_in_swarm': False,  # No failures
                'coll_result_occur': collision_occurred,  # Based on actual collision data
                'coll_result_type': collision_type,  # Based on actual collision detection
                'flight_time': round(flight_time_val,2),
                'flight_start': round(flight_start_val,2),
                'flight_end': round(flight_end_val,2)
            }
            
            dataframe_rows.append(row)
    
    # Create dataframe manually
    df = pd.DataFrame(dataframe_rows)
    
    # Ensure proper data types
    df = df.astype({
        'trial': int,
        'seed_num': float,
        'pop_size': int,
        'num_gen': int,
        'end_time': float,
        'start_x': float,
        'end_x': float,
        'num_agents': int,
        'fail_agent_num': int,
        'fail_time': float,
        'fail_in_swarm': bool,
        'coll_result_occur': bool,
        'coll_result_type': str,
        'flight_time': float,
        'flight_start': float,
        'flight_end': float
    })
    
    # Save results
    pickle_path = folder_path / "dataframe.pkl"
    df.to_pickle(pickle_path)
    csv_path = folder_path / "dataframe.csv"
    df.to_csv(csv_path, index=False, na_rep='NaN')
    
    # Shutdown simulator
    sim.shutdown_engine()
    
    return folder_path
