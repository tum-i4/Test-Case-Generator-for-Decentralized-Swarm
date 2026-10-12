"""
Calculations for research question: separation distance and fitness.
"""
import numpy as np
import pandas as pd
import pickle
import scipy.stats as stats
from pathlib import Path
from plotnine import (ggplot, aes, geom_point, scale_color_manual, scale_shape_manual, theme, theme_bw, geom_smooth,
                      scale_x_continuous, scale_y_continuous, labs
                      )
from src.postprocessing.data_loader import load_pickled_trial_results
from src.utils import create_results_dir


# Define color - same as plot_dumbbell_fitness
colors = {
    'PowerCutoff': '#bb7438',
    'VelocityX1': '#72ac5c',
    'VelocityX2': '#7f64b9',
    'VelocityX3': '#b94b75',
}

# Define distinct shapes for each type - same as plot_dumbbell_fitness
shapes = {
    'PowerCutoff': 'o',  # Circle (baseline)
    'VelocityX1': 's',  # Square
    'VelocityX2': '^',  # Triangle up
    'VelocityX3': 'D',  # Diamond
}


def extract_trial_data(out):
    """
    Extract and prepare all necessary data from a single trial output.

    This function extracts raw data and computes all relevant time indices for
    analysis, including scenario boundaries and failure windows.

    Args:
        out: Trial output dictionary

    Returns:
        dict: simulation output data and metadata
    """
    tol_time = 1E-5
    trial_num = out['trial']

    # Extract position coordinates (NED, [m]) and velocity
    pos_ned = np.array(out['pos_ned'][0])
    vel_ned = np.array(out['vel_ned'][0])

    # Extract times
    sim_time = np.array(out['time'][0])
    start_time = np.min(np.array(out['times_start'][0], dtype=float).flatten())
    end_time = np.max(sim_time)

    # Extract goal
    if out['p_swarm'][0]['is_active_migration'] and not out['p_swarm'][0]['is_active_goal']:
        u_ref = np.array(out['p_swarm'][0]['u_ref'], dtype=float).flatten()
        speed_ref = out['p_swarm'][0]['v_ref']
        v_ref = u_ref * speed_ref
    else:
        raise ValueError('Goal specification is not handled in current processing.')

    # Get fail agent number (velocX2 and X3 cases should be using same agent for all failures)
    fail_num = np.array(out['fail_config'][0]['fail_agent_num'], dtype=int)
    fail_time = np.array(out['fail_config'][0]['fail_time'])
    if fail_num.size > 1:
        if not np.all(fail_num == fail_num[0]):
            raise ValueError("Multiple, different fail agents have been specified in this trial; this is not yet accounted for here.")
        else:
            fail_num = fail_num[0][0]
            fail_time = min(fail_time[0])  # Take chronologically first failure time

    # Number of agents from column count
    num_agents = pos_ned.shape[1] // 3

    # in-sim states of UAVs
    in_sim = np.array(out['in_sim'][0], dtype=bool)

    # Reshape coordinates to 3d matrix [timesteps, agents, coordinates]
    pos_reshaped = pos_ned.reshape(pos_ned.shape[0], num_agents, 3)
    vel_reshaped = vel_ned.reshape(vel_ned.shape[0], num_agents, 3)

    # Get rerun fitness
    rerun_f = out['rerun_f'][0]

    # Calculate scenario start index (first timestep at or after start_time)
    scenario_start_idx_array = np.where(abs(sim_time-start_time) < tol_time)[0]
    if len(scenario_start_idx_array) > 0:
        scenario_start_idx = scenario_start_idx_array[0]
    else:
        raise ValueError(f"No start time detected for trial {trial_num}")

    # Get scenario end index
    scenario_end_idx = np.where(sim_time == end_time)[0][0]

    # Calculate failure start index
    fail_idx_array = np.where(abs(sim_time-fail_time) < tol_time)[0]
    if len(fail_idx_array) == 0:
        raise ValueError(f"No time steps at or after failure time for trial {trial_num}")

    fail_idx = fail_idx_array[0]

    # Calculate failure window end (when failing agent exits simulation)
    fail_agent_in_sim = in_sim[fail_idx:, fail_num-1]
    fail_agent_last_indices = np.where(fail_agent_in_sim)[0]
    fail_agent_last_relative_idx = fail_agent_last_indices[-1]
    failure_window_end_idx = fail_idx + fail_agent_last_relative_idx

    return {
        'trial_num': trial_num,
        'pos_reshaped': pos_reshaped,
        'vel_reshaped': vel_reshaped,
        'sim_time': sim_time,
        'in_sim': in_sim,
        'fail_num': fail_num,
        'fail_time': fail_time,
        'num_agents': num_agents,
        'v_ref': v_ref,
        'speed_ref': speed_ref,
        'rerun_f': rerun_f,
        'start_time': start_time,
        'end_time': end_time,
        'scenario_start_idx': scenario_start_idx,
        'scenario_end_idx': scenario_end_idx,
        'fail_start_idx': fail_idx,
        'fail_end_idx': failure_window_end_idx,
    }


def calc_separation_distances_over_time(pos_reshaped, in_sim, fail_num, fail_idx, end_idx, num_agents):
    """
    Calculate average separation distance between fail agent and teammates over time.

    Args:
        pos_reshaped: Position array [timesteps, agents, coordinates]
        in_sim: Boolean array of agent in-simulation status [timesteps, agents]
        fail_num: Failing agent number (1-indexed)
        fail_idx: Start index of time window
        end_idx: End index of time window
        num_agents: Total number of agents

    Returns:
        list: Average distances per timestep (only timesteps with teammates)
    """
    timestep_avg_distances = []

    for i in range(fail_idx, end_idx):
        distances_at_i = []
        for agent_idx in range(num_agents):
            if agent_idx != (fail_num-1) and in_sim[i, agent_idx]:
                # This agent is a teammate and is in simulation at this time
                dist = np.linalg.norm(
                    pos_reshaped[i, fail_num-1, :] - pos_reshaped[i, agent_idx, :]
                )
                distances_at_i.append(dist)

        # Average separation distance between fail agent and teammates at this time step
        if len(distances_at_i) > 0:
            timestep_avg_distances.append(np.mean(distances_at_i))

    return timestep_avg_distances


def calc_order_single_timestep(velocities):
    """
    Calculate order metric for a list of velocity vectors.

    Order is defined as the mean cosine similarity between all pairs of velocity vectors,
    representing how aligned the velocities are (metric demonstrated by Soria et al.).

    Args:
        velocities: list of velocity vectors (3D numpy arrays)
    """
    N = len(velocities)
    if N < 2:
        return np.nan

    order_sum = 0.0
    for i in range(N):
        for j in range(N):
            if i != j:
                v_i = velocities[i]
                v_j = velocities[j]
                norm_i = np.linalg.norm(v_i)
                norm_j = np.linalg.norm(v_j)
                if norm_i > 0 and norm_j > 0:
                    cos_similarity = np.dot(v_i, v_j) / (norm_i * norm_j)
                    order_sum += cos_similarity

    return (1.0 / (N * (N - 1))) * order_sum


def calc_order_over_time(vel_reshaped, in_sim, start_idx, end_idx, num_agents,
                         exclude_fail_agent=False, fail_agent_num=None, fail_active_start_idx=None):
    """
    Calculate order metric (velocity alignment) over time for all active agents.

    Args:
        vel_reshaped: Velocity array [timesteps, agents, coordinates]
        in_sim: Boolean array of agent in-simulation status [timesteps, agents]
        start_idx: Start index of time window
        end_idx: End index of time window
        num_agents: Total number of agents
        exclude_fail_agent: If True, exclude failing agent from order calculation during failure window
        fail_agent_num: Failing agent number (1-indexed), required if exclude_fail_agent=True
        fail_active_start_idx: Starting index when fail agent should be excluded

    Returns:
        list: Order metrics per timestep (only timesteps with >= 2 active agents)
    """
    timestep_order_metrics = []

    # Set defaults for failure window if not provided
    if exclude_fail_agent:
        if (fail_agent_num is None) or (fail_active_start_idx is None):
            raise ValueError('If excluding fail agent from order calculations, fail agent information must be provided.')

    # Iterate through time steps
    for i in range(start_idx, end_idx+1):  # i will be used to index rows
        active_velocities = []
        # Collect velocities of all active agents
        for agent_idx in range(num_agents):
            # Skip failing agent if we're in the failure window and exclusion is enabled
            if exclude_fail_agent:
                if (agent_idx == (fail_agent_num-1)) and (fail_active_start_idx <= i):  #
                    continue

            if in_sim[i, agent_idx]:
                active_velocities.append(vel_reshaped[i, agent_idx, :])

        # Calculate order metric for this time step
        order_metric = calc_order_single_timestep(active_velocities)  # note that if < 2 agents, function will return nan
        timestep_order_metrics.append(order_metric)

    return timestep_order_metrics


def calc_fail_alignment_over_time(vel_reshaped, in_sim, fail_num, fail_idx, end_idx, num_agents):
    """
    Calculate fail agent velocity alignment with teammates over time.

    Measures how aligned the failing agent's velocity is with its teammates' velocities.

    Args:
        vel_reshaped: Velocity array [timesteps, agents, coordinates]
        in_sim: Boolean array of agent in-simulation status [timesteps, agents]
        fail_num: Failing agent number (1-indexed)
        fail_idx: Start index of time window
        end_idx: End index of time window
        num_agents: Total number of agents

    Returns:
        list: Alignment metrics per timestep (only when fail agent is active with teammates)
    """
    timestep_fail_align_metrics = []

    for i in range(fail_idx, end_idx):
        # Calculate fail agent alignment metric (fail agent vs teammates only)
        if in_sim[i, fail_num-1]:  # fail agent is in simulation
            v_fail = vel_reshaped[i, fail_num-1, :]  # fail agent's velocity
            norm_fail = np.linalg.norm(v_fail)  # fail agent's speed

            # Continue only if fail agent has speed > 0
            if norm_fail > 0:
                fail_align_sum = 0.0
                teammate_count = 0

                # Iterate through teammates
                for agent_idx in range(num_agents):
                    if agent_idx != (fail_num-1) and in_sim[i, agent_idx]:  # only if teammate is still in_sim
                        v_teammate = vel_reshaped[i, agent_idx, :]  # velocity
                        norm_teammate = np.linalg.norm(v_teammate)  # speed

                        # Add to summation if applicable
                        if norm_teammate > 0:
                            cos_similarity = np.dot(v_fail, v_teammate) / (norm_fail * norm_teammate)
                            fail_align_sum += cos_similarity
                            teammate_count += 1

                # Divide similarity calc by # of in_sim teammates
                if teammate_count > 0:
                    fail_align_metric = fail_align_sum / teammate_count
                    timestep_fail_align_metrics.append(fail_align_metric)

    return timestep_fail_align_metrics


def calc_goal_alignment_over_time(vel_reshaped, in_sim, v_ref, speed_ref, fail_num, fail_idx, end_idx, num_agents):
    """
    Calculate teammate velocity alignment with goal velocity over time.

    Measures how aligned non-failing agents' velocities are with the goal direction.

    Args:
        vel_reshaped: Velocity array [timesteps, agents, coordinates]
        in_sim: Boolean array of agent in-simulation status [timesteps, agents]
        v_ref: Reference goal velocity vector
        speed_ref: Reference goal speed
        fail_num: Failing agent number (1-indexed, excluded from calculation)
        fail_idx: Start index of time window
        end_idx: End index of time window
        num_agents: Total number of agents

    Returns:
        list: Goal alignment metrics per timestep (only when teammates are active)
    """
    timestep_goal_align_metrics = []

    for t in range(fail_idx, end_idx):
        # Calculate goal alignment metric (teammates vs goal velocity)
        if speed_ref > 0:
            goal_align_sum = 0.0
            teammate_count = 0

            for agent_idx in range(num_agents):
                if agent_idx != (fail_num-1) and in_sim[t, agent_idx]:
                    v_teammate = vel_reshaped[t, agent_idx, :]
                    norm_teammate = np.linalg.norm(v_teammate)

                    if norm_teammate > 0:
                        cos_similarity = np.dot(v_teammate, v_ref) / (norm_teammate * speed_ref)
                        goal_align_sum += cos_similarity
                        teammate_count += 1

            if teammate_count > 0:
                goal_align_metric = goal_align_sum / teammate_count
                timestep_goal_align_metrics.append(goal_align_metric)

    return timestep_goal_align_metrics


def calc_avg_motion_metrics(out_all):
    """
    Calculate motion metrics for all trials.

    Iterates through trials and calculates various motion metrics including:
    - Average separation distance between fail agent and teammates
    - Order metric (velocity alignment) during failure window
    - Order metric from failure start to scenario end
    - Fail agent alignment with teammates
    - Teammate alignment with goal velocity

    Args:
        out_all: List of trial output dictionaries

    Returns:
        DataFrame with columns: trial, avg_dist, avg_order, avg_order_to_end,
                                avg_fail_align, avg_goal_align, rerun_f
    """
    results = []

    for out in out_all:
        # Extract trial data (includes all time indices)
        data = extract_trial_data(out)
        trial_num = data['trial_num']

        # Extract pre-computed indices
        fail_start_idx = data['fail_start_idx']
        fail_end_idx = data['fail_end_idx']
        scenario_end_idx = data['scenario_end_idx']
        scenario_start_idx = data['scenario_start_idx']

        # Calculate each metric for each time step over the failure window (failure start until failing agent exits)
        list_fail_window_avg_distances = calc_separation_distances_over_time(
            data['pos_reshaped'], data['in_sim'], data['fail_num'],
            fail_start_idx, fail_end_idx, data['num_agents']
        )

        list_fail_window_order_metrics = calc_order_over_time(
            data['vel_reshaped'], data['in_sim'],
            fail_start_idx, fail_end_idx, data['num_agents'],
            exclude_fail_agent=True, fail_agent_num=data['fail_num'], fail_active_start_idx=scenario_start_idx
        )

        list_fail_window_align_metrics = calc_fail_alignment_over_time(
            data['vel_reshaped'], data['in_sim'], data['fail_num'],
            fail_start_idx, fail_end_idx, data['num_agents']
        )

        list_fail_window_goal_align_metrics = calc_goal_alignment_over_time(
            data['vel_reshaped'], data['in_sim'], data['v_ref'], data['speed_ref'],
            data['fail_num'], fail_start_idx, fail_end_idx, data['num_agents']
        )

        # Calculate order for time steps from failure start to scenario end
        list_fail_window_order_to_end = calc_order_over_time(
            data['vel_reshaped'], data['in_sim'],
            fail_start_idx, scenario_end_idx, data['num_agents'],
            exclude_fail_agent=True, fail_agent_num=data['fail_num'], fail_active_start_idx=scenario_start_idx
        )

        # Calculate order for all time steps, i.e., from scenario start to scenario end
        list_all_time_order = calc_order_over_time(
            data['vel_reshaped'], data['in_sim'],
            scenario_start_idx, scenario_end_idx, data['num_agents'],
            exclude_fail_agent=True, fail_agent_num=data['fail_num'], fail_active_start_idx=scenario_start_idx
        )

        # For each: average across list of time steps
        fail_window_avg_dist = np.nanmean(list_fail_window_avg_distances)
        fail_window_avg_order = np.nanmean(list_fail_window_order_metrics)
        fail_window_avg_fail_align = np.nanmean(list_fail_window_align_metrics)
        fail_window_avg_goal_align = np.nanmean(list_fail_window_goal_align_metrics)
        fail_to_end_avg_order = np.nanmean(list_fail_window_order_to_end)
        all_time_avg_order = np.nanmean(list_all_time_order)

        # Calculate number of agents that did not reach end for this trial
        num_dnf = np.sum(np.isinf(np.array(out['times_end'][0], dtype=float)))

        # Append results
        results.append({
            'trial': trial_num,
            'fail_window_avg_dist': fail_window_avg_dist,
            'fail_window_avg_order': fail_window_avg_order,
            'fail_to_end_avg_order': fail_to_end_avg_order,
            'fail_window_avg_fail_align': fail_window_avg_fail_align,
            'fail_window_avg_goal_align': fail_window_avg_goal_align,
            'all_time_avg_order': all_time_avg_order,
            'num_dnf': num_dnf,
            'rerun_f': data['rerun_f']
        })

    # Convert to dataframe
    df = pd.DataFrame(results)

    return df


def figure_fitness_vs_metric(df, var, metric_name, x_min=None, x_max=None, y_min=None, y_max=None):
    """
    Creates a scatter plot with average distance on x-axis and fitness on y-axis.
    Points are colored and shaped by failure type.

    Parameters:
    df : DataFrame with columns: trial, avg_dist, rerun_f or fitness, type, obs, algo, r
    var : str, the variable to plot on x-axis (default: 'avg_dist')

    Returns:
    plotnine plot object
    """
    # Ensure we don't modify original
    df = df.copy()

    # Convert rerun_f to fitness column if needed
    if 'rerun_f' in df.columns:
        df['fitness'] = df['rerun_f'].apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)
    elif 'fitness' in df.columns:
        df['fitness'] = df['fitness'].apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)
    else:
        raise ValueError("DataFrame must contain either 'rerun_f' or 'fitness' column")

    # Get unique failure types
    failure_types = sorted(df['type'].unique())

    # Create the scatter plot with multiple layers
    plt = (
            ggplot(df, aes(x=var, y='fitness')) +
            # geom_smooth(method='lm', se=True, alpha=0.2, color='gray') +  # all data
            geom_smooth(aes(color='type'), method='lm', se=True, alpha=0.2, show_legend=False) +  # by type
            geom_point(
                aes(color='type', shape='type'),
                size=3,
                alpha=0.5
            ) +
            scale_color_manual(
                values={k: v for k, v in colors.items() if k in failure_types},  # color palette specified with 'colors'
                labels={
                    'PowerCutoff': 'PowerCutoff',
                    'VelocityX1': 'NavX1',
                    'VelocityX2': 'NavX2',
                    'VelocityX3': 'NavX3'
                },
                name='Failure Type'
            ) +
            scale_shape_manual(
                values={k: v for k, v in shapes.items() if k in failure_types},  # shapes specified with 'shapes'
                labels={
                    'PowerCutoff': 'PowerCutoff',
                    'VelocityX1': 'NavX1',
                    'VelocityX2': 'NavX2',
                    'VelocityX3': 'NavX3'
                },
                name='Failure Type'
            ) +
            labs(
                x=f'{metric_name}',
                y='Fitness'
            ) +
            scale_x_continuous(
                limits=(x_min, x_max) if x_min is not None and x_max is not None else None
            ) +
            scale_y_continuous(
                limits=(y_min, y_max) if y_min is not None and y_max is not None else None
            ) +
            theme_bw()
    )

    return plt


def figure_fitness_vs_metric_ranks(df, var, metric_name):
    """
    Creates a scatter plot using ranks instead of raw values. Ranks are calculated separately within each failure type.

    Parameters:
    df : DataFrame with columns: trial, var, rerun_f or fitness, type, obs, algo, r
    var : str, the variable to plot on x-axis

    Returns:
    plotnine plot object
    """
    # Ensure we don't modify original
    df = df.copy()

    # Convert rerun_f to fitness column if needed
    if 'rerun_f' in df.columns:
        df['fitness'] = df['rerun_f'].apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)
    elif 'fitness' in df.columns:
        df['fitness'] = df['fitness'].apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)
    else:
        raise ValueError("DataFrame must contain either 'rerun_f' or 'fitness' column")

    # Calculate ranks within each failure type
    # Higher values get higher ranks (rankdata default behavior)
    df['fitness_rank'] = df.groupby('type')['fitness'].transform(
        lambda x: stats.rankdata(x, method='average')
    )
    df[f'{var}_rank'] = df.groupby('type')[var].transform(
        lambda x: stats.rankdata(x, method='average')
    )

    # Set axis limits based on maximum rank within any failure type
    max_rank_per_type = df.groupby('type').size().max()
    x_lims = (0.5, max_rank_per_type + 0.5)
    y_lims = (0.5, max_rank_per_type + 0.5)

    # Get unique failure types
    failure_types = sorted(df['type'].unique())

    # Create the scatter plot with multiple layers
    plt = (
        ggplot(df, aes(x=f'{var}_rank', y='fitness_rank')) +
        # Add trend lines by type
        geom_smooth(aes(color='type'), method='lm', se=True, alpha=0.2, show_legend=False) +
        geom_point(
            aes(color='type', shape='type'),
            size=3,
            alpha=0.6
        ) +
        scale_color_manual(
            values={k: v for k, v in colors.items() if k in failure_types},
            labels={
                'PowerCutoff': 'PowerCutoff',
                'VelocityX1': 'NavX1',
                'VelocityX2': 'NavX2',
                'VelocityX3': 'NavX3'
            },
            name='Failure Model'
        ) +
        scale_shape_manual(
            values={k: v for k, v in shapes.items() if k in failure_types},
            labels={
                'PowerCutoff': 'PowerCutoff',
                'VelocityX1': 'NavX1',
                'VelocityX2': 'NavX2',
                'VelocityX3': 'NavX3'
            },
            name='Failure Model'
        ) +
        scale_x_continuous(limits=x_lims) +
        scale_y_continuous(limits=y_lims) +
        labs(
            x=f'{metric_name} Rank',
            y='Fitness Rank'
        ) +
        theme_bw() +
        theme(legend_position='top')
    )

    return plt


def run_rq_separation(config_dict, key_out_sim="out_sim", key_metrics="df_motion_metrics"):
    # Iterate through experiments and either load sim out or inform user that it does not exist
    all_results = []
    for key, value in config_dict.items():
        print(f"\nRQ compare separation: checking for data for {key}...")
        if value["path_sim"] is not None:
            # Check if pre-saved dataframes already exist.
            df = load_pickled_trial_results(value["path_sim"], key_metrics)

            if df is None:  # Pre-saved dataframes do not exist
                # Load sim output data and calculate motion-based metrics, then save to reduce time in the future.
                print("\tMetrics have not been calculated yet, running calculations now...")
                out_all = load_pickled_trial_results(value["path_sim"], key_out_sim)

                # For each config, calculate metrics between the failing agent and the rest of the swarm
                df = calc_avg_motion_metrics(out_all[0])

                # save to dataframe with: type, obs, algo, r, trial_num, rerun_f
                df['type'] = value['type']
                df['obs'] = value['obs']
                df['algo'] = value['algo']
                df['r'] = value['r']
                df_filename = f"{key}_df_motion_metrics_all_trials.pkl"
                df_save_path = Path(value["path_sim"]) / df_filename
                with open(df_save_path, "wb") as f:
                    pickle.dump(df, f)
                print("\tCalculations completed.")

            else:
                df = df[0]
                print("\tPre-calculated metrics loaded.")

        else:
            print(f"\tNo sim output save path specified for {key}.")
            continue

        # Append
        all_results.append(df)
        print(f"\tLoading/processing for {key} completed.")

    # Check if empty
    if not all_results:
        # Exit early
        print("No sim out data found, analysis cannot be run. Exiting...")
        return None
    else:
        # Combine into one df
        df = pd.concat(all_results, ignore_index=True)

    # Create results directory
    folder_path = create_results_dir('RQ_separation', prefix='post')

    # Create figures: fitness vs. average distance from centroid of teammates; color/shape by failure type
    # Get unique categories of data
    print("\nRQ analysis: Separation vs. Fitness")
    type_labels = df['type'].unique()
    print(f"\tPossible type: {type_labels}")
    r_labels = df['r'].unique()
    print(f"\tPossible r: {r_labels}")
    obs_labels = df['obs'].unique()
    print(f"\tPossible obs: {obs_labels}")
    algo_labels = df['algo'].unique()
    print(f"\tPossible algo: {algo_labels}")
    print("\n")

    # Iterate through algo, obs, and r (unique figures)
    for temp_algo in algo_labels:
        for temp_obs in obs_labels:
            for temp_r in r_labels:
                # Filter data for this combination
                df_filtered = df[(df['obs'] == temp_obs) & (df['r'] == temp_r) & (df['algo'] == temp_algo)].copy()

                # Convert fitness to float if needed
                if 'rerun_f' in df_filtered.columns:
                    df_filtered['fitness'] = df_filtered['rerun_f'].apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)
                elif 'fitness' in df_filtered.columns:
                    df_filtered['fitness'] = df_filtered['fitness'].apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)

                # Calculate Spearman correlations for all metrics
                print(f"\nSpearman Correlations for r={temp_r}, obs={temp_obs}, algo={temp_algo}:")
                print("=" * 80)

                # List metrics
                metrics = ['fail_window_avg_dist', 'fail_window_avg_order', 'fail_to_end_avg_order',
                           'fail_window_avg_fail_align', 'fail_window_avg_goal_align', 'all_time_avg_order']
                metric_names = ['Average Separation Distance to Fail Agent (Failure Window)', 'Average Order of Non-Failing (Failure Window)', 'Average Order of Non-Failing (Failure Start to Sim End)',
                                'Alignment to Fail Agent Velocity (Failure Window)', 'Alignment to Goal (Excl. Fail Agent) (Failure Window)','Order (Agents Without Failure Injection)']

                for metric, metric_name in zip(metrics, metric_names):
                    # Create and save figure
                    # if metric == 'all_time_avg_order':
                    #     temp_plt = figure_fitness_vs_metric(df_filtered, var=metric, metric_name=metric_name,
                    #                                         x_min=0.75, x_max=1.0, y_min=4.0, y_max=11.0)
                    # else:
                    #     temp_plt = figure_fitness_vs_metric(df_filtered, var=metric, metric_name=metric_name)
                    temp_plt = figure_fitness_vs_metric_ranks(df_filtered, var=metric, metric_name=metric_name)
                    temp_plt.save(folder_path / f"fitness_vs_{metric}_{temp_r}_obs{temp_obs}_{temp_algo}.png", dpi=300)

                    # Run Spearman correlation calculation
                    if len(df_filtered) >= 3:  # (need at least 3 points for meaningful correlation)
                        corr, p_value = stats.spearmanr(df_filtered['fitness'], df_filtered[metric])
                        med = np.median(df_filtered[metric])
                        print(f"  {metric_name:25s}: rho = {corr:7.4f}, p = {p_value:.4e} (n = {len(df_filtered)}); median = {med:.3f}")
                    else:
                        print(f"  {metric_name:25s}: Insufficient data (n = {len(df_filtered)})")

                    # Calculate Spearman correlation for each type separately
                    print(f"    Per-type correlations for {metric_name}:")
                    for failure_type in sorted(df_filtered['type'].unique()):
                        df_type = df_filtered[df_filtered['type'] == failure_type]
                        if len(df_type) >= 3:
                            corr_type, p_value_type = stats.spearmanr(df_type['fitness'], df_type[metric])
                            med_type = np.median(df_type[metric])
                            print(f"      {failure_type:15s}: rho = {corr_type:7.4f}, p = {p_value_type:.4e} (n = {len(df_type)}); median = {med_type:.3f}")
                        else:
                            print(f"      {failure_type:15s}: Insufficient data (n = {len(df_type)})")

                # Also print the number of agents that did not reach end per failure type
                print(f"  Number of agents that did not reach end (DNF), by failure type: Average & Median.")
                for failure_type in sorted(df_filtered['type'].unique()):
                    df_type = df_filtered[df_filtered['type'] == failure_type]
                    dnf_med = np.median(df_type['num_dnf'])
                    dnf_mean = np.mean(df_type['num_dnf'])
                    print(f"    {failure_type:15s}: median = {dnf_med:.3f}, mean = {dnf_mean:.3f} (n = {len(df_type)})")

                print("=" * 80)

    return None