"""
Visualizations and checks to get a sense of whether the search has reached convergence.
"""

import pandas as pd
from plotnine import ggplot, aes, geom_line, ylim
from src.utils import create_results_dir
from src.postprocessing.data_loader import load_pickled_trial_results

def plot_fmin_all_trials(df, ymin=0, ymax=12, line_alpha=0.5):
    #
    # Ensure we don't modify original
    df = df.copy()

    # Create plot with gen on x-axis, f_min on y-axis, grouped by trial
    plt = (ggplot(df, aes(x='gen', y='f_min', group='trial'))
           + geom_line(color='grey', alpha=line_alpha)
           + ylim(ymin, ymax))

    return plt

def run_check_convergence(config_dict):
    """
    For each config in config_dict:
      - Plot minimum f value over generations for all trials
      - Find, for each trial, the generation when the best (minimum) f value first occurs
      - Compile a summary table across trials per config:
            mean/median generation where best f is found,
            mean/median best f value,
            number of trials
            median best f_min at 20%, 40%, 60%, 80%, 100% of total generations
      - Print and save this summary table as CSV in the results directory
    """
    # Create results directory
    folder_path = create_results_dir('check_convergence', prefix='post')

    summary_rows = []  # Collect stats per config

    # Process each config separately
    for key, value in config_dict.items():
        # Load results for all trials for this config
        res_combined = load_pickled_trial_results(value["path"], "result", "checkpoint")
        
        # Get max generation across all trials for THIS config
        max_gen = 0
        for trial_res in res_combined:
            num_gens = len(trial_res['convergence_history'])
            if num_gens > max_gen:
                max_gen = num_gens
        
        total_generations = max_gen

        # Each row: one trial-gen pair
        df_rows = []
        for trial_res in res_combined:

            # Iterate through saved history of best f (this is created in run_optim_failure_modes)
            for i_gen, f_gen in enumerate(trial_res['convergence_history']):
                df_rows.append({
                    'trial': trial_res['trial'],
                    'gen': i_gen+1,
                    'f_min': f_gen
                })

        # Build DataFrame for this config
        df = pd.DataFrame(df_rows)
        df = df.sort_values(by=['trial', 'gen'])

        # Plot: f_min over generations for each trial
        plt = plot_fmin_all_trials(df)
        plt_name = f"convergence_{key}_{value['type']}_obs{value['obs']}_algo{value['algo']}_{value['r']}.png"
        plt.save(folder_path / plt_name, dpi=300)

        # Compute per-trial stats: when and what is the best f_min
        generations_found = []  # generation index where best f_min first occurs
        best_f_mins = []        # value of best f_min per trial
        
        # For milestone calculations: collect f_min values at each milestone across all trials
        milestone_percentages = [20, 40, 60, 80, 100]
        milestone_indices = [round(total_generations * pct / 100) for pct in milestone_percentages]
        # Convert to 0-indexed (generation numbers in df are 1-indexed, but convergence_history is 0-indexed)
        milestone_indices_0based = [idx - 1 for idx in milestone_indices]
        
        milestone_fmins = {pct: [] for pct in milestone_percentages}
        
        for trial, group in df.groupby('trial'):
            # Check for NaN values in f_min
            if group['f_min'].isna().any():
                raise ValueError(f"Config '{key}', trial '{trial}' contains NaN values in f_min")
            
            min_f = group['f_min'].min()
            min_row = group[group['f_min'] == min_f].iloc[0]  # first occurrence
            generations_found.append(min_row['gen'])
            best_f_mins.append(min_f)
            
            # Get the trial's max generation
            group_sorted = group.sort_values('gen')
            trial_max_gen = group_sorted['gen'].max()
            trial_best_fmin = group_sorted[group_sorted['gen'] == trial_max_gen]['f_min'].values[0]
            
            # Extract f_min values at each milestone for this trial
            for pct, target_gen in zip(milestone_percentages, milestone_indices):
                milestone_row = group_sorted[group_sorted['gen'] == target_gen]
                if not milestone_row.empty:
                    # Trial has data at this milestone
                    milestone_fmins[pct].append(milestone_row['f_min'].values[0])
                elif target_gen > trial_max_gen:
                    # Trial ended before this milestone, carry forward its best result
                    milestone_fmins[pct].append(trial_best_fmin)
                else:
                    # This should not happen, but raise error for safety
                    raise ValueError(f"Config '{key}', trial '{trial}': generation {target_gen} not found but is within trial's range")
        
        num_trials = len(generations_found)

        # Compute statistics (mean/median)
        mean_gen = pd.Series(generations_found).mean()
        median_gen = pd.Series(generations_found).median()
        mean_f_min = pd.Series(best_f_mins).mean()
        median_f_min = pd.Series(best_f_mins).median()
        
        # Calculate median f_min at each milestone
        median_milestone_fmins = {pct: pd.Series(milestone_fmins[pct]).median() 
                                  for pct in milestone_percentages}

        # Build summary row for table
        summary_row = {
            "config_key": key,
            "type": value.get("type", ""),
            "algo": value.get("algo", ""),
            "r": value.get("r", ""),
            "obs": value.get("obs", ""),
            "num_trials": num_trials,
            "mean_gen_found": mean_gen,
            "median_gen_found": median_gen,
            "mean_best_fmin": mean_f_min,
            "median_best_fmin": median_f_min,
            "median_best_fmin_20pct": median_milestone_fmins[20],
            "median_best_fmin_40pct": median_milestone_fmins[40],
            "median_best_fmin_60pct": median_milestone_fmins[60],
            "median_best_fmin_80pct": median_milestone_fmins[80],
            "median_best_fmin_100pct": median_milestone_fmins[100]
        }
        summary_rows.append(summary_row)

    # After all configs: print and save summary table
    summary_df = pd.DataFrame(summary_rows)
    print("\nConvergence summary table:")
    print(summary_df.to_string(index=False))
    csv_name = "convergence_summary_generations.csv"
    summary_df.to_csv(folder_path / csv_name, index=False)

