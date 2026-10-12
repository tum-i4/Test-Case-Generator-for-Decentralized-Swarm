"""
Calculations for research question: performance degradation.
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
from plotnine import (ggplot, aes, geom_point, geom_boxplot, position_jitterdodge, theme_bw, position_dodge,
                      stat_summary, scale_y_continuous, scale_x_continuous, geom_line,
                      scale_shape_manual, scale_color_manual, theme_matplotlib, theme, scale_x_discrete,
                      scale_color_discrete, xlab, ylab
                      )
from plotnine.themes.themeable import legend_position

from src.utils import create_results_dir
import scikit_posthocs as sp


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


def plot_scatter_fitness(df, y_min=0.0, y_max=12.0):
    """
    df   = dataframe with results from multiple trials of GA and PSO searches; should be combination of dfs from different failure modes.
    """
    # Ensure we don't modify original
    df = df.copy()

    # Each row of fitness column is an ndarray of size 1, need to convert to float
    df['fitness'] = df['fitness'].apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)

    # random seed for jitter (so that figures can be reproduced)
    np.random.seed(1)

    def median_line(y):
        med = np.median(y)
        return pd.DataFrame({'y': [med], 'ymin': [med], 'ymax': [med]})

    def median_text(y):
        med = np.median(y)
        return pd.DataFrame({'y': [med], 'label': [f'{med:.2f}']})

    # Get unique algo values and sort them (to determine left vs right positioning)
    algo_values = sorted(df['algo'].unique())

    # Create figure
    plt = (
        ggplot(df, aes(x='type', y='fitness', color='algo')) +
        geom_boxplot(
            show_legend=False,
            alpha=0.0,
            width=0.35,
            position=position_dodge(
                width=0.6)
        ) +
        geom_point(
            position=position_jitterdodge(
                jitter_width=0.1,    # Horizontal spread within each algo group
                dodge_width=0.6,     # Horizontal separation between algo groups
                jitter_height=0      # No vertical jitter - exact fitness values
            ),
            size=2,
            alpha=0.5
        ) +
        # stat_summary(
        #     aes(group='algo'),
        #     fun_data=median_line,
        #     geom='errorbar',
        #     position=position_dodge(width=0.6),
        #     width=0.5,
        #     size=1.5,
        #     color='dimgrey'
        # ) +
        # Text for left algo group (nudged left)
        stat_summary(
            aes(group='algo'),
            fun_data=median_text,
            geom='text',
            position=position_dodge(width=0.6),
            size=10,
            color='black',
            nudge_x=-0.4,
            va='center',
            angle=90,
            data=df[df['algo'] == algo_values[0]]
        ) +
        # Text for right algo group (nudged right)
        stat_summary(
            aes(group='algo'),
            fun_data=median_text,
            geom='text',
            position=position_dodge(width=0.6),
            size=10,
            color='black',
            nudge_x=0.4,
            va='center',
            angle=90,
            data=df[df['algo'] == algo_values[1]]
        ) +
        scale_y_continuous(
            limits=(y_min, y_max),
            breaks=range(int(y_min), int(y_max) + 1)
        ) +
        scale_x_discrete(
            labels={
                'PowerCutoff': 'PowerCutoff',
                'VelocityX1': 'NavX1',
                'VelocityX2': 'NavX2',
                'VelocityX3': 'NavX3'
            }
        ) +
        scale_color_discrete(
            name='[Search Method]',
            labels={
                'GA': 'GA (left)',
                'PSO': 'PSO (right)',
                'RANDOM': 'Random (right)'
            }
        ) +
        xlab('Fault Model') +
        ylab('Fitness') +
        theme_bw() +
        theme(legend_position="bottom")
    )

    # Need to update axes and legend labels

    return plt


def plot_dumbbell_fitness(df, x_min=None, x_max=None, normalize_to='PowerCutoff'):
    """
    Creates a dumbbell chart showing normalized fitness values across trials for different failure types.

    Fitness values are normalized on a trial-by-trial basis using percentage difference from a
    baseline failure type (default: PowerCutoff). For each trial, the baseline type's fitness
    value in that trial is used to calculate: ((fitness - baseline) / baseline) * 100

    Parameters:
    df : Dataframe with results from multiple trials and failure types
    x_min, x_max : X-axis limits for normalized fitness values (percentage). If None, uses data range.
    normalize_to : The failure type to use as baseline for normalization

    Returns:
    plotnine plot object
    """
    # Ensure we don't modify original
    df = df.copy()

    # Convert fitness to float if needed
    df['fitness'] = df['fitness'].apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)

    # Normalize fitness values on a trial-by-trial basis
    # For each trial, calculate difference from that trial's baseline
    def normalize_fitness(group):
        baseline_fitness = group[group['type'] == normalize_to]['fitness'].values
        if len(baseline_fitness) > 0:
            baseline = baseline_fitness[0]
            # group['fitness_diff'] = ((group['fitness'] - baseline) / baseline) * 100  # Percent diff
            group['fitness_diff'] = (group['fitness'] - baseline)  # Absolute diff
        else:
            # If baseline type not found in this trial, use 0 (no change)
            group['fitness_diff'] = 0
        return group

    df = df.groupby('trial', group_keys=False).apply(normalize_fitness)

    # Sort trials numerically and create ordered categorical for proper y-axis ordering
    sorted_trials = sorted(df['trial'].unique())
    df['trial_str'] = pd.Categorical(
        df['trial'].astype(str),
        categories=[str(t) for t in sorted_trials],
        ordered=True
    )

    # Get unique failure types to define shapes and colors
    failure_types = sorted(df['type'].unique())

    # Split data into baseline and other types for layered plotting
    df_baseline = df[df['type'] == normalize_to]
    df_other = df[df['type'] != normalize_to]

    # Create the plot
    plt = (
        ggplot(df, aes(x='fitness_diff', y='trial_str')) +
        # Add lines connecting points for each trial
        geom_line(
            aes(group='trial_str'),
            color='lightgrey',
            size=1.5,
            alpha=0.6
        ) +
        # Layer 1: Plot baseline (PowerCutoff) first - appears behind
        geom_point(
            aes(color='type', shape='type'),
            data=df_baseline,
            size=3,
            alpha=0.8
        ) +
        # Layer 2: Plot other types second - appears on top
        geom_point(
            aes(color='type', shape='type'),
            data=df_other,
            size=3,
            alpha=0.8
        ) +
        # Apply colorblind-friendly color palette
        scale_color_manual(
            values={k: v for k, v in colors.items() if k in failure_types},
            name='Failure Type'
        ) +
        # Apply distinct shapes
        scale_shape_manual(
            values={k: v for k, v in shapes.items() if k in failure_types},
            name='Failure Type'
        ) +
        theme_matplotlib()
    )

    # Add x-axis limits if provided
    if x_min is not None and x_max is not None:
        plt = plt + scale_x_continuous(limits=(x_min, x_max))

    return plt


def plot_by_type_fitness_diff(df, x_min=None, x_max=None, normalize_to='PowerCutoff'):
    """
    Creates a scatter plot with jittered points showing normalized fitness differences
    across trials for different failure types.

    Fitness values are normalized on a trial-by-trial basis using difference from a
    baseline failure type. For each trial, the baseline type's fitness
    value in that trial is used to calculate: (fitness - baseline)

    Parameters:
    df : DataFrame with results from multiple trials and failure types
    x_min, x_max : X-axis limits for normalized fitness values. If None, uses data range.
    normalize_to : The failure type to use as baseline for normalization

    Returns:
    plotnine plot object
    """
    # Ensure we don't modify original
    df = df.copy()

    # Convert fitness to float if needed
    df['fitness'] = df['fitness'].apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)

    # Normalize fitness values on a trial-by-trial basis
    # For each trial, calculate difference from that trial's baseline
    def normalize_fitness(group):
        baseline_fitness = group[group['type'] == normalize_to]['fitness'].values
        if len(baseline_fitness) > 0:
            baseline = baseline_fitness[0]
            group['fitness_diff'] = (group['fitness'] - baseline)  # Absolute diff
        else:
            # If baseline type not found in this trial, use 0 (no change)
            group['fitness_diff'] = 0
        return group

    df = df.groupby('trial', group_keys=False).apply(normalize_fitness)

    # Filter out the baseline type since it's used for normalization
    df = df[df['type'] != normalize_to]

    # Get unique failure types to define shapes and colors
    failure_types = sorted(df['type'].unique())

    # Set random seed for reproducible jitter
    np.random.seed(1)

    # Create the plot
    plt = (
        ggplot(df, aes(x='fitness_diff', y='type')) +
        geom_point(
            aes(color='type', shape='type'),
            position=position_jitterdodge(
                jitter_height=0.20,    # Vertical jitter for horizontal clouds
                jitter_width=0,        # No horizontal jitter - exact fitness_diff values
                dodge_width=0          # No dodging needed
            ),
            size=3,
            alpha=0.6
        ) +
        # Apply colorblind-friendly color palette
        scale_color_manual(
            values={k: v for k, v in colors.items() if k in failure_types},
            name='Failure Type'
        ) +
        # Apply distinct shapes
        scale_shape_manual(
            values={k: v for k, v in shapes.items() if k in failure_types},
            name='Failure Type'
        ) +
        theme_bw() +
        theme(legend_position='none')
    )

    # Add x-axis limits if provided
    if x_min is not None and x_max is not None:
        plt = plt + scale_x_continuous(limits=(x_min, x_max))

    return plt


def calc_normality(df):
    """
    df  = dataframe with results
    Tests normality for each failure type using Anderson-Darling test.
    Returns DataFrame with test statistics and critical values.
    """
    # Ensure we don't modify original
    df = df.copy()

    # Convert fitness to float if needed (in case it's still ndarray)
    df['fitness'] = df['fitness'].apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)

    # Group by type and calculate normality test for each
    results = []
    for failure_type, group in df.groupby('type'):
        fitness_values = group['fitness'].values
        n_samples = len(fitness_values)

        # Anderson test works with any sample size but needs at least 2 observations
        if n_samples >= 2:
            result = stats.anderson(fitness_values, dist='norm')
            # Critical values are at significance levels: [15%, 10%, 5%, 2.5%, 1%]
            # Index 2 corresponds to 5% significance level
            critical_value_5pct = result.critical_values[2]

            results.append({
                'type': failure_type,
                'n': n_samples,
                'statistic': result.statistic,
                'critical_5pct': critical_value_5pct,
                'normal': 'Yes' if result.statistic < critical_value_5pct else 'No'
            })
        else:
            results.append({
                'type': failure_type,
                'n': n_samples,
                'statistic': np.nan,
                'critical_5pct': np.nan,
                'normal': 'Insufficient data'
            })

    # Create DataFrame and print
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("Normality Test Results (Anderson-Darling):")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80)
    print("Note: normal='Yes' if statistic < critical_5pct (fail to reject null hypothesis at 5% level)")

    return results_df


def calc_hodges_lehmann_estimator(array1, array2):
    """
    Calculate comparison statistics between two paired data arrays.

    Returns a dictionary with:
        - 'hodges_lehmann': Hodges-Lehmann estimator (median of all pairwise averages of differences)
        - 'median_array1': Median of array1
        - 'median_array2': Median of array2
        - 'mean_array1': Mean of array1
        - 'mean_array2': Mean of array2
        - 'median_diff': Median of array2 minus median of array1
        - 'mean_diff': Mean of array2 minus mean of array1

    The function will raise a ValueError if either array contains NaN.
    Arrays must be convertible to numpy arrays of the same shape.
    """

    a1 = np.asarray(array1)
    a2 = np.asarray(array2)

    if np.isnan(a1).any() or np.isnan(a2).any():
        raise ValueError("Inputs contain NaN values.")
    if len(a1) != len(a2):
        raise ValueError("Input arrays must be of the same length.")

    diffs = a2 - a1
    n = len(diffs)

    # Calculate Walsh averages (all pairwise averages including self-pairs)
    walsh_avgs = []
    for i in range(n):
        for j in range(i, n):
            walsh_avgs.append((diffs[i] + diffs[j]) / 2)
    hodges_lehmann = np.median(walsh_avgs)

    median_array1 = np.median(a1)
    median_array2 = np.median(a2)
    mean_array1 = np.mean(a1)
    mean_array2 = np.mean(a2)
    median_diff = median_array2 - median_array1
    mean_diff = mean_array2 - mean_array1

    return {
        'hodges_lehmann': hodges_lehmann,
        'median_array1': median_array1,
        'median_array2': median_array2,
        'mean_array1': mean_array1,
        'mean_array2': mean_array2,
        'median_diff': median_diff,
        'mean_diff': mean_diff
    }


def calc_friedman_with_posthoc(df):
    """
    df = dataframe with results; should be combination of dfs from different failure mode types

    Performs Friedman test to compare related samples across failure types, followed by
    pairwise Wilcoxon signed-rank tests with Bonferroni correction.

    Assumes: All types have been tested on the same set of trials (repeated measures design).
    """
    # Ensure we don't modify original
    df = df.copy()

    # Convert fitness to float if needed
    df['fitness'] = df['fitness'].apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)

    # Get unique types and prepare data for test
    types = sorted(df['type'].unique())
    trial_sets = {}
    for t in types:
        trial_sets[t] = set(df[df['type'] == t]['trial'].unique())

    # Check if trial numbers (map numbers are identical between types)
    reference_trials = trial_sets[types[0]]
    for t in types[1:]:
        if trial_sets[t] != reference_trials:
            raise ValueError("Trial/map counts do not match across all types.")

    # Number of trials
    n_trials = len(reference_trials)

    # Select and sort to ensure data is lined up by type and trial
    df_subset = df[['type', 'trial', 'fitness']]
    df_subset = df_subset.sort_values(by=['type','trial'], ignore_index=True)
    groups = [df_subset[df_subset['type'] == t]['fitness'].values for t in types]

    # Run Friedman test
    fr_stat, p_value = stats.friedmanchisquare(*groups)

    print("\n" + "=" * 80)
    print("Friedman Test Results")
    print("=" * 80)
    print(f"Chi-square statistic: {fr_stat:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"Significant: {'Yes' if p_value < 0.05 else 'No'} (α = 0.05)")
    print(f"Number of groups (types): {len(types)}")
    print(f"Number of blocks (trials): {n_trials}")
    print(f"Group names: {', '.join(types)}")
    print("=" * 80)

    # If significant, run post-hoc pairwise Wilcoxon signed-rank tests and Hodges-Lehmann estimator
    if p_value < 0.05:
        print("\nPost-hoc Wilcoxon Signed-Rank Tests (Bonferroni correction):")
        print("=" * 80)

        # Wilcoxon test with Bonferroni correction using scikit-posthocs
        # Note: posthoc_wilcoxon expects data in long format with group_col
        posthoc_results = sp.posthoc_wilcoxon(df_subset, val_col='fitness', group_col='type', p_adjust='bonferroni')

        # Print the matrix
        print(posthoc_results.to_string())
        print("=" * 80)
        print("Note: Values are adjusted p-values. Significant if p < 0.05")

        # Hodges-Lehmann estimator; compare median of differences between types
        print("\nHodges-Lehmann Estimator applied to Delta F between Pairwise Results:")
        print("=" * 80)
        for t1 in types:
            for t2 in types:
                if t1 != t2:
                    results = calc_hodges_lehmann_estimator(
                        df_subset[df_subset["type"]==t1]["fitness"].values, df_subset[df_subset["type"]==t2]["fitness"].values)
                    print(f"(2){t2} - (1){t1}:")
                    print(f"\tHL: {results['hodges_lehmann']:.4f}\tmedian(1): {results['median_array1']:.4f}\tmedian(2): {results['median_array2']:.4f}\tmean(1): {results['mean_array1']:.4f}\tmean(2): {results['mean_array2']:.4f}\tmedian_diff: {results['median_diff']:.4f}\tmean_diff: {results['mean_diff']:.4f}")

        # Close the box
        print("=" * 80)

    else:
        print("\nFriedman test not significant. Skipping post-hoc tests.")
        posthoc_results = None

    # Create summary dictionary
    results = {
        'friedman': {
            'chi_square': fr_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'n_groups': len(types),
            'n_blocks': n_trials,
            'groups': list(types)
        },
        'posthoc_matrix': posthoc_results,
    }

    return results


def run_rq_quantify_degradation(df):
    # Ensure we don't modify original
    df = df.copy()

    # For scatter plot (current setup requires integers, otherwise will convert to int)
    # scatter_y_min = 4
    # scatter_y_max = 11
    scatter_y_min = 4.5
    scatter_y_max = 10.5

    # min/max for horizontal scatter
    horiz_scatter_x_min = -5.0
    horiz_scatter_x_max = 2.5

    # Create results directory
    folder_path = create_results_dir('RQ_quantify_degradation', prefix='post')

    # Get unique categories of data
    print("\nRQ analysis: Quantify Degradation")
    type_labels = df['type'].unique()
    print(f"\tPossible type: {type_labels}")
    r_labels = df['r'].unique()
    print(f"\tPossible r: {r_labels}")
    obs_labels = df['obs'].unique()
    print(f"\tPossible obs: {obs_labels}")
    algo_labels = df['algo'].unique()
    print(f"\tPossible algo: {algo_labels}")
    print("\n")

    # Generate scatter plots for GA-PSO result comparison (but skip if PSO data is not available yet)
    # Iterate: through obs (T/F) and r (rInf/r20)
    if len(algo_labels) < 2:
        print(f"\tCurrent config only includes data for one search type ({algo_labels}), skipping GA-PSO scatter comparison.")
    else:
        for temp_obs in obs_labels:
            for temp_r in r_labels:
                # Scatter plots
                temp_plt = plot_scatter_fitness(df[(df['obs']==temp_obs) & (df['r']==temp_r)],
                                                y_min=scatter_y_min, y_max=scatter_y_max)
                temp_plt.save(folder_path / f"jitter_{temp_r}_obs{temp_obs}.png", dpi=300)

    # Generate plots for algo (GA/PSO), obs (T/F) and r (rInf/r20) combinations:
    #   Dumbbell plot (change in fitness vs. trial)
    # Additionally calc statistics to check:
    #   Normality (to justify usage of non-parametric)
    #   Friedman with Wilcoxon post-hoc (if Friedman shows significance)
    for temp_algo in algo_labels:
        for temp_r in r_labels:
            for temp_obs in obs_labels:
                # Create dumbbell plots and save
                temp_plt2 = plot_dumbbell_fitness(df[(df['obs'] == temp_obs)
                                                     & (df['r'] == temp_r)
                                                     & (df['algo'] == temp_algo)
                                                     & (df['type'].isin(['PowerCutoff', 'VelocityX3']))])
                temp_plt2.save(folder_path / f"dumbbell_{temp_r}_obs{temp_obs}_{temp_algo}.png", dpi=300)

                # Create scatters to show diffs
                temp_plt3 = plot_by_type_fitness_diff(df[(df['obs']==temp_obs)
                                                         & (df['r']==temp_r)
                                                         & (df['algo']==temp_algo)],
                                                      x_min=horiz_scatter_x_min, x_max=horiz_scatter_x_max)
                temp_plt3.save(folder_path / f"scatter_fitness_diff_{temp_r}_obs{temp_obs}_{temp_algo}.png", dpi=300)

                # Start statistics calcs
                print(f"\nExperiment Configs: r={temp_r}, obs={temp_obs}, algorithm={temp_algo}")

                # Check normality with Anderson-Darling test
                calc_normality(df[(df['obs']==temp_obs)
                                  & (df['r']==temp_r)
                                  & (df['algo']==temp_algo)])

                # Run Friedman (will automatically run Wilcoxon post-hoc if applicable)
                calc_friedman_with_posthoc(df[(df['obs']==temp_obs)
                                              & (df['r']==temp_r)
                                              & (df['algo']==temp_algo)])

    return None
