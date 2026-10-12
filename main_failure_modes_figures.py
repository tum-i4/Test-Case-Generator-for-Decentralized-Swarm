from src.postprocessing import data_loader as data_loader
from src.postprocessing.RQ_quantify_degradation import run_rq_quantify_degradation
from src.postprocessing.RQ_compare_separation import run_rq_separation
from src.postprocessing.results_check_convergence import run_check_convergence
from src.postprocessing.results_check_solutions_with_reruns import run_rerun_best_and_verify

# ========================================
# Functional Test Results Directories
# ========================================

RESULT_PATHS_FUNCTIONAL = {
    "rInf": {
        "path": "results_to_save/exp_failure_modes_v202603/functional/result_20251009_163009_functional_rInf",
        "obs": True
    },
    "r20": {
        "path": "results_to_save/exp_failure_modes_v202603/functional/result_20251024_175212_functional_r20",
        "obs": True
    }
}

# ========================================
# Experiment Test Results Directories
# ========================================

RESULT_PATHS_EXPERIMENT_v2 = {
    "exp2": {
        "type": "PowerCutoff",
        "obs": True,
        "algo": "GA",
        "r": "rInf",
        "path": "results_to_save/exp_failure_modes_v202603/exp_GA_rInf_agents12/result_20260309_191935_GA_fail1_power_cutoff",
        "path_sim": "results_to_save/exp_failure_modes_v202603/rerun_sim_out/rerun_verify_20260320_183541_exp2"
    },
    "exp3": {
        "type": "VelocityX1",
        "obs": True,
        "algo": "GA",
        "r": "rInf",
        "path": "results_to_save/exp_failure_modes_v202603/exp_GA_rInf_agents12/result_20260310_210038_GA_fail1_velocity_single",
        "path_sim": "results_to_save/exp_failure_modes_v202603/rerun_sim_out/rerun_verify_20260320_184020_exp3"
    },
    "exp4": {
        "type": "VelocityX2",
        "obs": True,
        "algo": "GA",
        "r": "rInf",
        "path": "results_to_save/exp_failure_modes_v202603/exp_GA_rInf_agents12/result_20260311_133607_GA_fail2_velocity_double",
        "path_sim": "results_to_save/exp_failure_modes_v202603/rerun_sim_out/rerun_verify_20260320_184504_exp4"
    },
    "exp5": {
        "type": "VelocityX3",
        "obs": True,
        "algo": "GA",
        "r": "rInf",
        "path": "results_to_save/exp_failure_modes_v202603/exp_GA_rInf_agents12/result_20260312_072135_GA_fail3_velocity_triple",
        "path_sim": "results_to_save/exp_failure_modes_v202603/rerun_sim_out/rerun_verify_20260320_184951_exp5"
    },
    "exp7": {
        "type": "PowerCutoff",
        "obs": True,
        "algo": "GA",
        "r": "r20",
        "path": "results_to_save/exp_failure_modes_v202603/exp_GA_r20_agents12/result_20260313_013220_GA_fail1_power_cutoff_r20",
        "path_sim": "results_to_save/exp_failure_modes_v202603/rerun_sim_out/rerun_verify_20260320_185432_exp7"
    },
    "exp8": {
        "type": "VelocityX1",
        "obs": True,
        "algo": "GA",
        "r": "r20",
        "path": "results_to_save/exp_failure_modes_v202603/exp_GA_r20_agents12/result_20260313_183726_GA_fail1_velocity_single_r20",
        "path_sim": "results_to_save/exp_failure_modes_v202603/rerun_sim_out/rerun_verify_20260320_190014_exp8"
    },
    "exp9": {
        "type": "VelocityX2",
        "obs": True,
        "algo": "GA",
        "r": "r20",
        "path": "results_to_save/exp_failure_modes_v202603/exp_GA_r20_agents12/result_20260313_185431_GA_fail2_velocity_double_r20",
        "path_sim": "results_to_save/exp_failure_modes_v202603/rerun_sim_out/rerun_verify_20260320_190533_exp9"
    },
    "exp10": {
        "type": "VelocityX3",
        "obs": True,
        "algo": "GA",
        "r": "r20",
        "path": "results_to_save/exp_failure_modes_v202603/exp_GA_r20_agents12/result_20260314_074102_GA_fail3_velocity_triple_r20",
        "path_sim": "results_to_save/exp_failure_modes_v202603/rerun_sim_out/rerun_verify_20260320_191038_exp10"
    },
    "exp12": {
        "type": "PowerCutoff",
        "obs": True,
        "algo": "RANDOM",
        "r": "rInf",
        "path": "results_to_save/exp_failure_modes_v202603/exp_RANDOM_rInf_agents12/result_20260311_171111_RANDOM_fail1_power_cutoff_rInf",
        "path_sim": "results_to_save/exp_failure_modes_v202603/rerun_sim_out/rerun_verify_20260320_191510_exp12"
    },
    "exp13": {
        "type": "VelocityX1",
        "obs": True,
        "algo": "RANDOM",
        "r": "rInf",
        "path": "results_to_save/exp_failure_modes_v202603/exp_RANDOM_rInf_agents12/result_20260312_091256_RANDOM_fail1_velocity_single_rInf",
        "path_sim": "results_to_save/exp_failure_modes_v202603/rerun_sim_out/rerun_verify_20260320_191951_exp13"
    },
    "exp14": {
        "type": "VelocityX2",
        "obs": True,
        "algo": "RANDOM",
        "r": "rInf",
        "path": "results_to_save/exp_failure_modes_v202603/exp_RANDOM_rInf_agents12/result_20260313_024413_RANDOM_fail2_velocity_double_rInf",
        "path_sim": "results_to_save/exp_failure_modes_v202603/rerun_sim_out/rerun_verify_20260320_192432_exp14"
    },
    "exp15": {
        "type": "VelocityX3",
        "obs": True,
        "algo": "RANDOM",
        "r": "rInf",
        "path": "results_to_save/exp_failure_modes_v202603/exp_RANDOM_rInf_agents12/result_20260314_201110_RANDOM_fail3_velocity_triple_rInf",
        "path_sim": "results_to_save/exp_failure_modes_v202603/rerun_sim_out/rerun_verify_20260320_192910_exp15"
    },
    "exp17": {
        "type": "PowerCutoff",
        "obs": True,
        "algo": "RANDOM",
        "r": "r20",
        "path": "results_to_save/exp_failure_modes_v202603/exp_RANDOM_r20_agents12/result_20260315_071750_RANDOM_fail1_power_cutoff_r20",
        "path_sim": "results_to_save/exp_failure_modes_v202603/rerun_sim_out/rerun_verify_20260320_193334_exp17"
    },
    "exp18": {
        "type": "VelocityX1",
        "obs": True,
        "algo": "RANDOM",
        "r": "r20",
        "path": "results_to_save/exp_failure_modes_v202603/exp_RANDOM_r20_agents12/result_20260315_164414_RANDOM_fail1_velocity_single_r20",
        "path_sim": "results_to_save/exp_failure_modes_v202603/rerun_sim_out/rerun_verify_20260320_193846_exp18"
    },
    "exp19": {
        "type": "VelocityX2",
        "obs": True,
        "algo": "RANDOM",
        "r": "r20",
        "path": "results_to_save/exp_failure_modes_v202603/exp_RANDOM_r20_agents12/result_20260316_132648_RANDOM_fail2_velocity_double_r20",
        "path_sim": "results_to_save/exp_failure_modes_v202603/rerun_sim_out/rerun_verify_20260320_194358_exp19"
    },
    "exp20": {
        "type": "VelocityX3",
        "obs": True,
        "algo": "RANDOM",
        "r": "r20",
        "path": "results_to_save/exp_failure_modes_v202603/exp_RANDOM_r20_agents12/result_20260317_081048_RANDOM_fail3_velocity_triple_r20",
        "path_sim": "results_to_save/exp_failure_modes_v202603/rerun_sim_out/rerun_verify_20260320_194848_exp20"
    }
}

# ========================================
# MAIN FUNCTION
# ========================================

def main():
    RESULT_PATHS_EXPERIMENT = RESULT_PATHS_EXPERIMENT_v2
    # ========================================
    # Load data
    # ========================================

    # Load experiment results (summary data)
    df_exp = data_loader.load_all_pickled_dataframes(RESULT_PATHS_EXPERIMENT)

    # Load functionality test results
    df_fun = data_loader.load_all_pickled_dataframes(RESULT_PATHS_FUNCTIONAL)

    # ========================================
    # Run once: rerun best results, verify fitness, and save sim outputs
    # ========================================

    run_rerun_best_and_verify(RESULT_PATHS_EXPERIMENT, df_exp)

    # ========================================
    # Convergence Visualization
    # ========================================

    run_check_convergence(RESULT_PATHS_EXPERIMENT)

    # ========================================
    # Results Processing based on summary data
    # ========================================

    # RQ1 (& 3): degradation
    run_rq_quantify_degradation(df_exp)

    # # RQ2: separation
    run_rq_separation(RESULT_PATHS_EXPERIMENT)


if __name__ == "__main__":
    main()
