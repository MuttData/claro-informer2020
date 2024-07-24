from wavelet.constants import RESULTS_PATH, SPLIT_TRAIN_PROPORTION, SPLIT_VAL_PROPORTION
from wavelet.postprocess import obtain_metrics_for_split_size
from wavelet.utils_single_run import plot_signals_in_same_plot
from weekends.constants import RESULTS_PATH as RESULTS_PATH_DAY_FILTERED, FILTERED_DAY

import numpy as np
import polars as pl

normal_df_path = f"{RESULTS_PATH}/results.csv"
day_filtered_df_path = f"{RESULTS_PATH_DAY_FILTERED}/results.csv"
num_days_specific_evaluation = 14

def integrate():

    normal_df = pl.read_csv(normal_df_path)
    day_filtered_df = pl.read_csv(day_filtered_df_path)

    print(f"normal_df.columns: {normal_df.columns}, day_filtered_df.columns: {day_filtered_df.columns}")
    merged_df = normal_df.join(day_filtered_df, on="date", how="left", suffix="_day_filtered")
    print(f"merged_df.columns: {merged_df.columns}")
    
    # Replaces filtered days with separately trained filtered days
    result_df = merged_df.with_columns(
        pl.when(pl.col("pred_corrected_day_filtered").is_null())
        .then(pl.col("pred"))
        .otherwise(pl.col("pred_corrected_day_filtered"))
        .alias("pred_sust_day_filtered")
    )

    metrics_test = obtain_metrics_for_split_size(
        np.array(result_df["true"]),
        np.array(result_df["pred"]),
        split_prop=(SPLIT_TRAIN_PROPORTION + SPLIT_VAL_PROPORTION),
    )

    metrics_test_sustituted = obtain_metrics_for_split_size(
        np.array(result_df["true"]),
        np.array(result_df["pred_sust_day_filtered"]),
        split_prop=(SPLIT_TRAIN_PROPORTION + SPLIT_VAL_PROPORTION)
    )

    metrics_last_days = obtain_metrics_for_split_size(
        np.array(result_df["true"]),
        np.array(result_df["pred"]),
        split_prop=num_days_specific_evaluation,
    )

    metrics_last_days_sust_day_filtered = obtain_metrics_for_split_size(
        np.array(result_df["true"]),
        np.array(result_df["pred_sust_day_filtered"]),
        split_prop=num_days_specific_evaluation,
    )

    plot_signals_in_same_plot(
        result_df["true"],
        result_df["pred_sust_day_filtered"],
        title=f"7 days predictions corrected for cantidad_entregas total \
  MAE, MAPE, MdAPE, RMSE test = {metrics_test['mae'], metrics_test['mape'], metrics_test['mdape'], metrics_test['rmse']} \
| MAE, MAPE, MdAPE, RMSE test sustituted = {metrics_test_sustituted['mae'], metrics_test_sustituted['mape'], metrics_test_sustituted['mdape'], metrics_test_sustituted['rmse']} \
| MAE, MAPE, MdAPE, RMSE last {num_days_specific_evaluation} days = {metrics_last_days['mae'], metrics_last_days['mape'], metrics_last_days['mdape'], metrics_last_days['rmse']} \
| MAE, MAPE, MdAPE, RMSE last {num_days_specific_evaluation} days sustituted = {metrics_last_days_sust_day_filtered['mae'], metrics_last_days_sust_day_filtered['mape'], metrics_last_days_sust_day_filtered['mdape'], metrics_last_days_sust_day_filtered['rmse']}",
        legend=[
            "True signal",
            f"Predicted and corrected signal with sustituted {FILTERED_DAY}s",
        ],
        save_path=f"{RESULTS_PATH_DAY_FILTERED}/informer_no_corr_predicted_day_filtered_sustituted.png",
        x_labels=list(result_df["date"],),
        day_locator_interval=7,
        show=False
        )
    
    result_df.write_csv(f"{RESULTS_PATH_DAY_FILTERED}/results_no_corr_day_filtered_sustituted.csv")
    
    print(f"METRICS INFORMER LAST {num_days_specific_evaluation} days:                      ", metrics_last_days)
    print(f"METRICS INFORMER + {FILTERED_DAY} FILTERED SUSTITUTED LAST {num_days_specific_evaluation} days:", metrics_last_days_sust_day_filtered)
    print("Normaly path:", RESULTS_PATH)
    print("Day filtered path:", RESULTS_PATH_DAY_FILTERED)
integrate()