from wavelet.constants import RESULTS_PATH, SPLIT_TRAIN_PROPORTION, SPLIT_VAL_PROPORTION
from wavelet.postprocess import obtain_metrics_for_split_size
from wavelet.utils_single_run import plot_signals_in_same_plot
from weekly.constants import RESULTS_PATH as RESULTS_PATH_WEEKLY

import numpy as np
import polars as pl

normal_df_path = f"{RESULTS_PATH}/results.csv"
weekly_df_path = f"{RESULTS_PATH_WEEKLY}/results.csv"

def integrate():

    normal_df = pl.read_csv(normal_df_path)
    weekly_df = pl.read_csv(weekly_df_path)

    merged_df = normal_df.join(weekly_df, on="date", how="left", suffix="_weekend")

    result_df = merged_df.with_columns(
        pl.when(pl.col("pred_corrected_weekend").is_null())
        .then(pl.col("pred_corrected"))
        .otherwise(pl.col("pred_corrected_weekend"))
        .alias("pred_sust_weekend")
    )

    metrics_train = obtain_metrics_for_split_size(
        np.array(result_df["true"]),
        np.array(result_df["pred_corrected"]),
        split_prop=(SPLIT_TRAIN_PROPORTION + SPLIT_VAL_PROPORTION),
        test_set=False
    )

    metrics_test = obtain_metrics_for_split_size(
        np.array(result_df["true"]),
        np.array(result_df["pred_sust_weekend"]),
        split_prop=(SPLIT_TRAIN_PROPORTION + SPLIT_VAL_PROPORTION)
    )
    plot_signals_in_same_plot(
        result_df["true"],
        result_df["pred_sust_weekend"],
        title=f"7 days predictions corrected for cantidad_entregas for PLANT ALL. Corrector: LGBM. MAE, MAPE, MdAPE, RMSE train = {metrics_train['mae'], metrics_train['mape'], metrics_train['mdape'], metrics_train['rmse']} | MAE, MAPE, MdAPE, RMSE test = {metrics_test['mae'], metrics_test['mape'], metrics_test['mdape'], metrics_test['rmse']}",
        legend=[
            "True signal",
            "Predicted and corrected signal with sustituted weekends",
        ],
        save_path=f"{RESULTS_PATH}/informer_predicted_weekends_sustituted.png",
        x_labels=list(result_df["date"],),
        day_locator_interval=7,
        show=False
        )