import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import polars as pl

from weekly.constants import (
    FEATURES_SEQ_LEN,
    RESULTS_PATH,
    SPLIT_TRAIN_PROPORTION,
    SPLIT_VAL_PROPORTION
)
from wavelet.postprocess import obtain_metrics_for_split_size
from wavelet.utils_single_run import plot_signals_in_same_plot

def plot_true_and_predicted_signal(plant: str):

    pred_train = np.load(f"{RESULTS_PATH}/pred_train.npy")
    true_train = np.load(f"{RESULTS_PATH}/true_level_train.npy")
    per_sample_timestamp_train = np.load(f"{RESULTS_PATH}/per_sample_timestamps_train.npy", allow_pickle=True)

    pred_val = np.load(f"{RESULTS_PATH}/pred_val.npy")
    true_val = np.load(f"{RESULTS_PATH}/true_level_val.npy")
    per_sample_timestamp_val = np.load(f"{RESULTS_PATH}/per_sample_timestamps_val.npy", allow_pickle=True)

    pred_test = np.load(f"{RESULTS_PATH}/pred.npy")
    true_test = np.load(f"{RESULTS_PATH}/true_level.npy")
    per_sample_timestamp_test = np.load(f"{RESULTS_PATH}/per_sample_timestamps_test.npy", allow_pickle=True)

    metrics_train = np.load(f"{RESULTS_PATH}/metrics_train.npy")
    metrics_val = np.load(f"{RESULTS_PATH}/metrics_val.npy")
    metrics_test = np.load(f"{RESULTS_PATH}/metrics.npy")


    true_signal = np.concatenate(
        (
            true_train[:,-1].reshape(-1),
            true_val[:,-1].reshape(-1),
            true_test[:,-1].reshape(-1)
        )
    )

    pred_signal = np.concatenate(
        (
            pred_train[:,-1].reshape(-1),
            pred_val[:,-1].reshape(-1),
            pred_test[:,-1].reshape(-1)
        )
    )

    preds = np.concatenate(
        (pred_train[:,-1].reshape(-1), pred_val[:,-1].reshape(-1), pred_test[:,-1].reshape(-1))
        )
    trues = np.concatenate(
        (true_train[:,-1].reshape(-1), true_val[:,-1].reshape(-1), true_test[:,-1].reshape(-1))
        )
    x_axis_timestamps = np.concatenate((
        per_sample_timestamp_train[:,-1].reshape(-1), 
        per_sample_timestamp_val[:,-1].reshape(-1), 
        per_sample_timestamp_test[:,-1].reshape(-1)
        )
        )

    metrics_train = obtain_metrics_for_split_size(
        trues, 
        preds, 
        split_prop=(SPLIT_TRAIN_PROPORTION + SPLIT_VAL_PROPORTION),
        test_set=False
        )

    metrics_test = obtain_metrics_for_split_size(
        trues, 
        preds, 
        split_prop=(SPLIT_TRAIN_PROPORTION + SPLIT_VAL_PROPORTION)
        )
    
    plt.figure(figsize=(70, 14))
    # plt.subplot(2, 1, 1)

    plt.plot(range(len(true_signal)), true_signal)
    plt.plot(range(len(pred_train)), pred_train[:,-1].reshape(-1))
    plt.plot(range(len(pred_train), len(pred_train)+len(pred_val)), pred_val[:,-1].reshape(-1))
    plt.plot(range(len(pred_train)+len(pred_val), len(pred_train)+len(pred_val)+len(pred_test)), pred_test[:,-1].reshape(-1))
    
    plt.title(f"1 week predictions with weekly step for cantidad_entregas for PLANT {plant} | MAE, MAPE, MdAPE, RMSE train = {metrics_train['mae'], metrics_train['mape'], metrics_train['mdape'], metrics_train['rmse']} | MAE, MAPE, MdAPE, RMSE test = {metrics_test['mae'], metrics_test['mape'], metrics_test['mdape'], metrics_test['rmse']}")
    plt.legend([
        "True signal",
        "Pred signal train",
        "Pred signal val",
        "Pred signal test",
        ])
    # Setting the ticks and labels on the x-axis
    plt.xticks(range(len(x_axis_timestamps)), x_axis_timestamps, rotation=90)
    # Use a locator to display every tenth date
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))

    plt.savefig(f"{RESULTS_PATH}/informer_predicted.png")
    
    # pred_signal_corrected = correct_prediction_predicting_error(
    #     true_signal,
    #     pred_signal,
    #     )
    
    # metrics_train = obtain_metrics_for_split_size(
    #     true_signal,
    #     pred_signal_corrected,
    #     split_prop=(SPLIT_TRAIN_PROPORTION + SPLIT_VAL_PROPORTION),
    #     test_set=False
    # )

    # metrics_test = obtain_metrics_for_split_size(
    #     true_signal,
    #     pred_signal_corrected,
    #     split_prop=(SPLIT_TRAIN_PROPORTION + SPLIT_VAL_PROPORTION)
    # )

    # plot_signals_in_same_plot(
    #     true_signal, 
    #     pred_signal_corrected,
    #     title=f"7 days predictions corrected for cantidad_entregas for PLANT {plant}. Corrector: past weekdays avg. MAE, MAPE, MdAPE, RMSE train = {metrics_train['mae'], metrics_train['mape'], metrics_train['mdape'], metrics_train['rmse']} | MAE, MAPE, MdAPE, RMSE test = {metrics_test['mae'], metrics_test['mape'], metrics_test['mdape'], metrics_test['rmse']}",
    #     legend=[
    #         "True signal",
    #         "Predicted and corrected signal",
    #     ],
    #     save_path=f"{RESULTS_PATH}/informer_predicted_and_corrected.png",
    #     x_labels=list(x_axis_timestamps),
    #     day_locator_interval=1,
    #     show=False
    #     )
    
    np.save(f"{RESULTS_PATH}/true_signal.npy", true_signal)
    np.save(f"{RESULTS_PATH}/pred_signal.npy", pred_signal)
    # np.save(f"{RESULTS_PATH}/pred_corrected_signal.npy", pred_signal_corrected)

    pl.DataFrame({
        "date": x_axis_timestamps,
        "true": true_signal,
        "pred": pred_signal,
        # "pred_corrected": pred_signal_corrected
    }).write_csv(f"{RESULTS_PATH}/results.csv")
    

def estimate_errors_averaging_past_errors(errors, past_errors: np.array) -> np.array:
    estimated_errors = []

    for i in range(len(errors)):
        estimated_error = 0
        if i >= np.max(np.abs(past_errors)):
            estimated_error = np.mean(errors[i + past_errors])
        estimated_errors.append(estimated_error)
    return np.array(estimated_errors)

def correct_prediction_predicting_error(y_trues: np.array, y_preds: np.array):
    y_errors = y_trues - y_preds
    y_estimated_errors = estimate_errors_averaging_past_errors(y_errors, np.array([-4, -2]))
    y_preds_corrected = y_preds + y_estimated_errors
    return y_preds_corrected