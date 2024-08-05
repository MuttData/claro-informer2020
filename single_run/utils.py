from random import randint
from typing import Dict, List, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMRegressor, plot_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    TimeSeriesSplit,
    train_test_split,
)

from single_run.constants import (
    CV_N_SPLITS,
    ERROR_LOOK_BACK_DAYS,
    ERROR_LOOK_FORWARDS_DAYS,
    FEATURES_SEQ_LEN,
    GRID_SEARCH_ITERS,
    NUM_AUGMENTATIONS,
    PLOTS_SAVING_DIR,
    PRED_SEQ_LEN,
    RESULTS_PATH,
    SPLIT_RANDOM_STATE,
    SPLIT_TRAIN_PROPORTION,
    SPLIT_VAL_PROPORTION,
    USE_LINEAR_CORRECTOR,
)
from utils.metrics import MdAPE, metric


def plot_true_and_predicted_signal() -> None:
    """
    Plots the true and predicted signals, as well as the corrected predictions,
    for the time series data. Saves the plots to specified paths and stores the
    results in a CSV file.

    The function loads the true and predicted signals for training, validation,
    and test sets, computes metrics, and visualizes the results. It also applies
    a correction to the predictions using an error prediction model and plots
    the corrected predictions.

    Args:
        None

    Returns:
        None
    """

    pred_len_index = PRED_SEQ_LEN - 1

    pred_train = np.load(f"{RESULTS_PATH}/pred_train.npy")
    true_train = np.load(f"{RESULTS_PATH}/true_train.npy")
    per_sample_timestamp_train = np.load(
        f"{RESULTS_PATH}/per_sample_timestamps_train.npy", allow_pickle=True
    )

    pred_val = np.load(f"{RESULTS_PATH}/pred_val.npy")
    true_val = np.load(f"{RESULTS_PATH}/true_val.npy")
    per_sample_timestamp_val = np.load(
        f"{RESULTS_PATH}/per_sample_timestamps_val.npy", allow_pickle=True
    )

    pred_test = np.load(f"{RESULTS_PATH}/pred.npy")
    true_test = np.load(f"{RESULTS_PATH}/true.npy")
    per_sample_timestamp_test = np.load(
        f"{RESULTS_PATH}/per_sample_timestamps_test.npy", allow_pickle=True
    )

    metrics_train = np.load(f"{RESULTS_PATH}/metrics_train.npy")
    metrics_val = np.load(f"{RESULTS_PATH}/metrics_val.npy")
    metrics_test = np.load(f"{RESULTS_PATH}/metrics.npy")

    true_signal = np.concatenate(
        (
            true_train[:, pred_len_index].reshape(-1),
            true_val[:, pred_len_index].reshape(-1),
            true_test[:, pred_len_index].reshape(-1),
        )
    )

    pred_signal = np.concatenate(
        (
            pred_train[:, pred_len_index].reshape(-1),
            pred_val[:, pred_len_index].reshape(-1),
            pred_test[:, pred_len_index].reshape(-1),
        )
    )

    x_axis_timestamps = np.concatenate(
        (
            per_sample_timestamp_train[:, pred_len_index].reshape(-1),
            per_sample_timestamp_val[:, pred_len_index].reshape(-1),
            per_sample_timestamp_test[:, pred_len_index].reshape(-1),
        )
    )

    metrics_train = obtain_metrics_for_split_size(
        true_signal,
        pred_signal,
        split_prop=(SPLIT_TRAIN_PROPORTION + SPLIT_VAL_PROPORTION),
        test_set=False,
    )

    metrics_test = obtain_metrics_for_split_size(
        true_signal,
        pred_signal,
        split_prop=(SPLIT_TRAIN_PROPORTION + SPLIT_VAL_PROPORTION),
    )

    plt.figure(figsize=(70, 14))

    plt.plot(range(len(true_signal)), true_signal)
    plt.plot(range(len(pred_train)), pred_train[:, pred_len_index].reshape(-1))
    plt.plot(
        range(len(pred_train), len(pred_train) + len(pred_val)),
        pred_val[:, pred_len_index].reshape(-1),
    )
    plt.plot(
        range(
            len(pred_train) + len(pred_val),
            len(pred_train) + len(pred_val) + len(pred_test),
        ),
        pred_test[:, pred_len_index].reshape(-1),
    )

    plt.title(
        f"7 days predictions for total cantidad_entregas | MAE, MAPE, MdAPE, RMSE train = {metrics_train['mae'], metrics_train['mape'], metrics_train['mdape'], metrics_train['rmse']} | MAE, MAPE, MdAPE, RMSE test = {metrics_test['mae'], metrics_test['mape'], metrics_test['mdape'], metrics_test['rmse']}"
    )
    plt.legend(
        [
            "True signal",
            "Pred signal train",
            "Pred signal val",
            "Pred signal test",
        ]
    )
    # Setting the ticks and labels on the x-axis
    plt.xticks(range(len(x_axis_timestamps)), x_axis_timestamps, rotation=90)
    # Use a locator to display every tenth date
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))

    plt.savefig(f"{RESULTS_PATH}/informer_predicted.png")

    pred_signal_corrected = correct_prediction_predicting_error(
        true_signal,
        pred_signal,
    )

    metrics_train = obtain_metrics_for_split_size(
        true_signal,
        pred_signal_corrected,
        split_prop=(SPLIT_TRAIN_PROPORTION + SPLIT_VAL_PROPORTION),
        test_set=False,
    )

    metrics_test = obtain_metrics_for_split_size(
        true_signal,
        pred_signal_corrected,
        split_prop=(SPLIT_TRAIN_PROPORTION + SPLIT_VAL_PROPORTION),
    )

    plot_signals_in_same_plot(
        true_signal,
        pred_signal_corrected,
        title=f"7 days predictions corrected for total cantidad_entregas. Corrector: {'linear regressor' if USE_LINEAR_CORRECTOR else 'LGBM'}. MAE, MAPE, MdAPE, RMSE train = {metrics_train['mae'], metrics_train['mape'], metrics_train['mdape'], metrics_train['rmse']} | MAE, MAPE, MdAPE, RMSE test = {metrics_test['mae'], metrics_test['mape'], metrics_test['mdape'], metrics_test['rmse']}",
        legend=[
            "True signal",
            "Predicted and corrected signal",
        ],
        save_path=f"{RESULTS_PATH}/informer_predicted_and_corrected.png",
        x_labels=list(x_axis_timestamps),
        show=False,
    )

    np.save(f"{RESULTS_PATH}/true_signal.npy", true_signal)
    np.save(f"{RESULTS_PATH}/pred_corrected_signal.npy", pred_signal_corrected)
    pl.DataFrame(
        {
            "date": x_axis_timestamps,
            "true": true_signal,
            "pred": pred_signal,
            "pred_corrected": pred_signal_corrected,
        }
    ).write_csv(f"{RESULTS_PATH}/results.csv")


def correct_prediction_predicting_error(
    y_trues: np.array,
    y_preds: np.array,
    augment: bool = False,
    early_stopping: bool = False,
    shuffle_train: bool = False,
) -> np.array:
    """
    Corrects the predicted values using an error prediction model.

    The function calculates the error between the true and predicted values,
    then trains a model to predict the error. The corrected predictions are
    obtained by adding the predicted error to the original predictions.

    Args:
        y_trues (np.array): The true values.
        y_preds (np.array): The predicted values.
        augment (bool): If True, data augmentation is applied.
        early_stopping (bool): If True, early stopping is used during training.
        shuffle_train (bool): If True, the training data is shuffled.

    Returns:
        np.array: The corrected predictions.
    """

    y_error = y_trues - y_preds

    # scatter_plot_for_different_days_diff(
    #     y_error[int(len(y_error)*SPLIT_TRAIN_PROPORTION):],
    #     ERROR_LOOK_BACK_DAYS,
    #     y_error.min(),
    #     y_error.max()
    #     )

    print(f"Raw error shape: {y_error.shape}")

    error_features, error_targets = create_dataset(
        y_error, look_back=ERROR_LOOK_BACK_DAYS, look_forward=ERROR_LOOK_FORWARDS_DAYS
    )
    # error_features = error_features[:, [-28, -21, -14, -13, -9, -7, -2, -1]]
    error_targets = error_targets[:, -1]

    print(
        f"error features shape: {error_features.shape} | error targets shape: {error_targets.shape}"
    )

    X_error_train, X_error_test, y_error_train, y_error_test = train_test_split(
        error_features,
        error_targets,
        train_size=(SPLIT_TRAIN_PROPORTION + SPLIT_VAL_PROPORTION),
        random_state=SPLIT_RANDOM_STATE,
        shuffle=False,
    )

    print(
        f"error training features shape: {X_error_train.shape} | error train targets shape: {y_error_train.shape}"
    )
    print(
        f"error test features shape: {X_error_test.shape} | error test targets shape: {y_error_test.shape}"
    )

    if augment:
        # X_error_train, y_error_train = augment_data(X_error_train, y_error_train)
        X_error_train, y_error_train = augment_data_with_random_deviation(
            y_error[: int(len(y_error) * SPLIT_TRAIN_PROPORTION)]
        )
        print(
            f"error training augmented features shape: {X_error_train.shape} | error train augmented targets shape: {y_error_train.shape}"
        )

    params = {
        "objective": "regression",
        "reg_alpha": 0.1,
        "reg_lambda": 0.2,
    }
    if early_stopping:
        params["metric"] = "rmse"
        params["early_stopping_rounds"] = 200

    model = LinearRegression()

    if not USE_LINEAR_CORRECTOR:
        lgbm_param_grid = {
            "learning_rate": [0.005, 0.01, 0.05],
            "num_leaves": [31, 63, 127],
            "max_depth": [3, 5, 7],
            "min_child_samples": [10, 20, 50],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": [0.0, 0.1, 0.5],
            "reg_lambda": [0.0, 0.1, 0.5],
            "n_estimators": [100, 200, 500],
        }

        tscv = TimeSeriesSplit(n_splits=CV_N_SPLITS)

        grid_search = RandomizedSearchCV(
            estimator=LGBMRegressor(),
            param_distributions=lgbm_param_grid,
            n_iter=GRID_SEARCH_ITERS,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=1,
        )
        grid_search.fit(X_error_train, y_error_train)

        model = LGBMRegressor(**grid_search.best_params_)
        print(f"RandomizedGridSearchCV.best_params_: = {grid_search.best_params_}")

    if shuffle_train:
        shuffle_indices = np.random.permutation(len(X_error_train))
        X_error_train = X_error_train[shuffle_indices]
        y_error_train = y_error_train[shuffle_indices]

    if early_stopping:
        val_size = int(len(error_features) * SPLIT_VAL_PROPORTION)
        X_error_val, X_error_test, y_error_val, y_error_test = train_test_split(
            X_error_test,
            y_error_test,
            train_size=val_size,
            random_state=SPLIT_RANDOM_STATE,
            shuffle=False,
        )
        print(f"Early stopping with {val_size} samples")

        model.fit(
            X_error_train,
            y_error_train,
            eval_set=[(X_error_val, y_error_val)],
        )
    else:
        model.fit(X_error_train, y_error_train)

    # Plot the feature importances
    # plt.figure(figsize=(10, 6))
    # plot_importance(model, )
    # plt.title("Feature Importances")
    # plt.show()

    error_preds = model.predict(error_features)

    n_heading_paddings = ERROR_LOOK_BACK_DAYS + ERROR_LOOK_FORWARDS_DAYS - 1
    print(f"n_heading_paddings = {n_heading_paddings}")
    error_preds = np.pad(error_preds, (n_heading_paddings, 0), constant_values=0)
    y_preds_corrected = y_preds + error_preds
    return y_preds_corrected


def plot_signals_in_same_plot(
    *signals: np.array,
    title: str = None,
    legend: List[str] = None,
    show: bool = False,
    save_path=f"img/my_plots/plot.png",
    x_labels: List[str] = None,
    day_locator_interval: int = 7,
    linestyles: List[str] = None,
) -> None:
    """
    Plots multiple signals on the same plot with optional customization.

    The function allows plotting multiple time series signals on a single plot,
    with options to set the title, legend, x-axis labels, linestyles, and
    whether to show or save the plot.

    Args:
        *signals (np.array): Variable number of arrays representing the signals to plot.
        title (str, optional): Title of the plot.
        legend (List[str], optional): Legend labels for the plot.
        show (bool, optional): If True, displays the plot.
        save_path (str, optional): Path to save the plot image.
        x_labels (List[str], optional): Labels for the x-axis.
        day_locator_interval (int, optional): Interval for day locator on x-axis.
        linestyles (List[str], optional): Linestyles for each signal plot.

    Returns:
        None
    """

    # Create the figure with appropriate subplots
    plt.figure(figsize=(70, 14))  # Adjust figure size as needed

    for i, signal in enumerate(signals):
        linestyle = "-"
        if linestyles is not None:
            linestyle = linestyles[i]
        plt.plot(signal, linestyle=linestyle)
    plt.title(title)
    plt.xlabel("Time [days]")
    plt.legend(legend, loc="lower left")

    if x_labels:
        # Setting the ticks and labels on the x-axis
        plt.xticks(range(len(x_labels)), x_labels, rotation=90)
        # Use a locator to display every tenth date
        plt.gca().xaxis.set_major_locator(
            mdates.DayLocator(interval=day_locator_interval)
        )
    plt.savefig(save_path)
    if show:
        plt.show()


def augment_data_with_random_deviation(
    time_series: np.array,
) -> Tuple[np.array, np.array]:
    """
    Augments the time series data by adding random deviations.

    The function generates augmented data by adding random deviations to the
    original time series data, creating multiple augmented versions. It then
    creates a dataset with look-back and look-forward sequences.

    Args:
        time_series (np.array): The original time series data.

    Returns:
        Tuple[np.array, np.array]: Augmented features and targets.
    """

    MAX_DEVIATION_CONCATS = 5
    MAX_OFFSETS_PER_DEVIATION = 100
    time_series_l = len(time_series)
    aug_max_value = time_series.mean()
    accumulated_time_series_aug = np.array([])
    n_concatenations_list = []
    print(
        f"time_series before augmentation: {time_series.shape} | mean={time_series.mean()}, max={time_series.max()}, min={time_series.min()}"
    )
    while len(accumulated_time_series_aug) <= (time_series_l * NUM_AUGMENTATIONS):
        random_aug = np.random.uniform(
            low=-aug_max_value, high=aug_max_value, size=ERROR_LOOK_FORWARDS_DAYS
        )
        for _ in range(randint(1, MAX_OFFSETS_PER_DEVIATION)):
            n_concatenations = randint(1, MAX_DEVIATION_CONCATS)
            offset_random = randint(
                0, time_series_l - ERROR_LOOK_FORWARDS_DAYS * n_concatenations
            )
            concatenated_random_augs = np.tile(random_aug, (n_concatenations,))
            time_series_aug_slice = (
                time_series[
                    offset_random : offset_random
                    + ERROR_LOOK_FORWARDS_DAYS * n_concatenations
                ]
                + concatenated_random_augs
            )
            accumulated_time_series_aug = np.concatenate(
                (accumulated_time_series_aug, time_series_aug_slice)
            )
            n_concatenations_list.append(n_concatenations)
    time_series_aug = np.concatenate((time_series, accumulated_time_series_aug))
    print(
        f"time_series_aug after augmentation: {time_series_aug.shape} | max={time_series_aug.max()}, min={time_series_aug.min()}"
    )
    print(f"sum n_concatenations_list = {sum(n_concatenations_list)}")
    features, targets = create_dataset(
        time_series_aug,
        look_back=ERROR_LOOK_BACK_DAYS,
        look_forward=ERROR_LOOK_FORWARDS_DAYS,
    )
    return features, targets[:, -1]


def create_dataset(
    dataset: np.array, look_back: int = 1, look_forward: int = 7
) -> Tuple[np.array, np.array]:
    """
    Creates a dataset with look-back and look-forward sequences.

    The function generates input-output pairs from the provided dataset based
    on the specified look-back and look-forward lengths.

    Args:
        dataset (np.array): The original time series data.
        look_back (int, optional): The number of past time steps to consider.
        look_forward (int, optional): The number of future time steps to predict.

    Returns:
        Tuple[np.array, np.array]: Input features and output targets.
    """

    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - look_forward + 1):
        a = dataset[i : (i + look_back)]
        dataX.append(a)
        # Append the next 7 days as a single array to dataY
        dataY.append(dataset[i + look_back : i + look_back + look_forward])
    return np.array(dataX), np.array(dataY)


def scatter_plot_for_different_days_diff(
    signal: np.array,
    days_diff: int,
    range_min: float,
    range_max: float,
    show: bool = False,
) -> None:
    """
    Creates a scatter plot for the given signal with a specified days difference.

    The function generates a scatter plot showing the relationship between the
    original signal and its shifted version by a given number of days. It fits a
    linear regression line to the data.

    Args:
        signal (np.array): The signal to plot.
        days_diff (int): The number of days to shift the signal.
        range_min (float): The minimum value for the plot range.
        range_max (float): The maximum value for the plot range.
        show (bool, optional): If True, displays the plot.
    """
    signal_orig = signal[days_diff:]
    signal_shifted = signal[:-days_diff]

    m, b = np.polyfit(signal_shifted, signal_orig, 1)  # Linear regression (degree 1)
    x_fit = np.linspace(min(signal_shifted), max(signal_shifted), 100)
    # Calculate y values for the regression line using the formula y = mx + b
    y_fit = m * x_fit + b

    plt.figure()
    # Set axis limits based on the range
    plt.xlim(range_min, range_max)  # Set x-axis limits
    plt.ylim(range_min, range_max)  # Set y-axis limits
    plt.scatter(signal_shifted, signal_orig, label=f"error[i] vs. error[i-{days_diff}]")
    plt.plot(x_fit, y_fit, color="red", label="Best Fit Line")  # Red color and label
    plt.title(f"error[i] vs. error[i-{days_diff}] after IMRA")
    plt.xlabel(f"error[i-{days_diff}] [days]")
    plt.ylabel("error[i] [days]")
    plt.legend(loc="lower right")

    if show:
        plt.show()
    plt.savefig(f"{PLOTS_SAVING_DIR}/errors_shifted.png")


def obtain_metrics_for_split_size(
    true_signal: np.array,
    pred_signal: np.array,
    split_prop: float = SPLIT_TRAIN_PROPORTION,
    test_set: bool = True,
) -> Dict[str, float]:
    """
    Obtain evaluation metrics for a specified split of the signal.

    Parameters:
        true_signal (array-like): The true values.
        pred_signal (array-like): The predicted values.
        split_prop (float or int): Proportion of the data to use for training (if <= 1)
                                    or absolute number of samples (if > 1).
        test_set (bool): Whether to evaluate on the test set (True) or training set (False).

    Returns:
        Dict[str, float]: Metrics including MAE, MSE, RMSE, MAPE, MSPE, MdAPE.
    """
    split_len = 0
    if split_prop <= 1:
        split_len = int(len(true_signal) * split_prop)
    else:
        split_len = split_prop
        if test_set:
            split_prop = split_prop * (-1)

    true_signal_split = np.array([])
    pred_signal_split = np.array([])

    if test_set:
        true_signal_split = true_signal[split_len:]
        pred_signal_split = pred_signal[split_len:]
    else:
        true_signal_split = true_signal[:split_len]
        pred_signal_split = pred_signal[:split_len]
    mae, mse, rmse, mape, mspe = metric(pred_signal_split, true_signal_split)
    mdape = MdAPE(pred_signal_split, true_signal_split)
    metrics = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "mspe": mspe,
        "mdape": mdape,
    }
    for key, value in metrics.items():
        metrics[key] = round(value, 2)
    return metrics
