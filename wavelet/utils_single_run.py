import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from random import randint
from typing import List
from utils.metrics import RMSE

from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from lightgbm import LGBMRegressor, plot_importance
from sklearn.linear_model import LinearRegression

from wavelet.constants import (
    FEATURES_SEQ_LEN,
    ERROR_LOOK_BACK_DAYS, 
    ERROR_LOOK_FORWARDS_DAYS,
    SPLIT_TRAIN_PROPORTION,
    SPLIT_VAL_PROPORTION,
    SPLIT_RANDOM_STATE,
    CV_N_SPLITS,
    GRID_SEARCH_ITERS,
    RESULTS_PATH
    )
from wavelet.utils import (
    augment_data_with_random_deviation, 
    create_dataset, 
    scatter_plot_for_different_days_diff
    )
from wavelet.postprocess import obtain_metrics_for_split_size

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
    
    blanks_before_train = np.empty((FEATURES_SEQ_LEN,), dtype=object)
    pred_train_with_blanks = np.concatenate((blanks_before_train, pred_train[:,6].reshape(-1)))

    blanks_before_val = np.empty(pred_train_with_blanks.shape, dtype=object)
    pred_val_with_blanks = np.concatenate((blanks_before_val, pred_val[:, 6].reshape(-1)))

    blanks_before_test = np.empty(pred_val_with_blanks.shape, dtype=object)
    pred_test_with_blanks = np.concatenate((blanks_before_test, pred_test[:, 6].reshape(-1)))
    
    true_signal_with_blanks = np.concatenate(
        (
            blanks_before_train,
            true_train[:,6].reshape(-1),
            true_val[:,6].reshape(-1),
            true_test[:,6].reshape(-1)
        )
    )

    true_signal = np.concatenate(
        (
            true_train[:,6].reshape(-1),
            true_val[:,6].reshape(-1),
            true_test[:,6].reshape(-1)
        )
    )

    pred_signal = np.concatenate(
        (
            pred_train[:,6].reshape(-1),
            pred_val[:,6].reshape(-1),
            pred_test[:,6].reshape(-1)
        )
    )

    preds = np.concatenate(
        (pred_train[:,6].reshape(-1), pred_val[:,6].reshape(-1), pred_test[:,6].reshape(-1))
        )
    trues = np.concatenate(
        (true_train[:,6].reshape(-1), true_val[:,6].reshape(-1), true_test[:,6].reshape(-1))
        )
    x_axis_timestamps = np.concatenate((
        per_sample_timestamp_train[:,6].reshape(-1), 
        per_sample_timestamp_val[:,6].reshape(-1), 
        per_sample_timestamp_test[:,6].reshape(-1)
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
    plt.plot(range(len(pred_train)), pred_train[:,6].reshape(-1))
    plt.plot(range(len(pred_train), len(pred_train)+len(pred_val)), pred_val[:,6].reshape(-1))
    plt.plot(range(len(pred_train)+len(pred_val), len(pred_train)+len(pred_val)+len(pred_test)), pred_test[:,6].reshape(-1))
    
    plt.title(f"7 days predictions for cantidad_entregas for PLANT {plant} | MAE, MdAPE, RMSE train = {metrics_train['mae'], metrics_train['mdape'], metrics_train['rmse']} | MAE, MdAPE, RMSE test = {metrics_test['mae'], metrics_test['mdape'], metrics_test['rmse']}")
    plt.legend([
        "True signal",
        "Pred signal train",
        "Pred signal val",
        "Pred signal test",
        ])
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
        test_set=False
    )

    metrics_test = obtain_metrics_for_split_size(
        true_signal,
        pred_signal_corrected,
        split_prop=(SPLIT_TRAIN_PROPORTION + SPLIT_VAL_PROPORTION)
    )

    plot_signals_in_same_plot(
        true_signal, 
        pred_signal_corrected,
        title=f"7 days predictions corrected for cantidad_entregas for PLANT {plant}. Corrector: LGBM. No augmentation. MAE, MdAPE, RMSE train = {metrics_train['mae'], metrics_train['mdape'], metrics_train['rmse']} | MAE, MdAPE, RMSE test = {metrics_test['mae'], metrics_test['mdape'], metrics_test['rmse']}",
        legend=[
            "True signal",
            "Predicted and corrected signal",
        ],
        save_path=f"{RESULTS_PATH}/informer_predicted_and_corrected.png",
        x_labels=list(x_axis_timestamps),
        show=False
        )
    
    np.save(f"{RESULTS_PATH}/true_signal.npy", true_signal)
    np.save(f"{RESULTS_PATH}/pred_corrected_signal.npy", pred_signal_corrected)


def correct_prediction_predicting_error(
    y_trues: np.array, 
    y_preds: np.array, 
    augment: bool = False, 
    early_stopping: bool = False,
    shuffle_train: bool = False
    ):

    y_error = y_trues - y_preds

    # scatter_plot_for_different_days_diff(
    #     y_error[int(len(y_error)*SPLIT_TRAIN_PROPORTION):], 
    #     ERROR_LOOK_BACK_DAYS,
    #     y_error.min(),
    #     y_error.max()
    #     )

    print(f"Raw error shape: {y_error.shape}")
    
    error_features, error_targets = create_dataset(
        y_error, 
        look_back=ERROR_LOOK_BACK_DAYS, 
        look_forward=ERROR_LOOK_FORWARDS_DAYS
        )
    # error_features = error_features[:, [-28, -21, -14, -13, -9, -7, -2, -1]]
    error_targets = error_targets[:,-1]

    print(f"error features shape: {error_features.shape} | error targets shape: {error_targets.shape}")
    
    X_error_train, X_error_test, y_error_train, y_error_test = train_test_split(
        error_features, 
        error_targets, 
        train_size=(SPLIT_TRAIN_PROPORTION+SPLIT_VAL_PROPORTION), 
        random_state=SPLIT_RANDOM_STATE,
        shuffle=False
        )

    print(f"error training features shape: {X_error_train.shape} | error train targets shape: {y_error_train.shape}")
    print(f"error test features shape: {X_error_test.shape} | error test targets shape: {y_error_test.shape}")

    if augment:
        # X_error_train, y_error_train = augment_data(X_error_train, y_error_train)
        X_error_train, y_error_train = augment_data_with_random_deviation(y_error[:int(len(y_error)*SPLIT_TRAIN_PROPORTION)])
        print(f"error training augmented features shape: {X_error_train.shape} | error train augmented targets shape: {y_error_train.shape}")

    params = {
        'objective': 'regression',
        'reg_alpha': 0.1,
        'reg_lambda': 0.2,
    }
    if early_stopping:
        params["metric"] = 'rmse'
        params["early_stopping_rounds"] = 200

    model = LinearRegression() # lgb.LGBMRegressor(**params)

    lgbm_param_grid = {
        'learning_rate': [0.005, 0.01, 0.05],
        'num_leaves': [31, 63, 127],
        'max_depth': [3, 5, 7],
        'min_child_samples': [10, 20, 50],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0.0, 0.1, 0.5],
        'reg_lambda': [0.0, 0.1, 0.5],
        'n_estimators': [100, 200, 500]
    }

    tscv = TimeSeriesSplit(n_splits=CV_N_SPLITS)

    grid_search = RandomizedSearchCV(
        estimator=LGBMRegressor(), 
        param_distributions=lgbm_param_grid, 
        n_iter=GRID_SEARCH_ITERS,
        cv=tscv, 
        scoring='neg_root_mean_squared_error', 
        n_jobs=-1,
        verbose=1)
    grid_search.fit(X_error_train, y_error_train)
    
    model = LGBMRegressor(**grid_search.best_params_)

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
            shuffle=False
        )
        print(f"Early stopping with {val_size} samples")

        model.fit(
            X_error_train, 
            y_error_train,
            eval_set=[(X_error_val, y_error_val)],)
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
    print(f"RandomizedGridSearchCV.best_params_: = {grid_search.best_params_}")
    y_preds_corrected = y_preds + error_preds
    return y_preds_corrected

def plot_signals_in_same_plot(
    *signals, 
    title: str = None, 
    legend: List[str] = None, 
    show: bool = False, 
    save_path=f"img/my_plots/plot.png",
    x_labels: List[str] = None,
    day_locator_interval: int = 7
    ):

    # Create the figure with appropriate subplots
    plt.figure(figsize=(70, 14))  # Adjust figure size as needed
    
    for signal in signals:
        plt.plot(signal)
    plt.title(title)
    plt.xlabel("Time [days]")
    plt.legend(legend, loc="lower left")

    if x_labels:
        # Setting the ticks and labels on the x-axis
        plt.xticks(range(len(x_labels)), x_labels, rotation=90)
        # Use a locator to display every tenth date
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=day_locator_interval))
    plt.savefig(save_path)
    if show:
        plt.show()
    