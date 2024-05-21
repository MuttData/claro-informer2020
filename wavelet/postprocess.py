from typing import List
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pywt

from utils.metrics import metric

def plot_true_and_predicted_signal(plant:str, uthresh: int):
    results_path = f"results/informer_cantidad_entregas_denoised_thr{uthresh}_ftS_sl7_ll7_pl7_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_1"
    
    pred_train = np.load(f"{results_path}/pred_train.npy")
    true_train = np.load(f"{results_path}/true_train.npy")
    pred_val = np.load(f"{results_path}/pred_val.npy")
    true_val = np.load(f"{results_path}/true_val.npy")
    pred_test = np.load(f"{results_path}/pred.npy")
    true_test = np.load(f"{results_path}/true.npy")

    metrics_train = np.load(f"{results_path}/metrics_train.npy")
    metrics_val = np.load(f"{results_path}/metrics_val.npy")
    metrics_test = np.load(f"{results_path}/metrics.npy")
    
    blanks_before_train = np.empty((7,), dtype=object)
    pred_train_with_blanks = np.concatenate((blanks_before_train, pred_train[:,6].reshape(-1)))
    true_train_with_blanks = np.concatenate((blanks_before_train, true_train[:,6].reshape(-1)))

    blanks_before_val = np.empty(pred_train_with_blanks.shape, dtype=object)
    pred_val_with_blanks = np.concatenate((blanks_before_val, pred_val[:, 6].reshape(-1)))
    true_val_with_blanks = np.concatenate((blanks_before_val, true_val[:, 6].reshape(-1)))

    blanks_before_test = np.empty(pred_val_with_blanks.shape, dtype=object)
    pred_test_with_blanks = np.concatenate((blanks_before_test, pred_test[:, 6].reshape(-1)))
    true_test_with_blanks = np.concatenate((blanks_before_test, true_test[:, 6].reshape(-1)))
    plt.figure(figsize=(70, 7))
    plt.subplot(2, 1, 1)

    plt.plot(pred_train_with_blanks)
    plt.plot(true_train_with_blanks)

    plt.plot(pred_val_with_blanks)
    plt.plot(true_val_with_blanks)

    plt.plot(pred_test_with_blanks)
    plt.plot(true_test_with_blanks)
    
    plt.title(f"7 days predictions for PLANT {plant} | de-noising thr = {uthresh} | rmse (train, val, test) = {metrics_train[2], metrics_val[2], metrics_test[2]}")
    plt.legend([
        "Pred signal train",
        "True signal train",
        "Pred signal val",
        "True signal val",
        "Pred signal test",
        "True signal test",
        ])
    plt.savefig(f"{results_path}/plots_thr{uthresh}")

def recompose_mre_and_evaluate(plant: str, mra_signals_columns: List[str]):

    pred_train_mra = []
    pred_val_mra = []
    pred_test_mra = []

    previous_true_train = None
    previous_true_val = None
    previous_true_test = None

    for i, mra_signals_column in enumerate(reversed(mra_signals_columns)):
        
        results_path = f"results/informer_mra_{mra_signals_column}_ftS_sl7_ll7_pl7_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0"

        pred_train_level_i = np.load(f"{results_path}/pred_train.npy")
        true_train = np.load(f"{results_path}/true_train.npy")
        pred_val_level_i = np.load(f"{results_path}/pred_val.npy")
        true_val = np.load(f"{results_path}/true_val.npy")
        pred_test_level_i = np.load(f"{results_path}/pred.npy")
        true_test = np.load(f"{results_path}/true.npy")

        pred_train_mra.append(pred_train_level_i)
        pred_val_mra.append(pred_val_level_i)
        pred_test_mra.append(pred_test_level_i)

        # Check that trues are all the same
        if i > 0:
            assert np.all(true_train == previous_true_train), "true_train are not equal!"
            previous_true_train = true_train

            assert np.all(true_val == previous_true_val), "true_val are not equal!"
            previous_true_val = true_val

            assert np.all(true_test == previous_true_test), "true_test are not equal!"
            previous_true_test = true_test
        else: 
            previous_true_train = true_train
            previous_true_val = true_val
            previous_true_test = true_test

    pred_train = pywt.imra(pred_train_mra)
    pred_val = pywt.imra(pred_val_mra)
    pred_test = pywt.imra(pred_test_mra)


    metrics_train = metric(pred_train, true_train)
    metrics_val = metric(pred_val, true_val)
    metrics_test = metric(pred_test, true_test)
    
    blanks_before_train = np.empty((7,), dtype=object)
    pred_train_with_blanks = np.concatenate((blanks_before_train, pred_train[:,6].reshape(-1)))
    true_train_with_blanks = np.concatenate((blanks_before_train, true_train[:,6].reshape(-1)))

    blanks_before_val = np.empty(pred_train_with_blanks.shape, dtype=object)
    pred_val_with_blanks = np.concatenate((blanks_before_val, pred_val[:, 6].reshape(-1)))
    true_val_with_blanks = np.concatenate((blanks_before_val, true_val[:, 6].reshape(-1)))

    blanks_before_test = np.empty(pred_val_with_blanks.shape, dtype=object)
    pred_test_with_blanks = np.concatenate((blanks_before_test, pred_test[:, 6].reshape(-1)))
    true_test_with_blanks = np.concatenate((blanks_before_test, true_test[:, 6].reshape(-1)))
    
    plt.figure(figsize=(70, 7))
    plt.subplot(2, 1, 1)

    plt.plot(pred_train_with_blanks)
    plt.plot(true_train_with_blanks)

    plt.plot(pred_val_with_blanks)
    plt.plot(true_val_with_blanks)

    plt.plot(pred_test_with_blanks)
    plt.plot(true_test_with_blanks)
    
    plt.title(f"7 days predictions for PLANT {plant} | rmse (train, val, test) = {metrics_train[2], metrics_val[2], metrics_test[2]}")
    plt.legend([
        "Pred signal train",
        "True signal train",
        "Pred signal val",
        "True signal val",
        "Pred signal test",
        "True signal test",
        ])
    plt.savefig(f"img/my_plots/mra_preds_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.jpg")
