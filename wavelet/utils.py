import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt

from typing import List
from utils.metrics import RMSE

# Function to create dataset with multiple output steps
def create_dataset(dataset, look_back=1, look_forward=7):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - look_forward + 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        # Append the next 7 days as a single array to dataY
        dataY.append(dataset[i + look_back:i + look_back + look_forward, 0])
    return np.array(dataX), np.array(dataY)
    
def madev(d, axis=None):
    """ Mean absolute deviation of a signal """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def wavelet_denoising(x, wavelet='db4', level=1, uthresh=None):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    if uthresh is None:
        sigma = (1/0.6745) * madev(coeff[-level])
        uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    print(f"Obtained uthresh = {uthresh}")
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')

def decompose_signal_in_different_freq_bands(
    signal, 
    wavelet, 
    level=None, 
    axis=-1,
    transform='dwt', 
    mode='periodization'
    ) -> List[np.array]:
    mra = pywt.mra(signal, wavelet, level=level, axis=axis, transform=transform, mode=mode)
    return mra

def plot_signals_in_df(df: pd.DataFrame) -> None:
    # Number of columns
    nrows = len(df.columns)

    # Create the figure with appropriate subplots
    plt.figure(figsize=(70, 7))  # Adjust figure size as needed
    
    # Adjust spacing between subplots (adjust values as needed)
    top = 0.9  # Adjust top margin to avoid title overlapping
    bottom = 0.15  # Adjust bottom margin to avoid x-axis labels overlapping
    wspace = 0.3  # Adjust horizontal spacing between subplots
    plt.subplots_adjust(left=0.1, right=0.9, top=top, bottom=bottom, wspace=wspace)
    # Loop through each column
    for i, col in enumerate(df.columns):
        # Set subplot position based on number of columns
        plt.subplot(nrows, 1, i+1)  # 1 row, ncols columns, starting from subplot number i+1

        # Plot the column data (assuming numeric)
        plt.plot(df[col])

        # Customize plot elements (labels, title, etc.)
        plt.xlabel("Time [days]")
        plt.title(col, fontsize=8)

    # Adjust layout and display the plot
    # plt.tight_layout()
    plt.savefig("img/my_plots/mra/mra.jpg")
    # plt.show()
    
def evaluate_and_plot_per_level_preds(plant: str, mra_signals_columns: List[str]):
    
    # Number of columns
    nrows = len(mra_signals_columns)

    # Create the figure with appropriate subplots
    plt.figure(figsize=(70, 7))  # Adjust figure size as needed
    
    # Adjust spacing between subplots (adjust values as needed)
    top = 0.9  # Adjust top margin to avoid title overlapping
    bottom = 0.15  # Adjust bottom margin to avoid x-axis labels overlapping
    wspace = 0.3  # Adjust horizontal spacing between subplots
    plt.subplots_adjust(left=0.1, right=0.9, top=top, bottom=bottom, wspace=wspace, hspace=1)

    for i, mra_signals_column in enumerate(mra_signals_columns):
        
        results_path = f"results/informer_mra_err_{mra_signals_column}_ftS_sl7_ll7_pl7_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0"

        pred_train_level = np.load(f"{results_path}/pred_train.npy")
        true_train_level = np.load(f"{results_path}/true_level_train.npy")
        pred_val_level = np.load(f"{results_path}/pred_val.npy")
        true_val_level = np.load(f"{results_path}/true_level_val.npy")
        pred_test_level = np.load(f"{results_path}/pred.npy")
        true_test_level = np.load(f"{results_path}/true_level.npy")

        rmse_train = RMSE(pred_train_level, true_train_level)
        rmse_val = RMSE(pred_val_level, true_val_level)
        rmse_test = RMSE(pred_test_level, true_test_level)

        blanks_before_train = np.empty((7,), dtype=object)
        true_train_with_blanks = np.concatenate((blanks_before_train, true_train_level[:,6].reshape(-1)))
        pred_train_with_blanks = np.concatenate((blanks_before_train, pred_train_level[:,6].reshape(-1)))

        blanks_before_val = np.empty(pred_train_with_blanks.shape, dtype=object)
        true_val_with_blanks = np.concatenate((blanks_before_val, true_val_level[:, 6].reshape(-1)))
        pred_val_with_blanks = np.concatenate((blanks_before_val, pred_val_level[:, 6].reshape(-1)))
        
        blanks_before_test = np.empty(pred_val_with_blanks.shape, dtype=object)
        true_test_with_blanks = np.concatenate((blanks_before_test, true_test_level[:, 6].reshape(-1)))
        pred_test_with_blanks = np.concatenate((blanks_before_test, pred_test_level[:, 6].reshape(-1)))

        # Set subplot position based on number of columns
        plt.subplot(nrows, 1, i+1)  # 1 row, ncols columns, starting from subplot number i+1

        # Plot the column data (assuming numeric)
        plt.plot(np.concatenate((true_train_with_blanks, true_val_level[:, 6].reshape(-1), true_test_level[:, 6].reshape(-1))))
        plt.plot(pred_train_with_blanks)

        plt.plot(pred_val_with_blanks)

        plt.plot(pred_test_with_blanks)

        # Customize plot elements (labels, title, etc.)
        
        plt.title(f"{mra_signals_column} for {plant} | rmse (train, val, test) = {rmse_train, rmse_val, rmse_test}", fontsize=8)
    plt.xlabel("Time [days]")
    plt.legend([
        "True signal",
        "Pred train signal",
        "Pred val signal",
        "Pred test signal",
        ],
        loc="lower left")
    # plt.show()
    plt.savefig("img/my_plots/mra/mra_per_level_preds.png")