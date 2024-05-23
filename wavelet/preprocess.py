import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pywt
from typing import List
from wavelet.utils import wavelet_denoising, decompose_signal_in_different_freq_bands, plot_signals_in_df
from utils.metrics import RMSE

column_name_to_predict = "cantidad_entregas"
plant_to_select = "N001"
date_column_name = "CREATEDON"
data_dir = "data/logtel"
denoised_signals_dir = f"./{data_dir}/denoised_signals"

mra_signals_dir = f"./{data_dir}/mra_signals"

def create_denoised_signal(uthresh: float = None):
    # Obtain raw data
    dataframe = pd.read_csv(f"{data_dir}/cantidad_entregas.csv")
    signal = dataframe[column_name_to_predict].values

    # Rename column with _real
    dataframe.rename(columns={column_name_to_predict: f"{column_name_to_predict}_real"}, inplace=True)

    # De-noise signal using DWT
    signal_denoised = wavelet_denoising(signal, uthresh=uthresh)
    signal_denoised = signal_denoised[:-1]
    dataframe[column_name_to_predict] = signal_denoised

    # Plot the signals
    plt.figure(figsize=(70, 7))
    plt.subplot(2, 1, 1)
    plt.plot(signal)
    plt.plot(signal_denoised)
    plt.title(f"{column_name_to_predict} for {plant_to_select} | thr = {uthresh}")
    plt.legend(["Original signal", f"De-noised signal with thr={uthresh}"])
    plt.savefig(f"img/my_plots/thr{uthresh}.jpg")

    # Save de-noised signal
    if not os.path.exists(denoised_signals_dir):
        # Create the directory
        os.makedirs(denoised_signals_dir)
    denoised_signal_filename = f"cantidad_entregas_denoised_thr{uthresh:.2f}.csv"
    dataframe.to_csv(f"{denoised_signals_dir}/{denoised_signal_filename}", index=False)
    return denoised_signals_dir, denoised_signal_filename

def create_decomposed_wavelet_signals():
    # Obtain raw data
    mra_signals_df = pd.read_csv(f"{data_dir}/cantidad_entregas.csv")
    signal = mra_signals_df[column_name_to_predict].values
    wavelet = "db4"


    # mra_signals_df[column_name_to_predict] = signal
    mra_signals = decompose_signal_in_different_freq_bands(signal, wavelet)
    decomposition_level = len(mra_signals) - 1
    curr_level = 1
    for mra_signal in reversed(mra_signals):
        if curr_level == decomposition_level + 1:
            mra_signals_df[f"{column_name_to_predict}_approx_level_{decomposition_level}"] = mra_signal
        else:
            mra_signals_df[f"{column_name_to_predict}_details_level_{curr_level}"] = mra_signal
        curr_level += 1
    
    imra_reconstructed_signal = pywt.imra(mra_signals)
    reconstructing_error = signal - imra_reconstructed_signal
    print(f"RMSE(signal, imra_rec_signal) = {RMSE(signal, imra_reconstructed_signal)}")
    print(f"Count(reconstructing_error > 0) = {np.sum(reconstructing_error > 0)}")
    mra_signals_df[f"{column_name_to_predict}_reconstructing_error_level_{decomposition_level}"] = reconstructing_error

    plot_signals_in_df(mra_signals_df.drop(columns="date"))
    if not os.path.exists(mra_signals_dir):
    # Create the directory
        os.makedirs(mra_signals_dir)
    mra_signals_filename = f"{column_name_to_predict}_mra.csv"
    mra_signals_df.to_csv(f"{mra_signals_dir}/{mra_signals_filename}", index=False)
    return mra_signals_dir, mra_signals_filename, list(mra_signals_df.drop(columns=["date", column_name_to_predict]).columns)
    




