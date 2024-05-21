import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt

from typing import List

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
    