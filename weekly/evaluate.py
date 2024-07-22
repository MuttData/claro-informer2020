from wavelet.constants import RESULTS_PATH, SPLIT_TRAIN_PROPORTION, SPLIT_VAL_PROPORTION
from wavelet.postprocess import obtain_metrics_for_split_size
from wavelet.utils_single_run import plot_signals_in_same_plot
from weekly.constants import RESULTS_PATH as RESULTS_PATH_WEEKLY

import numpy as np
import pandas as pd
import polars as pl

normal_df_path = f"{RESULTS_PATH}/results.csv"
weekly_df_path = f"{RESULTS_PATH_WEEKLY}/results.csv"

def integrate():

    normal_df = pd.read_csv(normal_df_path)
    normal_df['date'] = pd.to_datetime(normal_df['date'], format="%Y-%m-%d")
    normal_test_df = normal_df[int(len(normal_df)*(SPLIT_TRAIN_PROPORTION+SPLIT_VAL_PROPORTION)):].reset_index(drop=True)
    indexes_to_drop = []
    for idx, row in normal_test_df.iterrows():
        if row["date"].dayofweek == 4:
            break
        else:
            indexes_to_drop.append(idx)

    normal_test_df.drop(indexes_to_drop, inplace=True) # Erase all samples before first Friday so that the first week starts of Friday
    normal_test_df['week_start'] = normal_test_df['date'] - pd.to_timedelta((normal_test_df['date'].dt.dayofweek + 3) % 7, unit='D') # + 3 so that week starts on Friday
    # Now group by the 'week_start' column
    normal_gb_weekly = normal_test_df.groupby('week_start').agg(
        true=('true', 'sum'),
        pred=('pred', 'sum')
    ).reset_index()
    normal_gb_weekly.rename(columns={"week_start": "date"}, inplace=True)

    print(f"normal_gb_weekly.shape: {normal_gb_weekly.shape}")

    weekly_df = pd.read_csv(weekly_df_path)
    weekly_df['date'] = pd.to_datetime(weekly_df['date'], format="%Y-%m-%d")
    weekly_joined_df = normal_gb_weekly.merge(weekly_df[['date', 'pred']], how='inner', on='date', suffixes=('','_weekly_trained'))
    print(f"weekly_joined_df.shape: {weekly_joined_df.shape}")

    metrics_daily_to_weekly = obtain_metrics_for_split_size(weekly_joined_df['true'], weekly_joined_df['pred'], split_prop=1, test_set=False)
    metrics_weekly = obtain_metrics_for_split_size(weekly_joined_df['pred_weekly_trained'], weekly_joined_df['pred'], split_prop=1, test_set=False)
    
    print("MODEL TRAINED DAILY GROUPED BY WEEK:", metrics_daily_to_weekly)
    print("               MODEL TRAINED WEEKLY:", metrics_weekly)


integrate()