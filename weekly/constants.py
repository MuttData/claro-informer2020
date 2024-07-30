SIGNALS_DIR = "data/logtel"
SIGNAL_FILENAME = "cantidad_entregas_total_weekly.csv"

FEATURES_SEQ_LEN = 7
PRED_LEN = 1

# ERROR_LOOK_BACK_DAYS = 28
# ERROR_LOOK_FORWARDS_DAYS = 1

SPLIT_TRAIN_PROPORTION = 0.67
SPLIT_VAL_PROPORTION = 0.13
# SPLIT_RANDOM_STATE = 42

# NUM_AUGMENTATIONS = 20
# MAX_AMPLITUDE_FACTOR = 3

# MODEL_SUFFIX = "lgbm"

PLOTS_SAVING_DIR = "img/my_plots/mra_2024"

# CV_N_SPLITS = 5
# GRID_SEARCH_ITERS = 1

DATA_PARAM = "weekly_20240712"
RESULTS_PATH = f"results/informer_{DATA_PARAM}_ftS_sl{FEATURES_SEQ_LEN}_ll7_pl1_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0"