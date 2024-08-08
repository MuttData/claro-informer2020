"""Variables that set the run of the script that trains and informer and plots its results."""

FEATURES_SEQ_LEN = 28 # Days to look back when generating samples
PRED_SEQ_LEN = 7 # Days to look forward when predicting

USE_LINEAR_CORRECTOR = True # Whether to use a linear corrector
ERROR_LOOK_BACK_DAYS = 28 # Days to look back the error when training the corrector
ERROR_LOOK_FORWARDS_DAYS = 1 # Days to look forward the error when predicting the error with the corrector

# The next three vars are used if the corrector is not linear
CV_N_SPLITS = 5 # Amount of cross validation splits to prevent overfitting
GRID_SEARCH_ITERS = 30 # Amount of RandomGridSearch iterations
NUM_AUGMENTATIONS = 20 # # If using data augmentation in the corrector

SPLIT_TRAIN_PROPORTION = 0.67
SPLIT_VAL_PROPORTION = 0.13
SPLIT_RANDOM_STATE = 42

# Raw data identifier
DATA_DATE = "20240801" # Next date to the last date of the dataset

# Directory where raw data is stored
SIGNALS_DIR = "data/logtel"
# Raw data filename
SIGNAL_FILENAME = f"cantidad_entregas_total_{DATA_DATE}.csv"

# Model run identified
DATA_PARAM = f"single_run_lgbm_total_{DATA_DATE}"
# Directory where to store results
RESULTS_PATH = f"results/informer_{DATA_PARAM}_ftS_sl{FEATURES_SEQ_LEN}_ll7_pl{PRED_SEQ_LEN}_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0"