FILTERED_DAY = "weekend"
PAST_ERRORS_CORRECTOR = [-4, -2, ] # [-4, -3, -2, -1]

RUN_ID = "20240712"
SIGNALS_DIR = "data/logtel"
SIGNAL_FILENAME = f"cantidad_entregas_total_{RUN_ID}_fb_{FILTERED_DAY}.csv"

FEATURES_SEQ_LEN = 28

SPLIT_TRAIN_PROPORTION = 0.67
SPLIT_VAL_PROPORTION = 0.13

PLOTS_SAVING_DIR = "img/my_plots/mra_2024"

DATA_PARAM = f"{FILTERED_DAY}_{RUN_ID}"
RESULTS_PATH = f"results/informer_{DATA_PARAM}_ftS_sl{FEATURES_SEQ_LEN}_ll7_pl7_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0"
