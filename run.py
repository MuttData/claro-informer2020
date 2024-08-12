"""
This script runs the `main_informer.py` script for time-series forecasting using the Informer model. It constructs and executes 
a command with specific parameters for model prediction, captures and processes the output, and visualizes the results.

Data generated in this run (predictions and their timestamps, graphs, ) are stored to the RESULTS_PATH path.
Finally, results are uploaded to the predictions table.
"""
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import subprocess

from report.upsert_predictions import upsert_predictions
from single_run.run_vars import (DATA_PARAM, FEATURES_SEQ_LEN, PRED_SEQ_LEN,
                                 RESULTS_PATH, SIGNAL_FILENAME, SIGNALS_DIR)
from single_run.utils import plot_true_and_predicted_corrected_signal

command = [
    "python", "-u", "main_informer.py",
    "--model", "informer",
    "--data", DATA_PARAM,
    "--root_path", SIGNALS_DIR,
    "--data_path", SIGNAL_FILENAME,
    "--features", "S",
    "--target", "cantidad_entregas",
    "--freq", "d",
    "--seq_len", f"{FEATURES_SEQ_LEN}",
    "--label_len", "7",
    "--pred_len", f"{PRED_SEQ_LEN}",
    "--enc_in", "1",
    "--dec_in", "1",
    "--c_out", "1",
    # "--skip_training",
    "--batch_size", "8",
    "--do_predict",
    "--itr", "1",
    "--inverse",
]
# Train Informer
logging.info("Running: " + " ".join(command))
process = subprocess.run(command, capture_output=True)
output, err = process.stdout, process.stderr
if err:
    logging.info(f"Error: {err.decode()}")
else:
    # Process the captured output (optional)
    logging.info(f"Output: {output.decode()}")

# Plot predictions
plot_true_and_predicted_corrected_signal()
logging.info(f"RESULTS_PATH: {RESULTS_PATH}")

# Upsert predictions to DB
logging.info(f"Uploading predictions to DB")
upsert_predictions()