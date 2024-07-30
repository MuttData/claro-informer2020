import subprocess

from weekends.constants import FEATURES_SEQ_LEN, DATA_PARAM, SIGNALS_DIR, SIGNAL_FILENAME, RESULTS_PATH
plant_to_select = "ALL"
from weekends.postprocess import plot_true_and_predicted_signal

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
    "--pred_len", "7",
    "--enc_in", "1",
    "--dec_in", "1",
    "--c_out", "1",
    # "--skip_training",
    "--batch_size", "8",
    "--do_predict",
    "--itr", "1",
    "--inverse",
]
print(" ".join(command))
process = subprocess.run(command, capture_output=True)
output, err = process.stdout, process.stderr
if err:
    print("Error:", err.decode())
else:
    # Process the captured output (optional)
    print("Output:", output.decode())
plot_true_and_predicted_signal(plant_to_select)

print(f"RESULTS_PATH: {RESULTS_PATH}")
