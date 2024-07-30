import subprocess

from wavelet.utils_single_run import plot_true_and_predicted_signal
from wavelet.constants import FEATURES_SEQ_LEN, DATA_PARAM
plant_to_select = "ALL"

signals_dir = "data/logtel"
signal_filename = "cantidad_entregas_total.csv"
command = [
    "python", "-u", "main_informer.py",
    "--model", "informer",
    "--data", DATA_PARAM,
    "--root_path", signals_dir,
    "--data_path", signal_filename,
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
    "--batch_size", "16",
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

