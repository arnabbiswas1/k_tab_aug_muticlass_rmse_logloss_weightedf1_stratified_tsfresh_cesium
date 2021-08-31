import os

import pandas as pd

COMPETITION_NAME = "tabular-playground-series-aug-2021"

SUBMISSION_DIR = "."
SUBMISSION_FILE = "sub_blending_2_public_kernels_plus_lgb_freq_enc_params_from_Kaggle_seed_20_update_2_0830_1958.csv"
SUBMISSION_MESSAGE = '"blending_2_public_kernels_plus_lgb_freq_enc_params_from_Kaggle_seed_20"'

df = pd.read_csv(f"{SUBMISSION_DIR}/{SUBMISSION_FILE}")
print(df.head())

submission_string = f"kaggle competitions submit {COMPETITION_NAME} -f {SUBMISSION_DIR}/{SUBMISSION_FILE} -m {SUBMISSION_MESSAGE}"

os.system(submission_string)
