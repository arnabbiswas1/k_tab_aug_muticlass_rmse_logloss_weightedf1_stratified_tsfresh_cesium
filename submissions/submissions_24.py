import os

import pandas as pd

COMPETITION_NAME = "tabular-playground-series-aug-2021"

SUBMISSION_DIR = "."
SUBMISSION_FILE = "sub_blending_public_7_8500_lgb_ts_log_loss_top_5_0825_1327.csv"
SUBMISSION_MESSAGE = '"Blending of public score 7.85000 and LGB ts log_loss top 5 features (almost lowest logloss & lowest RMSE)"'

df = pd.read_csv(f"{SUBMISSION_DIR}/{SUBMISSION_FILE}")
print(df.head())

submission_string = f"kaggle competitions submit {COMPETITION_NAME} -f {SUBMISSION_DIR}/{SUBMISSION_FILE} -m {SUBMISSION_MESSAGE}"

os.system(submission_string)
