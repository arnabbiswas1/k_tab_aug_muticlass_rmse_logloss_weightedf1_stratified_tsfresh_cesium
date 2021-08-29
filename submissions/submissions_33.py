import os

import pandas as pd

COMPETITION_NAME = "tabular-playground-series-aug-2021"

SUBMISSION_DIR = "."
SUBMISSION_FILE = "sub_blending_public_7_84995_lgb_ts_f1_score_top_50_0828_2121.csv"
SUBMISSION_MESSAGE = '"LGB, SK(10), freq enc, params from K, seed 20 & public sub with 7.84995 (0.09/0.01)"'

df = pd.read_csv(f"{SUBMISSION_DIR}/{SUBMISSION_FILE}")
print(df.head())

submission_string = f"kaggle competitions submit {COMPETITION_NAME} -f {SUBMISSION_DIR}/{SUBMISSION_FILE} -m {SUBMISSION_MESSAGE}"

os.system(submission_string)
