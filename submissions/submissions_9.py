import os

import pandas as pd

COMPETITION_NAME = "tabular-playground-series-aug-2021"

SUBMISSION_DIR = "."
SUBMISSION_FILE = "sub_xgb_benchmark_SK_freq_0815_1319_7.86044.csv"
SUBMISSION_MESSAGE = '"XGB Benchamrk with StratifiedKFold(10) frequency feature"'

df = pd.read_csv(f"{SUBMISSION_DIR}/{SUBMISSION_FILE}")
df.loss = df.loss.round()
print(df.head())

submission_string = f"kaggle competitions submit {COMPETITION_NAME} -f {SUBMISSION_DIR}/{SUBMISSION_FILE} -m {SUBMISSION_MESSAGE}"

print(submission_string)

os.system(submission_string)
