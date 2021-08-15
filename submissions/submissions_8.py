import os

import pandas as pd

COMPETITION_NAME = "tabular-playground-series-aug-2021"

SUBMISSION_DIR = "."
SUBMISSION_FILE = "sub_lgb_benchmark_SKFold_freq_0815_1234_7.85689.csv"
SUBMISSION_MESSAGE = '"LGB Benchamrk with StratifiedKFold (10) with frequency encoding"'

df = pd.read_csv(f"{SUBMISSION_DIR}/{SUBMISSION_FILE}")
df.loss = df.loss.round()
print(df.head())

submission_string = f"kaggle competitions submit {COMPETITION_NAME} -f {SUBMISSION_DIR}/{SUBMISSION_FILE} -m {SUBMISSION_MESSAGE}"

print(submission_string)

os.system(submission_string)
