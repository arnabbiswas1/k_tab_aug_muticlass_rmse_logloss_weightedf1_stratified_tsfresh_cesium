import os

import pandas as pd

COMPETITION_NAME = "tabular-playground-series-jul-2021"

SUBMISSION_DIR = "."
SUBMISSION_FILE = "sub_lgb_benchmark_holdout_w_date_log_benzene_0721_1212_0.39573.csv"
SUBMISSION_MESSAGE = '"LGB Benchamrk with date on holdout with log benzene"'

df = pd.read_csv(f"{SUBMISSION_DIR}/{SUBMISSION_FILE}")
print(df.head())

submission_string = f"kaggle competitions submit {COMPETITION_NAME} -f {SUBMISSION_DIR}/{SUBMISSION_FILE} -m {SUBMISSION_MESSAGE}"

print(submission_string)

os.system(submission_string)
