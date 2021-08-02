import os

import pandas as pd

COMPETITION_NAME = "tabular-playground-series-jul-2021"

SUBMISSION_DIR = "."
SUBMISSION_FILE = "sub_lgb_holdout_w_date_log_ben_dew_0721_1510_0.38658.csv"
SUBMISSION_MESSAGE = '"LGB with date, dew point on holdout with log benzene"'

df = pd.read_csv(f"{SUBMISSION_DIR}/{SUBMISSION_FILE}")
print(df.head())

submission_string = f"kaggle competitions submit {COMPETITION_NAME} -f {SUBMISSION_DIR}/{SUBMISSION_FILE} -m {SUBMISSION_MESSAGE}"

print(submission_string)

os.system(submission_string)
