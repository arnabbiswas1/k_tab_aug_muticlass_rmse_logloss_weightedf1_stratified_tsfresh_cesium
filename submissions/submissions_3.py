import os

import pandas as pd

COMPETITION_NAME = "tabular-playground-series-jul-2021"

SUBMISSION_DIR = "."
SUBMISSION_FILE = "sub_lgb_benchmark_date_features_0720_1657_0.31635.csv"
SUBMISSION_MESSAGE = '"LGB Benchamrk date features"'

df = pd.read_csv(f"{SUBMISSION_DIR}/{SUBMISSION_FILE}")
print(df.head())

submission_string = f"kaggle competitions submit {COMPETITION_NAME} -f {SUBMISSION_DIR}/{SUBMISSION_FILE} -m {SUBMISSION_MESSAGE}"

print(submission_string)

os.system(submission_string)
