import os

import pandas as pd

COMPETITION_NAME = "tabular-playground-series-aug-2021"

SUBMISSION_DIR = "."
SUBMISSION_FILE = "sub_lgb_ts_f1_weighted_SK_all_features_0825_1233_0.08641.csv"
SUBMISSION_MESSAGE = '"LGB ts f1-weighted SKFold 5 all features"'

df = pd.read_csv(f"{SUBMISSION_DIR}/{SUBMISSION_FILE}")
print(df.head())

submission_string = f"kaggle competitions submit {COMPETITION_NAME} -f {SUBMISSION_DIR}/{SUBMISSION_FILE} -m {SUBMISSION_MESSAGE}"

os.system(submission_string)
