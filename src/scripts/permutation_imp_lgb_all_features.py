"""
Permutation Importance using all created features
"""

import os
from timeit import default_timer as timer
from datetime import datetime

import pandas as pd
from sklearn.model_selection import StratifiedKFold

import src.common as common
import src.config.constants as constants
import src.modeling.train_util as model
import src.munging.process_data_util as process_data

common.set_timezone()
start = timer()

# Create RUN_ID
RUN_ID = datetime.now().strftime("%m%d_%H%M")
MODEL_NAME = os.path.basename(__file__).split('.')[0]

SEED = 42
EXP_DETAILS = "Permutation Importance using using all created features"
IS_TEST = False
PLOT_FEATURE_IMPORTANCE = False

N_SPLITS = 3

TARGET = "loss"

MODEL_TYPE = "lgb"
OBJECTIVE = "multiclass"
NUM_CLASSES = 43
# Note: Metric is set to custom
METRIC = "custom"
BOOSTING_TYPE = "gbdt"
VERBOSE = 100
N_THREADS = 8
NUM_LEAVES = 31
MAX_DEPTH = -1
N_ESTIMATORS = 1000
LEARNING_RATE = 0.1
EARLY_STOPPING_ROUNDS = 100

lgb_params = {
    "objective": OBJECTIVE,
    "boosting_type": BOOSTING_TYPE,
    "learning_rate": LEARNING_RATE,
    "num_class": NUM_CLASSES,
    "num_leaves": NUM_LEAVES,
    "tree_learner": "serial",
    "n_jobs": N_THREADS,
    "seed": SEED,
    "max_depth": MAX_DEPTH,
    "max_bin": 255,
    "metric": METRIC,
    "verbose": -1,
}

LOGGER_NAME = 'main'
logger = common.get_logger(LOGGER_NAME, MODEL_NAME, RUN_ID, constants.LOG_DIR)
common.set_seed(SEED)
logger.info(f'Running for Model Number [{MODEL_NAME}] & [{RUN_ID}]')

train_df, test_df, sample_submission_df = process_data.read_processed_data(
    logger,
    constants.PROCESSED_DATA_DIR,
    train=True,
    test=True,
    sample_submission=True,
)

features_df = pd.read_parquet(
    f"{constants.FEATURES_DATA_DIR}/cast/tsfresh_f_merged.parquet"
)

logger.info(f"Shape of the features {features_df.shape}")

df_cesium = pd.read_parquet(f"{constants.FEATURES_DATA_DIR}/cesium_final.parquet")
features_df = pd.concat([features_df, df_cesium], axis=1)

train_X = features_df.iloc[0: len(train_df)]
train_Y = train_df["loss"]
test_X = features_df.iloc[len(train_df):]

logger.info(f"Shape of train_X: {train_X.shape}, train_Y: {train_Y.shape}, test_X: {test_X.shape}")

predictors = list(train_X.columns)
logger.info(f"List of predictors {predictors}")

sk = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

permu_imp_df, top_imp_df = model.lgb_train_perm_importance_on_cv(
    logger,
    run_id=RUN_ID,
    train_X=train_X,
    train_Y=train_Y,
    kf=sk,
    features=predictors,
    seed=SEED,
    score_function=model.f1_score_weighted,
    params=lgb_params,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    cat_features=[],
    verbose_eval=100,
    display_imp=False,
    feval=model.evaluate_macroF1_lgb
)


common.save_permutation_imp_artifacts(
    logger,
    permu_imp_df,
    top_imp_df,
    RUN_ID,
    MODEL_NAME,
    constants.FI_DIR,
    constants.FI_FIG_DIR,
)
