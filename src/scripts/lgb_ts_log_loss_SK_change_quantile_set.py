"""
LGB ts log_loss SKFold 10 change_quantile_set
"""

import numpy as np
import pandas as pd

import os
from datetime import datetime
from timeit import default_timer as timer

from sklearn.model_selection import StratifiedKFold

import src.common as common
import src.config.constants as constants
import src.modeling.train_util as model
import src.munging.process_data_util as process_data

import src.common.com_util as util

if __name__ == "__main__":
    common.set_timezone()
    start = timer()

    # Create RUN_ID
    RUN_ID = datetime.now().strftime("%m%d_%H%M")
    MODEL_NAME = os.path.basename(__file__).split(".")[0]

    SEED = 42
    EXP_DETAILS = "LGB ts log_loss SKFold 10 change_quantile_set"
    IS_TEST = False
    PLOT_FEATURE_IMPORTANCE = False

    N_SPLITS = 10

    TARGET = "loss"

    MODEL_TYPE = "lgb"
    OBJECTIVE = "multiclass"
    NUM_CLASSES = 43
    METRIC = "multi_logloss"
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

    LOGGER_NAME = "sub_1"
    logger = common.get_logger(LOGGER_NAME, MODEL_NAME, RUN_ID, constants.LOG_DIR)
    common.set_seed(SEED)
    logger.info(f"Running for Model Number [{MODEL_NAME}] & [{RUN_ID}]")

    common.update_tracking(
        RUN_ID, "model_number", MODEL_NAME, drop_incomplete_rows=True
    )
    common.update_tracking(RUN_ID, "model_type", MODEL_TYPE)
    common.update_tracking(RUN_ID, "is_test", IS_TEST)
    common.update_tracking(RUN_ID, "n_estimators", N_ESTIMATORS)
    common.update_tracking(RUN_ID, "learning_rate", LEARNING_RATE)
    common.update_tracking(RUN_ID, "num_leaves", NUM_LEAVES)
    common.update_tracking(RUN_ID, "early_stopping_rounds", EARLY_STOPPING_ROUNDS)

    train_df, test_df, sample_submission_df = process_data.read_processed_data(
        logger,
        constants.PROCESSED_DATA_DIR,
        train=True,
        test=True,
        sample_submission=True,
    )

    features_df = pd.read_parquet(
        f"{constants.FEATURES_DATA_DIR}/cast/change_quantile_set_cast.parquet"
    )

    logger.info(f"Shape of the features {features_df.shape}")
    features_to_drop = [
    ]
    features_df = features_df.drop(features_to_drop, axis=1)
    logger.info(f"Shape of the features after dropping {features_df.shape}")

    train_X = features_df.iloc[0 : len(train_df)]
    train_Y = train_df["loss"]
    test_X = features_df.iloc[len(train_df) :]

    logger.info("Adding additional rows for loss=42")
    train_X_rare = train_X.loc[[96131, 131570, 212724]]
    train_X = train_X.append(
        [train_X_rare, train_X_rare, train_X_rare], ignore_index=True
    )

    train_Y_rare = train_Y.loc[[96131, 131570, 212724]]
    train_Y = train_Y.append(
        [train_Y_rare, train_Y_rare, train_Y_rare], ignore_index=True
    )

    logger.info(
        f"Shape of train_X : {train_X.shape}, test_X: {test_X.shape}, train_Y: {train_Y.shape}"
    )

    predictors = list(train_X.columns)
    sk = StratifiedKFold(n_splits=N_SPLITS, shuffle=True)

    common.update_tracking(RUN_ID, "no_of_features", len(predictors), is_integer=True)
    common.update_tracking(RUN_ID, "cv_method", "StratifiedKFold")
    common.update_tracking(RUN_ID, "n_splits", N_SPLITS, is_integer=True)

    results_dict = model.lgb_train_validate_on_cv(
        logger=logger,
        run_id=RUN_ID,
        train_X=train_X,
        train_Y=train_Y,
        test_X=test_X,
        metric="log_loss",
        num_class=NUM_CLASSES,
        kf=sk,
        features=predictors,
        params=lgb_params,
        n_estimators=N_ESTIMATORS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        cat_features="auto",
        is_test=False,
        verbose_eval=100,
    )

    train_index = train_X.index

    # Since we are using multiclass classification with logloss as metric
    # the prediction and y_oof consist of probablities for 43 classes.
    # Convert those to a label representing the number of the class
    results_dict_copy = results_dict.copy()
    logger.info(results_dict["prediction"])
    results_dict_copy["prediction"] = np.argmax(results_dict["prediction"], axis=1)
    results_dict_copy["y_oof"] = np.argmax(results_dict["y_oof"], axis=1)

    rmse_score = model._calculate_perf_metric(
        "rmse", train_Y.values, results_dict_copy["y_oof"]
    )
    logger.info(f"RMSE score {rmse_score}")
    util.update_tracking(run_id=RUN_ID, key="RMSE", value=rmse_score, is_integer=False)

    common.save_artifacts(
        logger,
        is_test=False,
        is_plot_fi=True,
        result_dict=results_dict_copy,
        submission_df=sample_submission_df,
        train_index=train_index,
        model_number=MODEL_NAME,
        run_id=RUN_ID,
        sub_dir=constants.SUBMISSION_DIR,
        oof_dir=constants.OOF_DIR,
        fi_dir=constants.FI_DIR,
        fi_fig_dir=constants.FI_FIG_DIR,
    )

    end = timer()
    common.update_tracking(RUN_ID, "training_time", end - start, is_integer=True)
    common.update_tracking(RUN_ID, "comments", EXP_DETAILS)
    logger.info("Execution Complete")
