"""
Cat Benchamrk date features
"""

import os
from datetime import datetime
from timeit import default_timer as timer

import pandas as pd

from sklearn.preprocessing import LabelEncoder

import src.common as common
import src.config.constants as constants
import src.modeling.train_util as model
import src.munging.process_data_util as process_data
import src.cv as cv

if __name__ == "__main__":
    common.set_timezone()
    start = timer()

    # Create RUN_ID
    RUN_ID = datetime.now().strftime("%m%d_%H%M")
    MODEL_NAME = os.path.basename(__file__).split(".")[0]

    SEED = 42
    EXP_DETAILS = "Cat Benchamrk date features"
    IS_TEST = False
    PLOT_FEATURE_IMPORTANCE = False

    TARGET = "target"

    MODEL_TYPE = "xgb"
    OBJECTIVE = "reg:squaredlogerror"
    METRIC = "rmsle"
    BOOSTING_TYPE = "gbtree"
    N_THREADS = -1
    NUM_LEAVES = 31
    MAX_DEPTH = 16
    N_ESTIMATORS = 1000
    LEARNING_RATE = 0.1
    EARLY_STOPPING_ROUNDS = 100
    MAX_BIN = 254
    VERBOSE_EVAL = 100

    xgb_params = {
        # Learning task parameters
        "objective": OBJECTIVE,
        "eval_metric": METRIC,
        "seed": SEED,
        # Type of the booster
        "booster": BOOSTING_TYPE,
        # parameters for tree booster
        "learning_rate": LEARNING_RATE,
        "max_depth": MAX_DEPTH,
        "max_leaves": NUM_LEAVES,
        "max_bin": MAX_BIN,
        # General parameters
        "nthread": N_THREADS,
        # Following generated too much of logging
        "verbosity": 0,
        "validate_parameters": True,
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

    features = pd.read_parquet(f"{constants.FEATURES_DATA_DIR}/features.parquet")

    targets = ["target_carbon_monoxide", "target_benzene", "target_nitrogen_oxides"]
    features_to_drop = ["date_time"]

    test_df_org = test_df.copy()
    targets_df = train_df[targets]

    combinded_df = pd.concat([train_df.drop(targets, axis=1), test_df])
    combinded_df = combinded_df.reset_index(drop=True)

    date_features = [
        "month",
        "quarter",
        "weekofyear",
        "hour",
        "day",
        "dayofweek",
        "day_type",
        "dayofyear",
        "is_month_start",
        "is_month_end",
        "is_quarter_start",
        "is_quarter_end",
        "us_season",
        "part_of_day",
    ]

    combinded_df = pd.concat([combinded_df, features[date_features]], axis=1)

    for name in combinded_df.select_dtypes("bool"):
        combinded_df[name] = combinded_df[name].astype(int)

    for name in combinded_df.select_dtypes(include=["object", "category"]):
        lb = LabelEncoder()
        combinded_df[name] = lb.fit_transform(combinded_df[name])

    logger.info(f"Dropping feature {features_to_drop}")
    combinded_df = combinded_df.drop(features_to_drop, axis=1)

    train_df = combinded_df.iloc[0: len(train_df)]
    test_df = combinded_df.iloc[len(train_df):]

    train_df = pd.concat([train_df, targets_df], axis=1)

    logger.info("train_df data Types")
    logger.info(train_df.info())
    logger.info("test_df data Types")
    logger.info(test_df.info())

    logger.info(
        f"Shape of train & test after adding features : {train_df.shape}, test_df: {test_df.shape}"
    )

    logger.info(
        f"Features present in train : {train_df.columns}, & in test_df: {test_df.columns}"
    )

    # Training on first six months
    train_months = [3, 4, 5, 6, 7, 8, 9]
    # Holdout on last three months
    validation_months = [10, 11, 12]

    training_df, validation_df = cv.get_data_splits_by_month(
        logger=logger,
        df=train_df,
        train_months=train_months,
        validation_months=validation_months,
    )

    predictors = [
        "deg_C",
        "relative_humidity",
        "absolute_humidity",
        "sensor_1",
        "sensor_2",
        "sensor_3",
        "sensor_4",
        "sensor_5",
        "month",
        "quarter",
        "weekofyear",
        "hour",
        "day",
        "dayofweek",
        "day_type",
        "dayofyear",
        "is_month_start",
        "is_month_end",
        "is_quarter_start",
        "is_quarter_end",
        "us_season",
        "part_of_day",
    ]
    logger.info(f"List of predictors {predictors}")

    common.update_tracking(RUN_ID, "no_of_features", len(predictors), is_integer=True)
    common.update_tracking(RUN_ID, "cv_method", "holdout")

    results_dict_co = model.xgb_train_validate_on_holdout(
        logger=logger,
        run_id=RUN_ID,
        training=training_df,
        validation=validation_df,
        predictors=predictors,
        target="target_carbon_monoxide",
        params=xgb_params.copy(),
        test_X=test_df,
        n_estimators=N_ESTIMATORS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=VERBOSE_EVAL,
        label_name="co",
        log_target=False,
    )

    results_dict_benzene = model.xgb_train_validate_on_holdout(
        logger=logger,
        run_id=RUN_ID,
        training=training_df,
        validation=validation_df,
        predictors=predictors,
        target="target_benzene",
        params=xgb_params.copy(),
        test_X=test_df,
        n_estimators=N_ESTIMATORS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=VERBOSE_EVAL,
        label_name="benzene",
        log_target=False,
    )

    results_dict_no = model.xgb_train_validate_on_holdout(
        logger=logger,
        run_id=RUN_ID,
        training=training_df,
        validation=validation_df,
        predictors=predictors,
        target="target_nitrogen_oxides",
        params=xgb_params.copy(),
        test_X=test_df,
        n_estimators=N_ESTIMATORS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=VERBOSE_EVAL,
        label_name="no",
        log_target=False,
    )

    common.save_artifacts_holdout(
        logger=logger,
        is_test=False,
        is_plot_fi=True,
        result_dict=results_dict_co,
        model_number=MODEL_NAME,
        run_id=RUN_ID,
        oof_dir=constants.OOF_DIR,
        fi_dir=constants.FI_DIR,
        fi_fig_dir=constants.FI_FIG_DIR,
        label_name="co",
    )

    common.save_artifacts_holdout(
        logger=logger,
        is_test=False,
        is_plot_fi=True,
        result_dict=results_dict_benzene,
        model_number=MODEL_NAME,
        run_id=RUN_ID,
        oof_dir=constants.OOF_DIR,
        fi_dir=constants.FI_DIR,
        fi_fig_dir=constants.FI_FIG_DIR,
        label_name="benzene",
    )

    common.save_artifacts_holdout(
        logger=logger,
        is_test=False,
        is_plot_fi=True,
        result_dict=results_dict_no,
        model_number=MODEL_NAME,
        run_id=RUN_ID,
        oof_dir=constants.OOF_DIR,
        fi_dir=constants.FI_DIR,
        fi_fig_dir=constants.FI_FIG_DIR,
        label_name="no",
    )

    agg_val_score = common.calculate_final_score(
        RUN_ID, results_dict_co, results_dict_benzene, results_dict_no
    )

    logger.info(f"agg_val_score: {agg_val_score}")
    common.create_submission_file(
        logger=logger,
        run_id=RUN_ID,
        model_number=MODEL_NAME,
        sub_dir=constants.SUBMISSION_DIR,
        score=agg_val_score,
        sub_df=sample_submission_df,
        test_df=test_df_org,
        results_dict_co=results_dict_co,
        results_dict_ben=results_dict_benzene,
        results_dict_no=results_dict_no,
    )

    end = timer()
    common.update_tracking(RUN_ID, "training_time", end - start, is_integer=True)
    common.update_tracking(RUN_ID, "comments", EXP_DETAILS)
    logger.info("Execution Complete")
