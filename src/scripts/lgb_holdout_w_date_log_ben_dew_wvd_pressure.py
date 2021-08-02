"""
LGB Benchamrk with date, dew point
on holdout with log benzene, cat features
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
    EXP_DETAILS = "LGB with date, dew point, s_vvd on holdout with log benzene, cat features"
    IS_TEST = False
    PLOT_FEATURE_IMPORTANCE = False

    TARGET = "target"

    MODEL_TYPE = "lgb"
    OBJECTIVE = "regression"
    NUM_CLASSES = 1
    METRIC = "rmse"
    BOOSTING_TYPE = "gbdt"
    VERBOSE = 100
    N_THREADS = -1
    NUM_LEAVES = 31
    MAX_DEPTH = -1
    N_ESTIMATORS = 1000
    LEARNING_RATE = 0.1
    EARLY_STOPPING_ROUNDS = 100

    lgb_params = {
        "objective": OBJECTIVE,
        "boosting_type": BOOSTING_TYPE,
        "learning_rate": LEARNING_RATE,
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

    features = pd.read_parquet(f"{constants.FEATURES_DATA_DIR}/features.parquet")

    targets = ["target_carbon_monoxide", "target_benzene", "target_nitrogen_oxides"]
    date = ["date_time"]

    test_df_org = test_df.copy()
    targets_df = train_df[targets]

    combinded_df = pd.concat([train_df.drop(targets, axis=1), test_df])
    combinded_df = combinded_df.reset_index(drop=True)

    sel_features = [
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
        "dew_point"
    ]

    cat_features = [
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

    combinded_df = pd.concat([combinded_df, features[sel_features]], axis=1)

    for name in combinded_df.select_dtypes("bool"):
        combinded_df[name] = combinded_df[name].astype(int)

    for name in combinded_df.select_dtypes(include=["object", "category"]):
        lb = LabelEncoder()
        combinded_df[name] = lb.fit_transform(combinded_df[name])

    train_df = combinded_df.iloc[0: len(train_df)]
    test_df = combinded_df.iloc[len(train_df):]

    train_df = pd.concat([train_df, targets_df], axis=1)

    logger.info(
        f"Shape of train after adding features : {train_df.shape}, test_df: {test_df.shape}"
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
        "deg_C",
        "relative_humidity",
        "absolute_humidity",
        "sensor_1",
        "sensor_2",
        "sensor_3",
        "sensor_4",
        "sensor_5",
        "dew_point",
    ]
    logger.info(f"List of predictors {predictors}")

    common.update_tracking(RUN_ID, "no_of_features", len(predictors), is_integer=True)
    common.update_tracking(RUN_ID, "cv_method", "holdout")

    results_dict_co = model.lgb_train_validate_on_holdout(
        logger=logger,
        run_id=RUN_ID,
        training=training_df,
        validation=validation_df,
        test_X=test_df,
        predictors=predictors,
        target="target_carbon_monoxide",
        params=lgb_params,
        n_estimators=10000,
        early_stopping_rounds=100,
        cat_features=cat_features,
        verbose_eval=100,
        label_name="co",
        log_target=False,
    )

    results_dict_benzene = model.lgb_train_validate_on_holdout(
        logger=logger,
        run_id=RUN_ID,
        training=training_df,
        validation=validation_df,
        test_X=test_df,
        predictors=predictors,
        target="target_benzene",
        params=lgb_params,
        n_estimators=10000,
        early_stopping_rounds=100,
        cat_features=cat_features,
        verbose_eval=100,
        label_name="benzene",
        log_target=True,
    )

    results_dict_no = model.lgb_train_validate_on_holdout(
        logger=logger,
        run_id=RUN_ID,
        training=training_df,
        validation=validation_df,
        test_X=test_df,
        predictors=predictors,
        target="target_nitrogen_oxides",
        params=lgb_params,
        n_estimators=10000,
        early_stopping_rounds=100,
        cat_features=cat_features,
        verbose_eval=100,
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
