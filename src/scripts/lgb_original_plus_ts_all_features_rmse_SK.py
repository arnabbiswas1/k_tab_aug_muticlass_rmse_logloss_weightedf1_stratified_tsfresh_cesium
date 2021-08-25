"""
LGB original_fetaures, ts top 50 features, rmse SKFold 10
"""

import os
from datetime import datetime
from timeit import default_timer as timer

import pandas as pd

from sklearn.model_selection import StratifiedKFold

import src.common as common
import src.config.constants as constants
import src.modeling.train_util as model
import src.munging.process_data_util as process_data

if __name__ == "__main__":
    common.set_timezone()
    start = timer()

    # Create RUN_ID
    RUN_ID = datetime.now().strftime("%m%d_%H%M")
    MODEL_NAME = os.path.basename(__file__).split(".")[0]

    SEED = 42
    EXP_DETAILS = "LGB original_fetaures, ts top 50 features, rmse SKFold 10"
    IS_TEST = False
    PLOT_FEATURE_IMPORTANCE = False
    N_SPLITS = 10
    TARGET = "loss"

    MODEL_TYPE = "lgb"
    OBJECTIVE = "root_mean_squared_error"
    METRIC = "RMSE"
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

    features_df = pd.read_parquet(
        f"{constants.FEATURES_DATA_DIR}/cast/tsfresh_f_merged.parquet"
    )
    logger.info(f"Shape of the features {features_df.shape}")

    combined_df = pd.concat([train_df.drop("loss", axis=1), test_df])
    orginal_features = list(test_df.columns)
    combined_df = pd.concat([combined_df, features_df], axis=1)
    logger.info(f"Shape of combined data with features {combined_df.shape}")

    features_to_include = [
        "loan__fft_coefficient__attr_imag__coeff_21",
        "loan__fft_coefficient__attr_imag__coeff_27",
        "loan__fft_coefficient__attr_imag__coeff_19",
        "loan__fft_coefficient__attr_real__coeff_44",
        "loan__fft_coefficient__attr_real__coeff_27",
        "loan__fft_coefficient__attr_imag__coeff_36",
        "loan__fft_coefficient__attr_real__coeff_29",
        "loan__fft_coefficient__attr_real__coeff_13",
        "loan__fft_coefficient__attr_real__coeff_37",
        "loan__fft_coefficient__attr_imag__coeff_37",
        "loan__fft_coefficient__attr_real__coeff_11",
        "loan__fft_coefficient__attr_real__coeff_6",
        "loan__fft_coefficient__attr_imag__coeff_43",
        "loan__fft_coefficient__attr_real__coeff_22",
        "loan__fft_coefficient__attr_imag__coeff_45",
        "loan__cwt_coefficients__coeff_3__w_2__widths_251020",
        "loan__fft_coefficient__attr_real__coeff_46",
        "loan__fft_coefficient__attr_real__coeff_40",
        "loan__fft_coefficient__attr_imag__coeff_35",
        "loan__fft_coefficient__attr_angle__coeff_46",
        "loan__fft_coefficient__attr_real__coeff_39",
        "loan__fft_coefficient__attr_real__coeff_48",
        "loan__fft_coefficient__attr_imag__coeff_3",
        "loan__fft_coefficient__attr_real__coeff_31",
        "loan__fft_coefficient__attr_imag__coeff_20",
        "loan__energy_ratio_by_chunks__num_segments_10__segment_focus_7",
        "loan__fft_coefficient__attr_imag__coeff_30",
        "loan__fft_coefficient__attr_real__coeff_21",
        "loan__fft_coefficient__attr_abs__coeff_27",
        "loan__fft_coefficient__attr_imag__coeff_46",
        "loan__fft_coefficient__attr_abs__coeff_42",
        "loan__fft_coefficient__attr_angle__coeff_37",
        "loan__fft_coefficient__attr_real__coeff_12",
        "loan__fft_coefficient__attr_imag__coeff_42",
        "loan__fft_coefficient__attr_angle__coeff_6",
        "loan__fft_coefficient__attr_imag__coeff_9",
        "loan__fft_coefficient__attr_angle__coeff_29",
        "loan__fft_coefficient__attr_imag__coeff_4",
        "loan__fft_coefficient__attr_angle__coeff_21",
        "loan__fft_coefficient__attr_real__coeff_17",
        "loan__fft_coefficient__attr_imag__coeff_34",
        "loan__fft_coefficient__attr_angle__coeff_45",
        "loan__fft_coefficient__attr_imag__coeff_26",
        "loan__fft_coefficient__attr_real__coeff_2",
        "loan__fft_coefficient__attr_imag__coeff_11",
        "loan__energy_ratio_by_chunks__num_segments_10__segment_focus_6",
        "loan__fft_coefficient__attr_imag__coeff_10",
        "loan__fft_coefficient__attr_angle__coeff_36",
        "loan__fft_coefficient__attr_imag__coeff_6",
        "loan__fft_coefficient__attr_abs__coeff_9",
    ]

    combined_df = combined_df.loc[:, orginal_features + features_to_include]
    logger.info(f"Shape of the data after selecting features {combined_df.shape}")

    train_X = combined_df.iloc[0: len(train_df)]
    train_Y = train_df["loss"]
    test_X = combined_df.iloc[len(train_df):]

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
        metric="rmse",
        num_class=None,
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

    common.save_artifacts(
        logger,
        is_test=False,
        is_plot_fi=True,
        result_dict=results_dict,
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
