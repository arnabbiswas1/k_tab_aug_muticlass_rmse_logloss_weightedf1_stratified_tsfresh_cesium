"""
LGB Benchamrk without date
"""

import os
from datetime import datetime
from timeit import default_timer as timer

from sklearn.model_selection import TimeSeriesSplit

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
    EXP_DETAILS = "LGB Benchamrk without date"
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

    targets = ["target_carbon_monoxide", "target_benzene", "target_nitrogen_oxides"]
    date = ["date_time"]

    train_X = train_df.drop(date + targets, axis=1)
    train_Ys = train_df[targets]
    test_X = test_df.drop(date, axis=1)

    logger.info(
        f"Shape of train_X : {train_X.shape}, test_X: {test_X.shape}, train_Y: {train_Ys.shape}"
    )

    predictors = list(train_X.columns)
    logger.info(f"List of predictors {predictors}")

    common.update_tracking(RUN_ID, "no_of_features", len(predictors), is_integer=True)
    common.update_tracking(RUN_ID, "cv_method", "ts split")

    ts_split = TimeSeriesSplit(n_splits=5, test_size=train_df.shape[0] // 5, gap=0)

    train_Y = train_Ys["target_carbon_monoxide"]
    results_dict_co = model.lgb_train_validate_on_cv(
        logger,
        run_id=RUN_ID,
        train_X=train_X,
        train_Y=train_Y,
        test_X=test_X,
        num_class=1,
        kf=ts_split,
        features=predictors,
        params=lgb_params,
        n_estimators=1000,
        early_stopping_rounds=100,
        cat_features="auto",
        is_test=False,
        verbose_eval=100,
        label_name="co",
    )
    train_index = train_df.index
    common.save_artifacts(
        logger,
        is_test=False,
        is_plot_fi=True,
        result_dict=results_dict_co,
        submission_df=sample_submission_df,
        train_index=train_index,
        model_number=MODEL_NAME,
        run_id=RUN_ID,
        sub_dir=constants.SUBMISSION_DIR,
        oof_dir=constants.OOF_DIR,
        fi_dir=constants.FI_DIR,
        fi_fig_dir=constants.FI_FIG_DIR,
        label_name="co",
    )

    train_Y = train_Ys["target_benzene"]
    results_dict_ben = model.lgb_train_validate_on_cv(
        logger,
        run_id=RUN_ID,
        train_X=train_X,
        train_Y=train_Y,
        test_X=test_X,
        num_class=1,
        kf=ts_split,
        features=predictors,
        params=lgb_params,
        n_estimators=1000,
        early_stopping_rounds=100,
        cat_features="auto",
        is_test=False,
        verbose_eval=100,
        label_name="ben",
    )

    train_index = train_df.index
    common.save_artifacts(
        logger,
        is_test=False,
        is_plot_fi=True,
        result_dict=results_dict_ben,
        submission_df=sample_submission_df,
        train_index=train_index,
        model_number=MODEL_NAME,
        run_id=RUN_ID,
        sub_dir=constants.SUBMISSION_DIR,
        oof_dir=constants.OOF_DIR,
        fi_dir=constants.FI_DIR,
        fi_fig_dir=constants.FI_FIG_DIR,
        label_name="ben",
    )

    train_Y = train_Ys["target_nitrogen_oxides"]
    results_dict_no = model.lgb_train_validate_on_cv(
        logger,
        run_id=RUN_ID,
        train_X=train_X,
        train_Y=train_Y,
        test_X=test_X,
        num_class=1,
        kf=ts_split,
        features=predictors,
        params=lgb_params,
        n_estimators=1000,
        early_stopping_rounds=100,
        cat_features="auto",
        is_test=False,
        verbose_eval=100,
        label_name="no",
    )

    train_index = train_df.index
    common.save_artifacts(
        logger,
        is_test=False,
        is_plot_fi=True,
        result_dict=results_dict_no,
        submission_df=sample_submission_df,
        train_index=train_index,
        model_number=MODEL_NAME,
        run_id=RUN_ID,
        sub_dir=constants.SUBMISSION_DIR,
        oof_dir=constants.OOF_DIR,
        fi_dir=constants.FI_DIR,
        fi_fig_dir=constants.FI_FIG_DIR,
        label_name="no",
    )

    (agg_oof_score, agg_avg_cv_scores) = common.calculate_final_score(
        RUN_ID, results_dict_co, results_dict_ben, results_dict_no
    )

    logger.info(f"agg_oof_score: {agg_oof_score}, agg_avg_cv_scores: {agg_avg_cv_scores}")

    common.create_submission_file(
        logger=logger,
        run_id=RUN_ID,
        model_number=MODEL_NAME,
        sub_dir=constants.SUBMISSION_DIR,
        score=agg_avg_cv_scores,
        sub_df=sample_submission_df,
        test_df=test_df,
        results_dict_co=results_dict_co,
        results_dict_ben=results_dict_ben,
        results_dict_no=results_dict_no,
    )

    end = timer()
    common.update_tracking(RUN_ID, "training_time", end - start, is_integer=True)
    common.update_tracking(RUN_ID, "comments", EXP_DETAILS)
    logger.info("Execution Complete")
