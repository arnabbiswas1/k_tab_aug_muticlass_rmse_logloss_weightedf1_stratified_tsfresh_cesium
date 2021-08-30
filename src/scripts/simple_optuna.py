import os
import gc
from datetime import datetime
import logging

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

import optuna as optuna
import lightgbm as lgb

import src.common as common
import src.modeling as train_util
import src.munging as process_data
import src.config.constants as constants


def weightedF1(y_hat, data):
    """
    Custom F1 Score to be used for multiclass classification using lightgbm.
    This function should be passed as a value to the parameter feval.

    weighted average takes care of imbalance

    https://stackoverflow.com/questions/57222667/light-gbm-early-stopping-does-not-work-for-custom-metric
    https://stackoverflow.com/questions/52420409/lightgbm-manual-scoring-function-f1-score
    https://stackoverflow.com/questions/51139150/how-to-write-custom-f1-score-metric-in-light-gbm-python-in-multiclass-classifica
    """
    y = data.get_label()
    y_hat = y_hat.reshape(-1, len(np.unique(y))).argmax(axis=1)
    f1 = metrics.f1_score(y_true=y, y_pred=y_hat, average="weighted")
    return ("weightedF1", f1, True)


def get_data():
    TARGET = "loss"
    # Read the processed data. Read the features with which best result has been
    # received so far.
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
    fetaures_to_use = [
        "loan__agg_linear_trend__attr_stderr__chunk_len_5__f_agg_mean",
        "loan__cwt_coefficients__coeff_6__w_2__widths_251020",
    ]
    features_df = features_df[fetaures_to_use]
    logger.info(f"Shape of the selected features {features_df.shape}")

    train_X = features_df.iloc[0: len(train_df)]
    train_Y = train_df[TARGET]
    test_X = features_df.iloc[len(train_df):]

    logger.info(
        f"Shape of train_X : {train_X.shape}, test_X: {test_X.shape}, train_Y: {train_Y.shape}"
    )

    logger.debug(f"Shape of train_X: {train_X.shape}, train_Y: {train_Y.shape}")

    predictors = list(train_X.columns)

    return train_X, train_Y, predictors


def objective(trial):
    # Para Definition from : https://www.kaggle.com/isaienkov/lgbm-optuna-rfe
    params = {
        "objective": "multiclass",
        "metric": "custom",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": 42,
        "metric": "custom",
        "num_class": 43,

        "num_leaves": trial.suggest_int("num_leaves", 2, 31),
        "n_estimators": trial.suggest_int("n_estimators", 20, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "min_child_samples": trial.suggest_int("min_child_samples", 100, 1200),
        "learning_rate": trial.suggest_uniform("learning_rate", 0.0001, 0.99),
        "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.0001, 1.0),
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.0001, 1.0),
    }

    train_X, train_Y, predictors = get_data()

    skf = StratifiedKFold(n_splits=3, shuffle=False)
    n_folds = skf.get_n_splits()
    fold = 0
    
    y_oof = np.zeros(shape=(len(train_X), 43))
    cv_scores = []
    for train_index, validation_index in skf.split(X=train_X, y=train_Y):
        fold += 1
        logger.info(f"fold {fold} of {n_folds}")

        X_train, X_validation, y_train, y_validation = train_util._get_X_Y_from_CV(
            train_X, train_Y, train_index, validation_index
        )

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_validation, y_validation, reference=lgb_train)

        # Use custom metrics
        # https://github.com/optuna/optuna/issues/861
        # https://gist.github.com/smly/5a5ddf968d59492b79e4cbf90b2d3430
        pruning_callback = optuna.integration.LightGBMPruningCallback(
            trial, "weightedF1"
        )

        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_eval],
            verbose_eval=False,
            early_stopping_rounds=100,
            feature_name=predictors,
            callbacks=[pruning_callback],
            feval=weightedF1
        )

        del lgb_train, lgb_eval, train_index, X_train, y_train
        gc.collect()

        y_oof[validation_index] = model.predict(X_validation)

        logger.info(y_oof[validation_index])

        cv_oof_score = train_util._calculate_perf_metric(
            "f1_score_weighted", y_validation, y_oof[validation_index]
        )
        logger.info(f"CV Score for fold {fold}: {cv_oof_score}")
        cv_scores.append(cv_oof_score)

    mean_cv_score = sum(cv_scores) / len(cv_scores)
    logger.info(f"Mean CV Score {mean_cv_score}")
    return mean_cv_score


if __name__ == "__main__":

    RUN_ID = datetime.now().strftime("%m%d_%H%M")
    MODEL_NAME = os.path.basename(__file__).split(".")[0]

    LOGGER_NAME = "hpo"
    logger = common.get_logger(LOGGER_NAME, MODEL_NAME, RUN_ID, constants.LOG_DIR)
    logger.setLevel(logging.WARNING)

    EARLY_STOPPING_ROUNDS = 100

    # Optimization
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=100), direction="maximize"
    )
    study.optimize(
        objective,
        n_trials=3,
        # timeout=3600 * 4,
    )

    logger.warning(f"Number of finished trials: {len(study.trials)}")

    best_score = study.best_value
    best_params = study.best_params
    # A dictionary
    param_importance = optuna.importance.get_param_importances(study)

    logger.warning(f"Best score: {best_score}")
    logger.warning(f"Best params: {best_params}")
    logger.warning(f"Parameter importnace: {param_importance}")

    # Save the best params
    common.save_optuna_artifacts(
        logger,
        best_score=best_score,
        best_params=best_params,
        param_importance=param_importance,
        run_id=RUN_ID,
        hpo_dir=constants.HPO_DIR,
    )