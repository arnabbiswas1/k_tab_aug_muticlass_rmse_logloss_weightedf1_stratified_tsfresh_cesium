"""
A simple and custom class for greedy feature selection.

> src$ python -m fs.greedy_feature_selection
"""
import gc
import os
from datetime import datetime
from timeit import default_timer as timer

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import src.common as common
import src.config.constants as constants
import src.munging.process_data_util as process_data
from src.modeling import _calculate_perf_metric as calculate_metric
from src.modeling import evaluate_macroF1_lgb


__all__ = [
    "GreedyFeatureSelection"
]


class GreedyFeatureSelection:
    def __init__(self, logger):
        self.logger = logger

    def __get_X_Y_from_CV(self, train_X, train_Y, train_index, validation_index):
        X_train, X_validation = (
            train_X.iloc[train_index].values,
            train_X.iloc[validation_index].values,
        )
        y_train, y_validation = (
            train_Y.iloc[train_index].values,
            train_Y.iloc[validation_index].values,
        )
        return X_train, X_validation, y_train, y_validation

    def __calculate_perf_metric(self, y, y_hat):
        return calculate_metric(metric_name="f1_score_weighted", y=y, y_hat=y_hat)

    def evalaute_score(self, train_X, train_Y):
        kf = StratifiedKFold(n_splits=2, shuffle=True)

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

        y_oof = np.zeros(shape=(len(train_X), NUM_CLASSES))

        features = list(train_X.columns)
        fold = 0
        # n_folds = kf.get_n_splits()
        for train_index, validation_index in kf.split(X=train_X, y=train_Y):
            fold += 1
            # logger.info(f"fold {fold} of {n_folds}")
            X_train, X_validation, y_train, y_validation = self.__get_X_Y_from_CV(
                train_X, train_Y, train_index, validation_index
            )

            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_eval = lgb.Dataset(X_validation, y_validation, reference=lgb_train)

            model = lgb.train(
                lgb_params,
                lgb_train,
                valid_sets=[lgb_train, lgb_eval],
                verbose_eval=VERBOSE,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                num_boost_round=N_ESTIMATORS,
                feature_name=features,
                # categorical_feature=[],
                feval=evaluate_macroF1_lgb
            )

            del lgb_train, lgb_eval, train_index, X_train, y_train
            gc.collect()

            y_oof[validation_index] = model.predict(
                X_validation, num_iteration=model.best_iteration
            )

        oof_score = round(self.__calculate_perf_metric(train_Y, y_oof), 5)
        return oof_score

    def _feature_selection(self, X, y):
        good_features = []
        best_scores = []

        all_features = list(X.columns)
        logger.info(f"All features present in the dataset {all_features}")

        while True:
            self.logger.info("Starting iteration over all the features...")
            self.logger.info(f"Good features so far {good_features}")
            self.logger.info(f"Best scores so far {best_scores}")
            this_feature = None
            best_score = 0

            # Loop over all the features
            # for feature in range(num_features):
            for feature in all_features:
                logger.info(f"Picking up feature {feature}")
                if feature in good_features:
                    continue
                selected_features = good_features + [feature]
                self.logger.info(
                    f"Features to be used for evaluation: {selected_features}"
                )
                # TODO : Sort the feature selection thing
                xtrain = X.loc[:, selected_features]
                score = self.evalaute_score(xtrain, y)
                self.logger.info(f"Resultant Score: [{score}]")
                if score > best_score:
                    logger.info(
                        f"With feature {feature}, scrore {score} is better than last best score {best_score}"
                    )
                    this_feature = feature
                    best_score = score
                else:
                    logger.info(
                        f"Dropping feature {feature}. Scrore {score} is not better than last best score {best_score}."
                    )

            if this_feature is not None:
                good_features.append(this_feature)
                best_scores.append(best_score)
                self.logger.info(
                    f"Good Features at the end of iteration: [{good_features}]"
                )
                self.logger.info(
                    f"Best Scores at the end of iteration: [{best_scores}]"
                )

            if len(best_scores) > 2:
                if best_scores[-1] < best_scores[-2]:
                    logger.info("Ending greedy selcetion loop")
                    break

        return best_scores[-2], good_features[:-1]

    def select_features(self, X, y):
        scores, features = self._feature_selection(X, y)
        return scores, features


if __name__ == "__main__":

    common.set_timezone()
    start = timer()

    # Create RUN_ID
    RUN_ID = datetime.now().strftime("%m%d_%H%M")
    MODEL_NAME = os.path.basename(__file__).split(".")[0]

    SEED = 42
    TARGET = "loss"

    LOGGER_NAME = "gfs"
    logger = common.get_logger(LOGGER_NAME, MODEL_NAME, RUN_ID, constants.LOG_DIR)
    common.set_seed(SEED)
    logger.info(f"Staring run for Model Number [{MODEL_NAME}] & [{RUN_ID}]")

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
        "loan__agg_linear_trend__attr_stderr__chunk_len_10__f_agg_mean",
        "loan__change_quantiles__f_agg_mean__isabs_False__qh_06__ql_02",
        "loan__fft_coefficient__attr_real__coeff_27",
        "loan__cwt_coefficients__coeff_8__w_2__widths_251020",
        "loan__fft_coefficient__attr_angle__coeff_27",
        "loan__ar_coefficient__coeff_10__k_10",
        "loan__fft_coefficient__attr_imag__coeff_46",
        "loan__fft_coefficient__attr_imag__coeff_12",
        "loan__skewness",
        "loan__agg_linear_trend__attr_stderr__chunk_len_5__f_agg_mean",
        "loan__fft_coefficient__attr_abs__coeff_27",
        "loan__fft_coefficient__attr_abs__coeff_46",
        "loan__autocorrelation__lag_1",
        "loan__fft_coefficient__attr_angle__coeff_12",
        "loan__cwt_coefficients__coeff_7__w_2__widths_251020",
        "loan__ar_coefficient__coeff_0__k_10",
        "loan__cwt_coefficients__coeff_9__w_2__widths_251020",
        "loan__autocorrelation__lag_2",
        "loan__fft_coefficient__attr_abs__coeff_12",
        "loan__fft_coefficient__attr_imag__coeff_27",
        "loan__change_quantiles__f_agg_mean__isabs_True__qh_10__ql_06",
        "loan__cwt_coefficients__coeff_12__w_2__widths_251020",
        "loan__autocorrelation__lag_3",
        "loan__autocorrelation__lag_4",
        "loan__fft_coefficient__attr_angle__coeff_46",
        "loan__kurtosis",
        "loan__autocorrelation__lag_5",
        "loan__change_quantiles__f_agg_mean__isabs_False__qh_04__ql_02",
        "loan__change_quantiles__f_agg_mean__isabs_False__qh_06__ql_00",
        "loan__change_quantiles__f_agg_mean__isabs_False__qh_06__ql_04",
        "loan__autocorrelation__lag_6",
        "loan__agg_linear_trend__attr_rvalue__chunk_len_10__f_agg_mean",
        "loan__change_quantiles__f_agg_mean__isabs_False__qh_08__ql_02",
        "loan__autocorrelation__lag_7",
        "loan__change_quantiles__f_agg_mean__isabs_False__qh_10__ql_04",
        "loan__autocorrelation__lag_8",
        "loan__agg_autocorrelation__f_agg_var__maxlag_40",
        "loan__change_quantiles__f_agg_mean__isabs_False__qh_02__ql_00",
        "loan__cwt_coefficients__coeff_11__w_2__widths_251020",
        "loan__change_quantiles__f_agg_var__isabs_False__qh_10__ql_06",
        "loan__agg_autocorrelation__f_agg_mean__maxlag_40",
        "loan__fft_coefficient__attr_real__coeff_13",
        "loan__autocorrelation__lag_9",
        "loan__agg_autocorrelation__f_agg_median__maxlag_40",
        "loan__fft_coefficient__attr_real__coeff_12",
        "loan__change_quantiles__f_agg_mean__isabs_False__qh_04__ql_00",
        "loan__cwt_coefficients__coeff_4__w_2__widths_251020",
        "loan__fft_coefficient__attr_real__coeff_46",
        "loan__fft_coefficient__attr_imag__coeff_28",
        "loan__change_quantiles__f_agg_mean__isabs_False__qh_08__ql_04",
        "loan__fft_coefficient__attr_real__coeff_31",
        "loan__fft_coefficient__attr_real__coeff_9",
        "loan__change_quantiles__f_agg_var__isabs_False__qh_02__ql_00",
        "loan__fft_coefficient__attr_abs__coeff_9",
        "loan__change_quantiles__f_agg_mean__isabs_False__qh_08__ql_00",
        "loan__agg_linear_trend__attr_stderr__chunk_len_5__f_agg_max",
        "loan__cwt_coefficients__coeff_3__w_2__widths_251020",
        "loan__change_quantiles__f_agg_mean__isabs_True__qh_04__ql_02",
        "loan__fft_coefficient__attr_angle__coeff_29",
        "loan__cwt_coefficients__coeff_5__w_2__widths_251020",
        "loan__fft_coefficient__attr_imag__coeff_31",
        "loan__fft_coefficient__attr_imag__coeff_37",
        "loan__fft_coefficient__attr_abs__coeff_5",
        "loan__change_quantiles__f_agg_mean__isabs_True__qh_02__ql_00",
        "loan__cwt_coefficients__coeff_6__w_2__widths_251020",
        "loan__cwt_coefficients__coeff_13__w_2__widths_251020",
        "loan__fft_coefficient__attr_abs__coeff_1",
        "loan__fft_coefficient__attr_abs__coeff_31",
        "loan__change_quantiles__f_agg_mean__isabs_False__qh_10__ql_00",
        "loan__fft_coefficient__attr_angle__coeff_13",
        "loan__change_quantiles__f_agg_mean__isabs_False__qh_10__ql_08",
        "loan__fft_coefficient__attr_angle__coeff_2",
        "loan__change_quantiles__f_agg_var__isabs_False__qh_04__ql_00",
        "loan__change_quantiles__f_agg_mean__isabs_False__qh_08__ql_06",
        "loan__change_quantiles__f_agg_var__isabs_False__qh_06__ql_00",
        "loan__change_quantiles__f_agg_mean__isabs_False__qh_10__ql_02",
        "loan__fft_coefficient__attr_real__coeff_8",
        "loan__change_quantiles__f_agg_var__isabs_True__qh_02__ql_00",
        "loan__fft_coefficient__attr_real__coeff_11",
        "loan__change_quantiles__f_agg_var__isabs_True__qh_10__ql_06",
        "loan__fft_coefficient__attr_abs__coeff_30",
        "loan__fft_coefficient__attr_abs__coeff_8",
        "loan__change_quantiles__f_agg_var__isabs_True__qh_06__ql_02",
        "loan__fft_coefficient__attr_abs__coeff_2",
        "loan__fft_coefficient__attr_imag__coeff_9",
        "loan__fft_coefficient__attr_angle__coeff_5",
        "loan__fft_coefficient__attr_abs__coeff_29",
        "loan__approximate_entropy__m_2__r_01",
        "loan__fft_coefficient__attr_angle__coeff_9",
        "loan__fft_coefficient__attr_abs__coeff_39",
        "loan__fft_coefficient__attr_real__coeff_26",
        "loan__change_quantiles__f_agg_var__isabs_True__qh_10__ql_04",
        "loan__change_quantiles__f_agg_mean__isabs_False__qh_10__ql_06",
        "loan__fft_coefficient__attr_abs__coeff_19",
        "loan__fft_coefficient__attr_abs__coeff_6",
        "loan__change_quantiles__f_agg_var__isabs_True__qh_04__ql_00",
        "loan__agg_linear_trend__attr_stderr__chunk_len_5__f_agg_min",
        "loan__agg_linear_trend__attr_stderr__chunk_len_10__f_agg_min",
        "loan__change_quantiles__f_agg_var__isabs_True__qh_06__ql_00",
        "loan__fft_coefficient__attr_abs__coeff_32",
    ]
    features_df = features_df[fetaures_to_use]
    logger.info(f"Shape of the selected features {features_df.shape}")

    train_X = features_df.iloc[0: len(train_df)]
    train_Y = train_df["loss"]
    test_X = features_df.iloc[len(train_df):]

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

    gfs = GreedyFeatureSelection(logger)
    scores, features = gfs.select_features(train_X, train_Y)
    logger.info(f"Final Best Score: {scores}")
    logger.info(f"Final Best Features: {features}")
