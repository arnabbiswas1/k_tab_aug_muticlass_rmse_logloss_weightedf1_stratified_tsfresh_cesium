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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import src.common as common
import src.config.constants as constants
import src.munging.process_data_util as process_data


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
        return roc_auc_score(y, y_hat)

    def evalaute_score(self, train_X, train_Y):
        kf = StratifiedKFold(n_splits=5, shuffle=False)

        OBJECTIVE = "binary"
        BOOSTING_TYPE = "gbdt"
        METRIC = "auc"
        VERBOSE = 100
        N_THREADS = -1
        NUM_LEAVES = 31
        MAX_DEPTH = -1
        N_ESTIMATORS = 10000
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

        y_oof = np.zeros(len(train_X))

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
                categorical_feature="auto",
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
    TARGET = "target"

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

    # Prepare data
    combined_df = pd.concat([train_df.drop("target", axis=1), test_df])
    target = train_df[TARGET]

    cat_fetaures = [name for name in train_df.columns if "cat" in name]

    logger.info("Label Encoding the categorcal features")
    for name in cat_fetaures:
        lb = LabelEncoder()
        combined_df[name] = lb.fit_transform(combined_df[name])

    train_df = combined_df.loc[train_df.index]
    train_df[TARGET] = target

    test_df = combined_df.loc[test_df.index]

    train_X = train_df.drop([TARGET], axis=1)
    train_Y = train_df[TARGET]

    gfs = GreedyFeatureSelection(logger)
    scores, features = gfs.select_features(train_X, train_Y)
    logger.info(f"Final Best Score: {scores}")
    logger.info(f"Final Best Features: {features}")
