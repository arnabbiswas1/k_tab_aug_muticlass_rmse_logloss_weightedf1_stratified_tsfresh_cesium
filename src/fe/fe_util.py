""" A library for feature engineering

Reference: https://www.kaggle.com/c/ieee-fraud-detection/discussion/108575
"""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.preprocessing import PolynomialFeatures


__all__ = [
    "calc_percentile",
    "calc_quantile",
    "calc_quantile",
    "create_row_wise_stat_features",
    "create_ploynomial_features",
    "bin_cut_cont_features",
    "bin_qcut_cont_features",
    "log_transform",
    "sqrt_transform",
    "create_power_features",
    "create_continuous_feature_interaction",
    "create_categorical_feature_interaction",
    "create_frequency_encoding",
    "get_group_stat",
    "get_rare_categories",
    "create_rare_category",
    ]


def calc_percentile(x, percentile):
    """Returns the percentile for a numpy array
    """
    return np.percentile(x, percentile)


def calc_quantile(x, quantile):
    """Returns the quantile for a numpy array
    """
    return np.quantile(x, quantile)


def create_row_wise_stat_features(logger, source_df, target_df, features):
    """Returns features based on the statistics for each row
    """
    logger.info("Creating statistical features...")
    target_df["max"] = source_df[features].max(axis=1)
    target_df["min"] = source_df[features].min(axis=1)
    target_df["sum"] = source_df[features].sum(axis=1)
    target_df["mean"] = source_df[features].mean(axis=1)
    target_df["std"] = source_df[features].std(axis=1)
    target_df["skew"] = source_df[features].skew(axis=1)
    target_df["kurt"] = source_df[features].kurt(axis=1)
    target_df["med"] = source_df[features].median(axis=1)
    target_df["ptp"] = source_df[features].apply(np.ptp, axis=1)

    target_df["percentile_10"] = source_df[features].apply(
        lambda x: calc_percentile(x, 10), axis=1
    )
    target_df["percentile_60"] = source_df[features].apply(
        lambda x: calc_percentile(x, 60), axis=1
    )
    target_df["percentile_90"] = source_df[features].apply(
        lambda x: calc_percentile(x, 90), axis=1
    )

    target_df["quantile_5"] = source_df[features].apply(
        lambda x: calc_quantile(x, 0.5), axis=1
    )
    target_df["quantile_95"] = source_df[features].apply(
        lambda x: calc_quantile(x, 0.95), axis=1
    )
    target_df["quantile_99"] = source_df[features].apply(
        lambda x: calc_quantile(x, 0.99), axis=1
    )

    return target_df


def create_ploynomial_features(logger, source_df, target_df, features, degree=2):
    """
    Creates features using `sklearn.preprocessing.PolynomialFeatures`
    """
    logger.info("Creating polynomial features")
    pf = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    ploy_features = pf.fit_transform(source_df[features])
    # Get the names of the generated features
    feature_names = pf.get_feature_names(source_df[features].columns)
    # Append poly to the feature names
    feature_names = [f'ploy_{name.replace(" ", "*")}' for name in feature_names]
    # Create a DF from the generated features
    df_transformed = pd.DataFrame(ploy_features, columns=feature_names)
    # Drop the first few features sinces those are the original ones
    df_transformed = df_transformed.iloc[:, len(features):]
    # Append to the target DF
    target_df = pd.concat([target_df, df_transformed], axis=1)
    return target_df


def bin_cut_cont_features(logger, source_df, target_df, features, bin_size=10):
    """Creates bins from the continous fetaures
    """
    logger.info("Creating bins out of continous features")
    for name in features:
        logger.info(f"Creating bins out of using {name}")
        target_df[f"{name}_cut_bin_{bin_size}"] = pd.cut(
            source_df[name], bins=bin_size, labels=False
        )
    return target_df


def bin_qcut_cont_features(logger, source_df, target_df, features, bin_size=10):
    """Creates bins from the continous fetaures
    """
    logger.info("Creating bins out of continous features")
    for name in features:
        logger.info(f"Creating bins out of using {name}")
        target_df[f"{name}_qcut_bin_{bin_size}"] = pd.qcut(
            source_df[name], bins=bin_size, labels=False
        )
    return target_df


def log_transform(logger, source_df, target_df, features):
    """Given with an array/series retuns log(1+x) of it
    """
    logger.info("Creating log feature")
    temp = source_df[features].apply(lambda x: np.log(1 + x), axis=1)
    temp.columns = [f"{name}_log" for name in temp.columns]
    target_df = pd.concat([target_df, temp], axis=1)
    return target_df


def sqrt_transform(logger, source_df, target_df, features):
    """Given with an array/series retuns sqrt(1+x) of it
    """
    logger.info("Creating sqrt feature...")
    temp = source_df[features].apply(lambda x: np.sqrt(1 + x), axis=1)
    temp.columns = [f"{name}_sqrt" for name in temp.columns]
    target_df = pd.concat([target_df, temp], axis=1)
    return target_df


def create_power_features(logger, source_df, target_df, features):
    """Calculates power & ranks of each feature
    """
    for name in features:
        logger.info(f"Generating power features for {name}...")
        ser_norm = (source_df[name] - source_df[name].mean()) / source_df[name].std()
        target_df[name + "_square"] = np.power(ser_norm, 2)
        target_df[name + "_cube"] = np.power(ser_norm, 3)
        target_df[name + "_fourth"] = np.power(ser_norm, 4)
        # Cumulative percentile (not normalized)
        target_df[name + "_cp"] = rankdata(ser_norm).astype("float32")
        # Cumulative normal percentile
        target_df[name + "_cnp"] = norm.cdf(ser_norm).astype("float32")
    return target_df


def create_continuous_feature_interaction(logger, source_df, target_df, features):
    """Creates interation features for continous features
    """
    for col1 in features:
        for col2 in features:
            logger.info(f"Concatenating continous features {col1} and {col2}...")
            target_df[col1 + "_" + col2 + "_mul"] = source_df[col1] * source_df[col2]
            target_df[col1 + "_" + col2 + "_add"] = source_df[col1] + source_df[col2]
            target_df[col1 + "_" + col2 + "_sub"] = source_df[col1] - source_df[col2]
            target_df[col1 + "_" + col2 + "_div"] = source_df[col1] / source_df[col2]
    return target_df


def create_categorical_feature_interaction(logger, source_df, target_df, features):
    """Creates interation features for categorical columns
    """
    combinations = list(itertools.combinations(features, 2))
    for c1, c2 in combinations:
        target_df.loc[:, c1 + "_" + c2] = (
            source_df[c1].astype(str) + "_" + source_df[c2].astype(str)
        )
    return target_df


def create_frequency_encoding(logger, source_df, target_df, features):
    """Returns frequency encoding for categorical features
    """
    logger.info("Creating frequency encoding features...")
    for name in features:
        logger.info(f"Creating frequency encoding for {name}...")
        freq_map_dict = source_df[name].value_counts().to_dict()
        target_df[f"{name}_freq"] = source_df[name].map(freq_map_dict).astype(np.int32)
    return target_df


def get_group_stat(
    logger, source_df, target_df, cat_feat_name, cont_feat_name, agg_func_list
):
    """
    Generates statistical aggregate on a continous variable grouped by a categorical variable

    Example Usage:
        target_df = get_group_stat(
            logger, train_df, target_df, 'cat0', 'cont0', ['mean', 'max', 'min'])
    """
    col_name_dict = {}
    # dict for resultant column names
    for name in agg_func_list:
        col_name_dict[name] = f"{cat_feat_name}_{cont_feat_name}_{name}"
    temp = (
        source_df.groupby(cat_feat_name)[cont_feat_name]
        .agg(agg_func_list)
        .rename(col_name_dict, axis=1)
    )
    merged_df = pd.merge(source_df[cat_feat_name], temp, on=cat_feat_name, how="left")
    target_df = pd.concat([target_df, merged_df.iloc[:, 1:]], axis=1)
    return target_df


def get_rare_categories(source_df, feature_name, threshold):
    """ Returns a list of categories which occur lesser than a threshold value
    for a categorical variable.

    Args:
        source_df: DataFrame consiting of the categorical variable
        feature_name: Categorical variable for which rare categories are being identified
        threshold: If number of occurence of a category is lesser than this integer value,
        the category will be identified as rare
    """
    counts = source_df[feature_name].value_counts()
    rare_categories = counts[counts < threshold].index
    return list(rare_categories)


def create_rare_category(df, feature_name, threshold):
    """Marks all categories which occur lesser than a threshold value
    for a categorical variable with a value "RARE"

    Args:
        source_df: DataFrame consiting of the categorical variable
        feature_name: Categorical variable for which rare categories are being identified
        threshold: If number of occurence of a category is lesser than this integer value,
        the category will be identified as rare
    """
    rare_categories = get_rare_categories(df, feature_name, threshold)
    # Assumption: Variable is of type 'category'
    df[feature_name] = df[feature_name].astype(str)
    df.loc[df[feature_name].isin(rare_categories), feature_name] = "RARE"
    df[feature_name] = df[feature_name].astype("category")
    return df
