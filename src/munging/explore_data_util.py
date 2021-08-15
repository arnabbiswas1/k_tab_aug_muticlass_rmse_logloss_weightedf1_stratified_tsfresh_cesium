import matplotlib.pyplot as plt
import pandas as pd


__all__ = [
    "check_null",
    "check_duplicate",
    "count_unique_values",
    "do_value_counts",
    "check_id",
    "get_feature_names",
    "check_value_counts_across_train_test",
    "get_freq_encoding_feature_names",
    "get_bin_feature_names",
    "get_power_feature_names",
    "get_row_wise_stat_feature_names",
    "get_cat_interaction_features"
]


def check_null(df):
    return df.isna().sum() * 100 / len(df)


def check_duplicate(df, subset):
    if subset is not None:
        return df.duplicated(subset=subset, keep=False).sum()
    else:
        return df.duplicated(keep=False).sum()


def count_unique_values(df, feature_name):
    return df[feature_name].nunique()


def do_value_counts(df, feature_name):
    return (
        df[feature_name]
        .value_counts(normalize=True, dropna=False)
        .sort_values(ascending=False)
        * 100
    )


def check_id(df, column_name, data_set_name):
    """
    Check if the identifier column is continous and monotonically increasing
    """
    print(f"Is the {column_name} monotonic : {df[column_name].is_monotonic}")
    # Plot the column
    df[column_name].plot(title=data_set_name)
    plt.show()


def get_feature_names(df, feature_name_substring):
    """
    Returns the list of features with name matching 'feature_name_substring'
    """
    return [
        col_name
        for col_name in df.columns
        if col_name.find(feature_name_substring) != -1
    ]


def check_value_counts_across_train_test(
    train_df, test_df, feature_name, normalize=True
):
    """
    Create a DF consisting of value_counts of a particular feature for
    train and test
    """
    train_counts = (
        train_df[feature_name]
        .sort_index()
        .value_counts(normalize=normalize, dropna=True)
        * 100
    )
    test_counts = (
        test_df[feature_name]
        .sort_index()
        .value_counts(normalize=normalize, dropna=True)
        * 100
    )
    count_df = pd.concat([train_counts, test_counts], axis=1).reset_index(drop=True)
    count_df.columns = [feature_name, "train", "test"]
    return count_df


def get_freq_encoding_feature_names(df):
    return get_feature_names(df, "freq")


def get_power_feature_names(df):
    power_features = []
    power_feature_keys = ["_square", "_cube", "_fourth", "_cp", "_cnp"]
    for name in df.columns:
        for key in power_feature_keys:
            if key in name:
                power_features.append(name)
    return power_features


def get_row_wise_stat_feature_names():
    return [
        "max",
        "min",
        "sum",
        "mean",
        "std",
        "skew",
        "kurt",
        "med",
        "ptp",
        "percentile_10",
        "percentile_60",
        "percentile_90",
        "quantile_5",
        "quantile_95",
        "quantile_99",
    ]


def get_bin_feature_names(df, bin_size=10):
    return get_feature_names(df, f"cut_bin_{bin_size}")


def get_cat_interaction_features():
    return ["f1_f86", "f1_f55",	"f1_f27", "f86_f55", "f86_f27", "f55_f27"]
