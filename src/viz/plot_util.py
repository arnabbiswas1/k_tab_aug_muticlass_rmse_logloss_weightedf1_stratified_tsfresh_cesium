import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.tsa.api as tsa

import src.munging as munging


__all__ = [
    "plot_desnity_train_test_overlapping",
    "plot_hist_train_test_overlapping",
    "plot_barh_train_test_side_by_side",
    "plot_line_train_test_overlapping",
    "plot_hist",
    "plot_barh",
    "plot_boxh",
    "plot_line",
    "plot_boxh_groupby",
    "plot_hist_groupby",
    "save_feature_importance_as_fig",
    "save_permutation_importance_as_fig",
    "save_optuna_param_importance_as_fig",
    "save_rfecv_plot",
    "plot_seasonal_decomposition",
    "plot_seasonality",
    "plot_trend",
    "plot_ts_line_groupby",
    "plot_ts_point_groupby",
    "plot_ts_bar_groupby",
    "plot_multiple_seasonalities",
]


def plot_desnity_train_test_overlapping(df_train, df_test, feature_name):
    """
    Plot density for a particular feature both for train and test.

    """
    df_train[feature_name].plot.density(
        figsize=(15, 5),
        label="train",
        alpha=0.4,
        color="red",
        title=f"Train vs Test {feature_name} distribution",
    )
    df_test[feature_name].plot.density(label="test", alpha=0.4, color="blue")
    plt.legend()
    plt.show()


def plot_hist_train_test_overlapping(df_train, df_test, feature_name, kind="hist"):
    """
    Plot histogram for a particular feature both for train and test.

    kind : Type of the plot

    """
    df_train[feature_name].plot(
        kind=kind,
        figsize=(10, 10),
        label="train",
        bins=50,
        alpha=0.4,
        color="red",
        title=f"Train vs Test {feature_name} distribution",
    )
    df_test[feature_name].plot(
        kind="hist",
        figsize=(10, 10),
        label="test",
        bins=50,
        alpha=0.4,
        color="darkgreen",
    )
    plt.legend()
    plt.show()


def plot_barh_train_test_side_by_side(
    df_train, df_test, feature_name, normalize=True, sort_index=False
):
    """
    Plot histogram for a particular feature both for train and test.

    kind : Type of the plot

    """
    print(
        f"Number of unique values in train: {munging.count_unique_values(df_train, feature_name)}"
    )
    print(
        f"Number of unique values in test: {munging.count_unique_values(df_test, feature_name)}"
    )

    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 8))

    if sort_index:
        df_train[feature_name].value_counts(
            normalize=normalize, dropna=False
        ).sort_index().plot(
            kind="barh",
            figsize=(15, 6),
            ax=ax1,
            grid=True,
            title=f"Bar plot for {feature_name} for train",
        )

        df_test[feature_name].value_counts(
            normalize=normalize, dropna=False
        ).sort_index().plot(
            kind="barh",
            figsize=(15, 6),
            ax=ax2,
            grid=True,
            title=f"Bar plot for {feature_name} for test",
        )
    else:
        df_train[feature_name].value_counts(
            normalize=normalize, dropna=False
        ).sort_values().plot(
            kind="barh",
            figsize=(15, 6),
            ax=ax1,
            grid=True,
            title=f"Bar plot for {feature_name} for train",
        )

        df_test[feature_name].value_counts(
            normalize=normalize, dropna=False
        ).sort_values().plot(
            kind="barh",
            figsize=(15, 6),
            ax=ax2,
            grid=True,
            title=f"Bar plot for {feature_name} for test",
        )

    plt.legend()
    plt.show()


def plot_line_train_test_overlapping(df_train, df_test, feature_name, figsize=(10, 5)):
    """
    Plot line for a particular feature both for train and test
    """
    df_train[feature_name].plot(
        kind="line",
        figsize=figsize,
        label="train",
        alpha=0.4,
        title=f"Train vs Test {feature_name} distribution",
    )
    df_test[feature_name].plot(kind="line", label="test", alpha=0.4)
    plt.ylabel(f"Value of {feature_name}")
    plt.legend()
    plt.show()


def plot_line(df, feature_name, figsize=(10, 5)):
    """
    Plot line for a particular feature for the DF
    """
    df[feature_name].plot(
        kind="line",
        figsize=figsize,
        label="train",
        alpha=0.4,
        title=f"Line plot for {feature_name} distribution",
    )
    plt.ylabel(f"Value of {feature_name}")
    plt.legend()
    plt.show()


def plot_hist(df, feature_name, kind="hist", bins=100, log=True):
    """
    Plot either for train or test
    """
    if log:
        df[feature_name].apply(np.log1p).plot(
            kind="hist",
            bins=bins,
            figsize=(15, 5),
            title=f"Distribution of log1p[{feature_name}]",
        )
    else:
        df[feature_name].plot(
            kind="hist",
            bins=bins,
            figsize=(15, 5),
            title=f"Distribution of {feature_name}",
        )
    plt.show()


def plot_barh(
    df, feature_name, normalize=True, kind="barh", figsize=(15, 5), sort_index=False
):
    """
    Plot barh for a particular feature both for train and test.

    kind : Type of the plot

    """
    if sort_index:
        df[feature_name].value_counts(
            normalize=normalize, dropna=False
        ).sort_index().plot(
            kind=kind, figsize=figsize, grid=True, title=f"Bar plot for {feature_name}"
        )
    else:
        df[feature_name].value_counts(
            normalize=normalize, dropna=False
        ).sort_values().plot(
            kind=kind, figsize=figsize, grid=True, title=f"Bar plot for {feature_name}"
        )

    plt.legend()
    plt.show()


def plot_boxh(df, feature_name, kind="box", log=True):
    """
    Box plot either for train or test
    """
    if log:
        df[feature_name].apply(np.log1p).plot(
            kind="box",
            vert=False,
            figsize=(10, 6),
            title=f"Distribution of log1p[{feature_name}]",
        )
    else:
        df[feature_name].plot(
            kind="box",
            vert=False,
            figsize=(10, 6),
            title=f"Distribution of {feature_name}",
        )
    plt.show()


def plot_boxh_groupby(df, feature_name, by):
    """
    Box plot with groupby feature
    """
    df.boxplot(column=feature_name, by=by, vert=False, figsize=(10, 6))
    plt.title(f"Distribution of {feature_name} by {by}")
    plt.show()


def plot_hist_groupby(df, feature_name, by, bins=100, figsize=(15, 5)):
    """
    Box plot with groupby feature
    """
    df.hist(column=feature_name, by=by, figsize=figsize, bins=100, legend=False)
    plt.suptitle(f"Distribution of {feature_name} by {by}")
    plt.show()


def save_feature_importance_as_fig(best_features_df, dir_name, file_name):
    plt.figure(figsize=(16, 12))
    sns.barplot(
        x="importance",
        y="feature",
        data=best_features_df.sort_values(by="importance", ascending=False),
    )
    plt.title("Features (avg over folds)")
    plt.savefig(f"{dir_name}/{file_name}")


def save_permutation_importance_as_fig(best_features_df, dir_name, file_name):
    plt.figure(figsize=(16, 12))
    sns.barplot(
        x="weight",
        y="feature",
        data=best_features_df.sort_values(by="weight", ascending=False),
    )
    plt.title("Permutation Importance (avg over folds)")
    plt.savefig(f"{dir_name}/{file_name}")


def save_optuna_param_importance_as_fig(params_df, dir_name, file_name):
    plt.figure(figsize=(16, 12))
    sns.barplot(
        x="importance",
        y="param_name",
        data=params_df.sort_values(by="importance", ascending=False),
    )
    plt.title("Importance of hyperparameters")
    plt.savefig(f"{dir_name}/{file_name}")


def save_rfecv_plot(rfecv, dir_name, file_name):
    plt.figure(figsize=(14, 8))
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.title("CV Score vs Features Selected (REFCV)")
    plt.savefig(f"{dir_name}/{file_name}")


def plot_seasonal_decomposition(
    df, feature, freq, freq_type="daily", model="additive", figsize=(20, 10)
):
    plt.rcParams["figure.figsize"] = figsize
    decomposition = tsa.seasonal_decompose(df[feature], model=model, period=freq)
    decomposition.plot()
    plt.title(f"{model} {freq_type} seasonal decomposition of {feature}")
    plt.show()


def plot_seasonality(
    df, feature, freq, freq_type="daily", model="additive", figsize=(20, 10)
):
    plt.rcParams["figure.figsize"] = figsize
    decomposition = tsa.seasonal_decompose(df[feature], model=model, period=freq)
    decomposition.seasonal.plot(color="blue", linewidth=0.5)
    plt.title(f"{model} {freq_type} seasonality of {feature}")
    plt.show()


def plot_trend(
    df, feature, freq, freq_type="daily", model="additive", figsize=(20, 10)
):
    plt.rcParams["figure.figsize"] = figsize
    decomposition = tsa.seasonal_decompose(df[feature], model=model, period=freq)
    decomposition.trend.plot(color="blue", linewidth=0.5)
    plt.title(f"{model} {freq_type} seasonality of {feature}")
    plt.show()


def plot_ts_line_groupby(
    df,
    ts_index_feature,
    groupby_feature,
    value_feature,
    title,
    xlabel,
    ylabel,
    figsize=(15, 8),
):
    fig, ax = plt.subplots(figsize=figsize)
    for label, df in df.groupby(groupby_feature):
        df.set_index(ts_index_feature)[value_feature].plot(
            kind="line", alpha=0.3, ax=ax, color="blue", linewidth=0.5
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_ts_point_groupby(
    df,
    ts_index_feature,
    groupby_feature,
    value_feature,
    title,
    xlabel,
    ylabel,
    figsize=(15, 8),
):
    fig, ax = plt.subplots(figsize=figsize)
    for label, df in df.groupby(groupby_feature):
        df.set_index(ts_index_feature)[value_feature].plot(
            style=".", kind="line", alpha=0.2, ax=ax, linewidth=0.5
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_ts_bar_groupby(
    df,
    ts_index_feature,
    groupby_feature,
    value_feature,
    title,
    xlabel,
    ylabel,
    figsize=(15, 8),
):
    fig, ax = plt.subplots(figsize=figsize)
    for label, df in df.groupby(groupby_feature):
        df.set_index(ts_index_feature)[value_feature].plot(
            kind="bar", alpha=0.05, ax=ax, linewidth=0.5
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_multiple_seasonalities(df, feature_name, figsize=(20, 6)):
    period_names = ["daily", "weekly", "monthly", "quarterly"]
    periods = [24, 24 * 7, 24 * 30, 24 * 90]

    for name, period in zip(period_names, periods):
        plot_seasonality(
            df.set_index("date_time")[0: period * 3],
            feature=feature_name,
            freq=period,
            freq_type=name,
            figsize=(20, 6),
        )