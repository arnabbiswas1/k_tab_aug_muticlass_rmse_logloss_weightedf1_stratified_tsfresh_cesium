import pandas as pd


__all__ = [
    "read_raw_data",
    "read_processed_data",
    "change_dtype",
    "map_variable",
    "merge_df",
]


def read_raw_data(
    logger, data_dir, index_col_name=None, train=True, test=True, sample_submission=True
):
    """Read all the different data files
    """
    logger.info(f"Reading Data from {data_dir}...")

    train_df = None
    test_df = None
    sample_submission_df = None

    if train:
        train_df = pd.read_csv(f"{data_dir}/train.csv", index_col=index_col_name)
        logger.info(f"Shape of train_df : {train_df.shape}")
    if test:
        test_df = pd.read_csv(f"{data_dir}/test.csv", index_col=index_col_name)
        logger.info(f"Shape of test_df : {test_df.shape}")
    if sample_submission:
        sample_submission_df = pd.read_csv(
            f"{data_dir}/sample_submission.csv", index_col=index_col_name
        )
        logger.info(f"Shape of sample_submission_df : {sample_submission_df.shape}")

    return train_df, test_df, sample_submission_df


def read_processed_data(
    logger, data_dir, train=True, test=True, sample_submission=True, frac=None
):
    """Read all the processed data files. If frac has a valid value, all
    the DFs will be sampled and returned
    """
    logger.info(f"Reading Data from {data_dir}...")

    train_df = None
    test_df = None
    sample_submission_df = None

    if train:
        train_df = pd.read_parquet(f"{data_dir}/train_processed.parquet")
        if frac is not None:
            train_df = train_df.sample(frac=frac)
        logger.info(f"Shape of train_df : {train_df.shape}")
    if test:
        test_df = pd.read_parquet(f"{data_dir}/test_processed.parquet")
        if frac is not None:
            test_df = test_df.sample(frac=frac)
        logger.info(f"Shape of test_df : {test_df.shape}")
    if sample_submission:
        sample_submission_df = pd.read_parquet(f"{data_dir}/sub_processed.parquet")
        if frac is not None:
            sample_submission_df = sample_submission_df.sample(frac=frac)
        logger.info(f"Shape of sample_submission_df : {sample_submission_df.shape}")

    return train_df, test_df, sample_submission_df


def change_dtype(logger, df, source_dtype, target_dtype):
    df = df.copy()
    for col in df.select_dtypes([source_dtype]).columns:
        logger.info(
            f"Changing dtype of [{col}] from [{source_dtype}] to [{target_dtype}]"
        )
        df[col] = df[col].astype(target_dtype)
    return df


def map_variable(df: pd.DataFrame, feature_name: str, map_dict: dict):
    """
    Map values of a feature to a different set of values.
    This mapping is defined in the `map_dict`
    """
    df[feature_name] = df[feature_name].map(map_dict)
    return df


def merge_df(logger, left_df, right_df, how, on):
    """
    Wrapper on top of Pandas merge. Prints the shape & missing
    values before and after merge
    """
    logger.info("Before merge missing values on left_df")
    logger.info(left_df.isna().sum())
    logger.info("Before merge missing values on right_df")
    logger.info(right_df.isna().sum())
    logger.info("Before merge shape of left_df: {left_df.shape}")
    logger.info("Before merge shape of right_df: {right_df.shape}")
    merged_df = pd.merge(left_df, right_df, how=how, on=on)
    logger.info("After merge missing values in merged_df")
    logger.info(merged_df.isna().sum())
    logger.info("After merge shape of merged_df {merged_df.shape}")
    return merged_df
