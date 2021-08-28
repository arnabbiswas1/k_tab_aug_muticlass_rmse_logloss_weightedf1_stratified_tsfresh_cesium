import pandas as pd

from src.config import constants
import src.munging as process_data
import src.common as common


if __name__ == "__main__":
    logger = common.get_logger("main")

    logger.info("Reading cesium features")
    df1 = pd.read_parquet(f"{constants.FEATURES_DATA_DIR}/cesium_features_1.parquet")
    df2 = pd.read_parquet(f"{constants.FEATURES_DATA_DIR}/cesium_features_2.parquet")
    df3 = pd.read_parquet(f"{constants.FEATURES_DATA_DIR}/cesium_features_3.parquet")
    df4 = pd.read_parquet(f"{constants.FEATURES_DATA_DIR}/cesium_features_4.parquet")

    df_merged = pd.concat([df1, df2, df3, df4])

    df_null = process_data.check_null(df_merged)

    logger.info("Dropping Null, no variance, duplicate features")
    # Features with Null values
    null_features = [
        "all_times_nhist_peak1_bin",
        "all_times_nhist_peak2_bin",
        "all_times_nhist_peak3_bin",
        "all_times_nhist_peak4_bin",
        "all_times_nhist_peak_1_to_2",
        "all_times_nhist_peak_1_to_3",
        "all_times_nhist_peak_1_to_4",
        "all_times_nhist_peak_2_to_3",
        "all_times_nhist_peak_2_to_4",
        "all_times_nhist_peak_3_to_4",
    ]
    df_merged = df_merged.drop(null_features, axis=1)

    # Features with 0 variance
    no_variance_features = process_data.get_features_with_no_variance(df_merged)
    df_merged = df_merged.drop(no_variance_features, axis=1)

    # Features which are also present in tsfresh
    duplicate_fetaures = ["mean", "maximum", "median", "minimum", "skew", "std"]
    df_merged = df_merged.drop(
        ["mean", "maximum", "median", "minimum", "skew", "std"], axis=1
    )

    df_merged = df_merged.reset_index(drop=True)

    logger.info(f"Writing the final parquet file with shape {df_merged.shape}")
    df_merged.to_parquet(
        f"{constants.FEATURES_DATA_DIR}/cesium_final.parquet", index=False
    )
    logger.info("Completed")
