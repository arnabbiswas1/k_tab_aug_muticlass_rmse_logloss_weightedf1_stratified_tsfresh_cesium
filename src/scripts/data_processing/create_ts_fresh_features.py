"""
Script to generate fetaures using tsfresh

TODO : Work In Progress
"""

import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import MinMaxScaler

from tsfresh import extract_features

import src.munging as process_data
import src.common as common
import src.config.constants as constants
import src.config.tsfresh_config as tsfresh_config


def flatten_df(df):
    """
    Organize the data in a form
    which is understandable by tsfresh
    """
    flat_array = df.values.flatten()
    flat_df = pd.DataFrame(flat_array)
    flat_df.columns = ["loan"]
    flat_df["row_no"] = flat_df.reset_index().index
    flat_df = flat_df[["row_no", "loan"]]
    flat_df.row_no = flat_df.row_no // 100
    return flat_df


def main():
    # Create a Stream only logger
    logger = common.get_logger("tsfresh")
    logger.info("Starting to generate tsfresh features")

    train_df, test_df, _ = process_data.read_processed_data(
        logger,
        constants.PROCESSED_DATA_DIR,
        train=True,
        test=True,
        sample_submission=True,
    )

    combined_df = pd.concat([train_df.drop("loss", axis=1), test_df])

    combined_df_min_max = combined_df.copy()
    for name in combined_df.columns:
        mm = MinMaxScaler()
        combined_df_min_max.loc[:, name] = mm.fit_transform(combined_df[[name]])

    combined_df_min_max = process_data.change_dtype(
        logger, combined_df_min_max, np.float64, np.float32
    )

    ts_df = flatten_df(combined_df_min_max)
    print(f"Shape of the falttened data {ts_df.shape}")

    ts_df = process_data.change_dtype(logger, ts_df, np.int64, np.int32)

    tsfresh_feature_set = [
        "mixed_1_set",
        "symmetry_large_std_quantile_set",
        "acf_pacf_set",
        "cwt_coeff_set",
        "change_quantile_set",
        "fft_real_set",
        "fft_imag_set",
        "fft_abs_set",
        "fft_angle_set",
        "fft_agg_set", # TODO
        "liner_agg_linear_set",
        "mixed_2_set",
        "mixed_3_set",
        "mixed_4_set",
    ]

    for name in tsfresh_feature_set:
        logger.info(f"Generating features for {name}")
        parameters = getattr(tsfresh_config, name)
        features_df = extract_features(
            ts_df, default_fc_parameters=parameters, column_id="row_no", n_jobs=10,
        )
        logger.info(f"Shape of the feature {features_df.shape}")
        logger.info("Removing JSON charachters from the column names")
        # features_df.columns = [name.replace("\"", "") for name in features_df.columns]
        df_renamed = features_df.rename(columns=lambda x: re.sub('-', 'minus', x))
        df_renamed = df_renamed.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

        features_df = process_data.change_dtype(logger, features_df, np.int64, np.int32)
        features_df = process_data.change_dtype(logger, features_df, np.float64, np.float32)
        logger.info(f"Writing featurs to {constants.FEATURES_DATA_DIR}/{name}.parquet")
        features_df.to_parquet(f"{constants.FEATURES_DATA_DIR}/{name}.parquet", index=True)


if __name__ == "__main__":
    main()
