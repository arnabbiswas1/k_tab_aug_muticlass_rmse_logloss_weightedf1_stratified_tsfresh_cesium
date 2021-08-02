import pandas as pd
import numpy as np

import src.munging as process_data
import src.common as common
import src.config.constants as constants
import src.ts as ts


if __name__ == "__main__":
    logger = common.get_logger("fe")
    logger.info("Starting to engineer features")

    train_df, test_df, sample_submission_df = process_data.read_processed_data(
        logger,
        constants.PROCESSED_DATA_DIR,
        train=True,
        test=True,
        sample_submission=True,
    )

    TARGETS = ["target_carbon_monoxide", "target_benzene", "target_nitrogen_oxides"]

    targets = train_df[TARGETS]
    combined_df = pd.concat([train_df.drop(TARGETS, axis=1), test_df])

    features_df = pd.DataFrame()
    logger.info("Cretaing date related features")
    features_df = ts.create_date_features(
        source_df=combined_df, target_df=features_df, feature_name="date_time"
    )
    features_df = ts.create_us_season(
        source_df=combined_df, target_df=features_df, feature_name="date_time"
    )

    features_df = ts.create_part_of_day(
        source_df=combined_df, target_df=features_df, feature_name="date_time"
    )

    # Saturated water vapour density
    features_df["s_wvd"] = (combined_df["absolute_humidity"] * 100) / combined_df[
        "relative_humidity"
    ]

    # https://www.calcunation.com/calculator/dew-point.php
    features_df["dew_point"] = (
        243.12
        * (
            np.log(combined_df["relative_humidity"] * 0.01)
            + (17.62 * combined_df["deg_C"]) / (243.12 + combined_df["deg_C"])
        )
        / (
            17.62
            - (
                np.log(combined_df["relative_humidity"] * 0.01)
                + 17.62 * combined_df["deg_C"] / (243.12 + combined_df["deg_C"])
            )
        )
    )

    features_df["partial_pressure"] = (
        ((237.7 + combined_df["deg_C"]) * 286.8) * combined_df["absolute_humidity"]
    ) / 100000

    logger.info(f"Writing features to parquet file of shape {features_df.shape}")
    features_df.to_parquet(
        path=f"{constants.FEATURES_DATA_DIR}/features.parquet", index=False
    )
