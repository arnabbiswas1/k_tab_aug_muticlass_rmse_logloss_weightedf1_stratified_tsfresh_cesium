import pandas as pd
import numpy as np

import src.munging as process_data
import src.common as common
import src.config.constants as constants
import src.fe as fe


def main():
    # Create a Stream only logger
    logger = common.get_logger("generate_features")
    logger.info("Starting to generate features")

    train_df, test_df, _ = process_data.read_processed_data(
        logger,
        constants.PROCESSED_DATA_DIR,
        train=True,
        test=True,
        sample_submission=True,
    )

    combined_df = pd.concat([train_df.drop("loss", axis=1), test_df])

    cat_within_2 = ["f1", "f86", "f55"]
    cat_within_9 = ["f27"]
    cat_within_17 = ["f12", "f9", "f79", "f64", "f34", "f26", "f30"]

    cat_features = cat_within_2 + cat_within_9 + cat_within_17
    cont_features = list(
        combined_df.columns.drop(cat_features)
    )

    logger.info(f"Categorical features {cat_features}")
    logger.info(f"Continous features {cont_features}")

    features_df = pd.DataFrame()

    features_df = fe.create_frequency_encoding(
        logger, combined_df, features_df, cat_features
    )

    features_df = fe.create_categorical_feature_interaction(
        logger, combined_df, features_df, cat_within_2 + cat_within_9
    )

    # This is going to be huge. Dropping for now
    # features_df = fe.create_continuous_feature_interaction(
    #     logger, combined_df, features_df, cont_feature
    # )

    features_df = fe.create_power_features(
        logger, combined_df, features_df, cont_features
    )

    features_df = fe.create_row_wise_stat_features(
        logger, combined_df, features_df, cont_features
    )

    features_df = fe.bin_cut_cont_features(
        logger, combined_df, features_df, cont_features, bin_size=10
    )

    # This is going to be huge. Dropping for now
    # features_df = fe.create_ploynomial_features(
    #     logger, combined_df, features_df, cont_feature
    # )

    logger.info(f"Shape of the generated features {features_df.shape}")
    logger.info(f"Name of the features generated {list(features_df.columns)}")

    logger.info("Changing data type ..")
    combined_df = process_data.change_dtype(logger, features_df, np.int64, np.int32)
    combined_df = process_data.change_dtype(logger, features_df, np.float64, np.float32)
    combined_df = process_data.change_dtype(logger, features_df, np.object, "category")

    logger.info(f"Writing generated features to {constants.FEATURES_DATA_DIR}")
    features_df.to_parquet(
        f"{constants.FEATURES_DATA_DIR}/generated_features.parquet", index=True
    )


if __name__ == "__main__":
    main()
