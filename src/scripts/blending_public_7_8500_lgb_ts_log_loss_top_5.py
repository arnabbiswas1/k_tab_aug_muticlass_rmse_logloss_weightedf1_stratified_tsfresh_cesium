import os
from datetime import datetime

import pandas as pd

import src.config.constants as constants
import src.munging as process_data
import src.common as common

if __name__ == "__main__":
    RUN_ID = datetime.now().strftime("%m%d_%H%M")
    MODEL_NAME = os.path.basename(__file__).split(".")[0]

    logger = common.get_logger("blend")

    train_df, test_df, sample_submission_df = process_data.read_processed_data(
        logger,
        constants.PROCESSED_DATA_DIR,
        train=True,
        test=True,
        sample_submission=True,
    )

    # File with public score 7.85000
    # https://www.kaggle.com/yus002/blending-tool-tps-aug-2021?select=file1_7.85003_file2_7.85022_blend.csv
    df_sub_ext = pd.read_csv(
        f"{constants.PUB_SUBMISSION_DIR}/file1_7.85003_file2_7.85022_blend.csv"
    )

    # LGB ts log_loss SKFold 10 top 5 features (almost lowest logloss & lowest RMSE)
    df_lgb_log_loss_top_10 = pd.read_csv(
        f"{constants.SUBMISSION_DIR}/sub_lgb_ts_log_loss_SK_top_100_features_0824_1218_2.93948.csv"
    )

    # Giving more importnace to external submission
    sample_submission_df.loss = (
        0.99 * df_sub_ext.loss + 0.01 * df_lgb_log_loss_top_10.loss
    ).values

    file_name = f"sub_{MODEL_NAME}_{RUN_ID}.csv"

    logger.info(f"Saving to submission file {constants.SUBMISSION_DIR}/{file_name}")
    sample_submission_df.to_csv(f"{constants.SUBMISSION_DIR}/{file_name}")

    logger.info(pd.read_csv(f"{constants.SUBMISSION_DIR}/{file_name}"))
