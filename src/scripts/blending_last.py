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

    # File with public score 10.78437
    df_1 = pd.read_csv(f"{constants.SUBMISSION_DIR}/sub_lgb_ts_f1_weighted_SK_10_tsfresh_top_2_greedy_selection_optuna_param_last_run_0831_1740_0.09326.csv")

    # PL: 10.53201
    df_2 = pd.read_csv(
        f"{constants.SUBMISSION_DIR}/sub_lgb_ts_f1_weighted_SK_10_tsfresh_top_2_greedy_selection_set_2_0831_1804_0.09330.csv"
    )

    # Giving more importnace to external submission
    sample_submission_df.loss = (
        0.05 * df_1.loss + 0.95 * df_2.loss
    ).values

    file_name = f"sub_{MODEL_NAME}_{RUN_ID}.csv"

    logger.info(f"Saving to submission file {constants.SUBMISSION_DIR}/{file_name}")
    sample_submission_df.to_csv(f"{constants.SUBMISSION_DIR}/{file_name}")

    logger.info(pd.read_csv(f"{constants.SUBMISSION_DIR}/{file_name}"))
