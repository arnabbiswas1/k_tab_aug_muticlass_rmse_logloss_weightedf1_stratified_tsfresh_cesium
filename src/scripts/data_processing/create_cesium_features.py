"""
Module to generate time series features using cesium

Sample Usage:
    <PROJECT_HOME>$python -m src.scripts.create_cesium_features

This script internally starts a Local Dask Cluster with 6 workers and
distributes the job across those.
"""

import pandas as pd
import numpy as np

from dask.distributed import LocalCluster, Client

from cesium import featurize

from sklearn.preprocessing import MinMaxScaler

import src.common as common
import src.munging as process_data
import src.config.constants as constants


def main():
    logger = common.get_logger("main")

    train_df, test_df, combined_df = process_data.read_processed_data(
        logger=logger, data_dir=constants.PROCESSED_DATA_DIR
    )

    combined_df = pd.concat([train_df.drop("loss", axis=1), test_df])

    combined_df_min_max = combined_df.copy()
    for name in combined_df.columns:
        mm = MinMaxScaler()
        combined_df_min_max.loc[:, name] = mm.fit_transform(combined_df[[name]])

    combined_df_min_max = process_data.change_dtype(
        logger, combined_df_min_max, np.float64, np.float32
    )

    # logger.info("Creating Dask Client..")

    # cluster = LocalCluster(
    #     n_workers=4, threads_per_worker=1, scheduler_port=8786, memory_limit="4GB"
    # )
    # client = Client(cluster)

    features_to_use = [
        # Cadence/Error
        "all_times_nhist_numpeaks",
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
        "all_times_nhist_peak_val",
        # Commenting out because of divide by zero encountered in true_divide
        # "avg_double_to_single_step",
        "avg_err",
        "avgt",
        "cad_probs_1",
        "cad_probs_10",
        "cad_probs_20",
        "cad_probs_30",
        "cad_probs_40",
        "cad_probs_50",
        "cad_probs_100",
        "cad_probs_500",
        "cad_probs_1000",
        "cad_probs_5000",
        "cad_probs_10000",
        "cad_probs_50000",
        "cad_probs_100000",
        "cad_probs_500000",
        "cad_probs_1000000",
        "cad_probs_5000000",
        "cad_probs_10000000",
        "cads_avg",
        "cads_med",
        "cads_std",
        "mean",
        # "med_double_to_single_step",
        "med_err",
        "n_epochs",
        # "std_double_to_single_step",
        "std_err",
        "total_time",
        # General features
        "amplitude",
        "flux_percentile_ratio_mid20",
        "flux_percentile_ratio_mid35",
        "flux_percentile_ratio_mid50",
        "flux_percentile_ratio_mid65",
        "flux_percentile_ratio_mid80",
        "max_slope",
        "maximum",
        "median",
        "median_absolute_deviation",
        "minimum",
        "percent_amplitude",
        "percent_beyond_1_std",
        "percent_close_to_median",
        "percent_difference_flux_percentile",
        "period_fast",
        "qso_log_chi2_qsonu",
        "qso_log_chi2nuNULL_chi2nu",
        "skew",
        "std",
        "stetson_j",
        "stetson_k",
        "weighted_average",
        # Commenting for error
        # Lomb-Scargle (Periodic) fetaures
        # "fold2P_slope_10percentile",
        # "fold2P_slope_90percentile",
        # "freq1_amplitude1",
        # "freq1_amplitude2",
        # "freq1_amplitude3",
        # "freq1_amplitude4",
        # "freq1_freq",
        # "freq1_lambda",
        # "freq1_rel_phase2",
        # "freq1_rel_phase3",
        # "freq1_rel_phase4",
        # "freq1_signif",
        # "freq2_amplitude1",
        # "freq2_amplitude2",
        # "freq2_amplitude3",
        # "freq2_amplitude4",
        # "freq2_freq",
        # "freq2_rel_phase2",
        # "freq2_rel_phase3",
        # "freq2_rel_phase4",
        # "freq3_amplitude1",
        # "freq3_amplitude2",
        # "freq3_amplitude3",
        # "freq3_amplitude4",
        # "freq3_freq",
        # "freq3_rel_phase2",
        # "freq3_rel_phase3",
        # "freq3_rel_phase4",
        # "freq_amplitude_ratio_21",
        # "freq_amplitude_ratio_31",
        # "freq_frequency_ratio_21",
        # "freq_frequency_ratio_31",
        # "freq_model_max_delta_mags",
        # "freq_model_min_delta_mags",
        # "freq_model_phi1_phi2",
        # "freq_n_alias",
        # "freq_signif_ratio_21",
        # "freq_signif_ratio_31",
        # "freq_varrat",
        # "freq_y_offset",
        # "linear_trend",
        # "medperc90_2p_p",
        # "p2p_scatter_2praw",
        # "p2p_scatter_over_mad",
        # "p2p_scatter_pfold_over_mad",
        # "p2p_ssqr_diff_over_var",
        # "scatter_res_raw",
    ]

    logger.info("Preparing the data...")
    a = []
    # for i in range(0, len(combined_df)):
    for i in range(0, 100000):
        a.append(np.arange(start=0, stop=100, step=1))

    ts = list(combined_df_min_max[300000:400000].values)

    logger.info("Engineering the features...")
    fset_cesium = featurize.featurize_time_series(
        times=a,
        values=ts,
        errors=None,
        features_to_use=features_to_use,
        scheduler="processes",
    )

    logger.info("Feature engineering is done..")
    logger.info(f"Shape of the features is {fset_cesium.shape}")
    fset_cesium.columns = fset_cesium.columns.get_level_values(0)

    logger.info("Writing to parquet filee")
    fset_cesium.to_parquet(
        f"{constants.FEATURES_DATA_DIR}/cesium_features_4.parquet", index=True
    )
    logger.info("Completed...")


if __name__ == "__main__":
    main()
