"""
Script to combine the features generated for tsfresh and write the
combined DF back to disk
"""

from src import common

import pandas as pd

import src.config.constants as constants


def select_features(logger, df, features_to_drop):
    logger.info(f"Shape of the features {df.shape}")
    df = df.drop(features_to_drop, axis=1)
    logger.info(f"Shape of the features after dropping {df.shape}")
    return df


def load_data(logger, name, features_to_drop):
    df = pd.read_parquet(f"{constants.FEATURES_DATA_DIR}/cast/{name}_cast.parquet")
    logger.info(f"Shape of {name} before droipping {df.shape}")
    df = select_features(df, features_to_drop)
    logger.info(f"Shape of {name} after droipping {df.shape}")
    return df


def combine_features(logger):
    name = "mixed_1_set"
    features_to_drop = [
        "loan__has_duplicate_min",
        "loan__length",
        "loan__sample_entropy",
    ]
    df_mixed_1_set = load_data(logger, name, features_to_drop)

    name = "symmetry_large_std_quantile_set"
    features_to_drop = features_to_drop = [
        "loan__symmetry_looking__r_0.0",
        "loan__symmetry_looking__r_0.1",
        "loan__symmetry_looking__r_0.15000000000000002",
        "loan__symmetry_looking__r_0.2",
        "loan__symmetry_looking__r_0.25",
        "loan__symmetry_looking__r_0.30000000000000004",
        "loan__symmetry_looking__r_0.35000000000000003",
        "loan__symmetry_looking__r_0.4",
        "loan__symmetry_looking__r_0.45",
        "loan__symmetry_looking__r_0.5",
        "loan__symmetry_looking__r_0.55",
        "loan__symmetry_looking__r_0.6000000000000001",
        "loan__symmetry_looking__r_0.65",
        "loan__symmetry_looking__r_0.7000000000000001",
        "loan__symmetry_looking__r_0.75",
        "loan__symmetry_looking__r_0.8",
        "loan__symmetry_looking__r_0.8500000000000001",
        "loan__symmetry_looking__r_0.9",
        "loan__symmetry_looking__r_0.9500000000000001",
        "loan__large_standard_deviation__r_0.05",
        "loan__large_standard_deviation__r_0.1",
        "loan__large_standard_deviation__r_0.15000000000000002",
        "loan__large_standard_deviation__r_0.30000000000000004",
        "loan__large_standard_deviation__r_0.35000000000000003",
        "loan__large_standard_deviation__r_0.4",
        "loan__large_standard_deviation__r_0.45",
        "loan__large_standard_deviation__r_0.5",
        "loan__large_standard_deviation__r_0.55",
        "loan__large_standard_deviation__r_0.6000000000000001",
        "loan__large_standard_deviation__r_0.65",
        "loan__large_standard_deviation__r_0.7000000000000001",
        "loan__large_standard_deviation__r_0.75",
        "loan__large_standard_deviation__r_0.8",
        "loan__large_standard_deviation__r_0.8500000000000001",
        "loan__large_standard_deviation__r_0.9",
        "loan__large_standard_deviation__r_0.9500000000000001",
    ]
    df_sym = load_data(logger, name, features_to_drop)

    name = "acf_pacf_set"
    features_to_drop = ["loan__partial_autocorrelation__lag_0"]
    df_acf_pacf_set = load_data(logger, name, features_to_drop)

    name = "cwt_coeff_set"
    features_to_drop = []
    df_cwt_coeff_set = load_data(logger, name, features_to_drop)

    name = "change_quantile_set"
    features_to_drop = []
    df_change_quantile_set = load_data(logger, name, features_to_drop)

    name = "liner_agg_linear_set"
    features_to_drop = [
        "loan__agg_linear_trend__attr_stderr__chunk_len_50__f_agg_max",
        "loan__agg_linear_trend__attr_stderr__chunk_len_50__f_agg_min",
        "loan__agg_linear_trend__attr_stderr__chunk_len_50__f_agg_mean",
        "loan__agg_linear_trend__attr_stderr__chunk_len_50__f_agg_var",
    ]
    df_liner_agg_linear_set = load_data(logger, name, features_to_drop)

    name = "mixed_2_set"
    features_to_drop = [
        "loan__count_above__t_0",
        "loan__query_similarity_count__query_None__threshold_00",
        "loan__matrix_profile__feature_min__threshold_098",
        "loan__matrix_profile__feature_max__threshold_098",
        "loan__matrix_profile__feature_mean__threshold_098",
        "loan__matrix_profile__feature_median__threshold_098",
        "loan__matrix_profile__feature_25__threshold_098",
        "loan__matrix_profile__feature_75__threshold_098",
    ]
    df_mixed_2_set = load_data(logger, name, features_to_drop)

    name = "mixed_3_set"
    features_to_drop = []
    df_mixed_3_set = load_data(logger, name, features_to_drop)

    name = "mixed_4_set"
    features_to_drop = [
        "loan__value_count__value_minus1",
        "loan__range_count__max_0__min_10000000000000",
        "loan__range_count__max_10000000000000__min_0",
        "loan__number_crossing_m__m_minus1",
        "loan__ratio_beyond_r_sigma__r_5",
        "loan__ratio_beyond_r_sigma__r_6",
        "loan__ratio_beyond_r_sigma__r_7",
        "loan__ratio_beyond_r_sigma__r_10",
    ]
    df_mixed_4_set = load_data(logger, name, features_to_drop)

    name = "fft_real_set"
    features_to_drop = [
        "loan__fft_coefficient__attr_real__coeff_51",
        "loan__fft_coefficient__attr_real__coeff_52",
        "loan__fft_coefficient__attr_real__coeff_53",
        "loan__fft_coefficient__attr_real__coeff_54",
        "loan__fft_coefficient__attr_real__coeff_55",
        "loan__fft_coefficient__attr_real__coeff_56",
        "loan__fft_coefficient__attr_real__coeff_57",
        "loan__fft_coefficient__attr_real__coeff_58",
        "loan__fft_coefficient__attr_real__coeff_59",
        "loan__fft_coefficient__attr_real__coeff_60",
        "loan__fft_coefficient__attr_real__coeff_61",
        "loan__fft_coefficient__attr_real__coeff_62",
        "loan__fft_coefficient__attr_real__coeff_63",
        "loan__fft_coefficient__attr_real__coeff_64",
        "loan__fft_coefficient__attr_real__coeff_65",
        "loan__fft_coefficient__attr_real__coeff_66",
        "loan__fft_coefficient__attr_real__coeff_67",
        "loan__fft_coefficient__attr_real__coeff_68",
        "loan__fft_coefficient__attr_real__coeff_69",
        "loan__fft_coefficient__attr_real__coeff_70",
        "loan__fft_coefficient__attr_real__coeff_71",
        "loan__fft_coefficient__attr_real__coeff_72",
        "loan__fft_coefficient__attr_real__coeff_73",
        "loan__fft_coefficient__attr_real__coeff_74",
        "loan__fft_coefficient__attr_real__coeff_75",
        "loan__fft_coefficient__attr_real__coeff_76",
        "loan__fft_coefficient__attr_real__coeff_77",
        "loan__fft_coefficient__attr_real__coeff_78",
        "loan__fft_coefficient__attr_real__coeff_79",
        "loan__fft_coefficient__attr_real__coeff_80",
        "loan__fft_coefficient__attr_real__coeff_81",
        "loan__fft_coefficient__attr_real__coeff_82",
        "loan__fft_coefficient__attr_real__coeff_83",
        "loan__fft_coefficient__attr_real__coeff_84",
        "loan__fft_coefficient__attr_real__coeff_85",
        "loan__fft_coefficient__attr_real__coeff_86",
        "loan__fft_coefficient__attr_real__coeff_87",
        "loan__fft_coefficient__attr_real__coeff_88",
        "loan__fft_coefficient__attr_real__coeff_89",
        "loan__fft_coefficient__attr_real__coeff_90",
        "loan__fft_coefficient__attr_real__coeff_91",
        "loan__fft_coefficient__attr_real__coeff_92",
        "loan__fft_coefficient__attr_real__coeff_93",
        "loan__fft_coefficient__attr_real__coeff_94",
        "loan__fft_coefficient__attr_real__coeff_95",
        "loan__fft_coefficient__attr_real__coeff_96",
        "loan__fft_coefficient__attr_real__coeff_97",
        "loan__fft_coefficient__attr_real__coeff_98",
        "loan__fft_coefficient__attr_real__coeff_99",
    ]
    df_fft_real_set = load_data(logger, name, features_to_drop)

    name = "fft_imag_set"
    features_to_drop = [
        "loan__fft_coefficient__attr_imag__coeff_0",
        "loan__fft_coefficient__attr_imag__coeff_50",
        "loan__fft_coefficient__attr_imag__coeff_51",
        "loan__fft_coefficient__attr_imag__coeff_52",
        "loan__fft_coefficient__attr_imag__coeff_53",
        "loan__fft_coefficient__attr_imag__coeff_54",
        "loan__fft_coefficient__attr_imag__coeff_55",
        "loan__fft_coefficient__attr_imag__coeff_56",
        "loan__fft_coefficient__attr_imag__coeff_57",
        "loan__fft_coefficient__attr_imag__coeff_58",
        "loan__fft_coefficient__attr_imag__coeff_59",
        "loan__fft_coefficient__attr_imag__coeff_60",
        "loan__fft_coefficient__attr_imag__coeff_61",
        "loan__fft_coefficient__attr_imag__coeff_62",
        "loan__fft_coefficient__attr_imag__coeff_63",
        "loan__fft_coefficient__attr_imag__coeff_64",
        "loan__fft_coefficient__attr_imag__coeff_65",
        "loan__fft_coefficient__attr_imag__coeff_66",
        "loan__fft_coefficient__attr_imag__coeff_67",
        "loan__fft_coefficient__attr_imag__coeff_68",
        "loan__fft_coefficient__attr_imag__coeff_69",
        "loan__fft_coefficient__attr_imag__coeff_70",
        "loan__fft_coefficient__attr_imag__coeff_71",
        "loan__fft_coefficient__attr_imag__coeff_72",
        "loan__fft_coefficient__attr_imag__coeff_73",
        "loan__fft_coefficient__attr_imag__coeff_74",
        "loan__fft_coefficient__attr_imag__coeff_75",
        "loan__fft_coefficient__attr_imag__coeff_76",
        "loan__fft_coefficient__attr_imag__coeff_77",
        "loan__fft_coefficient__attr_imag__coeff_78",
        "loan__fft_coefficient__attr_imag__coeff_79",
        "loan__fft_coefficient__attr_imag__coeff_80",
        "loan__fft_coefficient__attr_imag__coeff_81",
        "loan__fft_coefficient__attr_imag__coeff_82",
        "loan__fft_coefficient__attr_imag__coeff_83",
        "loan__fft_coefficient__attr_imag__coeff_84",
        "loan__fft_coefficient__attr_imag__coeff_85",
        "loan__fft_coefficient__attr_imag__coeff_86",
        "loan__fft_coefficient__attr_imag__coeff_87",
        "loan__fft_coefficient__attr_imag__coeff_88",
        "loan__fft_coefficient__attr_imag__coeff_89",
        "loan__fft_coefficient__attr_imag__coeff_90",
        "loan__fft_coefficient__attr_imag__coeff_91",
        "loan__fft_coefficient__attr_imag__coeff_92",
        "loan__fft_coefficient__attr_imag__coeff_93",
        "loan__fft_coefficient__attr_imag__coeff_94",
        "loan__fft_coefficient__attr_imag__coeff_95",
        "loan__fft_coefficient__attr_imag__coeff_96",
        "loan__fft_coefficient__attr_imag__coeff_97",
        "loan__fft_coefficient__attr_imag__coeff_98",
        "loan__fft_coefficient__attr_imag__coeff_99",
    ]
    df_fft_imag_set = load_data(logger, name, features_to_drop)

    name = "fft_abs_set"
    features_to_drop = [
        "loan__fft_coefficient__attr_abs__coeff_51",
        "loan__fft_coefficient__attr_abs__coeff_52",
        "loan__fft_coefficient__attr_abs__coeff_53",
        "loan__fft_coefficient__attr_abs__coeff_54",
        "loan__fft_coefficient__attr_abs__coeff_55",
        "loan__fft_coefficient__attr_abs__coeff_56",
        "loan__fft_coefficient__attr_abs__coeff_57",
        "loan__fft_coefficient__attr_abs__coeff_58",
        "loan__fft_coefficient__attr_abs__coeff_59",
        "loan__fft_coefficient__attr_abs__coeff_60",
        "loan__fft_coefficient__attr_abs__coeff_61",
        "loan__fft_coefficient__attr_abs__coeff_62",
        "loan__fft_coefficient__attr_abs__coeff_63",
        "loan__fft_coefficient__attr_abs__coeff_64",
        "loan__fft_coefficient__attr_abs__coeff_65",
        "loan__fft_coefficient__attr_abs__coeff_66",
        "loan__fft_coefficient__attr_abs__coeff_67",
        "loan__fft_coefficient__attr_abs__coeff_68",
        "loan__fft_coefficient__attr_abs__coeff_69",
        "loan__fft_coefficient__attr_abs__coeff_70",
        "loan__fft_coefficient__attr_abs__coeff_71",
        "loan__fft_coefficient__attr_abs__coeff_72",
        "loan__fft_coefficient__attr_abs__coeff_73",
        "loan__fft_coefficient__attr_abs__coeff_74",
        "loan__fft_coefficient__attr_abs__coeff_75",
        "loan__fft_coefficient__attr_abs__coeff_76",
        "loan__fft_coefficient__attr_abs__coeff_77",
        "loan__fft_coefficient__attr_abs__coeff_78",
        "loan__fft_coefficient__attr_abs__coeff_79",
        "loan__fft_coefficient__attr_abs__coeff_80",
        "loan__fft_coefficient__attr_abs__coeff_81",
        "loan__fft_coefficient__attr_abs__coeff_82",
        "loan__fft_coefficient__attr_abs__coeff_83",
        "loan__fft_coefficient__attr_abs__coeff_84",
        "loan__fft_coefficient__attr_abs__coeff_85",
        "loan__fft_coefficient__attr_abs__coeff_86",
        "loan__fft_coefficient__attr_abs__coeff_87",
        "loan__fft_coefficient__attr_abs__coeff_88",
        "loan__fft_coefficient__attr_abs__coeff_89",
        "loan__fft_coefficient__attr_abs__coeff_90",
        "loan__fft_coefficient__attr_abs__coeff_91",
        "loan__fft_coefficient__attr_abs__coeff_92",
        "loan__fft_coefficient__attr_abs__coeff_93",
        "loan__fft_coefficient__attr_abs__coeff_94",
        "loan__fft_coefficient__attr_abs__coeff_95",
        "loan__fft_coefficient__attr_abs__coeff_96",
        "loan__fft_coefficient__attr_abs__coeff_97",
        "loan__fft_coefficient__attr_abs__coeff_98",
        "loan__fft_coefficient__attr_abs__coeff_99",
    ]
    df_fft_abs_set = load_data(logger, name, features_to_drop)

    name = "fft_angle_set"
    features_to_drop = [
        "loan__fft_coefficient__attr_angle__coeff_0",
        "loan__fft_coefficient__attr_angle__coeff_51",
        "loan__fft_coefficient__attr_angle__coeff_52",
        "loan__fft_coefficient__attr_angle__coeff_53",
        "loan__fft_coefficient__attr_angle__coeff_54",
        "loan__fft_coefficient__attr_angle__coeff_55",
        "loan__fft_coefficient__attr_angle__coeff_56",
        "loan__fft_coefficient__attr_angle__coeff_57",
        "loan__fft_coefficient__attr_angle__coeff_58",
        "loan__fft_coefficient__attr_angle__coeff_59",
        "loan__fft_coefficient__attr_angle__coeff_60",
        "loan__fft_coefficient__attr_angle__coeff_61",
        "loan__fft_coefficient__attr_angle__coeff_62",
        "loan__fft_coefficient__attr_angle__coeff_63",
        "loan__fft_coefficient__attr_angle__coeff_64",
        "loan__fft_coefficient__attr_angle__coeff_65",
        "loan__fft_coefficient__attr_angle__coeff_66",
        "loan__fft_coefficient__attr_angle__coeff_67",
        "loan__fft_coefficient__attr_angle__coeff_68",
        "loan__fft_coefficient__attr_angle__coeff_69",
        "loan__fft_coefficient__attr_angle__coeff_70",
        "loan__fft_coefficient__attr_angle__coeff_71",
        "loan__fft_coefficient__attr_angle__coeff_72",
        "loan__fft_coefficient__attr_angle__coeff_73",
        "loan__fft_coefficient__attr_angle__coeff_74",
        "loan__fft_coefficient__attr_angle__coeff_75",
        "loan__fft_coefficient__attr_angle__coeff_76",
        "loan__fft_coefficient__attr_angle__coeff_77",
        "loan__fft_coefficient__attr_angle__coeff_78",
        "loan__fft_coefficient__attr_angle__coeff_79",
        "loan__fft_coefficient__attr_angle__coeff_80",
        "loan__fft_coefficient__attr_angle__coeff_81",
        "loan__fft_coefficient__attr_angle__coeff_82",
        "loan__fft_coefficient__attr_angle__coeff_83",
        "loan__fft_coefficient__attr_angle__coeff_84",
        "loan__fft_coefficient__attr_angle__coeff_85",
        "loan__fft_coefficient__attr_angle__coeff_86",
        "loan__fft_coefficient__attr_angle__coeff_87",
        "loan__fft_coefficient__attr_angle__coeff_88",
        "loan__fft_coefficient__attr_angle__coeff_89",
        "loan__fft_coefficient__attr_angle__coeff_90",
        "loan__fft_coefficient__attr_angle__coeff_91",
        "loan__fft_coefficient__attr_angle__coeff_92",
        "loan__fft_coefficient__attr_angle__coeff_93",
        "loan__fft_coefficient__attr_angle__coeff_94",
        "loan__fft_coefficient__attr_angle__coeff_95",
        "loan__fft_coefficient__attr_angle__coeff_96",
        "loan__fft_coefficient__attr_angle__coeff_97",
        "loan__fft_coefficient__attr_angle__coeff_98",
        "loan__fft_coefficient__attr_angle__coeff_99",
    ]
    df_fft_angle_set = load_data(logger, name, features_to_drop)

    dfs = [
        df_acf_pacf_set,
        df_change_quantile_set,
        df_cwt_coeff_set,
        df_fft_abs_set,
        df_fft_angle_set,
        df_fft_imag_set,
        df_fft_real_set,
        df_liner_agg_linear_set,
        df_mixed_1_set,
        df_mixed_2_set,
        df_mixed_3_set,
        df_mixed_4_set,
        df_sym,
    ]

    result_df = pd.concat(dfs, axis=1)
    logger.info(f"Shape of the combined Data Frame {result_df.shape}")
    return result_df


if __name__ == "__main__":

    # Create a Stream only logger
    logger = common.get_logger("generate_features")
    logger.info("Starting to generate features")

    results_df = combine_features(logger=logger)
    logger.info(
        f"Writing the combined parquet to {constants.FEATURES_DATA_DIR}/cast/tsfresh_f_merged.parquet"
    )
    results_df.to_parquet(
        f"{constants.FEATURES_DATA_DIR}/cast/tsfresh_f_merged.parquet", index=True
    )
