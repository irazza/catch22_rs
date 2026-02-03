mod catch22;
mod statistics;

pub const N_CATCH22: usize = 25;
const MIN_INPUT_LEN: usize = 4;

// Performance-focused implementation notes:
// - Input validation is centralized in `compute_basic_stats_checked`, which also computes
//   mean/std/slope in a single pass to avoid repeated traversals of the input.
// - `compute_all` reuses a single autocorrelation (and derived tau) for multiple features,
//   and routes FC_LocalSimple_mean1_tauresrat through that shared autocorr to avoid extra FFTs.
// - PD_PeriodicityWang now uses an FFT-based autocovariance (O(n log n)) instead of per-lag
//   autocovariance (O(n^2)) in the hot path.
// - Unchecked variants exist for benchmarking or trusted call sites to skip validation costs.
#[derive(Debug, Clone, Copy)]
struct BasicStats {
    mean: f64,
    std_dev: f64,
    slope: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Catch22Error {
    InputTooShort { len: usize, min_len: usize },
    NonFiniteValue { index: usize, value: f64 },
    InvalidFeatureIndex { index: usize, max: usize },
}

impl std::fmt::Display for Catch22Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Catch22Error::InputTooShort { len, min_len } => {
                write!(f, "input length {len} is smaller than minimum {min_len}")
            }
            Catch22Error::NonFiniteValue { index, value } => {
                write!(f, "input value at index {index} is not finite: {value}")
            }
            Catch22Error::InvalidFeatureIndex { index, max } => {
                write!(f, "feature index {index} is out of range (max {max})")
            }
        }
    }
}

impl std::error::Error for Catch22Error {}

fn validate_input(x: &[f64]) -> Result<(), Catch22Error> {
    if x.len() < MIN_INPUT_LEN {
        return Err(Catch22Error::InputTooShort {
            len: x.len(),
            min_len: MIN_INPUT_LEN,
        });
    }

    for (index, &value) in x.iter().enumerate() {
        if !value.is_finite() {
            return Err(Catch22Error::NonFiniteValue { index, value });
        }
    }

    Ok(())
}

fn compute_basic_stats_checked(x: &[f64]) -> Result<BasicStats, Catch22Error> {
    if x.len() < MIN_INPUT_LEN {
        return Err(Catch22Error::InputTooShort {
            len: x.len(),
            min_len: MIN_INPUT_LEN,
        });
    }

    let mut mean = 0.0;
    let mut m2 = 0.0;
    let mut sum_xy = 0.0;
    let mut count = 0usize;

    for (index, &value) in x.iter().enumerate() {
        if !value.is_finite() {
            return Err(Catch22Error::NonFiniteValue { index, value });
        }

        count += 1;
        let delta = value - mean;
        mean += delta / count as f64;
        let delta2 = value - mean;
        m2 += delta * delta2;
        sum_xy += (index + 1) as f64 * value;
    }

    let variance = m2 / (count - 1) as f64;
    let std_dev = variance.sqrt();

    let n_f = count as f64;
    let sum_x = n_f * (n_f + 1.0) / 2.0;
    let sum_x2 = n_f * (n_f + 1.0) * (2.0 * n_f + 1.0) / 6.0;
    let sum_y = mean * n_f;
    let slope = (n_f * sum_xy - sum_x * sum_y) / (n_f * sum_x2 - sum_x * sum_x);
    debug_assert!(slope.is_finite());

    Ok(BasicStats {
        mean,
        std_dev,
        slope,
    })
}

fn compute_basic_stats_unchecked(x: &[f64]) -> BasicStats {
    debug_assert!(x.len() >= MIN_INPUT_LEN);

    let mut mean = 0.0;
    let mut m2 = 0.0;
    let mut sum_xy = 0.0;
    let mut count = 0usize;

    for (index, &value) in x.iter().enumerate() {
        count += 1;
        let delta = value - mean;
        mean += delta / count as f64;
        let delta2 = value - mean;
        m2 += delta * delta2;
        sum_xy += (index + 1) as f64 * value;
    }

    let variance = m2 / (count - 1) as f64;
    let std_dev = variance.sqrt();

    let n_f = count as f64;
    let sum_x = n_f * (n_f + 1.0) / 2.0;
    let sum_x2 = n_f * (n_f + 1.0) * (2.0 * n_f + 1.0) / 6.0;
    let sum_y = mean * n_f;
    let slope = (n_f * sum_xy - sum_x * sum_y) / (n_f * sum_x2 - sum_x * sum_x);
    debug_assert!(slope.is_finite());

    BasicStats {
        mean,
        std_dev,
        slope,
    }
}

pub fn compute(x: &[f64], n: usize) -> Result<f64, Catch22Error> {
    validate_input(x)?;
    if n >= N_CATCH22 {
        return Err(Catch22Error::InvalidFeatureIndex {
            index: n,
            max: N_CATCH22 - 1,
        });
    }

    Ok(compute_unchecked(x, n))
}

pub fn compute_all(x: &[f64]) -> Result<[f64; N_CATCH22], Catch22Error> {
    let stats = compute_basic_stats_checked(x)?;
    Ok(compute_all_inner(x, stats))
}

pub fn compute_unchecked(x: &[f64], n: usize) -> f64 {
    debug_assert!(n < N_CATCH22);
    match n {
        0 => catch22::dn_outlier_include_np_001_mdrmd(x, false),
        1 => catch22::dn_outlier_include_np_001_mdrmd(x, true),
        2 => catch22::dn_histogram_mode_n(x, 5),
        3 => catch22::dn_histogram_mode_n(x, 10),
        4 => catch22::co_embed2_dist_tau_d_expfit_meandiff(x),
        5 => catch22::co_f1ecac(x),
        6 => catch22::co_first_min_ac(x),
        7 => catch22::co_histogram_ami_even_tau_bins(x, 2, 5),
        8 => catch22::co_trev_1_num(x),
        9 => catch22::fc_local_simple_mean_tauresrat(x, 1),
        10 => catch22::fc_local_simple_mean_stderr(x, 3),
        11 => catch22::in_auto_mutual_info_stats_tau_gaussian_fmmi(x, 40.0),
        12 => catch22::md_hrv_classic_pnn(x, 40),
        13 => catch22::sb_binary_stats_diff_longstretch0(x),
        14 => catch22::sb_binary_stats_mean_longstretch1(x),
        15 => catch22::sb_motif_three_quantile_hh(x),
        16 => catch22::sc_fluct_anal_2_50_1_logi_prop_r1(x, 1, "rsrangefit"),
        17 => catch22::sc_fluct_anal_2_50_1_logi_prop_r1(x, 2, "dfa"),
        18 => catch22::sp_summaries_welch_rect(x, "area_5_1"),
        19 => catch22::sp_summaries_welch_rect(x, "centroid"),
        20 => catch22::sb_transition_matrix_3ac_sumdiagcov(x),
        21 => catch22::pd_periodicity_wang_th0_01(x),
        22 => statistics::mean(x),
        23 => statistics::std_dev(x),
        24 => statistics::slope(x),
        _ => unreachable!("validated feature index"),
    }
}

pub fn compute_all_unchecked(x: &[f64]) -> [f64; N_CATCH22] {
    let stats = compute_basic_stats_unchecked(x);
    compute_all_inner(x, stats)
}

fn compute_all_inner(x: &[f64], stats: BasicStats) -> [f64; N_CATCH22] {
    let mut out = [0.0; N_CATCH22];

    let autocorr = statistics::autocorr(x);
    let tau = statistics::first_zero_from_autocorr(&autocorr, x.len());

    out[0] = catch22::dn_outlier_include_np_001_mdrmd(x, false);
    out[1] = catch22::dn_outlier_include_np_001_mdrmd(x, true);
    out[2] = catch22::dn_histogram_mode_n(x, 5);
    out[3] = catch22::dn_histogram_mode_n(x, 10);
    out[4] = catch22::co_embed2_dist_tau_d_expfit_meandiff_with_tau(x, tau);
    out[5] = catch22::co_f1ecac_from_autocorr(x, &autocorr);
    out[6] = catch22::co_first_min_ac_from_autocorr(x, &autocorr);
    out[7] = catch22::co_histogram_ami_even_tau_bins(x, 2, 5);
    out[8] = catch22::co_trev_1_num(x);
    out[9] = catch22::fc_local_simple_mean_tauresrat_from_autocorr(x, &autocorr, 1);
    out[10] = catch22::fc_local_simple_mean_stderr(x, 3);
    out[11] = catch22::in_auto_mutual_info_stats_tau_gaussian_fmmi(x, 40.0);
    out[12] = catch22::md_hrv_classic_pnn(x, 40);
    out[13] = catch22::sb_binary_stats_diff_longstretch0(x);
    out[14] = catch22::sb_binary_stats_mean_longstretch1(x);
    out[15] = catch22::sb_motif_three_quantile_hh(x);
    out[16] = catch22::sc_fluct_anal_2_50_1_logi_prop_r1(x, 1, "rsrangefit");
    out[17] = catch22::sc_fluct_anal_2_50_1_logi_prop_r1(x, 2, "dfa");
    out[18] = catch22::sp_summaries_welch_rect(x, "area_5_1");
    out[19] = catch22::sp_summaries_welch_rect(x, "centroid");
    out[20] = catch22::sb_transition_matrix_3ac_sumdiagcov_with_tau(x, tau);
    out[21] = catch22::pd_periodicity_wang_th0_01(x);
    out[22] = stats.mean;
    out[23] = stats.std_dev;
    out[24] = stats.slope;

    out
}

pub fn zscore(x: &[f64]) -> Vec<f64> {
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    let std = (x.iter().map(|val| (val - mean).powi(2)).sum::<f64>() / x.len() as f64).sqrt();
    x.iter().map(|val| (val - mean) / std).collect()
}
