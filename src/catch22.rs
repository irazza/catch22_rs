use crate::statistics::{
    autocorr, autocorr_lag, autocovariance, coarsegrain, covariance_matrix, f_entropy, first_zero,
    first_zero_from_autocorr, histbinassign, histcount_edges, histcounts, is_constant, linreg,
    max_, mean, median, min_, norm, num_bins_auto, splinefit, std_dev, welch,
};

pub fn dn_outlier_include_np_001_mdrmd(a: &[f64], is_pos: bool) -> f64 {
    // constant check
    if is_constant(a) {
        return 0.0;
    }
    // sign is false if we want to represent -1

    let mut a = a.to_vec();
    let inc = 0.01;

    if !is_pos {
        a.iter_mut().for_each(|x| *x = -*x);
    }

    let tot = a.iter().filter(|&x| *x >= 0.0).count();
    let max_val = *a.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();

    if max_val < inc {
        return 0.0;
    }

    let n_thresh = ((max_val / inc) + 1.0) as usize;

    let mut r = vec![0.0; a.len()];

    let mut msdti1 = vec![0.0; n_thresh];
    let mut msdti3 = vec![0.0; n_thresh];
    let mut msdti4 = vec![0.0; n_thresh];

    for i in 0..n_thresh {
        let mut high_size = 0;

        for j in 0..a.len() {
            if a[j] >= i as f64 * inc {
                r[high_size] = (j + 1) as f64;
                high_size += 1;
            }
        }

        if high_size > 1 {
            let mut sum_dt = 0.0;
            for j in 0..high_size - 1 {
                sum_dt += r[j + 1] - r[j];
            }
            msdti1[i] = sum_dt / (high_size - 1) as f64;
        } else {
            msdti1[i] = 0.0;
        }
        msdti3[i] = ((high_size.saturating_sub(1)) as f64 * 100.0) / tot as f64;
        msdti4[i] = median(&r[..high_size]) / (a.len() as f64 / 2.0) - 1.0;
    }

    let trim_tr = 2.0;
    let mut mj = 0;
    let mut fbi = n_thresh - 1;

    for i in 0..n_thresh {
        if msdti3[i] > trim_tr {
            mj = i;
        }
        if msdti1[n_thresh - i - 1].is_nan() {
            fbi = n_thresh - i - 1;
        }
    }

    let trim_lim = mj.min(fbi);
    return median(&msdti4[..trim_lim + 1]);
}

pub fn dn_histogram_mode_n(a: &[f64], n_bins: usize) -> f64 {
    let (bin_counts, bin_edges) = histcounts(a, n_bins);

    let mut max_count = 0;
    let mut num_maxs = 1;
    let mut res = 0.0;

    for i in 0..n_bins {
        if bin_counts[i] > max_count {
            max_count = bin_counts[i];
            num_maxs = 1;
            res = (bin_edges[i] + bin_edges[i + 1]) / 2.0;
        } else if bin_counts[i] == max_count {
            num_maxs += 1;
            res += (bin_edges[i] + bin_edges[i + 1]) / 2.0;
        }
    }

    return res / num_maxs as f64;
}

pub fn co_embed2_dist_tau_d_expfit_meandiff(a: &[f64]) -> f64 {
    let tau = first_zero(a, a.len());
    co_embed2_dist_tau_d_expfit_meandiff_with_tau(a, tau)
}

pub(crate) fn co_embed2_dist_tau_d_expfit_meandiff_with_tau(a: &[f64], tau: usize) -> f64 {
    if a.len() < 2 {
        return 0.0;
    }

    let mut tau = tau;
    if tau > a.len() / 10 {
        tau = a.len() / 10;
    }
    if tau == 0 || a.len() <= tau + 1 {
        return 0.0;
    }

    let d_len = a.len() - tau - 1;
    if d_len < 2 {
        return 0.0;
    }

    let mut d = vec![0.0; d_len];

    for i in 0..d_len {
        d[i] = ((a[i + 1] - a[i]).powi(2) + (a[i + tau] - a[i + tau + 1]).powi(2)).sqrt();

        if d[i].is_nan() {
            return f64::NAN;
        }
    }

    let l = mean(&d);

    let n_bins = num_bins_auto(&d);

    if n_bins == 0 {
        return 0.0;
    }
    let (hist_counts, bin_edges) = histcounts(&d, n_bins);
    let mut hist_counts_norm = vec![0.0; n_bins];

    for i in 0..n_bins {
        hist_counts_norm[i] = hist_counts[i] as f64 / d_len as f64;
    }

    let mut d_expfit_diff = vec![0.0; n_bins];

    for i in 0..n_bins {
        let mut expf = (-(bin_edges[i] + bin_edges[i + 1]) * 0.5 / l).exp() / l;
        if expf < 0.0 {
            expf = 0.0;
        }
        d_expfit_diff[i] = (hist_counts_norm[i] - expf).abs();
    }

    return mean(&d_expfit_diff);
}

pub fn co_f1ecac(a: &[f64]) -> f64 {
    let autocorr = autocorr(a);
    co_f1ecac_from_autocorr(a, &autocorr)
}

pub(crate) fn co_f1ecac_from_autocorr(a: &[f64], autocorr: &[f64]) -> f64 {
    if autocorr.is_empty() {
        return 0.0;
    }

    let thresh = 1.0 / 1.0f64.exp();

    let mut out = a.len() as f64;

    for i in 0..a.len() - 2 {
        if autocorr[i + 1] < thresh {
            let m = autocorr[i + 1] - autocorr[i];
            let dy = thresh - autocorr[i];
            let dx = dy / m;
            out = i as f64 + dx;
            return out;
        }
    }
    return out;
}

pub fn co_first_min_ac(a: &[f64]) -> f64 {
    let autocorr = autocorr(a);
    co_first_min_ac_from_autocorr(a, &autocorr)
}

pub(crate) fn co_first_min_ac_from_autocorr(a: &[f64], autocorr: &[f64]) -> f64 {
    if autocorr.len() < 3 {
        return a.len() as f64;
    }

    let mut min_ind = a.len();

    for i in 1..a.len() - 1 {
        if autocorr[i] < autocorr[i - 1] && autocorr[i] < autocorr[i + 1] {
            min_ind = i;
            break;
        }
    }

    return min_ind as f64;
}

pub fn co_histogram_ami_even_tau_bins(a: &[f64], tau: usize, n_bins: usize) -> f64 {
    let mut y1 = vec![0.0; a.len() - tau];
    let mut y2 = vec![0.0; a.len() - tau];

    for i in 0..a.len() - tau {
        y1[i] = a[i];
        y2[i] = a[i + tau];
    }

    let max_val = max_(a);
    let min_val = min_(a);

    let bin_step = (max_val - min_val + 0.2) / n_bins as f64;

    let mut bin_edges = vec![0.0; n_bins + 1];

    for i in 0..n_bins + 1 {
        bin_edges[i] = min_val + (i as f64 * bin_step) - 0.1;
    }

    let bins1 = histbinassign(&y1, &bin_edges);
    let bins2 = histbinassign(&y2, &bin_edges);

    let mut bins12 = vec![0.0; a.len() - tau];
    let mut bin_edges12 = vec![0.0; (n_bins + 1) * (n_bins + 1)];

    for i in 0..a.len() - tau {
        bins12[i] = ((bins1[i] - 1) * (n_bins + 1) + bins2[i]) as f64;
    }

    for i in 0..(n_bins + 1) * (n_bins + 1) {
        bin_edges12[i] = (i + 1) as f64;
    }

    let joint_hist_linear = histcount_edges(&bins12, &bin_edges12);

    let mut pij = vec![vec![0.0; n_bins]; n_bins];

    let mut sum_bins = 0.0;

    for i in 0..n_bins {
        for j in 0..n_bins {
            pij[j][i] = joint_hist_linear[i * (n_bins + 1) + j] as f64;
            sum_bins += pij[j][i];
        }
    }

    for i in 0..n_bins {
        for j in 0..n_bins {
            pij[j][i] /= sum_bins;
        }
    }

    let mut pi = vec![0.0; n_bins];
    let mut pj = vec![0.0; n_bins];

    for i in 0..n_bins {
        for j in 0..n_bins {
            pi[i] += pij[i][j];
            pj[j] += pij[i][j];
        }
    }

    let mut ami = 0.0;
    for i in 0..n_bins {
        for j in 0..n_bins {
            if pij[i][j] > 0.0 {
                ami += pij[i][j] * (pij[i][j] / (pi[i] * pj[j])).ln();
            }
        }
    }

    return ami;
}

pub fn co_trev_1_num(a: &[f64]) -> f64 {
    if a.len() < 2 {
        return 0.0;
    }

    let mut sum = 0.0;
    for i in 0..a.len() - 1 {
        let diff = a[i + 1] - a[i];
        sum += diff * diff * diff;
    }

    sum / (a.len() - 1) as f64
}

fn local_mean_residuals(a: &[f64], train_length: usize) -> Vec<f64> {
    if a.len() <= train_length {
        return Vec::new();
    }

    let res_len = a.len() - train_length;
    let mut res = vec![0.0; res_len];
    let mut window_sum = a[..train_length].iter().sum::<f64>();

    for i in 0..res_len {
        let yest = window_sum / train_length as f64;
        res[i] = a[i + train_length] - yest;
        window_sum += a[i + train_length] - a[i];
    }

    res
}

pub fn fc_local_simple_mean_tauresrat(a: &[f64], train_length: usize) -> f64 {
    let res = local_mean_residuals(a, train_length);

    let res_ac1st_z = first_zero(&res, res.len()) as f64;
    let y_ac1st_z = first_zero(a, a.len()) as f64;

    let out = res_ac1st_z / y_ac1st_z;
    return out;
}

pub(crate) fn fc_local_simple_mean_tauresrat_from_autocorr(
    a: &[f64],
    autocorr: &[f64],
    train_length: usize,
) -> f64 {
    let res = local_mean_residuals(a, train_length);

    let res_ac1st_z = first_zero(&res, res.len()) as f64;
    let y_ac1st_z = first_zero_from_autocorr(autocorr, a.len()) as f64;

    let out = res_ac1st_z / y_ac1st_z;
    return out;
}

pub fn fc_local_simple_mean_stderr(a: &[f64], train_length: usize) -> f64 {
    let res = local_mean_residuals(a, train_length);

    let out = std_dev(&res);
    return out;
}

pub fn in_auto_mutual_info_stats_tau_gaussian_fmmi(a: &[f64], tau: f64) -> f64 {
    let mut tau = tau;

    if tau > (a.len() as f64 / 2.0).ceil() {
        tau = (a.len() as f64 / 2.0).ceil();
    }

    let mut ami = vec![0.0; a.len()];

    // let prefix_mean_a = a
    //     .iter()
    //     .enumerate()
    //     .rev()
    //     .scan(0.0, |state, (i, x)| {
    //         *state += x;
    //         Some(*state / (i + 1) as f64)
    //     })
    //     .collect::<Vec<f64>>();
    for i in 0..tau as usize {
        let ac = autocorr_lag(a, i + 1);
        ami[i] = -0.5 * (1.0 - ac * ac).ln();
    }

    let mut fmmi = tau;

    for i in 1..tau as usize - 1 {
        if ami[i] < ami[i - 1] && ami[i] < ami[i + 1] {
            fmmi = i as f64;
            break;
        }
    }
    return fmmi;
}

pub fn md_hrv_classic_pnn(a: &[f64], pnn: usize) -> f64 {
    let mut pnn40 = 0.0;

    if a.len() < 2 {
        return 0.0;
    }

    for i in 0..a.len() - 1 {
        let diff = a[i + 1] - a[i];
        if diff.abs() * 1000.0 > pnn as f64 {
            pnn40 += 1.0;
        }
    }

    return pnn40 / (a.len() - 1) as f64;
}

pub fn sb_binary_stats_diff_longstretch0(a: &[f64]) -> f64 {
    let mut y_bin = vec![0; a.len() - 1];

    for i in 0..a.len() - 1 {
        let diff_temp = a[i + 1] - a[i];
        if diff_temp < 0.0 {
            y_bin[i] = 0
        } else {
            y_bin[i] = 1
        }
    }

    let mut max_stretch = 0;
    let mut last1 = 0;

    for i in 0..a.len() - 1 {
        if y_bin[i] == 1 || i == a.len() - 2 {
            let stretch = i - last1;

            if stretch > max_stretch {
                max_stretch = stretch;
            }

            last1 = i;
        }
    }

    return max_stretch as f64;
}

pub fn sb_binary_stats_mean_longstretch1(a: &[f64]) -> f64 {
    let mut y_bin = vec![0; a.len() - 1];
    let a_mean = mean(a);
    for i in 0..a.len() - 1 {
        if a[i] - a_mean <= 0.0 {
            y_bin[i] = 0
        } else {
            y_bin[i] = 1
        }
    }

    let mut max_stretch = 0;
    let mut last1 = 0;

    for i in 0..a.len() - 1 {
        if y_bin[i] == 0 || i == a.len() - 2 {
            let stretch = i - last1;

            if stretch > max_stretch {
                max_stretch = stretch;
            }

            last1 = i;
        }
    }

    return max_stretch as f64;
}

pub fn sb_motif_three_quantile_hh(a: &[f64]) -> f64 {
    let alphabet_size = 3;
    let yt = coarsegrain(a, alphabet_size);

    let mut r1 = vec![Vec::with_capacity(a.len()); alphabet_size];
    for i in 0..alphabet_size {
        for j in 0..a.len() {
            if yt[j] == i + 1 {
                r1[i].push(j);
            }
        }
    }

    for i in 0..alphabet_size {
        if r1[i].last() == Some(&(a.len() - 1)) {
            r1[i].pop();
        }
    }

    let mut r2 = vec![vec![Vec::new(); alphabet_size]; alphabet_size];
    let mut out2 = vec![vec![0.0; alphabet_size]; alphabet_size];

    for i in 0..alphabet_size {
        for j in 0..alphabet_size {
            for k in 0..r1[i].len() {
                let tmp_idx = yt[r1[i][k] + 1];
                if tmp_idx == (j + 1) {
                    r2[i][j].push(r1[i][k]);
                }
            }
            let tmp = r2[i][j].len() as f64 / (a.len() - 1) as f64;
            out2[i][j] = tmp;
        }
    }

    let mut hh = 0.0;
    for i in 0..alphabet_size {
        hh += f_entropy(&out2[i]);
    }
    return hh;
}

pub fn sc_fluct_anal_2_50_1_logi_prop_r1(a: &[f64], lag: usize, how: &str) -> f64 {
    let lin_low = (5.0f64).ln();
    let lin_high = ((a.len() / 2) as f64).ln();

    let n_tau_steps = 50;
    let tau_step = (lin_high - lin_low) / (n_tau_steps - 1) as f64;

    let mut tau = vec![0.0; n_tau_steps];
    for i in 0..n_tau_steps {
        tau[i] = (lin_low + i as f64 * tau_step).exp().round();
    }

    let mut n_tau = n_tau_steps;
    for i in 0..n_tau_steps - 1 {
        while tau[i] == tau[i + 1] && i < n_tau - 1 {
            for j in i + 1..n_tau_steps - 1 {
                tau[j] = tau[j + 1];
            }
            n_tau -= 1;
        }
    }

    if n_tau < 12 {
        return 0.0;
    }

    let size_cs = a.len() / lag;
    let mut y_cs = vec![0.0; size_cs];

    y_cs[0] = a[0];
    for i in 0..size_cs - 1 {
        y_cs[i + 1] = y_cs[i] + a[(i + 1) * lag];
    }

    let mut x_reg = vec![0.0; tau[n_tau - 1] as usize];
    for i in 0..tau[n_tau - 1] as usize {
        x_reg[i] = (i + 1) as f64;
    }

    let mut f = vec![0.0; n_tau];
    for i in 0..n_tau {
        let n_buffer = (size_cs as f64 / tau[i]) as usize;
        let mut buffer = vec![0.0; tau[i] as usize];

        f[i] = 0.0;

        for j in 0..n_buffer {
            let (m, b) = linreg(tau[i] as usize, &x_reg, &y_cs[j * tau[i] as usize..]);

            for k in 0..tau[i] as usize {
                buffer[k] = y_cs[j * tau[i] as usize + k] - (m * (k + 1) as f64 + b);
            }

            match how {
                "rsrangefit" => {
                    let max = max_(&buffer);
                    let min = min_(&buffer);
                    f[i] += (max - min).powi(2);
                }
                "dfa" => {
                    for k in 0..tau[i] as usize {
                        f[i] += buffer[k].powi(2);
                    }
                }
                _ => return 0.0,
            }
        }

        match how {
            "rsrangefit" => f[i] = (f[i] / n_buffer as f64).sqrt(),
            "dfa" => f[i] = (f[i] / n_buffer as f64 * tau[i]).sqrt(),
            _ => unreachable!(),
        }
    }

    let mut logtt = vec![0.0; n_tau];
    let mut logff = vec![0.0; n_tau];

    let ntt = n_tau;

    for i in 0..n_tau {
        logtt[i] = tau[i].ln();
        logff[i] = f[i].ln();
    }

    let min_points = 6;

    let nsserr = ntt - 2 * min_points + 1;

    let mut sserr = vec![0.0; nsserr];
    let mut buffer = vec![0.0; ntt - min_points + 1];

    for i in min_points..ntt - min_points + 1 {
        let (m1, b1) = linreg(i, &logtt, &logff);
        let (m2, b2) = linreg(ntt - i + 1, &logtt[i - 1..], &logff[i - 1..]);

        for j in 0..i {
            buffer[j] = logtt[j] * m1 + b1 - logff[j];
        }

        sserr[i - min_points] += norm(&buffer[..i]);

        for j in 0..ntt - i + 1 {
            buffer[j] = logtt[j + i - 1] * m2 + b2 - logff[j + i - 1];
        }

        sserr[i - min_points] += norm(&buffer[..ntt - i + 1]);
    }

    let mut first_min_ind = 0;
    let minimum = min_(&sserr);
    for i in 0..nsserr {
        if sserr[i] == minimum {
            first_min_ind = i + min_points - 1;
            break;
        }
    }
    return (first_min_ind + 1) as f64 / ntt as f64;
}

pub fn sp_summaries_welch_rect(a: &[f64], what: &str) -> f64 {
    let window = (0..a.len()).map(|_| 1.0).collect::<Vec<f64>>();
    let fs = 1.0;

    let (s, f) = welch(a, fs, &window);

    let mut w = vec![0.0; s.len()];
    let mut sw = vec![0.0; s.len()];

    for i in 0..s.len() {
        w[i] = 2.0 * std::f64::consts::PI * f[i];
        sw[i] = s[i] / (2.0 * std::f64::consts::PI);

        if sw[i].is_infinite() {
            return 0.0;
        }
    }

    let dw = w[1] - w[0];

    // cum sum of sw
    let s_cs = sw
        .iter()
        .scan(0.0, |state, x| {
            *state += x;
            Some(*state)
        })
        .collect::<Vec<f64>>();

    match what {
        "centroid" => {
            let s_cs_thresh = s_cs[s.len() - 1] / 2.0;
            let mut centroid = 0.0;
            for i in 0..s.len() {
                if s_cs[i] > s_cs_thresh {
                    centroid = w[i];
                    break;
                }
            }
            centroid
        }
        "area_5_1" => {
            let mut area_5_1 = 0.0;
            for i in 0..s.len() / 5 {
                if w[i] >= 5.0 && w[i] <= 1.0 {
                    area_5_1 += sw[i];
                }
            }
            area_5_1 * dw
        }
        _ => unimplemented!("Not implemented yet"),
    }
}

pub fn sb_transition_matrix_3ac_sumdiagcov(a: &[f64]) -> f64 {
    if is_constant(a) {
        return f64::NAN;
    }

    let tau = first_zero(a, a.len());
    sb_transition_matrix_3ac_sumdiagcov_with_tau(a, tau)
}

pub(crate) fn sb_transition_matrix_3ac_sumdiagcov_with_tau(a: &[f64], tau: usize) -> f64 {
    let num_groups = 3;

    let y_filt = a.to_vec();

    let n_down = (a.len() - 1) / tau + 1;
    let mut y_down = vec![0.0; n_down];
    for i in 0..n_down {
        y_down[i] = y_filt[i * tau];
    }

    let y_cg = coarsegrain(&y_down, num_groups);

    let mut t = vec![vec![0.0; 3]; 3];

    for i in 0..n_down - 1 {
        t[y_cg[i] - 1][y_cg[i + 1] - 1] += 1.0;
    }

    for i in 0..num_groups {
        for j in 0..num_groups {
            t[i][j] /= (n_down - 1) as f64;
        }
    }

    let cm = covariance_matrix(t);
    let mut diag_sum = 0.0;

    for i in 0..num_groups {
        diag_sum += cm[i][i];
    }

    return diag_sum;
}

pub fn pd_periodicity_wang_th0_01(a: &[f64]) -> f64 {
    let th = 0.01;
    let mut y_sub = splinefit(a);
    for i in 0..a.len() {
        y_sub[i] = a[i] - y_sub[i];
    }

    let ac_max = (a.len() as f64 / 3.0).ceil() as usize;
    let autocov = autocovariance(&y_sub);
    let acf = &autocov[1..=ac_max];

    let mut troughs = vec![0.0; ac_max];
    let mut peaks = vec![0.0; ac_max];
    let mut n_troughs = 0;
    let mut n_peaks = 0;

    for i in 1..ac_max - 1 {
        let slope_in = acf[i] - acf[i - 1];
        let slope_out = acf[i + 1] - acf[i];

        if slope_in < 0.0 && slope_out > 0.0 {
            troughs[n_troughs] = i as f64;
            n_troughs += 1;
        } else if slope_in > 0.0 && slope_out < 0.0 {
            peaks[n_peaks] = i as f64;
            n_peaks += 1;
        }
    }

    let mut out = 0.0;

    for i in 0..n_peaks {
        let i_peak = peaks[i];
        let the_peak = acf[i_peak as usize];

        let mut j: isize = -1;

        while (j + 1) < n_troughs as isize && troughs[(j + 1) as usize] < i_peak as f64 {
            j += 1;
        }

        if j == -1 {
            continue;
        }

        let i_trough = troughs[j as usize];
        let the_trough = acf[i_trough as usize];

        if the_peak - the_trough < th {
            continue;
        }

        if the_peak < 0.0 {
            continue;
        }

        out = i_peak as f64;
        break;
    }

    return out;
}
