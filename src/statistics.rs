use std::vec;

use rustfft::{
    algorithm::Radix4,
    num_complex::{Complex, ComplexFloat},
    Fft,
};

pub fn min_(a: &[f64]) -> f64 {
    let mut min = a[0];
    for i in 1..a.len() {
        if a[i] < min {
            min = a[i];
        }
    }
    return min;
}

pub fn max_(a: &[f64]) -> f64 {
    let mut max = a[0];
    for i in 1..a.len() {
        if a[i] > max {
            max = a[i];
        }
    }
    return max;
}

pub fn is_constant(a: &[f64]) -> bool {
    a.iter().all(|&x| x == a[0])
}

pub fn mean(a: &[f64]) -> f64 {
    if a.len() == 0 {
        return 0.0;
    }
    a.iter().sum::<f64>() / a.len() as f64
}

pub fn median(a: &[f64]) -> f64 {
    if a.len() == 0 {
        return 0.0;
    }

    let mut a = a.to_vec();
    a.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap());
    let n = a.len();
    if n % 2 == 0 {
        (a[n / 2] + a[n / 2 - 1]) / 2.0
    } else {
        a[n / 2]
    }
}

pub fn std_dev(a: &[f64]) -> f64 {
    if a.len() < 2 {
        return 0.0;
    }

    let mut mean = 0.0;
    let mut m2 = 0.0;
    let mut count = 0usize;

    for &value in a {
        count += 1;
        let delta = value - mean;
        mean += delta / count as f64;
        let delta2 = value - mean;
        m2 += delta * delta2;
    }

    let variance = m2 / (count - 1) as f64;
    variance.sqrt()
}

pub fn slope(a: &[f64]) -> f64 {
    let n = a.len();
    if n == 0 {
        return 0.0;
    }

    let n_f = n as f64;
    let x_mean = (n_f + 1.0) / 2.0;
    let x2_mean = (n_f + 1.0) * (2.0 * n_f + 1.0) / 6.0;

    let mut y_sum = 0.0;
    let mut xy_sum = 0.0;
    for (i, &value) in a.iter().enumerate() {
        let x = (i + 1) as f64;
        y_sum += value;
        xy_sum += x * value;
    }

    let y_mean = y_sum / n_f;
    let xy_mean = xy_sum / n_f;
    let slope = (xy_mean - x_mean * y_mean) / (x2_mean - x_mean * x_mean);
    assert!(slope.is_finite());
    slope
}

pub fn histcounts(a: &[f64], n_bins: usize) -> (Vec<usize>, Vec<f64>) {
    let mut n_bins = n_bins;

    let max_val = max_(a);
    let min_val = min_(a);

    if n_bins <= 0 {
        n_bins = ((max_val - min_val) / (3.5 * std_dev(a) * (a.len() as f64).powf(-1.0 / 3.0)))
            .ceil() as usize;
    }

    let bin_step = (max_val - min_val) / n_bins as f64;

    let mut bin_counts = vec![0; n_bins];

    for i in 0..a.len() {
        let mut bin_ind = ((a[i] - min_val) / bin_step) as usize;
        bin_ind = bin_ind.min(n_bins - 1);

        bin_counts[bin_ind] += 1;
    }
    let mut bin_edges = vec![0.0; n_bins + 1];

    for i in 0..n_bins + 1 {
        bin_edges[i] = min_val + i as f64 * bin_step;
    }

    (bin_counts, bin_edges)
}

pub fn autocorr(a: &[f64]) -> Vec<f64> {
    if a.is_empty() {
        return Vec::new();
    }

    let n = a.len().next_power_of_two() << 1;
    let m = mean(a);

    let mut buffer = vec![Complex::new(0.0, 0.0); n];

    for (i, value) in a.iter().enumerate() {
        buffer[i].re = value - m;
    }

    let fft = Radix4::new(n, rustfft::FftDirection::Forward);
    let ifft = Radix4::new(n, rustfft::FftDirection::Inverse);

    fft.process(&mut buffer);
    for i in 0..n {
        buffer[i] = buffer[i] * buffer[i].conj();
    }
    ifft.process(&mut buffer);

    let norm = buffer[0];
    let mut out = Vec::with_capacity(a.len());
    for value in buffer.iter().take(a.len()) {
        out.push((value / norm).re);
    }

    out
}

pub fn autocovariance(a: &[f64]) -> Vec<f64> {
    if a.is_empty() {
        return Vec::new();
    }

    let n = a.len().next_power_of_two() << 1;
    let mut buffer = vec![Complex::new(0.0, 0.0); n];

    for (i, value) in a.iter().enumerate() {
        buffer[i].re = *value;
    }

    let fft = Radix4::new(n, rustfft::FftDirection::Forward);
    let ifft = Radix4::new(n, rustfft::FftDirection::Inverse);

    fft.process(&mut buffer);
    for i in 0..n {
        buffer[i] = buffer[i] * buffer[i].conj();
    }
    ifft.process(&mut buffer);

    let inv_n = 1.0 / n as f64;
    let mut out = Vec::with_capacity(a.len());
    for (lag, value) in buffer.iter().take(a.len()).enumerate() {
        let sum = value.re * inv_n;
        let denom = (a.len() - lag) as f64;
        out.push(sum / denom);
    }

    out
}

pub fn first_zero(a: &[f64], max_tau: usize) -> usize {
    let autocorr = autocorr(a);
    first_zero_from_autocorr(&autocorr, max_tau)
}

pub fn first_zero_from_autocorr(autocorr: &[f64], max_tau: usize) -> usize {
    let mut zero_cross_ind = 0;
    let max_tau = max_tau.min(autocorr.len());

    while zero_cross_ind < max_tau && autocorr[zero_cross_ind] > 0.0 {
        zero_cross_ind += 1;
    }

    zero_cross_ind
}

pub fn num_bins_auto(a: &[f64]) -> usize {
    let max_val = max_(a);
    let min_val = min_(a);

    if std_dev(a) < 0.001 {
        return 0;
    }

    let n_bins = ((max_val - min_val) / (3.5 * std_dev(a) * (a.len() as f64).powf(-1.0 / 3.0)))
        .ceil() as usize;
    return n_bins;
}

pub fn histbinassign(a: &[f64], bin_edges: &[f64]) -> Vec<usize> {
    let mut bin_identity = vec![0; a.len()];

    for i in 0..a.len() {
        for j in 0..bin_edges.len() {
            if a[i] < bin_edges[j] {
                bin_identity[i] = j;
                break;
            }
        }
    }

    return bin_identity;
}

pub fn histcount_edges(a: &[f64], bin_edges: &[f64]) -> Vec<usize> {
    let mut histcounts = vec![0; bin_edges.len()];

    for i in 0..a.len() {
        for j in 0..bin_edges.len() {
            if a[i] <= bin_edges[j] {
                histcounts[j] += 1;
                break;
            }
        }
    }

    return histcounts;
}
pub fn autocov_lag(a: &[f64], lag: usize) -> f64 {
    cov_(&a[..a.len() - lag], &a[lag..])
}
fn cov_(a: &[f64], b: &[f64]) -> f64 {
    let mut covariance = 0.0;
    for i in 0..a.len() {
        covariance += a[i] * b[i];
    }

    return covariance / a.len() as f64;
}

pub fn autocorr_lag(a: &[f64], lag: usize) -> f64 {
    let mean_a = mean(&a[..a.len() - lag]);
    let mean_b = mean(&a[lag..]);

    corr(&a[..a.len() - lag], &a[lag..], mean_a, mean_b)
}

pub fn corr(a: &[f64], b: &[f64], mean_a: f64, mean_b: f64) -> f64 {
    let mut nom = 0.0;
    let mut denom_a = 0.0;
    let mut denom_b = 0.0;

    for i in 0..b.len() {
        nom += (a[i] - mean_a) * (b[i] - mean_b);
        denom_a += (a[i] - mean_a) * (a[i] - mean_a);
        denom_b += (b[i] - mean_b) * (b[i] - mean_b);
    }

    return nom / (denom_a * denom_b).sqrt();
}

pub fn diff(a: &[f64]) -> Vec<f64> {
    if a.len() < 2 {
        return Vec::new();
    }

    let mut out = vec![0.0; a.len() - 1];

    for i in 1..a.len() {
        out[i - 1] = a[i] - a[i - 1];
    }

    out
}

pub fn coarsegrain(a: &[f64], num_groups: usize) -> Vec<usize> {
    let mut labels = vec![0; a.len()];
    let mut th = vec![0.0; num_groups + 1];
    let ls = linspace(0.0, 1.0, num_groups + 1);

    let mut sorted = a.to_vec();
    sorted.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap());

    for i in 0..num_groups + 1 {
        th[i] = quantile_sorted(&sorted, ls[i]);
    }

    th[0] -= 1.0;

    for i in 0..num_groups {
        for j in 0..a.len() {
            if a[j] > th[i] && a[j] <= th[i + 1] {
                labels[j] = i + 1;
            }
        }
    }
    labels
}

pub fn linspace(start: f64, end: f64, num_groups: usize) -> Vec<f64> {
    let mut out = vec![0.0; num_groups];
    let mut start = start;
    let step_size = (end - start) / (num_groups - 1) as f64;
    for i in 0..num_groups {
        out[i] = start;
        start += step_size;
    }
    return out;
}

pub fn quantile(a: &[f64], quantile: f64) -> f64 {
    let mut a = a.to_vec();
    a.sort_unstable_by(|x, y| x.partial_cmp(y).unwrap());

    quantile_sorted(&a, quantile)
}

fn quantile_sorted(a: &[f64], quantile: f64) -> f64 {
    let q = 0.5 / a.len() as f64;

    if quantile < q {
        return a[0];
    } else if quantile > (1.0 - q) {
        return a[a.len() - 1];
    }

    let quant_idx = a.len() as f64 * quantile - 0.5;
    let idx_left = quant_idx.floor() as usize;
    let idx_right = quant_idx.ceil() as usize;
    let value = a[idx_left]
        + (quant_idx - idx_left as f64) * (a[idx_right] - a[idx_left])
            / (idx_right - idx_left) as f64;
    value
}

pub fn f_entropy(a: &[f64]) -> f64 {
    let mut f = 0.0;
    for i in 0..a.len() {
        if a[i] > 0.0 {
            f += a[i] * a[i].ln();
        }
    }
    return -1.0 * f;
}

pub fn linreg(n: usize, x: &[f64], y: &[f64]) -> (f64, f64) {
    let mut sumx = 0.0;
    let mut sumx2 = 0.0;
    let mut sumxy = 0.0;
    let mut sumy = 0.0;

    for i in 0..n {
        sumx += x[i];
        sumx2 += x[i] * x[i];
        sumxy += x[i] * y[i];
        sumy += y[i];
    }

    let denom = n as f64 * sumx2 - sumx * sumx;

    if denom == 0.0 {
        return (0.0, 0.0);
    }

    return (
        (n as f64 * sumxy - sumx * sumy) / denom,
        (sumy * sumx2 - sumx * sumxy) / denom,
    );
}

pub fn norm(a: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i] * a[i];
    }
    return sum.sqrt();
}

pub fn welch(a: &[f64], fs: f64, window: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let dt = 1.0 / fs;
    let df = 1.0 / (window.len().next_power_of_two() as f64) / dt;
    let m = mean(a);
    let nfft = a.len().next_power_of_two();

    let k = ((a.len() as f64 / window.len() as f64).floor() - 1.0) as usize;

    let kmu = k as f64 * (norm(window).powi(2));

    let mut p = vec![Complex::new(0.0, 0.0); nfft];
    let mut f = vec![Complex::new(0.0, 0.0); nfft];
    let fft = Radix4::new(nfft, rustfft::FftDirection::Forward);

    let mut xw = vec![0.0; window.len()];

    for i in 0..k {
        for j in 0..window.len() {
            xw[j] = window[j] * a[i * (window.len() as f64 / 2.0) as usize + j];
        }

        for j in 0..window.len() {
            f[i].re = xw[j] - m;
        }

        fft.process(&mut f);

        for j in 0..nfft {
            p[j] += f[j].abs().powi(2);
        }
    }

    let n_out = nfft / 2 + 1;
    let mut pxx = vec![0.0; n_out];
    for i in 0..n_out {
        pxx[i] = p[i].re / kmu * dt;
        if i > 0 && i < n_out - 1 {
            pxx[i] *= 2.0;
        }
    }

    let f = (0..n_out).map(|x| x as f64 * df).collect::<Vec<f64>>();

    return (f, pxx);
}

pub fn cov(a: &[f64], b: &[f64]) -> f64 {
    let mut covariance = 0.0;

    let mean_x = mean(a);
    let mean_y = mean(b);

    for i in 0..a.len() {
        covariance += (a[i] - mean_x) * (b[i] - mean_y);
    }

    return covariance / (a.len() - 1) as f64;
}

pub fn covariance_matrix(a: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let rows = a.len();
    let cols = if rows > 0 { a[0].len() } else { 0 };
    let mut covariance_m = vec![vec![0.0; cols]; cols];

    for i in 0..cols {
        for j in 0..cols {
            let column_i: Vec<f64> = a.iter().map(|row| row[i]).collect();
            let column_j: Vec<f64> = a.iter().map(|row| row[j]).collect();
            covariance_m[i][j] = cov(&column_i, &column_j);
        }
    }

    covariance_m
}

pub fn splinefit(a: &[f64]) -> Vec<f64> {
    let deg = 3;
    let pieces = 2;
    let breaks = [0, (a.len() as f64 / 2.0).floor() as usize - 1, a.len() - 1];
    let h0 = [breaks[1] - breaks[0], breaks[2] - breaks[1]];

    let h_copy = [h0[0], h0[1], h0[0], h0[1]];

    let hl = [h_copy[3], h_copy[2], h_copy[1]];

    let hl_cs = hl
        .iter()
        .scan(0, |acc, &x| {
            *acc += x;
            Some(*acc)
        })
        .collect::<Vec<usize>>();

    let bl = hl_cs
        .iter()
        .map(|x| breaks[0] as f64 - *x as f64)
        .collect::<Vec<f64>>();

    let hr = [h_copy[0], h_copy[1], h_copy[2]];

    let hr_cs = hr
        .iter()
        .scan(0, |acc, &x| {
            *acc += x;
            Some(*acc)
        })
        .collect::<Vec<usize>>();

    let br = hr_cs
        .iter()
        .map(|x| breaks[2] as f64 + *x as f64)
        .collect::<Vec<f64>>();

    let mut breaks_ext = vec![0.0; 3 * deg];

    for i in 0..deg {
        breaks_ext[i] = bl[deg - i - 1] as f64;
        breaks_ext[i + deg] = breaks[i] as f64;
        breaks_ext[i + 2 * deg] = br[i] as f64;
    }

    let mut h_ext = vec![0.0; 3 * deg - 1];
    for i in 0..3 * deg - 1 {
        h_ext[i] = breaks_ext[i + 1] - breaks_ext[i];
    }

    let n_spline = 4;
    let pieces_ext = 3 * deg - 1;

    let mut coefs = vec![vec![0.0; n_spline + 1]; n_spline * pieces_ext];

    for i in (0..n_spline * pieces_ext).step_by(n_spline) {
        coefs[i][0] = 1.0;
    }

    let mut ii = vec![vec![0.0; pieces_ext]; deg + 1];

    for i in 0..pieces_ext {
        ii[0][i] = i.min(pieces_ext - 1) as f64;
        ii[1][i] = (i + 1).min(pieces_ext - 1) as f64;
        ii[2][i] = (i + 2).min(pieces_ext - 1) as f64;
        ii[3][i] = (i + 3).min(pieces_ext - 1) as f64;
    }

    let mut h = vec![0.0; (deg + 1) * pieces_ext];
    for i in 0..n_spline * pieces_ext {
        let ii_flat = ii[i % n_spline][i / n_spline] as usize;
        h[i] = h_ext[ii_flat];
    }

    let mut q = vec![vec![0.0; pieces_ext]; n_spline];

    for i in 1..n_spline {
        for j in 0..i {
            for k in 0..n_spline * pieces_ext {
                coefs[k][j] *= h[k] / (i - j) as f64;
            }
        }

        for j in 0..n_spline * pieces_ext {
            q[j % n_spline][j / n_spline] = 0.0;
            for k in 0..n_spline {
                q[j % n_spline][j / n_spline] += coefs[j][k];
            }
        }

        for j in 0..pieces_ext {
            for k in 1..n_spline {
                q[k][j] += q[k - 1][j];
            }
        }

        for j in 0..n_spline * pieces_ext {
            if j % n_spline == 0 {
                coefs[j][i] = 0.0;
            } else {
                coefs[j][i] = q[j % n_spline - 1][j / n_spline];
            }
        }

        let mut fmax = vec![0.0; pieces_ext * n_spline];
        for j in 0..pieces_ext {
            for k in 0..n_spline {
                fmax[j * n_spline + k] = q[n_spline - 1][j];
            }
        }

        for j in 0..i + 1 {
            for k in 0..n_spline * pieces_ext {
                coefs[k][j] /= fmax[k];
            }
        }

        // diff to adjacent antiderivatives
        for j in 0..(n_spline * pieces_ext) - deg {
            for k in 0..i + 1 {
                coefs[j][k] -= coefs[deg + j][k];
            }
        }
        for j in (0..n_spline * pieces_ext).step_by(n_spline) {
            coefs[j][i] = 0.0;
        }
    }

    let mut scale = vec![1.0; n_spline * pieces_ext];
    for i in 0..n_spline - 1 {
        for j in 0..n_spline * pieces_ext {
            scale[j] /= h[j];
        }
        for j in 0..n_spline * pieces_ext {
            coefs[j][(n_spline - 1) - (i + 1)] *= scale[j];
        }
    }

    let mut jj = vec![vec![0; pieces]; n_spline];
    for i in 0..n_spline {
        for j in 0..pieces {
            if i == 0 {
                jj[i][j] = n_spline * (1 + j);
            } else {
                jj[i][j] = deg;
            }
        }
    }

    for i in 1..n_spline {
        for j in 0..pieces {
            jj[i][j] += jj[i - 1][j];
        }
    }

    let mut coefs_out = vec![vec![0.0; n_spline]; n_spline * pieces];

    for i in 0..n_spline * pieces {
        let jj_flat = jj[i % n_spline][i / n_spline] - 1;
        for j in 0..n_spline {
            coefs_out[i][j] = coefs[jj_flat][j];
        }
    }

    let mut xs_b = vec![0; a.len() * n_spline];
    let mut index_b = vec![0; a.len() * n_spline];

    let mut break_ind = 1;

    for i in 0..a.len() {
        if i >= breaks[break_ind] && break_ind < breaks.len() - 1 {
            break_ind += 1;
        }
        for j in 0..n_spline {
            xs_b[i * n_spline + j] = i - breaks[break_ind - 1];
            index_b[i * n_spline + j] = j + (break_ind - 1) * n_spline;
        }
    }

    let mut v_b = vec![0.0; a.len() * n_spline];
    for i in 0..a.len() * n_spline {
        v_b[i] = coefs_out[index_b[i]][0];
    }

    for i in 1..n_spline {
        for j in 0..a.len() * n_spline {
            v_b[j] = v_b[j] * xs_b[j] as f64 + coefs_out[index_b[j]][i];
        }
    }

    let mut a_ = vec![0.0; a.len() * (n_spline + 1)];
    let mut break_ind = 0;
    for i in 0..a.len() * n_spline {
        if i / n_spline >= breaks[1] {
            break_ind = 1;
        }
        a_[(i % n_spline) + break_ind + (i / n_spline) * (n_spline + 1)] = v_b[i];
    }

    let x = lsqsolve_sub(a.len(), n_spline + 1, &a_, a);

    let mut c = vec![vec![0.0; n_spline * pieces]; pieces + n_spline - 1];
    for i in 0..n_spline * n_spline * pieces {
        let crow = i % n_spline + (i / n_spline) % 2;
        let ccol = i / n_spline;
        let coef_row = i % (n_spline * 2);
        let coef_col = i / (n_spline * 2);
        c[crow][ccol] = coefs_out[coef_row][coef_col];
    }

    let mut coefs_spline = vec![vec![0.0; n_spline]; pieces];

    for i in 0..n_spline * pieces {
        let coef_col = i / pieces;
        let coef_row = i % pieces;
        for j in 0..n_spline + 1 {
            coefs_spline[coef_row][coef_col] += c[j][i] * x[j];
        }
    }

    let mut y_out = vec![0.0; a.len()];
    for i in 0..a.len() {
        let second_half = if i < breaks[1] as usize { 0 } else { 1 };
        y_out[i] = coefs_spline[second_half][0];
    }

    for i in 1..n_spline {
        for j in 0..a.len() {
            let second_half = if j < breaks[1] as usize { 0 } else { 1 };
            y_out[j] =
                y_out[j] * (j - breaks[1] * second_half) as f64 + coefs_spline[second_half][i];
        }
    }

    return y_out;
}

// const int sizeA1, const int sizeA2, const double *A, const int sizeb, const double *b,
pub fn lsqsolve_sub(size_a1: usize, size_a2: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut at = vec![0.0; size_a2 * size_a1];

    for i in 0..size_a1 {
        for j in 0..size_a2 {
            at[j * size_a1 + i] = a[i * size_a2 + j];
        }
    }

    // ATA = AT * A
    let ata = matrix_multiply(size_a2, size_a1, &at, size_a1, size_a2, a);

    // ATb = AT * b
    let atb = matrix_times_vector(size_a2, size_a1, &at, size_a1, b);

    // Gauss-Jordan elimination

    let x = gauss_elimination(size_a2, ata, atb);
    return x;
}

pub fn gauss_elimination(size_a2: usize, a: Vec<f64>, b: Vec<f64>) -> Vec<f64> {
    let mut x = vec![0.0; size_a2];

    let mut a_elim = vec![vec![0.0; size_a2]; size_a2];
    let mut b_elim = vec![0.0; size_a2];

    for i in 0..size_a2 {
        for j in 0..size_a2 {
            a_elim[i][j] = a[i * size_a2 + j];
        }
        b_elim[i] = b[i];
    }

    for i in 0..size_a2 {
        for j in i + 1..size_a2 {
            let factor = a_elim[j][i] / a_elim[i][i];
            b_elim[j] = b_elim[j] - factor * b_elim[i];

            for k in i..size_a2 {
                a_elim[j][k] = a_elim[j][k] - factor * a_elim[i][k];
            }
        }
    }

    let mut b_mines_a_temp;
    for i in (0..size_a2).rev() {
        b_mines_a_temp = b_elim[i];
        for j in i + 1..size_a2 {
            b_mines_a_temp -= x[j] * a_elim[i][j];
        }
        x[i] = b_mines_a_temp / a_elim[i][i];
    }

    return x;
}

pub fn matrix_multiply(
    size_a1: usize,
    size_a2: usize,
    a: &[f64],
    size_b1: usize,
    size_b2: usize,
    b: &[f64],
) -> Vec<f64> {
    let mut c = vec![0.0; size_a1 * size_a1];

    for i in 0..size_a1 {
        for j in 0..size_b2 {
            c[i * size_b2 + j] = 0.0;
            for k in 0..size_b1 {
                c[i * size_b2 + j] += a[i * size_a2 + k] * b[k * size_b2 + j];
            }
        }
    }
    return c;
}

pub fn matrix_times_vector(
    size_a1: usize,
    size_a2: usize,
    a: &[f64],
    sizeb: usize,
    b: &[f64],
) -> Vec<f64> {
    //c[sizeb]
    let mut c = vec![0.0; size_a1];
    for i in 0..size_a1 {
        c[i] = 0.0;
        for k in 0..sizeb {
            c[i] += a[i * size_a2 + k] * b[k];
        }
    }
    return c;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn autocovariance_matches_autocov_lag() {
        let data = vec![1.0, -2.0, 3.5, 4.0, -1.5, 2.25, 0.75, -3.0, 1.2, 0.1];
        let autocov = autocovariance(&data);
        for lag in 0..data.len() {
            let expected = autocov_lag(&data, lag);
            let actual = autocov[lag];
            assert!(
                (actual - expected).abs() < 1e-10,
                "lag {lag}: expected {expected}, got {actual}"
            );
        }
    }
}
