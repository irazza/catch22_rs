use catch22::{compute, compute_all, Catch22Error, N_CATCH22};

#[test]
fn test_catch22() {
    let time_series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let n_features = 22;

    let features = (0..n_features)
        .map(|i| compute(&time_series, i).unwrap())
        .collect::<Vec<_>>();
    println!("Catch22 features: {:?}", features);
}

#[test]
fn test_compute_all_matches_compute() {
    let time_series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let all = compute_all(&time_series).unwrap();

    for i in 0..N_CATCH22 {
        let single = compute(&time_series, i).unwrap();
        let combined = all[i];
        assert!(
            (combined - single).abs() < 1e-12 || (combined.is_nan() && single.is_nan()),
            "feature {i} mismatch: compute_all={combined}, compute={single}"
        );
    }
}

#[test]
fn test_invalid_index() {
    let time_series = vec![1.0, 2.0, 3.0, 4.0];
    let err = compute(&time_series, N_CATCH22).unwrap_err();
    assert!(matches!(
        err,
        Catch22Error::InvalidFeatureIndex { .. }
    ));
}

#[test]
fn test_non_finite_input() {
    let time_series = vec![1.0, 2.0, f64::NAN, 4.0];
    let err = compute_all(&time_series).unwrap_err();
    assert!(matches!(err, Catch22Error::NonFiniteValue { .. }));
}

#[test]
fn test_input_too_short() {
    let time_series = vec![1.0, 2.0, 3.0];
    let err = compute_all(&time_series).unwrap_err();
    assert!(matches!(err, Catch22Error::InputTooShort { .. }));
}
