# Catch22 Features in Rust

This repository provides a Rust implementation of the `catch22` feature set, a collection of 22 time-series analysis features designed for fast and interpretable classification. The implementation has been tested against the original C and implementation from the [catch22 repository](https://github.com/DynamicsAndNeuralSystems/catch22) to ensure correctness and consistency.

## Features Overview

The `catch22` feature set includes a diverse range of statistical and signal-processing methods, such as:

- Autocorrelation-based features
- Histogram-based features
- Entropy measures
- Linear regression and slope-based features
- Frequency-domain features

These features are designed to capture various properties of time-series data, making them suitable for tasks like classification, clustering, and anomaly detection.

## Installation

To use this library, add it to your `Cargo.toml`:

```toml
[dependencies]
catch22 = { git = "https://github.com/albertoazzari/catch22_rs.git", version = "0.1.0" }
```

## Usage Example
Here is an example of how to compute the `catch22` features for a time series:
```rust
use catch22::compute;

fn main() {
    let time_series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let n_features = 22;

    let features = (0..n_features)
        .map(|i| compute(&time_series, i).unwrap())
        .collect::<Vec<_>>();
    println!("Catch22 features: {:?}", features);
}
```

## Performance Improvements

This Rust implementation offers significant performance improvements over the original implementations:

- **Speed**: Some features exhibit at least a **10x improvement in speed**, making this library ideal for large-scale time-series analysis.
- **Memory Management**: Memory leaks present in the original implementations have been resolved, ensuring efficient and reliable execution.
