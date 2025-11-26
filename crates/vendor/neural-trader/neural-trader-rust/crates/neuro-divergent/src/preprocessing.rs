//! Data preprocessing utilities

use anyhow::Result;
use ndarray::{Array1, Array2};

/// Normalize data using min-max scaling
pub fn normalize(data: &Array1<f64>) -> Result<(Array1<f64>, f64, f64)> {
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let normalized = data.mapv(|x| (x - min) / (max - min));

    Ok((normalized, min, max))
}

/// Denormalize data
pub fn denormalize(data: &Array1<f64>, min: f64, max: f64) -> Array1<f64> {
    data.mapv(|x| x * (max - min) + min)
}

/// Create sequences for time series
pub fn create_sequences(
    data: &Array1<f64>,
    input_size: usize,
    horizon: usize,
) -> Result<(Array2<f64>, Array2<f64>)> {
    let n = data.len();
    let n_sequences = n - input_size - horizon + 1;

    let mut x = Array2::zeros((n_sequences, input_size));
    let mut y = Array2::zeros((n_sequences, horizon));

    for i in 0..n_sequences {
        for j in 0..input_size {
            x[[i, j]] = data[i + j];
        }
        for j in 0..horizon {
            y[[i, j]] = data[i + input_size + j];
        }
    }

    Ok((x, y))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_normalize() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let (normalized, min, max) = normalize(&data).unwrap();

        assert_relative_eq!(min, 1.0);
        assert_relative_eq!(max, 5.0);
        assert_relative_eq!(normalized[0], 0.0);
        assert_relative_eq!(normalized[4], 1.0);
    }

    #[test]
    fn test_denormalize() {
        let normalized = Array1::from_vec(vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        let denormalized = denormalize(&normalized, 1.0, 5.0);

        assert_relative_eq!(denormalized[0], 1.0);
        assert_relative_eq!(denormalized[4], 5.0);
    }
}
