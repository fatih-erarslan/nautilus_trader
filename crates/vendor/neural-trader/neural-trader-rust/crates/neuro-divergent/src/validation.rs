//! Model validation utilities

use ndarray::Array1;

/// Calculate Mean Absolute Error
pub fn mae(predictions: &Array1<f64>, actuals: &Array1<f64>) -> f64 {
    predictions
        .iter()
        .zip(actuals.iter())
        .map(|(p, a)| (p - a).abs())
        .sum::<f64>()
        / predictions.len() as f64
}

/// Calculate Mean Squared Error
pub fn mse(predictions: &Array1<f64>, actuals: &Array1<f64>) -> f64 {
    predictions
        .iter()
        .zip(actuals.iter())
        .map(|(p, a)| (p - a).powi(2))
        .sum::<f64>()
        / predictions.len() as f64
}

/// Calculate Root Mean Squared Error
pub fn rmse(predictions: &Array1<f64>, actuals: &Array1<f64>) -> f64 {
    mse(predictions, actuals).sqrt()
}

/// Calculate Mean Absolute Percentage Error
pub fn mape(predictions: &Array1<f64>, actuals: &Array1<f64>) -> f64 {
    predictions
        .iter()
        .zip(actuals.iter())
        .map(|(p, a)| ((p - a) / a).abs())
        .sum::<f64>()
        / predictions.len() as f64
        * 100.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_mae() {
        let predictions = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let actuals = Array1::from_vec(vec![1.1, 2.2, 2.9]);

        let error = mae(&predictions, &actuals);
        assert_relative_eq!(error, 0.1333, max_relative = 0.01);
    }

    #[test]
    fn test_mse() {
        let predictions = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let actuals = Array1::from_vec(vec![1.1, 2.2, 2.9]);

        let error = mse(&predictions, &actuals);
        assert!(error > 0.0);
    }
}
