//! Forecasting accuracy metrics

/// Mean Absolute Percentage Error
pub fn mape(predictions: &[f64], actuals: &[f64]) -> f64 {
    if predictions.len() != actuals.len() {
        panic!("Length mismatch");
    }

    predictions
        .iter()
        .zip(actuals.iter())
        .map(|(&pred, &actual)| {
            if actual.abs() < 1e-10 {
                0.0 // Avoid division by zero
            } else {
                ((actual - pred) / actual).abs()
            }
        })
        .sum::<f64>()
        / predictions.len() as f64
}

/// Root Mean Squared Error
pub fn rmse(predictions: &[f64], actuals: &[f64]) -> f64 {
    if predictions.len() != actuals.len() {
        panic!("Length mismatch");
    }

    let mse = predictions
        .iter()
        .zip(actuals.iter())
        .map(|(&pred, &actual)| (actual - pred).powi(2))
        .sum::<f64>()
        / predictions.len() as f64;

    mse.sqrt()
}

/// Mean Absolute Error
pub fn mae(predictions: &[f64], actuals: &[f64]) -> f64 {
    if predictions.len() != actuals.len() {
        panic!("Length mismatch");
    }

    predictions
        .iter()
        .zip(actuals.iter())
        .map(|(&pred, &actual)| (actual - pred).abs())
        .sum::<f64>()
        / predictions.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_mape() {
        let predictions = vec![1.0, 2.0, 3.0];
        let actuals = vec![1.1, 1.9, 3.2];

        let error = mape(&predictions, &actuals);
        assert!(error > 0.0 && error < 1.0);
    }

    #[test]
    fn test_rmse() {
        let predictions = vec![1.0, 2.0, 3.0];
        let actuals = vec![1.1, 1.9, 3.2];

        let error = rmse(&predictions, &actuals);
        assert!(error > 0.0);
    }

    #[test]
    fn test_perfect_prediction() {
        let predictions = vec![1.0, 2.0, 3.0];
        let actuals = vec![1.0, 2.0, 3.0];

        assert_relative_eq!(mape(&predictions, &actuals), 0.0, epsilon = 1e-10);
        assert_relative_eq!(rmse(&predictions, &actuals), 0.0, epsilon = 1e-10);
        assert_relative_eq!(mae(&predictions, &actuals), 0.0, epsilon = 1e-10);
    }
}
