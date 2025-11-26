//! Training metrics computation

use ndarray::Array1;

/// Calculate Mean Absolute Error
pub fn mae(predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
    (predictions - targets).mapv(|x| x.abs()).mean().unwrap_or(0.0)
}

/// Calculate Mean Squared Error
pub fn mse(predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
    (predictions - targets).mapv(|x| x.powi(2)).mean().unwrap_or(0.0)
}

/// Calculate Root Mean Squared Error
pub fn rmse(predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
    mse(predictions, targets).sqrt()
}

/// Calculate Mean Absolute Percentage Error
pub fn mape(predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
    let mut total = 0.0;
    let mut count = 0;

    for (pred, target) in predictions.iter().zip(targets.iter()) {
        if target.abs() > 1e-8 {
            total += ((target - pred) / target).abs();
            count += 1;
        }
    }

    if count > 0 {
        100.0 * total / count as f64
    } else {
        0.0
    }
}

/// Calculate R-squared (coefficient of determination)
pub fn r2_score(predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
    let mean_target = targets.mean().unwrap_or(0.0);
    let ss_total: f64 = targets.iter().map(|&t| (t - mean_target).powi(2)).sum();
    let ss_residual: f64 = predictions.iter().zip(targets.iter())
        .map(|(&p, &t)| (t - p).powi(2))
        .sum();

    if ss_total > 0.0 {
        1.0 - (ss_residual / ss_total)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_mae() {
        let pred = arr1(&[1.0, 2.0, 3.0]);
        let target = arr1(&[1.5, 2.5, 3.5]);
        approx::assert_abs_diff_eq!(mae(&pred, &target), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_mse() {
        let pred = arr1(&[1.0, 2.0, 3.0]);
        let target = arr1(&[1.0, 2.0, 3.0]);
        approx::assert_abs_diff_eq!(mse(&pred, &target), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_r2_score() {
        let pred = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let target = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        approx::assert_abs_diff_eq!(r2_score(&pred, &target), 1.0, epsilon = 1e-10);
    }
}
