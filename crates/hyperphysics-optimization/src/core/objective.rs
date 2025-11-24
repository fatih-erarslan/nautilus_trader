//! Objective function traits and implementations.
//!
//! # Formal Properties
//!
//! **Property O1**: `evaluate(x)` is deterministic for fixed `x`
//! **Property O2**: `evaluate` returns finite values for feasible inputs

use crate::core::Bounds;
use ndarray::ArrayView1;

/// Trait for optimization objective functions.
///
/// Implementations must be thread-safe for parallel evaluation.
pub trait ObjectiveFunction: Send + Sync {
    /// Evaluate the objective function at point x.
    ///
    /// # Arguments
    /// * `x` - Decision variable vector
    ///
    /// # Returns
    /// Objective function value (lower is better for minimization)
    fn evaluate(&self, x: ArrayView1<f64>) -> f64;

    /// Get the search space bounds.
    fn bounds(&self) -> &Bounds;

    /// Get problem dimensionality.
    fn dimension(&self) -> usize;

    /// Whether this is a minimization problem.
    fn is_minimization(&self) -> bool {
        true
    }

    /// Get optimal value if known (for benchmarking).
    fn optimal_value(&self) -> Option<f64> {
        None
    }

    /// Get optimal solution if known (for benchmarking).
    fn optimal_solution(&self) -> Option<Vec<f64>> {
        None
    }

    /// Get function name for logging.
    fn name(&self) -> &str {
        "unnamed"
    }
}

/// Wrapper to convert minimization to maximization.
pub struct MaximizationWrapper<F: ObjectiveFunction> {
    inner: F,
}

impl<F: ObjectiveFunction> MaximizationWrapper<F> {
    /// Create a maximization wrapper.
    #[must_use]
    pub fn new(inner: F) -> Self {
        Self { inner }
    }
}

impl<F: ObjectiveFunction> ObjectiveFunction for MaximizationWrapper<F> {
    fn evaluate(&self, x: ArrayView1<f64>) -> f64 {
        -self.inner.evaluate(x)
    }

    fn bounds(&self) -> &Bounds {
        self.inner.bounds()
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn is_minimization(&self) -> bool {
        false
    }

    fn optimal_value(&self) -> Option<f64> {
        self.inner.optimal_value().map(|v| -v)
    }

    fn name(&self) -> &str {
        self.inner.name()
    }
}

/// Simple sphere function for testing.
///
/// f(x) = Σ x_i²
/// Global minimum: f(0, ..., 0) = 0
pub struct SphereFunction {
    bounds: Bounds,
}

impl SphereFunction {
    /// Create sphere function with given dimension.
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self {
            bounds: Bounds::symmetric(dimension, 5.12),
        }
    }
}

impl ObjectiveFunction for SphereFunction {
    fn evaluate(&self, x: ArrayView1<f64>) -> f64 {
        x.iter().map(|v| v.powi(2)).sum()
    }

    fn bounds(&self) -> &Bounds {
        &self.bounds
    }

    fn dimension(&self) -> usize {
        self.bounds.dimension()
    }

    fn optimal_value(&self) -> Option<f64> {
        Some(0.0)
    }

    fn optimal_solution(&self) -> Option<Vec<f64>> {
        Some(vec![0.0; self.dimension()])
    }

    fn name(&self) -> &str {
        "Sphere"
    }
}

/// Rosenbrock function (banana function).
///
/// f(x) = Σ [100(x_{i+1} - x_i²)² + (1 - x_i)²]
/// Global minimum: f(1, ..., 1) = 0
pub struct RosenbrockFunction {
    bounds: Bounds,
}

impl RosenbrockFunction {
    /// Create Rosenbrock function with given dimension.
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self {
            bounds: Bounds::symmetric(dimension, 5.0),
        }
    }
}

impl ObjectiveFunction for RosenbrockFunction {
    fn evaluate(&self, x: ArrayView1<f64>) -> f64 {
        let n = x.len();
        (0..n - 1)
            .map(|i| {
                100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2)
            })
            .sum()
    }

    fn bounds(&self) -> &Bounds {
        &self.bounds
    }

    fn dimension(&self) -> usize {
        self.bounds.dimension()
    }

    fn optimal_value(&self) -> Option<f64> {
        Some(0.0)
    }

    fn optimal_solution(&self) -> Option<Vec<f64>> {
        Some(vec![1.0; self.dimension()])
    }

    fn name(&self) -> &str {
        "Rosenbrock"
    }
}

/// Rastrigin function (highly multimodal).
///
/// f(x) = 10n + Σ [x_i² - 10cos(2πx_i)]
/// Global minimum: f(0, ..., 0) = 0
pub struct RastriginFunction {
    bounds: Bounds,
}

impl RastriginFunction {
    /// Create Rastrigin function with given dimension.
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self {
            bounds: Bounds::symmetric(dimension, 5.12),
        }
    }
}

impl ObjectiveFunction for RastriginFunction {
    fn evaluate(&self, x: ArrayView1<f64>) -> f64 {
        let n = x.len() as f64;
        let a = 10.0;
        a * n + x.iter().map(|xi| xi.powi(2) - a * (2.0 * std::f64::consts::PI * xi).cos()).sum::<f64>()
    }

    fn bounds(&self) -> &Bounds {
        &self.bounds
    }

    fn dimension(&self) -> usize {
        self.bounds.dimension()
    }

    fn optimal_value(&self) -> Option<f64> {
        Some(0.0)
    }

    fn optimal_solution(&self) -> Option<Vec<f64>> {
        Some(vec![0.0; self.dimension()])
    }

    fn name(&self) -> &str {
        "Rastrigin"
    }
}

/// Ackley function (multimodal with global basin).
///
/// Global minimum: f(0, ..., 0) = 0
pub struct AckleyFunction {
    bounds: Bounds,
    a: f64,
    b: f64,
    c: f64,
}

impl AckleyFunction {
    /// Create Ackley function with given dimension.
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self {
            bounds: Bounds::symmetric(dimension, 32.768),
            a: 20.0,
            b: 0.2,
            c: 2.0 * std::f64::consts::PI,
        }
    }
}

impl ObjectiveFunction for AckleyFunction {
    fn evaluate(&self, x: ArrayView1<f64>) -> f64 {
        let n = x.len() as f64;
        let sum_sq: f64 = x.iter().map(|xi| xi.powi(2)).sum::<f64>() / n;
        let sum_cos: f64 = x.iter().map(|xi| (self.c * xi).cos()).sum::<f64>() / n;

        -self.a * (-self.b * sum_sq.sqrt()).exp() - sum_cos.exp() + self.a + std::f64::consts::E
    }

    fn bounds(&self) -> &Bounds {
        &self.bounds
    }

    fn dimension(&self) -> usize {
        self.bounds.dimension()
    }

    fn optimal_value(&self) -> Option<f64> {
        Some(0.0)
    }

    fn optimal_solution(&self) -> Option<Vec<f64>> {
        Some(vec![0.0; self.dimension()])
    }

    fn name(&self) -> &str {
        "Ackley"
    }
}

/// Schwefel function (deceptive multimodal).
///
/// Global minimum: f(420.9687, ...) ≈ 0
pub struct SchwefelFunction {
    bounds: Bounds,
}

impl SchwefelFunction {
    /// Create Schwefel function with given dimension.
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self {
            bounds: Bounds::symmetric(dimension, 500.0),
        }
    }
}

impl ObjectiveFunction for SchwefelFunction {
    fn evaluate(&self, x: ArrayView1<f64>) -> f64 {
        let n = x.len() as f64;
        418.9829 * n - x.iter().map(|xi| xi * xi.abs().sqrt().sin()).sum::<f64>()
    }

    fn bounds(&self) -> &Bounds {
        &self.bounds
    }

    fn dimension(&self) -> usize {
        self.bounds.dimension()
    }

    fn optimal_value(&self) -> Option<f64> {
        Some(0.0)
    }

    fn optimal_solution(&self) -> Option<Vec<f64>> {
        Some(vec![420.9687; self.dimension()])
    }

    fn name(&self) -> &str {
        "Schwefel"
    }
}

/// Griewank function (many local minima).
///
/// Global minimum: f(0, ..., 0) = 0
pub struct GriewankFunction {
    bounds: Bounds,
}

impl GriewankFunction {
    /// Create Griewank function with given dimension.
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self {
            bounds: Bounds::symmetric(dimension, 600.0),
        }
    }
}

impl ObjectiveFunction for GriewankFunction {
    fn evaluate(&self, x: ArrayView1<f64>) -> f64 {
        let sum_sq: f64 = x.iter().map(|xi| xi.powi(2)).sum::<f64>() / 4000.0;
        let prod_cos: f64 = x
            .iter()
            .enumerate()
            .map(|(i, xi)| (xi / ((i + 1) as f64).sqrt()).cos())
            .product();
        sum_sq - prod_cos + 1.0
    }

    fn bounds(&self) -> &Bounds {
        &self.bounds
    }

    fn dimension(&self) -> usize {
        self.bounds.dimension()
    }

    fn optimal_value(&self) -> Option<f64> {
        Some(0.0)
    }

    fn optimal_solution(&self) -> Option<Vec<f64>> {
        Some(vec![0.0; self.dimension()])
    }

    fn name(&self) -> &str {
        "Griewank"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_sphere_optimum() {
        let f = SphereFunction::new(3);
        let optimum = array![0.0, 0.0, 0.0];
        let value = f.evaluate(optimum.view());
        assert!((value - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_rosenbrock_optimum() {
        let f = RosenbrockFunction::new(3);
        let optimum = array![1.0, 1.0, 1.0];
        let value = f.evaluate(optimum.view());
        assert!((value - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_rastrigin_optimum() {
        let f = RastriginFunction::new(3);
        let optimum = array![0.0, 0.0, 0.0];
        let value = f.evaluate(optimum.view());
        assert!((value - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_ackley_optimum() {
        let f = AckleyFunction::new(3);
        let optimum = array![0.0, 0.0, 0.0];
        let value = f.evaluate(optimum.view());
        assert!(value.abs() < 1e-10);
    }

    #[test]
    fn test_griewank_optimum() {
        let f = GriewankFunction::new(3);
        let optimum = array![0.0, 0.0, 0.0];
        let value = f.evaluate(optimum.view());
        assert!((value - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_maximization_wrapper() {
        let f = SphereFunction::new(2);
        let max_f = MaximizationWrapper::new(f);

        let x = array![1.0, 1.0];
        assert_eq!(max_f.evaluate(x.view()), -2.0);
        assert!(!max_f.is_minimization());
    }
}
