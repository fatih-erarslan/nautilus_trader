/// Integration examples for circuit breakers with various operations
///
/// This module demonstrates how to apply circuit breakers to:
/// - External API calls
/// - E2B sandbox operations
/// - Neural network operations
/// - Database operations

use super::{CircuitBreaker, CircuitBreakerConfig};
use anyhow::Result;
use std::time::Duration;

/// Circuit breaker for external API calls
pub struct ApiCircuitBreaker {
    breaker: CircuitBreaker,
}

impl ApiCircuitBreaker {
    pub fn new(service_name: &str) -> Self {
        let config = CircuitBreakerConfig {
            failure_threshold: 5,
            success_threshold: 2,
            timeout: Duration::from_secs(10),
            reset_timeout: Duration::from_secs(30),
        };

        Self {
            breaker: CircuitBreaker::new(format!("api:{}", service_name), config),
        }
    }

    /// Execute HTTP GET request with circuit breaker protection
    pub async fn get<T>(&self, url: &str) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
    {
        self.breaker
            .call(async {
                // Actual HTTP request implementation
                // This is a placeholder - integrate with your HTTP client
                let response = reqwest::get(url).await?;
                let data = response.json::<T>().await?;
                Ok(data)
            })
            .await
    }

    /// Execute HTTP POST request with circuit breaker protection
    pub async fn post<T, B>(&self, url: &str, body: &B) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
        B: serde::Serialize,
    {
        self.breaker
            .call(async {
                let client = reqwest::Client::new();
                let response = client.post(url).json(body).send().await?;
                let data = response.json::<T>().await?;
                Ok(data)
            })
            .await
    }

    pub async fn get_state(&self) -> String {
        self.breaker.get_state().await
    }
}

/// Circuit breaker for E2B sandbox operations
pub struct E2BSandboxCircuitBreaker {
    breaker: CircuitBreaker,
}

impl E2BSandboxCircuitBreaker {
    pub fn new() -> Self {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            success_threshold: 2,
            timeout: Duration::from_secs(60),
            reset_timeout: Duration::from_secs(120),
        };

        Self {
            breaker: CircuitBreaker::new("e2b:sandbox".to_string(), config),
        }
    }

    /// Create sandbox with circuit breaker protection
    pub async fn create_sandbox(&self, template: &str) -> Result<String> {
        self.breaker
            .call(async {
                // Placeholder for E2B sandbox creation
                // Integrate with your E2B client
                log::info!("Creating E2B sandbox with template: {}", template);

                // Simulate sandbox creation
                tokio::time::sleep(Duration::from_millis(100)).await;

                Ok("sandbox-id-123".to_string())
            })
            .await
    }

    /// Execute code in sandbox with circuit breaker protection
    pub async fn execute_code(&self, sandbox_id: &str, code: &str) -> Result<String> {
        self.breaker
            .call(async {
                log::info!("Executing code in sandbox: {}", sandbox_id);

                // Simulate code execution
                tokio::time::sleep(Duration::from_millis(50)).await;

                Ok("execution-result".to_string())
            })
            .await
    }

    /// Stop sandbox with circuit breaker protection
    pub async fn stop_sandbox(&self, sandbox_id: &str) -> Result<()> {
        self.breaker
            .call(async {
                log::info!("Stopping sandbox: {}", sandbox_id);

                // Simulate sandbox stop
                tokio::time::sleep(Duration::from_millis(50)).await;

                Ok(())
            })
            .await
    }

    pub async fn get_state(&self) -> String {
        self.breaker.get_state().await
    }
}

impl Default for E2BSandboxCircuitBreaker {
    fn default() -> Self {
        Self::new()
    }
}

/// Circuit breaker for neural network operations
pub struct NeuralCircuitBreaker {
    breaker: CircuitBreaker,
}

impl NeuralCircuitBreaker {
    pub fn new(model_name: &str) -> Self {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            success_threshold: 2,
            timeout: Duration::from_secs(30),
            reset_timeout: Duration::from_secs(60),
        };

        Self {
            breaker: CircuitBreaker::new(format!("neural:{}", model_name), config),
        }
    }

    /// Perform inference with circuit breaker protection
    pub async fn predict(&self, input: Vec<f64>) -> Result<Vec<f64>> {
        self.breaker
            .call(async {
                log::info!("Running neural network inference");

                // Simulate inference
                tokio::time::sleep(Duration::from_millis(100)).await;

                // Return dummy prediction
                Ok(vec![0.8, 0.2])
            })
            .await
    }

    /// Train model with circuit breaker protection
    pub async fn train(&self, data: Vec<Vec<f64>>, labels: Vec<f64>) -> Result<()> {
        self.breaker
            .call(async {
                log::info!("Training neural network with {} samples", data.len());

                // Simulate training
                tokio::time::sleep(Duration::from_millis(200)).await;

                Ok(())
            })
            .await
    }

    pub async fn get_state(&self) -> String {
        self.breaker.get_state().await
    }
}

/// Circuit breaker for database operations
pub struct DatabaseCircuitBreaker {
    breaker: CircuitBreaker,
}

impl DatabaseCircuitBreaker {
    pub fn new(db_name: &str) -> Self {
        let config = CircuitBreakerConfig {
            failure_threshold: 5,
            success_threshold: 3,
            timeout: Duration::from_secs(5),
            reset_timeout: Duration::from_secs(30),
        };

        Self {
            breaker: CircuitBreaker::new(format!("db:{}", db_name), config),
        }
    }

    /// Execute query with circuit breaker protection
    pub async fn query<T>(&self, sql: &str) -> Result<Vec<T>>
    where
        T: Default,
    {
        self.breaker
            .call(async {
                log::info!("Executing database query: {}", sql);

                // Simulate query execution
                tokio::time::sleep(Duration::from_millis(50)).await;

                Ok(vec![T::default()])
            })
            .await
    }

    /// Execute transaction with circuit breaker protection
    pub async fn transaction<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce() -> Result<T> + Send,
        T: Send,
    {
        self.breaker
            .call(async {
                log::info!("Executing database transaction");

                // Simulate transaction
                tokio::time::sleep(Duration::from_millis(100)).await;

                f()
            })
            .await
    }

    pub async fn get_state(&self) -> String {
        self.breaker.get_state().await
    }
}

/// Example: Composing multiple circuit breakers
pub struct TradingSystemCircuitBreakers {
    pub market_data_api: ApiCircuitBreaker,
    pub order_execution_api: ApiCircuitBreaker,
    pub e2b_sandbox: E2BSandboxCircuitBreaker,
    pub neural_predictor: NeuralCircuitBreaker,
    pub database: DatabaseCircuitBreaker,
}

impl TradingSystemCircuitBreakers {
    pub fn new() -> Self {
        Self {
            market_data_api: ApiCircuitBreaker::new("market_data"),
            order_execution_api: ApiCircuitBreaker::new("order_execution"),
            e2b_sandbox: E2BSandboxCircuitBreaker::new(),
            neural_predictor: NeuralCircuitBreaker::new("price_predictor"),
            database: DatabaseCircuitBreaker::new("trading_db"),
        }
    }

    /// Get health status of all circuit breakers
    pub async fn health_status(&self) -> Vec<(String, String)> {
        vec![
            ("market_data_api".to_string(), self.market_data_api.get_state().await),
            ("order_execution_api".to_string(), self.order_execution_api.get_state().await),
            ("e2b_sandbox".to_string(), self.e2b_sandbox.get_state().await),
            ("neural_predictor".to_string(), self.neural_predictor.get_state().await),
            ("database".to_string(), self.database.get_state().await),
        ]
    }
}

impl Default for TradingSystemCircuitBreakers {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_e2b_sandbox_circuit_breaker() {
        let cb = E2BSandboxCircuitBreaker::new();

        // Test successful sandbox creation
        let result = cb.create_sandbox("base").await;
        assert!(result.is_ok());

        // Test code execution
        let result = cb.execute_code("sandbox-123", "print('hello')").await;
        assert!(result.is_ok());

        // Test sandbox stop
        let result = cb.stop_sandbox("sandbox-123").await;
        assert!(result.is_ok());

        let state = cb.get_state().await;
        assert!(state.starts_with("CLOSED"));
    }

    #[tokio::test]
    async fn test_neural_circuit_breaker() {
        let cb = NeuralCircuitBreaker::new("test_model");

        // Test prediction
        let result = cb.predict(vec![1.0, 2.0, 3.0]).await;
        assert!(result.is_ok());

        // Test training
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let labels = vec![0.0, 1.0];
        let result = cb.train(data, labels).await;
        assert!(result.is_ok());

        let state = cb.get_state().await;
        assert!(state.starts_with("CLOSED"));
    }

    #[tokio::test]
    async fn test_database_circuit_breaker() {
        let cb = DatabaseCircuitBreaker::new("test_db");

        // Test query
        let result: Result<Vec<String>> = cb.query("SELECT * FROM users").await;
        assert!(result.is_ok());

        let state = cb.get_state().await;
        assert!(state.starts_with("CLOSED"));
    }

    #[tokio::test]
    async fn test_trading_system_health() {
        let system = TradingSystemCircuitBreakers::new();
        let health = system.health_status().await;

        assert_eq!(health.len(), 5);
        for (name, state) in health {
            assert!(!name.is_empty());
            assert!(state.starts_with("CLOSED"));
        }
    }
}
