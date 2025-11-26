//! Runtime management for the Neural Trader system.

use crate::{Config, Error, Result};
use std::sync::Arc;
use tokio::runtime::Runtime as TokioRuntime;
use tracing::{info, warn};

/// Runtime manager for the Neural Trader system.
///
/// Manages the async runtime and ensures proper initialization and shutdown.
pub struct Runtime {
    config: Arc<Config>,
    handle: tokio::runtime::Handle,
}

impl Runtime {
    /// Creates a new runtime with the given configuration.
    pub fn new(config: &Config) -> Result<Self> {
        let mut builder = tokio::runtime::Builder::new_multi_thread();

        // Configure worker threads
        if let Some(workers) = config.runtime.worker_threads {
            builder.worker_threads(workers);
        }

        // Configure blocking threads
        if let Some(max_blocking) = config.runtime.max_blocking_threads {
            builder.max_blocking_threads(max_blocking);
        }

        // Enable all features
        builder
            .thread_name("neural-trader-worker")
            .enable_all();

        // Get the current runtime handle (we're already in a Tokio runtime)
        let handle = tokio::runtime::Handle::current();

        info!("Runtime initialized successfully");

        Ok(Self {
            config: Arc::new(config.clone()),
            handle,
        })
    }

    /// Returns a handle to the runtime.
    pub fn handle(&self) -> &tokio::runtime::Handle {
        &self.handle
    }

    /// Spawns a task on the runtime.
    pub fn spawn<F>(&self, future: F) -> tokio::task::JoinHandle<F::Output>
    where
        F: std::future::Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.handle.spawn(future)
    }

    /// Spawns a blocking task on the runtime.
    pub fn spawn_blocking<F, R>(&self, f: F) -> tokio::task::JoinHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.handle.spawn_blocking(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_runtime_creation() {
        let _config = Config::default();
        let runtime = Runtime::new(&config).expect("Failed to create runtime");

        let handle = runtime.spawn(async {
            42
        });

        let result = handle.await.expect("Task failed");
        assert_eq!(result, 42);
    }
}
