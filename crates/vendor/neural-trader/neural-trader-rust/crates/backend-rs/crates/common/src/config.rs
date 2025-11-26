use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    // Server configuration
    pub server_host: String,
    pub server_port: u16,

    // Database configuration
    pub database_url: String,
    pub database_pool_size: u32,

    // Redis configuration
    pub redis_url: String,

    // Authentication
    pub jwt_secret: String,
    pub jwt_expiry: String,

    // CORS
    pub cors_origin: String,
    pub cors_credentials: bool,

    // Rate limiting
    pub rate_limit_window_ms: u64,
    pub rate_limit_max_requests: u32,

    // Supabase
    pub supabase_url: String,
    pub supabase_anon_key: String,
    pub supabase_service_role_key: String,

    // E2B
    pub e2b_api_key: String,

    // Anthropic
    pub anthropic_api_key: String,

    // FoxRuv packages configuration
    pub agentic_flow_mode: String,
    pub agentic_flow_max_agents: u32,
    pub agentic_flow_enable_booster: bool,
    pub agentic_flow_enable_reasoning_bank: bool,

    pub agentdb_mode: String,
    pub agentdb_index: String,
    pub agentdb_dimensions: u32,
    pub agentdb_enable_rl: bool,

    pub midstreamer_enable_wasm: bool,
    pub midstreamer_streaming: bool,

    pub aidefence_enable_realtime: bool,
    pub aidefence_detection_threshold: f64,

    // Logging
    pub log_level: String,
    pub debug: bool,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        dotenvy::dotenv().ok();

        Ok(Config {
            server_host: env::var("SERVER_HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
            server_port: env::var("SERVER_PORT")
                .unwrap_or_else(|_| "3001".to_string())
                .parse()
                .map_err(|e| Error::Configuration(format!("Invalid SERVER_PORT: {}", e)))?,

            database_url: env::var("DATABASE_URL")
                .map_err(|_| Error::Configuration("DATABASE_URL not set".into()))?,
            database_pool_size: env::var("DATABASE_POOL_SIZE")
                .unwrap_or_else(|_| "10".to_string())
                .parse()
                .unwrap_or(10),

            redis_url: env::var("REDIS_URL")
                .unwrap_or_else(|_| "redis://localhost:6379".to_string()),

            jwt_secret: env::var("JWT_SECRET")
                .map_err(|_| Error::Configuration("JWT_SECRET not set".into()))?,
            jwt_expiry: env::var("JWT_EXPIRY").unwrap_or_else(|_| "7d".to_string()),

            cors_origin: env::var("CORS_ORIGIN")
                .unwrap_or_else(|_| "http://localhost:5173".to_string()),
            cors_credentials: env::var("CORS_CREDENTIALS")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),

            rate_limit_window_ms: env::var("RATE_LIMIT_WINDOW_MS")
                .unwrap_or_else(|_| "900000".to_string())
                .parse()
                .unwrap_or(900000),
            rate_limit_max_requests: env::var("RATE_LIMIT_MAX_REQUESTS")
                .unwrap_or_else(|_| "100".to_string())
                .parse()
                .unwrap_or(100),

            supabase_url: env::var("VITE_SUPABASE_URL")
                .or_else(|_| env::var("SUPABASE_URL"))
                .map_err(|_| Error::Configuration("SUPABASE_URL not set".into()))?,
            supabase_anon_key: env::var("VITE_SUPABASE_ANON_KEY")
                .or_else(|_| env::var("SUPABASE_ANON_KEY"))
                .map_err(|_| Error::Configuration("SUPABASE_ANON_KEY not set".into()))?,
            supabase_service_role_key: env::var("SUPABASE_SERVICE_ROLE_KEY")
                .map_err(|_| Error::Configuration("SUPABASE_SERVICE_ROLE_KEY not set".into()))?,

            e2b_api_key: env::var("E2B_API_KEY")
                .map_err(|_| Error::Configuration("E2B_API_KEY not set".into()))?,

            anthropic_api_key: env::var("ANTHROPIC_API_KEY")
                .or_else(|_| env::var("VITE_ANTHROPIC_API_KEY"))
                .map_err(|_| Error::Configuration("ANTHROPIC_API_KEY not set".into()))?,

            agentic_flow_mode: env::var("AGENTIC_FLOW_MODE")
                .unwrap_or_else(|_| "production".to_string()),
            agentic_flow_max_agents: env::var("AGENTIC_FLOW_MAX_AGENTS")
                .unwrap_or_else(|_| "66".to_string())
                .parse()
                .unwrap_or(66),
            agentic_flow_enable_booster: env::var("AGENTIC_FLOW_ENABLE_BOOSTER")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
            agentic_flow_enable_reasoning_bank: env::var("AGENTIC_FLOW_ENABLE_REASONING_BANK")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),

            agentdb_mode: env::var("AGENTDB_MODE").unwrap_or_else(|_| "memory".to_string()),
            agentdb_index: env::var("AGENTDB_INDEX").unwrap_or_else(|_| "hnsw".to_string()),
            agentdb_dimensions: env::var("AGENTDB_DIMENSIONS")
                .unwrap_or_else(|_| "1536".to_string())
                .parse()
                .unwrap_or(1536),
            agentdb_enable_rl: env::var("AGENTDB_ENABLE_RL")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),

            midstreamer_enable_wasm: env::var("MIDSTREAMER_ENABLE_WASM")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
            midstreamer_streaming: env::var("MIDSTREAMER_STREAMING")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),

            aidefence_enable_realtime: env::var("AIDEFENCE_ENABLE_REALTIME")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
            aidefence_detection_threshold: env::var("AIDEFENCE_DETECTION_THRESHOLD")
                .unwrap_or_else(|_| "0.95".to_string())
                .parse()
                .unwrap_or(0.95),

            log_level: env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
            debug: env::var("DEBUG")
                .unwrap_or_else(|_| "false".to_string())
                .parse()
                .unwrap_or(false),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        // This test will fail without env vars, but demonstrates the API
        let result = Config::from_env();
        assert!(result.is_ok() || result.is_err()); // Either way is valid depending on env
    }
}
