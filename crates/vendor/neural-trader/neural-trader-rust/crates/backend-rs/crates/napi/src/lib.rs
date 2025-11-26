#![deny(clippy::all)]

use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi]
pub fn hello() -> String {
    "Hello from Rust via NAPI-RS!".to_string()
}

#[napi(object)]
pub struct WorkflowConfig {
    pub name: String,
    pub steps: Vec<String>,
}

#[napi]
pub fn create_workflow(config: WorkflowConfig) -> Result<String> {
    Ok(format!("Workflow '{}' created with {} steps", config.name, config.steps.len()))
}

#[napi]
pub async fn execute_workflow_async(workflow_id: String) -> Result<String> {
    Ok(format!("Workflow {} executed successfully", workflow_id))
}
