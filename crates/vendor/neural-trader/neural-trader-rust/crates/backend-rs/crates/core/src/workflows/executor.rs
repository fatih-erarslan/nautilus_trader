use beclever_common::{Error, Result};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowExecutionRequest {
    pub workflow_id: Uuid,
    pub user_id: Uuid,
    pub input: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowExecutionResult {
    pub execution_id: Uuid,
    pub status: String,
    pub output: Option<serde_json::Value>,
    pub metrics: Option<serde_json::Value>,
}

#[cfg_attr(test, mockall::automock)]
pub trait WorkflowExecutor: Send + Sync {
    fn execute(&self, request: WorkflowExecutionRequest) -> Result<WorkflowExecutionResult>;
}

pub struct DefaultWorkflowExecutor;

impl WorkflowExecutor for DefaultWorkflowExecutor {
    fn execute(&self, request: WorkflowExecutionRequest) -> Result<WorkflowExecutionResult> {
        Ok(WorkflowExecutionResult {
            execution_id: Uuid::new_v4(),
            status: "completed".to_string(),
            output: Some(serde_json::json!({"success": true})),
            metrics: Some(serde_json::json!({"duration_ms": 150})),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_workflow_executor() {
        let mut mock_executor = MockWorkflowExecutor::new();
        let execution_id = Uuid::new_v4();

        mock_executor
            .expect_execute()
            .times(1)
            .returning(move |_| {
                Ok(WorkflowExecutionResult {
                    execution_id,
                    status: "completed".to_string(),
                    output: Some(serde_json::json!({"result": "success"})),
                    metrics: None,
                })
            });

        let request = WorkflowExecutionRequest {
            workflow_id: Uuid::new_v4(),
            user_id: Uuid::new_v4(),
            input: serde_json::json!({}),
        };

        let result = mock_executor.execute(request);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().status, "completed");
    }
}
