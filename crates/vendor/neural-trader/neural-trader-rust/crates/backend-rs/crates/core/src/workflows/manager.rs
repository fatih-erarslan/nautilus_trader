use beclever_common::{Error, Result};
use beclever_db::{Workflow, NewWorkflow};
use uuid::Uuid;

#[cfg_attr(test, mockall::automock)]
pub trait WorkflowManager: Send + Sync {
    fn create_workflow(&self, workflow: NewWorkflow) -> Result<Workflow>;
    fn get_workflow(&self, id: Uuid) -> Result<Option<Workflow>>;
    fn list_workflows(&self, user_id: Uuid) -> Result<Vec<Workflow>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_workflow_manager() {
        let mut mock_manager = MockWorkflowManager::new();
        let workflow_id = Uuid::new_v4();

        mock_manager
            .expect_get_workflow()
            .with(mockall::predicate::eq(workflow_id))
            .times(1)
            .returning(move |_| Ok(None));

        let result = mock_manager.get_workflow(workflow_id);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }
}
