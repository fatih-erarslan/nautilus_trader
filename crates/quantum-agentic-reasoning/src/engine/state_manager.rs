//! State Manager Module
//!
//! Comprehensive state management for quantum trading operations with persistence and synchronization.

use crate::core::{QarResult, FactorMap};
use crate::error::QarError;
use crate::quantum::QuantumState;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// State types for different components
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StateType {
    Quantum,
    Classical,
    Hybrid,
    Decision,
    Execution,
    Risk,
    Portfolio,
    Market,
}

/// State persistence levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PersistenceLevel {
    Memory,      // In-memory only
    Session,     // Session-scoped
    Persistent,  // Long-term storage
    Replicated,  // Distributed across nodes
}

/// State change event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateChangeEvent {
    pub id: String,
    pub state_id: String,
    pub component: String,
    pub change_type: StateChangeType,
    pub old_value: Option<String>,
    pub new_value: String,
    pub timestamp: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

/// Types of state changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StateChangeType {
    Created,
    Updated,
    Deleted,
    Synchronized,
    Persisted,
    Restored,
}

/// State entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateEntry {
    pub id: String,
    pub component: String,
    pub state_type: StateType,
    pub data: String, // Serialized state data
    pub version: u64,
    pub persistence_level: PersistenceLevel,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub tags: Vec<String>,
    pub metadata: HashMap<String, String>,
}

/// State snapshot for backup/restore
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSnapshot {
    pub id: String,
    pub name: String,
    pub description: String,
    pub states: Vec<StateEntry>,
    pub created_at: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

/// State manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateManagerConfig {
    pub max_memory_states: usize,
    pub default_ttl_seconds: u64,
    pub enable_persistence: bool,
    pub enable_compression: bool,
    pub enable_encryption: bool,
    pub backup_interval_seconds: u64,
    pub max_snapshots: usize,
    pub sync_interval_seconds: u64,
}

/// State manager implementation
#[derive(Debug)]
pub struct StateManager {
    config: StateManagerConfig,
    states: Arc<RwLock<HashMap<String, StateEntry>>>,
    state_history: Arc<RwLock<Vec<StateChangeEvent>>>,
    snapshots: Arc<RwLock<HashMap<String, StateSnapshot>>>,
    persistence_handler: Arc<dyn StatePersistenceHandler + Send + Sync>,
    sync_handler: Arc<dyn StateSyncHandler + Send + Sync>,
    event_listeners: Arc<Mutex<Vec<Arc<dyn StateEventListener + Send + Sync>>>>,
}

/// State persistence handler trait
#[async_trait::async_trait]
pub trait StatePersistenceHandler {
    async fn save_state(&self, entry: &StateEntry) -> QarResult<()>;
    async fn load_state(&self, id: &str) -> QarResult<Option<StateEntry>>;
    async fn delete_state(&self, id: &str) -> QarResult<()>;
    async fn list_states(&self, component: Option<&str>) -> QarResult<Vec<String>>;
    async fn create_snapshot(&self, snapshot: &StateSnapshot) -> QarResult<()>;
    async fn load_snapshot(&self, id: &str) -> QarResult<Option<StateSnapshot>>;
}

/// State synchronization handler trait
#[async_trait::async_trait]
pub trait StateSyncHandler {
    async fn sync_state(&self, entry: &StateEntry) -> QarResult<()>;
    async fn receive_state_update(&self, entry: StateEntry) -> QarResult<()>;
    async fn request_state_sync(&self, component: &str) -> QarResult<Vec<StateEntry>>;
}

/// State event listener trait
#[async_trait::async_trait]
pub trait StateEventListener {
    async fn on_state_change(&self, event: &StateChangeEvent) -> QarResult<()>;
}

impl StateManager {
    /// Create new state manager
    pub fn new(
        config: StateManagerConfig,
        persistence_handler: Arc<dyn StatePersistenceHandler + Send + Sync>,
        sync_handler: Arc<dyn StateSyncHandler + Send + Sync>,
    ) -> Self {
        Self {
            config,
            states: Arc::new(RwLock::new(HashMap::new())),
            state_history: Arc::new(RwLock::new(Vec::new())),
            snapshots: Arc::new(RwLock::new(HashMap::new())),
            persistence_handler,
            sync_handler,
            event_listeners: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Set state for a component
    pub async fn set_state<T: Serialize>(
        &self,
        component: &str,
        state_type: StateType,
        data: &T,
        persistence_level: PersistenceLevel,
    ) -> QarResult<String> {
        let state_id = format!("{}_{}", component, Uuid::new_v4());
        let serialized_data = serde_json::to_string(data)
            .map_err(|e| QarError::StateError(format!("Serialization failed: {}", e)))?;

        let entry = StateEntry {
            id: state_id.clone(),
            component: component.to_string(),
            state_type,
            data: serialized_data,
            version: 1,
            persistence_level: persistence_level.clone(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            expires_at: self.calculate_expiry(),
            tags: Vec::new(),
            metadata: HashMap::new(),
        };

        // Store in memory
        {
            let mut states = self.states.write().await;
            states.insert(state_id.clone(), entry.clone());
        }

        // Persist if required
        if matches!(persistence_level, PersistenceLevel::Persistent | PersistenceLevel::Replicated) {
            self.persistence_handler.save_state(&entry).await?;
        }

        // Sync if replicated
        if matches!(persistence_level, PersistenceLevel::Replicated) {
            self.sync_handler.sync_state(&entry).await?;
        }

        // Record state change
        let event = StateChangeEvent {
            id: Uuid::new_v4().to_string(),
            state_id: state_id.clone(),
            component: component.to_string(),
            change_type: StateChangeType::Created,
            old_value: None,
            new_value: entry.data.clone(),
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        };

        self.record_state_change(event).await?;

        Ok(state_id)
    }

    /// Get state for a component
    pub async fn get_state<T: for<'de> Deserialize<'de>>(
        &self,
        state_id: &str,
    ) -> QarResult<Option<T>> {
        // Try memory first
        {
            let states = self.states.read().await;
            if let Some(entry) = states.get(state_id) {
                if !self.is_expired(entry) {
                    return self.deserialize_state_data(&entry.data);
                }
            }
        }

        // Try persistence
        if self.config.enable_persistence {
            if let Some(entry) = self.persistence_handler.load_state(state_id).await? {
                if !self.is_expired(&entry) {
                    // Cache in memory
                    {
                        let mut states = self.states.write().await;
                        states.insert(state_id.to_string(), entry.clone());
                    }
                    return self.deserialize_state_data(&entry.data);
                }
            }
        }

        Ok(None)
    }

    /// Update existing state
    pub async fn update_state<T: Serialize>(
        &self,
        state_id: &str,
        data: &T,
    ) -> QarResult<()> {
        let serialized_data = serde_json::to_string(data)
            .map_err(|e| QarError::StateError(format!("Serialization failed: {}", e)))?;

        let old_value = {
            let mut states = self.states.write().await;
            if let Some(entry) = states.get_mut(state_id) {
                let old_data = entry.data.clone();
                entry.data = serialized_data.clone();
                entry.version += 1;
                entry.updated_at = Utc::now();

                // Persist if required
                if matches!(entry.persistence_level, PersistenceLevel::Persistent | PersistenceLevel::Replicated) {
                    self.persistence_handler.save_state(entry).await?;
                }

                // Sync if replicated
                if matches!(entry.persistence_level, PersistenceLevel::Replicated) {
                    self.sync_handler.sync_state(entry).await?;
                }

                Some(old_data)
            } else {
                return Err(QarError::StateError(format!("State not found: {}", state_id)));
            }
        };

        // Record state change
        let event = StateChangeEvent {
            id: Uuid::new_v4().to_string(),
            state_id: state_id.to_string(),
            component: String::new(), // Will be filled from the entry
            change_type: StateChangeType::Updated,
            old_value,
            new_value: serialized_data,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        };

        self.record_state_change(event).await?;

        Ok(())
    }

    /// Delete state
    pub async fn delete_state(&self, state_id: &str) -> QarResult<()> {
        let old_value = {
            let mut states = self.states.write().await;
            if let Some(entry) = states.remove(state_id) {
                // Delete from persistence if required
                if matches!(entry.persistence_level, PersistenceLevel::Persistent | PersistenceLevel::Replicated) {
                    self.persistence_handler.delete_state(state_id).await?;
                }

                Some(entry.data)
            } else {
                None
            }
        };

        if old_value.is_some() {
            // Record state change
            let event = StateChangeEvent {
                id: Uuid::new_v4().to_string(),
                state_id: state_id.to_string(),
                component: String::new(),
                change_type: StateChangeType::Deleted,
                old_value,
                new_value: String::new(),
                timestamp: Utc::now(),
                metadata: HashMap::new(),
            };

            self.record_state_change(event).await?;
        }

        Ok(())
    }

    /// List states by component
    pub async fn list_states(&self, component: Option<&str>) -> QarResult<Vec<StateEntry>> {
        let states = self.states.read().await;
        let filtered_states: Vec<StateEntry> = states
            .values()
            .filter(|entry| {
                if let Some(comp) = component {
                    entry.component == comp
                } else {
                    true
                }
            })
            .filter(|entry| !self.is_expired(entry))
            .cloned()
            .collect();

        Ok(filtered_states)
    }

    /// Create state snapshot
    pub async fn create_snapshot(&self, name: &str, description: &str) -> QarResult<String> {
        let snapshot_id = Uuid::new_v4().to_string();
        let states = self.states.read().await;
        
        let snapshot = StateSnapshot {
            id: snapshot_id.clone(),
            name: name.to_string(),
            description: description.to_string(),
            states: states.values().cloned().collect(),
            created_at: Utc::now(),
            metadata: HashMap::new(),
        };

        // Store snapshot
        {
            let mut snapshots = self.snapshots.write().await;
            snapshots.insert(snapshot_id.clone(), snapshot.clone());
        }

        // Persist snapshot
        if self.config.enable_persistence {
            self.persistence_handler.create_snapshot(&snapshot).await?;
        }

        Ok(snapshot_id)
    }

    /// Restore from snapshot
    pub async fn restore_snapshot(&self, snapshot_id: &str) -> QarResult<()> {
        let snapshot = {
            let snapshots = self.snapshots.read().await;
            if let Some(snapshot) = snapshots.get(snapshot_id) {
                snapshot.clone()
            } else if self.config.enable_persistence {
                self.persistence_handler.load_snapshot(snapshot_id).await?
                    .ok_or_else(|| QarError::StateError(format!("Snapshot not found: {}", snapshot_id)))?
            } else {
                return Err(QarError::StateError(format!("Snapshot not found: {}", snapshot_id)));
            }
        };

        // Clear current states
        {
            let mut states = self.states.write().await;
            states.clear();
        }

        // Restore states from snapshot
        {
            let mut states = self.states.write().await;
            for entry in snapshot.states {
                states.insert(entry.id.clone(), entry);
            }
        }

        // Record restore event
        let event = StateChangeEvent {
            id: Uuid::new_v4().to_string(),
            state_id: snapshot_id.to_string(),
            component: "state_manager".to_string(),
            change_type: StateChangeType::Restored,
            old_value: None,
            new_value: format!("Restored from snapshot: {}", snapshot.name),
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        };

        self.record_state_change(event).await?;

        Ok(())
    }

    /// Add event listener
    pub async fn add_event_listener(&self, listener: Arc<dyn StateEventListener + Send + Sync>) {
        let mut listeners = self.event_listeners.lock().await;
        listeners.push(listener);
    }

    /// Clean up expired states
    pub async fn cleanup_expired_states(&self) -> QarResult<usize> {
        let expired_ids: Vec<String> = {
            let states = self.states.read().await;
            states
                .iter()
                .filter(|(_, entry)| self.is_expired(entry))
                .map(|(id, _)| id.clone())
                .collect()
        };

        for id in &expired_ids {
            self.delete_state(id).await?;
        }

        Ok(expired_ids.len())
    }

    /// Get state history
    pub async fn get_state_history(&self, state_id: Option<&str>, limit: Option<usize>) -> QarResult<Vec<StateChangeEvent>> {
        let history = self.state_history.read().await;
        let filtered_events: Vec<StateChangeEvent> = history
            .iter()
            .filter(|event| {
                if let Some(id) = state_id {
                    event.state_id == id
                } else {
                    true
                }
            })
            .rev()
            .take(limit.unwrap_or(100))
            .cloned()
            .collect();

        Ok(filtered_events)
    }

    /// Record state change event
    async fn record_state_change(&self, event: StateChangeEvent) -> QarResult<()> {
        // Add to history
        {
            let mut history = self.state_history.write().await;
            history.push(event.clone());
            
            // Limit history size
            if history.len() > 10000 {
                history.drain(0..1000);
            }
        }

        // Notify listeners
        let listeners = self.event_listeners.lock().await;
        for listener in listeners.iter() {
            let _ = listener.on_state_change(&event).await;
        }

        Ok(())
    }

    /// Check if state entry is expired
    fn is_expired(&self, entry: &StateEntry) -> bool {
        if let Some(expires_at) = entry.expires_at {
            Utc::now() > expires_at
        } else {
            false
        }
    }

    /// Calculate expiry time
    fn calculate_expiry(&self) -> Option<DateTime<Utc>> {
        if self.config.default_ttl_seconds > 0 {
            Some(Utc::now() + chrono::Duration::seconds(self.config.default_ttl_seconds as i64))
        } else {
            None
        }
    }

    /// Deserialize state data
    fn deserialize_state_data<T: for<'de> Deserialize<'de>>(&self, data: &str) -> QarResult<Option<T>> {
        match serde_json::from_str(data) {
            Ok(deserialized) => Ok(Some(deserialized)),
            Err(e) => Err(QarError::StateError(format!("Deserialization failed: {}", e))),
        }
    }
}

/// Mock implementations for testing
pub struct MockStatePersistenceHandler;

#[async_trait::async_trait]
impl StatePersistenceHandler for MockStatePersistenceHandler {
    async fn save_state(&self, _entry: &StateEntry) -> QarResult<()> {
        Ok(())
    }

    async fn load_state(&self, _id: &str) -> QarResult<Option<StateEntry>> {
        Ok(None)
    }

    async fn delete_state(&self, _id: &str) -> QarResult<()> {
        Ok(())
    }

    async fn list_states(&self, _component: Option<&str>) -> QarResult<Vec<String>> {
        Ok(Vec::new())
    }

    async fn create_snapshot(&self, _snapshot: &StateSnapshot) -> QarResult<()> {
        Ok(())
    }

    async fn load_snapshot(&self, _id: &str) -> QarResult<Option<StateSnapshot>> {
        Ok(None)
    }
}

pub struct MockStateSyncHandler;

#[async_trait::async_trait]
impl StateSyncHandler for MockStateSyncHandler {
    async fn sync_state(&self, _entry: &StateEntry) -> QarResult<()> {
        Ok(())
    }

    async fn receive_state_update(&self, _entry: StateEntry) -> QarResult<()> {
        Ok(())
    }

    async fn request_state_sync(&self, _component: &str) -> QarResult<Vec<StateEntry>> {
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_state_manager() -> StateManager {
        let config = StateManagerConfig {
            max_memory_states: 1000,
            default_ttl_seconds: 3600,
            enable_persistence: false,
            enable_compression: false,
            enable_encryption: false,
            backup_interval_seconds: 300,
            max_snapshots: 10,
            sync_interval_seconds: 60,
        };

        StateManager::new(
            config,
            Arc::new(MockStatePersistenceHandler),
            Arc::new(MockStateSyncHandler),
        )
    }

    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct TestData {
        value: i32,
        text: String,
    }

    #[tokio::test]
    async fn test_set_and_get_state() {
        let manager = create_test_state_manager();
        let test_data = TestData {
            value: 42,
            text: "test".to_string(),
        };

        let state_id = manager
            .set_state("test_component", StateType::Classical, &test_data, PersistenceLevel::Memory)
            .await
            .unwrap();

        let retrieved: Option<TestData> = manager.get_state(&state_id).await.unwrap();
        assert_eq!(retrieved, Some(test_data));
    }

    #[tokio::test]
    async fn test_update_state() {
        let manager = create_test_state_manager();
        let test_data = TestData {
            value: 42,
            text: "test".to_string(),
        };

        let state_id = manager
            .set_state("test_component", StateType::Classical, &test_data, PersistenceLevel::Memory)
            .await
            .unwrap();

        let updated_data = TestData {
            value: 100,
            text: "updated".to_string(),
        };

        manager.update_state(&state_id, &updated_data).await.unwrap();

        let retrieved: Option<TestData> = manager.get_state(&state_id).await.unwrap();
        assert_eq!(retrieved, Some(updated_data));
    }

    #[tokio::test]
    async fn test_delete_state() {
        let manager = create_test_state_manager();
        let test_data = TestData {
            value: 42,
            text: "test".to_string(),
        };

        let state_id = manager
            .set_state("test_component", StateType::Classical, &test_data, PersistenceLevel::Memory)
            .await
            .unwrap();

        manager.delete_state(&state_id).await.unwrap();

        let retrieved: Option<TestData> = manager.get_state(&state_id).await.unwrap();
        assert_eq!(retrieved, None);
    }

    #[tokio::test]
    async fn test_list_states() {
        let manager = create_test_state_manager();
        let test_data = TestData {
            value: 42,
            text: "test".to_string(),
        };

        let _state_id1 = manager
            .set_state("component1", StateType::Classical, &test_data, PersistenceLevel::Memory)
            .await
            .unwrap();

        let _state_id2 = manager
            .set_state("component2", StateType::Quantum, &test_data, PersistenceLevel::Memory)
            .await
            .unwrap();

        let all_states = manager.list_states(None).await.unwrap();
        assert_eq!(all_states.len(), 2);

        let component1_states = manager.list_states(Some("component1")).await.unwrap();
        assert_eq!(component1_states.len(), 1);
        assert_eq!(component1_states[0].component, "component1");
    }

    #[tokio::test]
    async fn test_create_and_restore_snapshot() {
        let manager = create_test_state_manager();
        let test_data = TestData {
            value: 42,
            text: "test".to_string(),
        };

        let _state_id = manager
            .set_state("test_component", StateType::Classical, &test_data, PersistenceLevel::Memory)
            .await
            .unwrap();

        let snapshot_id = manager
            .create_snapshot("test_snapshot", "Test snapshot description")
            .await
            .unwrap();

        // Clear states
        let states = manager.list_states(None).await.unwrap();
        for state in states {
            manager.delete_state(&state.id).await.unwrap();
        }

        // Verify states are cleared
        let cleared_states = manager.list_states(None).await.unwrap();
        assert_eq!(cleared_states.len(), 0);

        // Restore from snapshot
        manager.restore_snapshot(&snapshot_id).await.unwrap();

        // Verify states are restored
        let restored_states = manager.list_states(None).await.unwrap();
        assert_eq!(restored_states.len(), 1);
    }

    #[tokio::test]
    async fn test_state_history() {
        let manager = create_test_state_manager();
        let test_data = TestData {
            value: 42,
            text: "test".to_string(),
        };

        let state_id = manager
            .set_state("test_component", StateType::Classical, &test_data, PersistenceLevel::Memory)
            .await
            .unwrap();

        let updated_data = TestData {
            value: 100,
            text: "updated".to_string(),
        };

        manager.update_state(&state_id, &updated_data).await.unwrap();
        manager.delete_state(&state_id).await.unwrap();

        let history = manager.get_state_history(Some(&state_id), None).await.unwrap();
        assert_eq!(history.len(), 3); // Created, Updated, Deleted
    }
}