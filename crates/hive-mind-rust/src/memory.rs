//! Collective memory and knowledge management system

use std::collections::{HashMap, BTreeMap};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use tracing::{info, error, debug, warn};

use crate::{
    config::MemoryConfig,
    metrics::MetricsCollector,
    error::{MemoryError, HiveMindError, Result},
};

/// Main collective memory system
#[derive(Debug)]
pub struct CollectiveMemory {
    /// Configuration
    config: MemoryConfig,
    
    /// Metrics collector
    metrics: Arc<MetricsCollector>,
    
    /// Knowledge graph
    knowledge_graph: Arc<RwLock<KnowledgeGraph>>,
    
    /// Memory pools organized by type
    memory_pools: Arc<RwLock<HashMap<MemoryType, MemoryPool>>>,
    
    /// Active sessions
    sessions: Arc<RwLock<HashMap<Uuid, MemorySession>>>,
    
    /// Memory manager for coordination
    manager: Arc<MemoryManager>,
}

/// Knowledge graph for semantic memory
#[derive(Debug)]
pub struct KnowledgeGraph {
    /// Graph nodes
    nodes: HashMap<Uuid, KnowledgeNode>,
    
    /// Graph edges (relationships)
    edges: HashMap<Uuid, Vec<KnowledgeEdge>>,
    
    /// Semantic index for fast search
    semantic_index: BTreeMap<String, Vec<Uuid>>,
    
    /// Node metadata
    metadata: HashMap<Uuid, NodeMetadata>,
}

/// A node in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeNode {
    /// Unique identifier
    pub id: Uuid,
    
    /// Node content
    pub content: serde_json::Value,
    
    /// Node type/category
    pub node_type: String,
    
    /// Tags for categorization
    pub tags: Vec<String>,
    
    /// Creation timestamp
    pub created_at: SystemTime,
    
    /// Last access timestamp
    pub last_accessed: SystemTime,
    
    /// Access count
    pub access_count: u64,
    
    /// Importance score (0.0 - 1.0)
    pub importance: f64,
}

/// An edge in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeEdge {
    /// Source node ID
    pub from: Uuid,
    
    /// Target node ID
    pub to: Uuid,
    
    /// Relationship type
    pub relationship_type: String,
    
    /// Relationship strength (0.0 - 1.0)
    pub strength: f64,
    
    /// Edge metadata
    pub metadata: serde_json::Value,
    
    /// Creation timestamp
    pub created_at: SystemTime,
}

/// Metadata for knowledge nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetadata {
    /// Creator information
    pub creator: Option<Uuid>,
    
    /// Source of information
    pub source: Option<String>,
    
    /// Confidence level (0.0 - 1.0)
    pub confidence: f64,
    
    /// Validation status
    pub validated: bool,
    
    /// Custom attributes
    pub attributes: HashMap<String, serde_json::Value>,
}

/// Memory pool for different types of data
#[derive(Debug)]
pub struct MemoryPool {
    /// Pool type
    pool_type: MemoryType,
    
    /// Stored items
    items: HashMap<String, MemoryItem>,
    
    /// Current size in bytes
    current_size: usize,
    
    /// Maximum size limit
    max_size: usize,
    
    /// Eviction policy
    eviction_policy: EvictionPolicy,
}

/// Types of memory pools
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MemoryType {
    /// Short-term working memory
    WorkingMemory,
    
    /// Long-term persistent memory
    LongTermMemory,
    
    /// Shared collective knowledge
    CollectiveKnowledge,
    
    /// Pattern memory for neural insights
    PatternMemory,
    
    /// Cache for frequently accessed data
    Cache,
}

/// Memory item stored in pools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryItem {
    /// Unique identifier
    pub id: String,
    
    /// Item content
    pub content: serde_json::Value,
    
    /// Item type
    pub item_type: String,
    
    /// Size in bytes
    pub size: usize,
    
    /// Priority level
    pub priority: Priority,
    
    /// Time-to-live (optional)
    pub ttl: Option<Duration>,
    
    /// Creation timestamp
    pub created_at: SystemTime,
    
    /// Last access timestamp
    pub last_accessed: SystemTime,
    
    /// Access frequency
    pub access_count: u64,
}

/// Priority levels for memory items
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

/// Eviction policies for memory pools
#[derive(Debug, Clone)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,
    
    /// Least Frequently Used
    LFU,
    
    /// Time-To-Live based
    TTL,
    
    /// Priority-based
    Priority,
    
    /// Hybrid policy combining multiple strategies
    Hybrid,
}

/// Memory session for tracking usage
#[derive(Debug, Clone)]
pub struct MemorySession {
    /// Session ID
    pub id: Uuid,
    
    /// Creator/owner
    pub owner: Option<Uuid>,
    
    /// Session start time
    pub started_at: SystemTime,
    
    /// Last activity
    pub last_activity: SystemTime,
    
    /// Memory allocations
    pub allocations: Vec<MemoryAllocation>,
    
    /// Session state
    pub state: SessionState,
}

/// Memory allocation tracking
#[derive(Debug, Clone)]
pub struct MemoryAllocation {
    /// Allocation ID
    pub id: Uuid,
    
    /// Memory type allocated
    pub memory_type: MemoryType,
    
    /// Size allocated
    pub size: usize,
    
    /// Allocation timestamp
    pub allocated_at: SystemTime,
    
    /// Whether still active
    pub active: bool,
}

/// Session states
#[derive(Debug, Clone, PartialEq)]
pub enum SessionState {
    Active,
    Inactive,
    Expired,
    Terminated,
}

/// Memory manager for coordination and optimization
#[derive(Debug)]
pub struct MemoryManager {
    /// Configuration
    config: MemoryConfig,
    
    /// Total memory usage tracking
    total_usage: Arc<RwLock<MemoryUsage>>,
    
    /// Cleanup scheduler
    cleanup_scheduler: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    /// Total allocated memory
    pub total_allocated: usize,
    
    /// Memory usage by type
    pub usage_by_type: HashMap<MemoryType, usize>,
    
    /// Number of active sessions
    pub active_sessions: usize,
    
    /// Number of knowledge nodes
    pub knowledge_nodes: usize,
    
    /// Cache hit rate
    pub cache_hit_rate: f64,
    
    /// Last cleanup time
    pub last_cleanup: SystemTime,
}

impl CollectiveMemory {
    /// Create a new collective memory system
    pub async fn new(
        config: &MemoryConfig,
        metrics: Arc<MetricsCollector>,
    ) -> Result<Self> {
        info!("Initializing collective memory system");
        
        let knowledge_graph = Arc::new(RwLock::new(KnowledgeGraph::new(config)?));
        let memory_pools = Arc::new(RwLock::new(Self::initialize_memory_pools(config)?));
        let sessions = Arc::new(RwLock::new(HashMap::new()));
        let manager = Arc::new(MemoryManager::new(config)?);
        
        let memory = Self {
            config: config.clone(),
            metrics,
            knowledge_graph,
            memory_pools,
            sessions,
            manager,
        };
        
        // Start background tasks
        memory.start_background_tasks().await?;
        
        info!("Collective memory system initialized");
        Ok(memory)
    }
    
    /// Store data in collective memory
    pub async fn store(&mut self, key: &str, value: serde_json::Value) -> Result<()> {
        debug!("Storing data with key: {}", key);
        
        // Determine appropriate memory type based on content
        let memory_type = self.determine_memory_type(&value).await?;
        
        // Create memory item
        let item = MemoryItem {
            id: key.to_string(),
            content: value.clone(),
            item_type: "data".to_string(),
            size: serde_json::to_vec(&value)?.len(),
            priority: Priority::Medium,
            ttl: None,
            created_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            access_count: 0,
        };
        
        // Store in appropriate pool
        {
            let mut pools = self.memory_pools.write().await;
            if let Some(pool) = pools.get_mut(&memory_type) {
                pool.insert(key, item)?;
            } else {
                return Err(MemoryError::AllocationFailed.into());
            }
        }
        
        // Update knowledge graph if applicable
        if memory_type == MemoryType::CollectiveKnowledge {
            self.add_to_knowledge_graph(key, &value).await?;
        }
        
        self.metrics.record_memory_operation("store", 1).await;
        Ok(())
    }
    
    /// Retrieve data from collective memory
    pub async fn retrieve(&self, key: &str) -> Result<Option<serde_json::Value>> {
        debug!("Retrieving data with key: {}", key);
        
        // Search across all memory pools
        let pools = self.memory_pools.read().await;
        for pool in pools.values() {
            if let Some(item) = pool.get(key) {
                // Update access statistics
                self.update_access_stats(key, &item.item_type).await;
                
                self.metrics.record_memory_operation("retrieve_hit", 1).await;
                return Ok(Some(item.content.clone()));
            }
        }
        
        self.metrics.record_memory_operation("retrieve_miss", 1).await;
        Ok(None)
    }
    
    /// Search memory using semantic queries
    pub async fn search(&self, query: &str) -> Result<Vec<serde_json::Value>> {
        debug!("Searching memory with query: {}", query);
        
        let mut results = Vec::new();
        
        // Search knowledge graph
        {
            let graph = self.knowledge_graph.read().await;
            let graph_results = graph.semantic_search(query)?;
            results.extend(graph_results);
        }
        
        // Search memory pools
        let pools = self.memory_pools.read().await;
        for pool in pools.values() {
            let pool_results = pool.search(query)?;
            results.extend(pool_results);
        }
        
        self.metrics.record_memory_operation("search", 1).await;
        Ok(results)
    }
    
    /// Remove data from memory
    pub async fn remove(&mut self, key: &str) -> Result<bool> {
        debug!("Removing data with key: {}", key);
        
        let mut removed = false;
        
        // Remove from memory pools
        {
            let mut pools = self.memory_pools.write().await;
            for pool in pools.values_mut() {
                if pool.remove(key)? {
                    removed = true;
                    break;
                }
            }
        }
        
        // Remove from knowledge graph
        {
            let mut graph = self.knowledge_graph.write().await;
            graph.remove_by_key(key)?;
        }
        
        if removed {
            self.metrics.record_memory_operation("remove", 1).await;
        }
        
        Ok(removed)
    }
    
    /// Get memory usage statistics
    pub async fn get_usage_stats(&self) -> Result<MemoryUsageStats> {
        let usage = self.manager.get_usage_stats().await?;
        
        Ok(MemoryUsageStats {
            used_capacity: usage.total_allocated,
            knowledge_nodes: usage.knowledge_nodes,
            active_sessions: usage.active_sessions,
        })
    }
    
    /// Create a new memory session
    pub async fn create_session(&self, owner: Option<Uuid>) -> Result<Uuid> {
        let session_id = Uuid::new_v4();
        
        let session = MemorySession {
            id: session_id,
            owner,
            started_at: SystemTime::now(),
            last_activity: SystemTime::now(),
            allocations: Vec::new(),
            state: SessionState::Active,
        };
        
        let mut sessions = self.sessions.write().await;
        sessions.insert(session_id, session);
        
        debug!("Created memory session: {}", session_id);
        Ok(session_id)
    }
    
    /// Initialize memory pools
    fn initialize_memory_pools(config: &MemoryConfig) -> Result<HashMap<MemoryType, MemoryPool>> {
        let mut pools = HashMap::new();
        
        // Calculate size allocations for different memory types
        let total_size = config.max_pool_size;
        let working_memory_size = total_size / 10; // 10%
        let long_term_memory_size = total_size / 2; // 50%
        let collective_knowledge_size = total_size * 3 / 10; // 30%
        let pattern_memory_size = total_size / 20; // 5%
        let cache_size = total_size / 20; // 5%
        
        pools.insert(
            MemoryType::WorkingMemory,
            MemoryPool::new(MemoryType::WorkingMemory, working_memory_size, EvictionPolicy::LRU),
        );
        
        pools.insert(
            MemoryType::LongTermMemory,
            MemoryPool::new(MemoryType::LongTermMemory, long_term_memory_size, EvictionPolicy::Priority),
        );
        
        pools.insert(
            MemoryType::CollectiveKnowledge,
            MemoryPool::new(MemoryType::CollectiveKnowledge, collective_knowledge_size, EvictionPolicy::LFU),
        );
        
        pools.insert(
            MemoryType::PatternMemory,
            MemoryPool::new(MemoryType::PatternMemory, pattern_memory_size, EvictionPolicy::TTL),
        );
        
        pools.insert(
            MemoryType::Cache,
            MemoryPool::new(MemoryType::Cache, cache_size, EvictionPolicy::LRU),
        );
        
        Ok(pools)
    }
    
    /// Determine appropriate memory type for data
    async fn determine_memory_type(&self, value: &serde_json::Value) -> Result<MemoryType> {
        // Simple heuristic based on content type and size
        if let Some(obj) = value.as_object() {
            if obj.contains_key("pattern") || obj.contains_key("neural") {
                return Ok(MemoryType::PatternMemory);
            }
            
            if obj.contains_key("knowledge") || obj.contains_key("fact") {
                return Ok(MemoryType::CollectiveKnowledge);
            }
            
            if obj.contains_key("temporary") || obj.contains_key("cache") {
                return Ok(MemoryType::Cache);
            }
        }
        
        // Default to working memory
        Ok(MemoryType::WorkingMemory)
    }
    
    /// Add data to knowledge graph
    async fn add_to_knowledge_graph(&self, key: &str, value: &serde_json::Value) -> Result<()> {
        let node = KnowledgeNode {
            id: Uuid::new_v4(),
            content: value.clone(),
            node_type: "data".to_string(),
            tags: vec![key.to_string()],
            created_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            access_count: 0,
            importance: 0.5, // Default importance
        };
        
        let mut graph = self.knowledge_graph.write().await;
        graph.add_node(node)?;
        
        Ok(())
    }
    
    /// Update access statistics
    async fn update_access_stats(&self, _key: &str, _item_type: &str) {
        // Implementation would update access statistics
        // This is a placeholder for the actual implementation
    }
    
    /// Start background tasks
    async fn start_background_tasks(&self) -> Result<()> {
        // Start cleanup task
        self.manager.start_cleanup_task().await?;
        
        Ok(())
    }
}

impl KnowledgeGraph {
    /// Create a new knowledge graph
    pub fn new(_config: &MemoryConfig) -> Result<Self> {
        Ok(Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            semantic_index: BTreeMap::new(),
            metadata: HashMap::new(),
        })
    }
    
    /// Add a node to the graph
    pub fn add_node(&mut self, node: KnowledgeNode) -> Result<()> {
        let node_id = node.id;
        
        // Add to semantic index
        for tag in &node.tags {
            self.semantic_index
                .entry(tag.clone())
                .or_insert_with(Vec::new)
                .push(node_id);
        }
        
        self.nodes.insert(node_id, node);
        self.edges.insert(node_id, Vec::new());
        
        Ok(())
    }
    
    /// Add an edge between nodes
    pub fn add_edge(&mut self, edge: KnowledgeEdge) -> Result<()> {
        if !self.nodes.contains_key(&edge.from) || !self.nodes.contains_key(&edge.to) {
            return Err(MemoryError::KnowledgeNotFound {
                key: "node".to_string(),
            }.into());
        }
        
        self.edges
            .entry(edge.from)
            .or_insert_with(Vec::new)
            .push(edge);
        
        Ok(())
    }
    
    /// Perform semantic search
    pub fn semantic_search(&self, query: &str) -> Result<Vec<serde_json::Value>> {
        let mut results = Vec::new();
        
        // Search in semantic index
        for (key, node_ids) in &self.semantic_index {
            if key.contains(query) {
                for &node_id in node_ids {
                    if let Some(node) = self.nodes.get(&node_id) {
                        results.push(node.content.clone());
                    }
                }
            }
        }
        
        Ok(results)
    }
    
    /// Remove node by key
    pub fn remove_by_key(&mut self, _key: &str) -> Result<()> {
        // Implementation would remove nodes matching the key
        Ok(())
    }
}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new(pool_type: MemoryType, max_size: usize, eviction_policy: EvictionPolicy) -> Self {
        Self {
            pool_type,
            items: HashMap::new(),
            current_size: 0,
            max_size,
            eviction_policy,
        }
    }
    
    /// Insert an item into the pool
    pub fn insert(&mut self, key: &str, mut item: MemoryItem) -> Result<()> {
        // Check if we need to make space
        while self.current_size + item.size > self.max_size {
            if !self.evict_item()? {
                return Err(MemoryError::CapacityExceeded.into());
            }
        }
        
        // Update access time
        item.last_accessed = SystemTime::now();
        
        self.current_size += item.size;
        self.items.insert(key.to_string(), item);
        
        Ok(())
    }
    
    /// Get an item from the pool
    pub fn get(&self, key: &str) -> Option<&MemoryItem> {
        self.items.get(key)
    }
    
    /// Remove an item from the pool
    pub fn remove(&mut self, key: &str) -> Result<bool> {
        if let Some(item) = self.items.remove(key) {
            self.current_size -= item.size;
            Ok(true)
        } else {
            Ok(false)
        }
    }
    
    /// Search items in the pool
    pub fn search(&self, query: &str) -> Result<Vec<serde_json::Value>> {
        let mut results = Vec::new();
        
        for item in self.items.values() {
            if item.id.contains(query) || item.item_type.contains(query) {
                results.push(item.content.clone());
            }
        }
        
        Ok(results)
    }
    
    /// Evict an item based on the eviction policy
    fn evict_item(&mut self) -> Result<bool> {
        let key_to_evict = match self.eviction_policy {
            EvictionPolicy::LRU => self.find_lru_item(),
            EvictionPolicy::LFU => self.find_lfu_item(),
            EvictionPolicy::TTL => self.find_expired_item(),
            EvictionPolicy::Priority => self.find_lowest_priority_item(),
            EvictionPolicy::Hybrid => self.find_hybrid_eviction_item(),
        };
        
        if let Some(key) = key_to_evict {
            self.remove(&key)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
    
    /// Find least recently used item
    fn find_lru_item(&self) -> Option<String> {
        self.items
            .iter()
            .min_by_key(|(_, item)| item.last_accessed)
            .map(|(key, _)| key.clone())
    }
    
    /// Find least frequently used item
    fn find_lfu_item(&self) -> Option<String> {
        self.items
            .iter()
            .min_by_key(|(_, item)| item.access_count)
            .map(|(key, _)| key.clone())
    }
    
    /// Find expired item
    fn find_expired_item(&self) -> Option<String> {
        let now = SystemTime::now();
        
        for (key, item) in &self.items {
            if let Some(ttl) = item.ttl {
                if let Ok(elapsed) = now.duration_since(item.created_at) {
                    if elapsed > ttl {
                        return Some(key.clone());
                    }
                }
            }
        }
        
        None
    }
    
    /// Find lowest priority item
    fn find_lowest_priority_item(&self) -> Option<String> {
        self.items
            .iter()
            .min_by_key(|(_, item)| &item.priority)
            .map(|(key, _)| key.clone())
    }
    
    /// Find item using hybrid eviction strategy
    fn find_hybrid_eviction_item(&self) -> Option<String> {
        // Combine multiple factors for eviction decision
        // This is a simplified implementation
        
        // First try expired items
        if let Some(key) = self.find_expired_item() {
            return Some(key);
        }
        
        // Then try lowest priority
        if let Some(key) = self.find_lowest_priority_item() {
            return Some(key);
        }
        
        // Finally, LRU
        self.find_lru_item()
    }
}

impl MemoryManager {
    /// Create a new memory manager
    pub fn new(config: &MemoryConfig) -> Result<Self> {
        let total_usage = Arc::new(RwLock::new(MemoryUsage {
            total_allocated: 0,
            usage_by_type: HashMap::new(),
            active_sessions: 0,
            knowledge_nodes: 0,
            cache_hit_rate: 0.0,
            last_cleanup: SystemTime::now(),
        }));
        
        Ok(Self {
            config: config.clone(),
            total_usage,
            cleanup_scheduler: Arc::new(RwLock::new(None)),
        })
    }
    
    /// Get memory usage statistics
    pub async fn get_usage_stats(&self) -> Result<MemoryUsage> {
        let usage = self.total_usage.read().await;
        Ok(usage.clone())
    }
    
    /// Start cleanup task
    pub async fn start_cleanup_task(&self) -> Result<()> {
        let interval = self.config.cleanup_interval;
        let total_usage = self.total_usage.clone();
        
        let task = tokio::spawn(async move {
            let mut cleanup_interval = tokio::time::interval(interval);
            
            loop {
                cleanup_interval.tick().await;
                
                // Update cleanup timestamp
                {
                    let mut usage = total_usage.write().await;
                    usage.last_cleanup = SystemTime::now();
                }
                
                debug!("Memory cleanup task executed");
            }
        });
        
        let mut scheduler = self.cleanup_scheduler.write().await;
        *scheduler = Some(task);
        
        Ok(())
    }
}

/// Public memory usage statistics structure
#[derive(Debug, Clone)]
pub struct MemoryUsageStats {
    pub used_capacity: usize,
    pub knowledge_nodes: usize,
    pub active_sessions: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_node_creation() {
        let node = KnowledgeNode {
            id: Uuid::new_v4(),
            content: serde_json::json!({"test": "data"}),
            node_type: "test".to_string(),
            tags: vec!["test".to_string()],
            created_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            access_count: 0,
            importance: 0.5,
        };
        
        assert_eq!(node.node_type, "test");
        assert_eq!(node.tags, vec!["test"]);
    }
    
    #[test]
    fn test_memory_pool_creation() {
        let pool = MemoryPool::new(
            MemoryType::WorkingMemory,
            1024,
            EvictionPolicy::LRU,
        );
        
        assert_eq!(pool.pool_type, MemoryType::WorkingMemory);
        assert_eq!(pool.max_size, 1024);
        assert_eq!(pool.current_size, 0);
    }
    
    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Medium);
        assert!(Priority::Medium > Priority::Low);
    }
}