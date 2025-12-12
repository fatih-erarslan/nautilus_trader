//! Shared memory coordination system for cross-agent state management.

use crate::error::{OrchestrationError, Result};
use crate::types::{AgentId, Timestamp};
use async_trait::async_trait;
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{broadcast, RwLock as TokioRwLock};
use tokio::time::{interval, sleep};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Memory region identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryRegionId(pub Uuid);

impl MemoryRegionId {
    /// Generate a new random memory region ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
    
    /// Create a memory region ID from a UUID
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }
    
    /// Get the underlying UUID
    pub fn uuid(&self) -> Uuid {
        self.0
    }
}

impl std::fmt::Display for MemoryRegionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for MemoryRegionId {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory access permissions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryPermission {
    /// Read-only access
    Read,
    /// Write-only access
    Write,
    /// Read-write access
    ReadWrite,
    /// No access
    None,
}

/// Memory region metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRegionMetadata {
    /// Region ID
    pub id: MemoryRegionId,
    /// Region name
    pub name: String,
    /// Region description
    pub description: String,
    /// Region owner
    pub owner: AgentId,
    /// Access permissions by agent
    pub permissions: HashMap<AgentId, MemoryPermission>,
    /// Creation timestamp
    pub created_at: Timestamp,
    /// Last modification timestamp
    pub modified_at: Timestamp,
    /// Region size in bytes
    pub size: u64,
    /// Region tags for categorization
    pub tags: Vec<String>,
    /// Region expiration time (if any)
    pub expires_at: Option<Timestamp>,
    /// Region version for optimistic locking
    pub version: u64,
}

impl MemoryRegionMetadata {
    /// Create new memory region metadata
    pub fn new(
        id: MemoryRegionId,
        name: String,
        description: String,
        owner: AgentId,
    ) -> Self {
        Self {
            id,
            name,
            description,
            owner,
            permissions: HashMap::new(),
            created_at: Timestamp::now(),
            modified_at: Timestamp::now(),
            size: 0,
            tags: Vec::new(),
            expires_at: None,
            version: 1,
        }
    }
    
    /// Check if agent has permission
    pub fn has_permission(&self, agent_id: AgentId, permission: MemoryPermission) -> bool {
        if agent_id == self.owner {
            return true; // Owner has all permissions
        }
        
        match self.permissions.get(&agent_id) {
            Some(MemoryPermission::ReadWrite) => true,
            Some(MemoryPermission::Read) => matches!(permission, MemoryPermission::Read),
            Some(MemoryPermission::Write) => matches!(permission, MemoryPermission::Write),
            Some(MemoryPermission::None) | None => false,
        }
    }
    
    /// Grant permission to an agent
    pub fn grant_permission(&mut self, agent_id: AgentId, permission: MemoryPermission) {
        self.permissions.insert(agent_id, permission);
        self.modified_at = Timestamp::now();
        self.version += 1;
    }
    
    /// Revoke permission from an agent
    pub fn revoke_permission(&mut self, agent_id: AgentId) {
        self.permissions.remove(&agent_id);
        self.modified_at = Timestamp::now();
        self.version += 1;
    }
    
    /// Check if region has expired
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            Timestamp::now() >= expires_at
        } else {
            false
        }
    }
    
    /// Update modification timestamp and version
    pub fn update_modified(&mut self) {
        self.modified_at = Timestamp::now();
        self.version += 1;
    }
}

/// Memory region containing data and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRegion {
    /// Region metadata
    pub metadata: MemoryRegionMetadata,
    /// Region data
    pub data: Vec<u8>,
    /// Region checksum for integrity
    pub checksum: u64,
}

impl MemoryRegion {
    /// Create a new memory region
    pub fn new(
        id: MemoryRegionId,
        name: String,
        description: String,
        owner: AgentId,
        data: Vec<u8>,
    ) -> Self {
        let checksum = Self::calculate_checksum(&data);
        let mut metadata = MemoryRegionMetadata::new(id, name, description, owner);
        metadata.size = data.len() as u64;
        
        Self {
            metadata,
            data,
            checksum,
        }
    }
    
    /// Calculate checksum for data integrity
    fn calculate_checksum(data: &[u8]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        data.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Update region data
    pub fn update_data(&mut self, data: Vec<u8>) -> Result<()> {
        self.data = data;
        self.checksum = Self::calculate_checksum(&self.data);
        self.metadata.size = self.data.len() as u64;
        self.metadata.update_modified();
        Ok(())
    }
    
    /// Verify data integrity
    pub fn verify_integrity(&self) -> bool {
        self.checksum == Self::calculate_checksum(&self.data)
    }
    
    /// Get region data (read-only)
    pub fn get_data(&self) -> &[u8] {
        &self.data
    }
    
    /// Get region size
    pub fn size(&self) -> u64 {
        self.metadata.size
    }
}

/// Memory event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryEvent {
    /// Region created
    RegionCreated {
        region_id: MemoryRegionId,
        owner: AgentId,
        name: String,
    },
    /// Region updated
    RegionUpdated {
        region_id: MemoryRegionId,
        updater: AgentId,
        version: u64,
    },
    /// Region deleted
    RegionDeleted {
        region_id: MemoryRegionId,
        deleter: AgentId,
    },
    /// Permission granted
    PermissionGranted {
        region_id: MemoryRegionId,
        agent_id: AgentId,
        permission: MemoryPermission,
    },
    /// Permission revoked
    PermissionRevoked {
        region_id: MemoryRegionId,
        agent_id: AgentId,
    },
    /// Region expired
    RegionExpired {
        region_id: MemoryRegionId,
    },
}

/// Shared memory trait for cross-agent coordination
#[async_trait]
pub trait SharedMemory: Send + Sync {
    /// Create a new memory region
    async fn create_region(
        &self,
        name: String,
        description: String,
        owner: AgentId,
        data: Vec<u8>,
    ) -> Result<MemoryRegionId>;
    
    /// Get a memory region
    async fn get_region(&self, region_id: MemoryRegionId, accessor: AgentId) -> Result<MemoryRegion>;
    
    /// Update a memory region
    async fn update_region(
        &self,
        region_id: MemoryRegionId,
        updater: AgentId,
        data: Vec<u8>,
        expected_version: Option<u64>,
    ) -> Result<()>;
    
    /// Delete a memory region
    async fn delete_region(&self, region_id: MemoryRegionId, deleter: AgentId) -> Result<()>;
    
    /// Grant permission to an agent
    async fn grant_permission(
        &self,
        region_id: MemoryRegionId,
        grantor: AgentId,
        grantee: AgentId,
        permission: MemoryPermission,
    ) -> Result<()>;
    
    /// Revoke permission from an agent
    async fn revoke_permission(
        &self,
        region_id: MemoryRegionId,
        revoker: AgentId,
        revokee: AgentId,
    ) -> Result<()>;
    
    /// List regions accessible by an agent
    async fn list_regions(&self, agent_id: AgentId) -> Result<Vec<MemoryRegionMetadata>>;
    
    /// Search regions by tags
    async fn search_regions(&self, tags: Vec<String>, agent_id: AgentId) -> Result<Vec<MemoryRegionMetadata>>;
    
    /// Get memory statistics
    async fn get_memory_stats(&self) -> Result<MemoryStatistics>;
    
    /// Subscribe to memory events
    async fn subscribe_events(&self) -> Result<broadcast::Receiver<MemoryEvent>>;
}

/// Memory statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStatistics {
    /// Total number of regions
    pub total_regions: u64,
    /// Total memory used in bytes
    pub total_memory_used: u64,
    /// Number of active regions
    pub active_regions: u64,
    /// Number of expired regions
    pub expired_regions: u64,
    /// Memory usage by agent
    pub memory_by_agent: HashMap<AgentId, u64>,
    /// Region count by tags
    pub regions_by_tags: HashMap<String, u64>,
    /// Average region size
    pub avg_region_size: f64,
    /// Total read operations
    pub total_read_ops: u64,
    /// Total write operations
    pub total_write_ops: u64,
    /// Cache hit ratio
    pub cache_hit_ratio: f64,
}

/// High-performance shared memory coordinator
#[derive(Debug)]
pub struct MemoryCoordinator {
    /// Memory regions storage
    regions: Arc<DashMap<MemoryRegionId, MemoryRegion>>,
    /// Memory region cache for frequently accessed regions
    cache: Arc<DashMap<MemoryRegionId, MemoryRegion>>,
    /// Event broadcaster
    event_broadcaster: broadcast::Sender<MemoryEvent>,
    /// Operation counters
    read_counter: Arc<AtomicU64>,
    write_counter: Arc<AtomicU64>,
    cache_hits: Arc<AtomicU64>,
    cache_misses: Arc<AtomicU64>,
    /// Memory statistics
    stats: Arc<RwLock<MemoryStatistics>>,
    /// Cache size limit
    cache_size_limit: usize,
    /// Region expiration checker
    expiration_checker: Arc<TokioRwLock<bool>>,
}

impl MemoryCoordinator {
    /// Create a new memory coordinator
    pub fn new(cache_size_limit: usize) -> Self {
        let (event_broadcaster, _) = broadcast::channel(1000);
        
        Self {
            regions: Arc::new(DashMap::new()),
            cache: Arc::new(DashMap::new()),
            event_broadcaster,
            read_counter: Arc::new(AtomicU64::new(0)),
            write_counter: Arc::new(AtomicU64::new(0)),
            cache_hits: Arc::new(AtomicU64::new(0)),
            cache_misses: Arc::new(AtomicU64::new(0)),
            stats: Arc::new(RwLock::new(MemoryStatistics {
                total_regions: 0,
                total_memory_used: 0,
                active_regions: 0,
                expired_regions: 0,
                memory_by_agent: HashMap::new(),
                regions_by_tags: HashMap::new(),
                avg_region_size: 0.0,
                total_read_ops: 0,
                total_write_ops: 0,
                cache_hit_ratio: 0.0,
            })),
            cache_size_limit,
            expiration_checker: Arc::new(TokioRwLock::new(false)),
        }
    }
    
    /// Start the memory coordinator
    pub async fn start(&self) -> Result<()> {
        // Start expiration checker
        self.start_expiration_checker().await?;
        
        // Start statistics updater
        self.start_statistics_updater().await?;
        
        info!("Memory coordinator started successfully");
        Ok(())
    }
    
    /// Start expiration checker
    async fn start_expiration_checker(&self) -> Result<()> {
        let regions = Arc::clone(&self.regions);
        let event_broadcaster = self.event_broadcaster.clone();
        let expiration_checker = Arc::clone(&self.expiration_checker);
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                if !*expiration_checker.read().await {
                    continue;
                }
                
                let mut expired_regions = Vec::new();
                
                // Find expired regions
                for entry in regions.iter() {
                    let region = entry.value();
                    if region.metadata.is_expired() {
                        expired_regions.push(region.metadata.id);
                    }
                }
                
                // Remove expired regions
                for region_id in expired_regions {
                    regions.remove(&region_id);
                    
                    // Broadcast expiration event
                    let event = MemoryEvent::RegionExpired { region_id };
                    let _ = event_broadcaster.send(event);
                    
                    debug!("Memory region {} expired and removed", region_id);
                }
            }
        });
        
        *self.expiration_checker.write().await = true;
        Ok(())
    }
    
    /// Start statistics updater
    async fn start_statistics_updater(&self) -> Result<()> {
        let regions = Arc::clone(&self.regions);
        let stats = Arc::clone(&self.stats);
        let read_counter = Arc::clone(&self.read_counter);
        let write_counter = Arc::clone(&self.write_counter);
        let cache_hits = Arc::clone(&self.cache_hits);
        let cache_misses = Arc::clone(&self.cache_misses);
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(10));
            
            loop {
                interval.tick().await;
                
                let mut new_stats = MemoryStatistics {
                    total_regions: regions.len() as u64,
                    total_memory_used: 0,
                    active_regions: 0,
                    expired_regions: 0,
                    memory_by_agent: HashMap::new(),
                    regions_by_tags: HashMap::new(),
                    avg_region_size: 0.0,
                    total_read_ops: read_counter.load(Ordering::Relaxed),
                    total_write_ops: write_counter.load(Ordering::Relaxed),
                    cache_hit_ratio: 0.0,
                };
                
                // Collect statistics
                for entry in regions.iter() {
                    let region = entry.value();
                    new_stats.total_memory_used += region.metadata.size;
                    
                    if region.metadata.is_expired() {
                        new_stats.expired_regions += 1;
                    } else {
                        new_stats.active_regions += 1;
                    }
                    
                    // Update memory by agent
                    *new_stats.memory_by_agent.entry(region.metadata.owner).or_insert(0) += region.metadata.size;
                    
                    // Update regions by tags
                    for tag in &region.metadata.tags {
                        *new_stats.regions_by_tags.entry(tag.clone()).or_insert(0) += 1;
                    }
                }
                
                // Calculate averages
                if new_stats.total_regions > 0 {
                    new_stats.avg_region_size = new_stats.total_memory_used as f64 / new_stats.total_regions as f64;
                }
                
                // Calculate cache hit ratio
                let total_cache_ops = cache_hits.load(Ordering::Relaxed) + cache_misses.load(Ordering::Relaxed);
                if total_cache_ops > 0 {
                    new_stats.cache_hit_ratio = cache_hits.load(Ordering::Relaxed) as f64 / total_cache_ops as f64;
                }
                
                // Update statistics
                *stats.write() = new_stats;
            }
        });
        
        Ok(())
    }
    
    /// Get region from cache or storage
    async fn get_region_cached(&self, region_id: MemoryRegionId) -> Option<MemoryRegion> {
        // Try cache first
        if let Some(region) = self.cache.get(&region_id) {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
            return Some(region.clone());
        }
        
        // Try storage
        if let Some(region) = self.regions.get(&region_id) {
            self.cache_misses.fetch_add(1, Ordering::Relaxed);
            
            // Add to cache if not full
            if self.cache.len() < self.cache_size_limit {
                self.cache.insert(region_id, region.clone());
            }
            
            return Some(region.clone());
        }
        
        None
    }
    
    /// Update cache when region is modified
    fn update_cache(&self, region_id: MemoryRegionId, region: &MemoryRegion) {
        self.cache.insert(region_id, region.clone());
    }
    
    /// Remove from cache
    fn remove_from_cache(&self, region_id: MemoryRegionId) {
        self.cache.remove(&region_id);
    }
    
    /// Broadcast memory event
    fn broadcast_event(&self, event: MemoryEvent) {
        let _ = self.event_broadcaster.send(event);
    }
}

#[async_trait]
impl SharedMemory for MemoryCoordinator {
    async fn create_region(
        &self,
        name: String,
        description: String,
        owner: AgentId,
        data: Vec<u8>,
    ) -> Result<MemoryRegionId> {
        let region_id = MemoryRegionId::new();
        let region = MemoryRegion::new(region_id, name.clone(), description, owner, data);
        
        // Verify data integrity
        if !region.verify_integrity() {
            return Err(OrchestrationError::memory("Data integrity check failed"));
        }
        
        // Store region
        self.regions.insert(region_id, region);
        
        // Update cache
        if let Some(region) = self.regions.get(&region_id) {
            self.update_cache(region_id, &region);
        }
        
        // Broadcast event
        self.broadcast_event(MemoryEvent::RegionCreated {
            region_id,
            owner,
            name,
        });
        
        self.write_counter.fetch_add(1, Ordering::Relaxed);
        
        debug!("Memory region {} created by agent {}", region_id, owner);
        Ok(region_id)
    }
    
    async fn get_region(&self, region_id: MemoryRegionId, accessor: AgentId) -> Result<MemoryRegion> {
        let region = self.get_region_cached(region_id).await
            .ok_or_else(|| OrchestrationError::not_found(format!("Memory region {}", region_id)))?;
        
        // Check read permission
        if !region.metadata.has_permission(accessor, MemoryPermission::Read) {
            return Err(OrchestrationError::permission_denied(
                format!("Agent {} does not have read permission for region {}", accessor, region_id)
            ));
        }
        
        // Verify data integrity
        if !region.verify_integrity() {
            return Err(OrchestrationError::memory("Data integrity check failed"));
        }
        
        self.read_counter.fetch_add(1, Ordering::Relaxed);
        
        debug!("Memory region {} accessed by agent {}", region_id, accessor);
        Ok(region)
    }
    
    async fn update_region(
        &self,
        region_id: MemoryRegionId,
        updater: AgentId,
        data: Vec<u8>,
        expected_version: Option<u64>,
    ) -> Result<()> {
        // Get current region
        let mut region = self.get_region_cached(region_id).await
            .ok_or_else(|| OrchestrationError::not_found(format!("Memory region {}", region_id)))?;
        
        // Check write permission
        if !region.metadata.has_permission(updater, MemoryPermission::Write) {
            return Err(OrchestrationError::permission_denied(
                format!("Agent {} does not have write permission for region {}", updater, region_id)
            ));
        }
        
        // Check version for optimistic locking
        if let Some(expected) = expected_version {
            if region.metadata.version != expected {
                return Err(OrchestrationError::invalid_state(
                    format!("Version mismatch: expected {}, got {}", expected, region.metadata.version)
                ));
            }
        }
        
        // Update data
        region.update_data(data)?;
        
        // Store updated region
        self.regions.insert(region_id, region.clone());
        
        // Update cache
        self.update_cache(region_id, &region);
        
        // Broadcast event
        self.broadcast_event(MemoryEvent::RegionUpdated {
            region_id,
            updater,
            version: region.metadata.version,
        });
        
        self.write_counter.fetch_add(1, Ordering::Relaxed);
        
        debug!("Memory region {} updated by agent {} (version {})", region_id, updater, region.metadata.version);
        Ok(())
    }
    
    async fn delete_region(&self, region_id: MemoryRegionId, deleter: AgentId) -> Result<()> {
        // Get region to check permissions
        let region = self.get_region_cached(region_id).await
            .ok_or_else(|| OrchestrationError::not_found(format!("Memory region {}", region_id)))?;
        
        // Only owner can delete
        if region.metadata.owner != deleter {
            return Err(OrchestrationError::permission_denied(
                format!("Agent {} is not the owner of region {}", deleter, region_id)
            ));
        }
        
        // Remove from storage
        self.regions.remove(&region_id);
        
        // Remove from cache
        self.remove_from_cache(region_id);
        
        // Broadcast event
        self.broadcast_event(MemoryEvent::RegionDeleted {
            region_id,
            deleter,
        });
        
        debug!("Memory region {} deleted by agent {}", region_id, deleter);
        Ok(())
    }
    
    async fn grant_permission(
        &self,
        region_id: MemoryRegionId,
        grantor: AgentId,
        grantee: AgentId,
        permission: MemoryPermission,
    ) -> Result<()> {
        // Get current region
        let mut region = self.get_region_cached(region_id).await
            .ok_or_else(|| OrchestrationError::not_found(format!("Memory region {}", region_id)))?;
        
        // Only owner can grant permissions
        if region.metadata.owner != grantor {
            return Err(OrchestrationError::permission_denied(
                format!("Agent {} is not the owner of region {}", grantor, region_id)
            ));
        }
        
        // Grant permission
        region.metadata.grant_permission(grantee, permission);
        
        // Store updated region
        self.regions.insert(region_id, region.clone());
        
        // Update cache
        self.update_cache(region_id, &region);
        
        // Broadcast event
        self.broadcast_event(MemoryEvent::PermissionGranted {
            region_id,
            agent_id: grantee,
            permission,
        });
        
        debug!("Permission {:?} granted to agent {} for region {} by agent {}", 
               permission, grantee, region_id, grantor);
        Ok(())
    }
    
    async fn revoke_permission(
        &self,
        region_id: MemoryRegionId,
        revoker: AgentId,
        revokee: AgentId,
    ) -> Result<()> {
        // Get current region
        let mut region = self.get_region_cached(region_id).await
            .ok_or_else(|| OrchestrationError::not_found(format!("Memory region {}", region_id)))?;
        
        // Only owner can revoke permissions
        if region.metadata.owner != revoker {
            return Err(OrchestrationError::permission_denied(
                format!("Agent {} is not the owner of region {}", revoker, region_id)
            ));
        }
        
        // Revoke permission
        region.metadata.revoke_permission(revokee);
        
        // Store updated region
        self.regions.insert(region_id, region.clone());
        
        // Update cache
        self.update_cache(region_id, &region);
        
        // Broadcast event
        self.broadcast_event(MemoryEvent::PermissionRevoked {
            region_id,
            agent_id: revokee,
        });
        
        debug!("Permission revoked from agent {} for region {} by agent {}", 
               revokee, region_id, revoker);
        Ok(())
    }
    
    async fn list_regions(&self, agent_id: AgentId) -> Result<Vec<MemoryRegionMetadata>> {
        let mut accessible_regions = Vec::new();
        
        for entry in self.regions.iter() {
            let region = entry.value();
            
            // Check if agent has any permission or is the owner
            if region.metadata.owner == agent_id || 
               region.metadata.has_permission(agent_id, MemoryPermission::Read) {
                accessible_regions.push(region.metadata.clone());
            }
        }
        
        Ok(accessible_regions)
    }
    
    async fn search_regions(&self, tags: Vec<String>, agent_id: AgentId) -> Result<Vec<MemoryRegionMetadata>> {
        let mut matching_regions = Vec::new();
        
        for entry in self.regions.iter() {
            let region = entry.value();
            
            // Check if agent has permission
            if region.metadata.owner != agent_id && 
               !region.metadata.has_permission(agent_id, MemoryPermission::Read) {
                continue;
            }
            
            // Check if region has any of the requested tags
            if tags.iter().any(|tag| region.metadata.tags.contains(tag)) {
                matching_regions.push(region.metadata.clone());
            }
        }
        
        Ok(matching_regions)
    }
    
    async fn get_memory_stats(&self) -> Result<MemoryStatistics> {
        Ok(self.stats.read().clone())
    }
    
    async fn subscribe_events(&self) -> Result<broadcast::Receiver<MemoryEvent>> {
        Ok(self.event_broadcaster.subscribe())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};
    
    #[tokio::test]
    async fn test_memory_region_creation() {
        let region_id = MemoryRegionId::new();
        let owner = AgentId::new();
        let data = b"test data".to_vec();
        
        let region = MemoryRegion::new(
            region_id,
            "Test Region".to_string(),
            "Test Description".to_string(),
            owner,
            data.clone(),
        );
        
        assert_eq!(region.metadata.id, region_id);
        assert_eq!(region.metadata.owner, owner);
        assert_eq!(region.data, data);
        assert!(region.verify_integrity());
    }
    
    #[tokio::test]
    async fn test_memory_coordinator() {
        let coordinator = MemoryCoordinator::new(100);
        coordinator.start().await.unwrap();
        
        let owner = AgentId::new();
        let data = b"test data".to_vec();
        
        // Create region
        let region_id = coordinator.create_region(
            "Test Region".to_string(),
            "Test Description".to_string(),
            owner,
            data.clone(),
        ).await.unwrap();
        
        // Get region
        let region = coordinator.get_region(region_id, owner).await.unwrap();
        assert_eq!(region.data, data);
        
        // Update region
        let new_data = b"updated data".to_vec();
        coordinator.update_region(region_id, owner, new_data.clone(), None).await.unwrap();
        
        let updated_region = coordinator.get_region(region_id, owner).await.unwrap();
        assert_eq!(updated_region.data, new_data);
        assert_eq!(updated_region.metadata.version, 2);
        
        // Delete region
        coordinator.delete_region(region_id, owner).await.unwrap();
        
        let result = coordinator.get_region(region_id, owner).await;
        assert!(result.is_err());
    }
    
    #[tokio::test]
    async fn test_memory_permissions() {
        let coordinator = MemoryCoordinator::new(100);
        coordinator.start().await.unwrap();
        
        let owner = AgentId::new();
        let other_agent = AgentId::new();
        let data = b"test data".to_vec();
        
        // Create region
        let region_id = coordinator.create_region(
            "Test Region".to_string(),
            "Test Description".to_string(),
            owner,
            data.clone(),
        ).await.unwrap();
        
        // Other agent should not have access
        let result = coordinator.get_region(region_id, other_agent).await;
        assert!(result.is_err());
        
        // Grant read permission
        coordinator.grant_permission(region_id, owner, other_agent, MemoryPermission::Read).await.unwrap();
        
        // Other agent should now have read access
        let region = coordinator.get_region(region_id, other_agent).await.unwrap();
        assert_eq!(region.data, data);
        
        // Other agent should not have write access
        let result = coordinator.update_region(region_id, other_agent, b"new data".to_vec(), None).await;
        assert!(result.is_err());
        
        // Grant write permission
        coordinator.grant_permission(region_id, owner, other_agent, MemoryPermission::ReadWrite).await.unwrap();
        
        // Other agent should now have write access
        let new_data = b"new data".to_vec();
        coordinator.update_region(region_id, other_agent, new_data.clone(), None).await.unwrap();
        
        let updated_region = coordinator.get_region(region_id, other_agent).await.unwrap();
        assert_eq!(updated_region.data, new_data);
        
        // Revoke permission
        coordinator.revoke_permission(region_id, owner, other_agent).await.unwrap();
        
        // Other agent should no longer have access
        let result = coordinator.get_region(region_id, other_agent).await;
        assert!(result.is_err());
    }
    
    #[tokio::test]
    async fn test_memory_events() {
        let coordinator = MemoryCoordinator::new(100);
        coordinator.start().await.unwrap();
        
        let mut event_receiver = coordinator.subscribe_events().await.unwrap();
        
        let owner = AgentId::new();
        let data = b"test data".to_vec();
        
        // Create region
        let region_id = coordinator.create_region(
            "Test Region".to_string(),
            "Test Description".to_string(),
            owner,
            data.clone(),
        ).await.unwrap();
        
        // Check for creation event
        let event = event_receiver.recv().await.unwrap();
        match event {
            MemoryEvent::RegionCreated { region_id: id, owner: o, name } => {
                assert_eq!(id, region_id);
                assert_eq!(o, owner);
                assert_eq!(name, "Test Region");
            }
            _ => panic!("Expected RegionCreated event"),
        }
        
        // Update region
        coordinator.update_region(region_id, owner, b"updated data".to_vec(), None).await.unwrap();
        
        // Check for update event
        let event = event_receiver.recv().await.unwrap();
        match event {
            MemoryEvent::RegionUpdated { region_id: id, updater, version } => {
                assert_eq!(id, region_id);
                assert_eq!(updater, owner);
                assert_eq!(version, 2);
            }
            _ => panic!("Expected RegionUpdated event"),
        }
    }
    
    #[tokio::test]
    async fn test_optimistic_locking() {
        let coordinator = MemoryCoordinator::new(100);
        coordinator.start().await.unwrap();
        
        let owner = AgentId::new();
        let data = b"test data".to_vec();
        
        // Create region
        let region_id = coordinator.create_region(
            "Test Region".to_string(),
            "Test Description".to_string(),
            owner,
            data.clone(),
        ).await.unwrap();
        
        // Update with correct version
        coordinator.update_region(region_id, owner, b"updated data".to_vec(), Some(1)).await.unwrap();
        
        // Try to update with old version (should fail)
        let result = coordinator.update_region(region_id, owner, b"another update".to_vec(), Some(1)).await;
        assert!(result.is_err());
        
        // Update with correct version
        coordinator.update_region(region_id, owner, b"final update".to_vec(), Some(2)).await.unwrap();
        
        let region = coordinator.get_region(region_id, owner).await.unwrap();
        assert_eq!(region.data, b"final update");
        assert_eq!(region.metadata.version, 3);
    }
    
    #[tokio::test]
    async fn test_memory_statistics() {
        let coordinator = MemoryCoordinator::new(100);
        coordinator.start().await.unwrap();
        
        let owner = AgentId::new();
        let data = b"test data".to_vec();
        
        // Create multiple regions
        for i in 0..5 {
            coordinator.create_region(
                format!("Region {}", i),
                format!("Description {}", i),
                owner,
                data.clone(),
            ).await.unwrap();
        }
        
        // Wait for statistics to update
        sleep(Duration::from_millis(100)).await;
        
        let stats = coordinator.get_memory_stats().await.unwrap();
        assert_eq!(stats.total_regions, 5);
        assert_eq!(stats.active_regions, 5);
        assert_eq!(stats.total_memory_used, 5 * data.len() as u64);
        assert!(stats.memory_by_agent.contains_key(&owner));
    }
}