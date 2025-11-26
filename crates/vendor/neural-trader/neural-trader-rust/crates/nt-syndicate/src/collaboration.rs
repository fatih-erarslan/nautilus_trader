//! Collaboration tools for syndicate communication

use dashmap::DashMap;
use napi_derive::napi;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Collaboration hub for syndicate communication
#[napi]
#[derive(Clone)]
pub struct CollaborationHub {
    syndicate_id: String,
    messages: Arc<DashMap<String, Message>>,
    channels: Arc<DashMap<String, Channel>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Message {
    id: Uuid,
    channel_id: String,
    author_id: Uuid,
    content: String,
    timestamp: DateTime<Utc>,
    message_type: String,
    attachments: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Channel {
    id: String,
    name: String,
    description: String,
    channel_type: String,
    members: Vec<String>,
    created_at: DateTime<Utc>,
}

#[napi]
impl CollaborationHub {
    /// Create new collaboration hub
    #[napi(constructor)]
    pub fn new(syndicate_id: String) -> Self {
        Self {
            syndicate_id,
            messages: Arc::new(DashMap::new()),
            channels: Arc::new(DashMap::new()),
        }
    }

    /// Create a new channel
    #[napi]
    pub fn create_channel(
        &self,
        name: String,
        description: String,
        channel_type: String,
    ) -> napi::Result<String> {
        let channel_id = format!("{}-{}", &name.to_lowercase().replace(" ", "-"), Uuid::new_v4());

        let channel = Channel {
            id: channel_id.clone(),
            name,
            description,
            channel_type,
            members: Vec::new(),
            created_at: Utc::now(),
        };

        self.channels.insert(channel_id.clone(), channel);

        Ok(channel_id)
    }

    /// Add member to channel
    #[napi]
    pub fn add_member_to_channel(&self, channel_id: String, member_id: String) -> napi::Result<()> {
        let mut channel = self.channels
            .get_mut(&channel_id)
            .ok_or_else(|| napi::Error::from_reason("Channel not found"))?;

        if !channel.members.contains(&member_id) {
            channel.members.push(member_id);
        }

        Ok(())
    }

    /// Post message to channel
    #[napi]
    pub fn post_message(
        &self,
        channel_id: String,
        author_id: String,
        content: String,
        message_type: String,
        attachments: Vec<String>,
    ) -> napi::Result<String> {
        let author_uuid = Uuid::parse_str(&author_id)
            .map_err(|e| napi::Error::from_reason(format!("Invalid author ID: {}", e)))?;

        let message_id = Uuid::new_v4();

        let message = Message {
            id: message_id,
            channel_id: channel_id.clone(),
            author_id: author_uuid,
            content,
            timestamp: Utc::now(),
            message_type,
            attachments,
        };

        self.messages.insert(message_id.to_string(), message);

        Ok(message_id.to_string())
    }

    /// Get channel messages
    #[napi]
    pub fn get_channel_messages(&self, channel_id: String, limit: Option<u32>) -> napi::Result<String> {
        let limit = limit.unwrap_or(50) as usize;

        let mut messages: Vec<_> = self.messages
            .iter()
            .filter(|entry| entry.value().channel_id == channel_id)
            .map(|entry| entry.value().clone())
            .collect();

        messages.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        messages.truncate(limit);

        serde_json::to_string(&messages)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// List all channels
    #[napi]
    pub fn list_channels(&self) -> napi::Result<String> {
        let channels: Vec<_> = self.channels
            .iter()
            .map(|entry| entry.value().clone())
            .collect();

        serde_json::to_string(&channels)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Get channel details
    #[napi]
    pub fn get_channel_details(&self, channel_id: String) -> napi::Result<String> {
        let channel = self.channels
            .get(&channel_id)
            .ok_or_else(|| napi::Error::from_reason("Channel not found"))?;

        serde_json::to_string(&*channel)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collaboration_hub() {
        let hub = CollaborationHub::new("test-syndicate".to_string());
        let result = hub.create_channel(
            "General".to_string(),
            "General discussion".to_string(),
            "public".to_string(),
        );

        assert!(result.is_ok());
    }
}
