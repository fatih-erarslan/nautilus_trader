"""
Collaboration System for Sports Betting Syndicates

This module provides comprehensive collaboration functionality including:
- Shared research workspace and document management
- Real-time chat and communication channels
- Collaborative bet analysis and decision making
- Strategy sharing and backtesting collaboration
- Knowledge base and resource library
- Team coordination and project management
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, field
import json
import uuid

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Types of collaborative documents"""
    RESEARCH_REPORT = "research_report"
    BET_ANALYSIS = "bet_analysis"
    STRATEGY_GUIDE = "strategy_guide"
    MARKET_ANALYSIS = "market_analysis"
    GAME_PREVIEW = "game_preview"
    POST_MORTEM = "post_mortem"
    TRAINING_MATERIAL = "training_material"
    MEETING_NOTES = "meeting_notes"
    PROPOSAL_DRAFT = "proposal_draft"


class ChannelType(Enum):
    """Types of communication channels"""
    GENERAL = "general"
    RESEARCH = "research"
    STRATEGY = "strategy"
    ALERTS = "alerts"
    PRIVATE = "private"
    PROJECT = "project"
    ANNOUNCEMENT = "announcement"


class MessageType(Enum):
    """Types of messages"""
    TEXT = "text"
    DOCUMENT = "document"
    BET_RECOMMENDATION = "bet_recommendation"
    ANALYSIS = "analysis"
    ALERT = "alert"
    POLL = "poll"
    FILE_ATTACHMENT = "file_attachment"
    SYSTEM = "system"


class ProjectStatus(Enum):
    """Status of collaborative projects"""
    PLANNING = "planning"
    ACTIVE = "active"
    REVIEW = "review"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ON_HOLD = "on_hold"


@dataclass
class Message:
    """Represents a message in a communication channel"""
    message_id: str
    channel_id: str
    sender_id: str
    message_type: MessageType
    content: str
    timestamp: datetime
    edited_at: Optional[datetime] = None
    
    # Rich content
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    mentions: Set[str] = field(default_factory=set)
    reactions: Dict[str, Set[str]] = field(default_factory=dict)  # emoji -> set of member_ids
    
    # Threading
    reply_to: Optional[str] = None
    thread_replies: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    pinned: bool = False
    deleted: bool = False


@dataclass
class Channel:
    """Represents a communication channel"""
    channel_id: str
    name: str
    description: str
    channel_type: ChannelType
    created_by: str
    created_at: datetime
    
    # Access control
    members: Set[str] = field(default_factory=set)
    admins: Set[str] = field(default_factory=set)
    private: bool = False
    
    # Settings
    notifications_enabled: bool = True
    message_retention_days: int = 365
    allow_file_uploads: bool = True
    
    # Statistics
    message_count: int = 0
    last_activity: datetime = field(default_factory=datetime.now)
    
    # Messages (in-memory for this example, would use database in production)
    messages: List[Message] = field(default_factory=list)


@dataclass
class Document:
    """Represents a collaborative document"""
    document_id: str
    title: str
    content: str
    document_type: DocumentType
    created_by: str
    created_at: datetime
    
    # Version control
    version: int = 1
    last_modified: datetime = field(default_factory=datetime.now)
    last_modified_by: str = ""
    
    # Collaboration
    collaborators: Set[str] = field(default_factory=set)
    editors: Set[str] = field(default_factory=set)
    viewers: Set[str] = field(default_factory=set)
    
    # Content metadata
    tags: Set[str] = field(default_factory=set)
    related_bets: List[str] = field(default_factory=list)
    related_games: List[str] = field(default_factory=list)
    
    # Review and approval
    review_required: bool = False
    reviewed_by: Set[str] = field(default_factory=set)
    approved: bool = False
    approved_by: Optional[str] = None
    
    # Analytics
    view_count: int = 0
    comment_count: int = 0
    
    # Change history
    change_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Project:
    """Represents a collaborative project"""
    project_id: str
    name: str
    description: str
    project_type: str  # research, strategy_development, analysis, etc.
    status: ProjectStatus
    
    # Team
    lead_id: str
    members: Set[str] = field(default_factory=set)
    
    # Dates
    created_at: datetime = field(default_factory=datetime.now)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    deadline: Optional[datetime] = None
    
    # Progress
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    milestones: List[Dict[str, Any]] = field(default_factory=list)
    completion_percentage: float = 0.0
    
    # Resources
    documents: Set[str] = field(default_factory=set)
    channels: Set[str] = field(default_factory=set)
    budget: Optional[Decimal] = None
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    priority: str = "medium"  # low, medium, high, critical


@dataclass
class KnowledgeItem:
    """Represents an item in the knowledge base"""
    item_id: str
    title: str
    content: str
    category: str
    subcategory: str
    
    # Authorship
    created_by: str
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    difficulty_level: str = "beginner"  # beginner, intermediate, advanced
    estimated_read_time: int = 5  # minutes
    
    # Engagement
    views: int = 0
    upvotes: int = 0
    downvotes: int = 0
    bookmarks: Set[str] = field(default_factory=set)
    
    # Content structure
    sections: List[Dict[str, Any]] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    attachments: List[Dict[str, Any]] = field(default_factory=list)


class CollaborationManager:
    """
    Advanced collaboration system for syndicate coordination
    
    Features:
    - Multi-channel real-time communication
    - Collaborative document editing and version control
    - Project management and task coordination
    - Knowledge base and resource sharing
    - Team analytics and performance insights
    - Integration with betting and analysis tools
    """
    
    def __init__(self, syndicate_id: str):
        self.syndicate_id = syndicate_id
        
        # Core components
        self.channels: Dict[str, Channel] = {}
        self.documents: Dict[str, Document] = {}
        self.projects: Dict[str, Project] = {}
        self.knowledge_base: Dict[str, KnowledgeItem] = {}
        
        # Real-time subscribers (for real implementation, use WebSocket/Redis)
        self.channel_subscribers: Dict[str, Set[str]] = {}
        self.document_subscribers: Dict[str, Set[str]] = {}
        
        # Analytics
        self.collaboration_metrics: Dict[str, Any] = {}
        self.activity_feed: List[Dict[str, Any]] = []
        
        # Initialize default channels
        self._initialize_default_channels()
        
        logger.info(f"Initialized CollaborationManager for syndicate {syndicate_id}")

    def _initialize_default_channels(self):
        """Create default communication channels"""
        default_channels = [
            ("general", "General Discussion", ChannelType.GENERAL),
            ("research", "Research & Analysis", ChannelType.RESEARCH),
            ("strategy", "Strategy Development", ChannelType.STRATEGY),
            ("alerts", "Important Alerts", ChannelType.ALERTS),
            ("announcements", "Official Announcements", ChannelType.ANNOUNCEMENT)
        ]
        
        for name, description, channel_type in default_channels:
            channel_id = f"CHAN_{name.upper()}_{uuid.uuid4().hex[:8]}"
            
            channel = Channel(
                channel_id=channel_id,
                name=name,
                description=description,
                channel_type=channel_type,
                created_by="system",
                created_at=datetime.now(),
                private=False
            )
            
            self.channels[channel_id] = channel
            self.channel_subscribers[channel_id] = set()

    async def create_channel(self, creator_id: str, name: str, description: str,
                           channel_type: ChannelType, private: bool = False) -> str:
        """Create a new communication channel"""
        try:
            # Check if name already exists
            existing_names = [c.name.lower() for c in self.channels.values()]
            if name.lower() in existing_names:
                raise ValueError(f"Channel name '{name}' already exists")
            
            channel_id = f"CHAN_{name.upper().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
            
            channel = Channel(
                channel_id=channel_id,
                name=name,
                description=description,
                channel_type=channel_type,
                created_by=creator_id,
                created_at=datetime.now(),
                private=private
            )
            
            # Creator is automatically admin
            channel.admins.add(creator_id)
            channel.members.add(creator_id)
            
            self.channels[channel_id] = channel
            self.channel_subscribers[channel_id] = set()
            
            # Log activity
            await self._log_activity("channel_created", {
                "channel_id": channel_id,
                "creator_id": creator_id,
                "name": name,
                "type": channel_type.value,
                "private": private
            })
            
            logger.info(f"Created channel {name} ({channel_id}) by {creator_id}")
            return channel_id
            
        except Exception as e:
            logger.error(f"Error creating channel: {e}")
            raise

    async def send_message(self, channel_id: str, sender_id: str, content: str,
                          message_type: MessageType = MessageType.TEXT,
                          attachments: Optional[List[Dict[str, Any]]] = None,
                          reply_to: Optional[str] = None) -> str:
        """Send a message to a channel"""
        try:
            if channel_id not in self.channels:
                raise ValueError(f"Channel {channel_id} not found")
            
            channel = self.channels[channel_id]
            
            # Check if sender has access
            if channel.private and sender_id not in channel.members:
                raise ValueError("No access to private channel")
            
            message_id = f"MSG_{uuid.uuid4().hex[:12]}"
            
            # Extract mentions from content
            mentions = set()
            words = content.split()
            for word in words:
                if word.startswith('@') and len(word) > 1:
                    mentions.add(word[1:])
            
            message = Message(
                message_id=message_id,
                channel_id=channel_id,
                sender_id=sender_id,
                message_type=message_type,
                content=content,
                timestamp=datetime.now(),
                attachments=attachments or [],
                mentions=mentions,
                reply_to=reply_to
            )
            
            # Add to channel
            channel.messages.append(message)
            channel.message_count += 1
            channel.last_activity = datetime.now()
            
            # Handle threading
            if reply_to:
                parent_message = next((m for m in channel.messages if m.message_id == reply_to), None)
                if parent_message:
                    parent_message.thread_replies.append(message_id)
            
            # Notify subscribers (in real implementation, use WebSocket/push notifications)
            await self._notify_channel_subscribers(channel_id, message)
            
            # Log activity
            await self._log_activity("message_sent", {
                "channel_id": channel_id,
                "sender_id": sender_id,
                "message_id": message_id,
                "type": message_type.value
            })
            
            logger.debug(f"Message sent to {channel_id} by {sender_id}")
            return message_id
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise

    async def create_document(self, creator_id: str, title: str, content: str,
                            document_type: DocumentType, collaborators: Optional[Set[str]] = None) -> str:
        """Create a new collaborative document"""
        try:
            document_id = f"DOC_{document_type.value.upper()}_{uuid.uuid4().hex[:8]}"
            
            document = Document(
                document_id=document_id,
                title=title,
                content=content,
                document_type=document_type,
                created_by=creator_id,
                created_at=datetime.now(),
                last_modified_by=creator_id
            )
            
            # Set permissions
            document.editors.add(creator_id)
            if collaborators:
                document.collaborators.update(collaborators)
                document.viewers.update(collaborators)
            
            # Initial change history entry
            document.change_history.append({
                "timestamp": datetime.now(),
                "user_id": creator_id,
                "action": "created",
                "version": 1,
                "summary": "Document created"
            })
            
            self.documents[document_id] = document
            self.document_subscribers[document_id] = set()
            
            # Log activity
            await self._log_activity("document_created", {
                "document_id": document_id,
                "creator_id": creator_id,
                "title": title,
                "type": document_type.value
            })
            
            logger.info(f"Created document {title} ({document_id}) by {creator_id}")
            return document_id
            
        except Exception as e:
            logger.error(f"Error creating document: {e}")
            raise

    async def edit_document(self, document_id: str, editor_id: str, new_content: str,
                          summary: str = "") -> bool:
        """Edit a collaborative document"""
        try:
            if document_id not in self.documents:
                raise ValueError(f"Document {document_id} not found")
            
            document = self.documents[document_id]
            
            # Check permissions
            if editor_id not in document.editors and editor_id not in document.collaborators:
                raise ValueError("No edit permission for this document")
            
            # Save previous version info
            old_content = document.content
            old_version = document.version
            
            # Update document
            document.content = new_content
            document.version += 1
            document.last_modified = datetime.now()
            document.last_modified_by = editor_id
            
            # Record change
            document.change_history.append({
                "timestamp": datetime.now(),
                "user_id": editor_id,
                "action": "edited",
                "version": document.version,
                "summary": summary or f"Content updated by {editor_id}",
                "changes": {
                    "content_length_before": len(old_content),
                    "content_length_after": len(new_content)
                }
            })
            
            # Notify subscribers
            await self._notify_document_subscribers(document_id, "document_updated", {
                "editor_id": editor_id,
                "version": document.version,
                "summary": summary
            })
            
            # Log activity
            await self._log_activity("document_edited", {
                "document_id": document_id,
                "editor_id": editor_id,
                "version": document.version,
                "summary": summary
            })
            
            logger.info(f"Document {document_id} edited by {editor_id} (v{document.version})")
            return True
            
        except Exception as e:
            logger.error(f"Error editing document {document_id}: {e}")
            return False

    async def create_project(self, lead_id: str, name: str, description: str,
                           project_type: str, deadline: Optional[datetime] = None) -> str:
        """Create a new collaborative project"""
        try:
            project_id = f"PROJ_{project_type.upper()}_{uuid.uuid4().hex[:8]}"
            
            project = Project(
                project_id=project_id,
                name=name,
                description=description,
                project_type=project_type,
                status=ProjectStatus.PLANNING,
                lead_id=lead_id,
                deadline=deadline
            )
            
            # Add lead as member
            project.members.add(lead_id)
            
            self.projects[project_id] = project
            
            # Create dedicated project channel
            channel_id = await self.create_channel(
                creator_id=lead_id,
                name=f"Project: {name}",
                description=f"Discussion for project: {description}",
                channel_type=ChannelType.PROJECT,
                private=True
            )
            
            project.channels.add(channel_id)
            
            # Log activity
            await self._log_activity("project_created", {
                "project_id": project_id,
                "lead_id": lead_id,
                "name": name,
                "type": project_type
            })
            
            logger.info(f"Created project {name} ({project_id}) led by {lead_id}")
            return project_id
            
        except Exception as e:
            logger.error(f"Error creating project: {e}")
            raise

    async def add_project_task(self, project_id: str, task_name: str, description: str,
                             assigned_to: Optional[str] = None, due_date: Optional[datetime] = None) -> str:
        """Add a task to a project"""
        try:
            if project_id not in self.projects:
                raise ValueError(f"Project {project_id} not found")
            
            project = self.projects[project_id]
            task_id = f"TASK_{uuid.uuid4().hex[:8]}"
            
            task = {
                "task_id": task_id,
                "name": task_name,
                "description": description,
                "assigned_to": assigned_to,
                "created_at": datetime.now(),
                "due_date": due_date,
                "status": "pending",  # pending, in_progress, completed, cancelled
                "completion_date": None,
                "notes": []
            }
            
            project.tasks.append(task)
            
            # Update project completion percentage
            await self._update_project_completion(project_id)
            
            # Log activity
            await self._log_activity("task_added", {
                "project_id": project_id,
                "task_id": task_id,
                "task_name": task_name,
                "assigned_to": assigned_to
            })
            
            logger.info(f"Added task {task_name} to project {project_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"Error adding task to project {project_id}: {e}")
            raise

    async def update_task_status(self, project_id: str, task_id: str, status: str,
                               notes: str = "") -> bool:
        """Update the status of a project task"""
        try:
            if project_id not in self.projects:
                raise ValueError(f"Project {project_id} not found")
            
            project = self.projects[project_id]
            task = next((t for t in project.tasks if t["task_id"] == task_id), None)
            
            if not task:
                raise ValueError(f"Task {task_id} not found in project {project_id}")
            
            old_status = task["status"]
            task["status"] = status
            
            if status == "completed":
                task["completion_date"] = datetime.now()
            
            if notes:
                task["notes"].append({
                    "timestamp": datetime.now(),
                    "note": notes
                })
            
            # Update project completion
            await self._update_project_completion(project_id)
            
            # Log activity
            await self._log_activity("task_status_updated", {
                "project_id": project_id,
                "task_id": task_id,
                "old_status": old_status,
                "new_status": status,
                "notes": notes
            })
            
            logger.info(f"Updated task {task_id} status from {old_status} to {status}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating task status: {e}")
            return False

    async def _update_project_completion(self, project_id: str):
        """Update project completion percentage based on task completion"""
        project = self.projects[project_id]
        
        if not project.tasks:
            project.completion_percentage = 0.0
            return
        
        completed_tasks = len([t for t in project.tasks if t["status"] == "completed"])
        project.completion_percentage = (completed_tasks / len(project.tasks)) * 100
        
        # Update project status based on completion
        if project.completion_percentage == 100:
            project.status = ProjectStatus.COMPLETED
        elif project.completion_percentage > 0:
            project.status = ProjectStatus.ACTIVE

    async def add_knowledge_item(self, creator_id: str, title: str, content: str,
                               category: str, subcategory: str = "") -> str:
        """Add an item to the knowledge base"""
        try:
            item_id = f"KB_{category.upper()}_{uuid.uuid4().hex[:8]}"
            
            # Estimate reading time (average 200 words per minute)
            word_count = len(content.split())
            estimated_read_time = max(1, word_count // 200)
            
            knowledge_item = KnowledgeItem(
                item_id=item_id,
                title=title,
                content=content,
                category=category,
                subcategory=subcategory,
                created_by=creator_id,
                estimated_read_time=estimated_read_time
            )
            
            self.knowledge_base[item_id] = knowledge_item
            
            # Log activity
            await self._log_activity("knowledge_item_added", {
                "item_id": item_id,
                "creator_id": creator_id,
                "title": title,
                "category": category
            })
            
            logger.info(f"Added knowledge item {title} ({item_id}) by {creator_id}")
            return item_id
            
        except Exception as e:
            logger.error(f"Error adding knowledge item: {e}")
            raise

    async def search_knowledge_base(self, query: str, category: Optional[str] = None,
                                  difficulty: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search the knowledge base"""
        results = []
        query_lower = query.lower()
        
        for item in self.knowledge_base.values():
            # Check category filter
            if category and item.category.lower() != category.lower():
                continue
            
            # Check difficulty filter
            if difficulty and item.difficulty_level != difficulty:
                continue
            
            # Calculate relevance score
            relevance_score = 0
            
            if query_lower in item.title.lower():
                relevance_score += 10
            if query_lower in item.content.lower():
                relevance_score += 5
            if any(query_lower in tag.lower() for tag in item.tags):
                relevance_score += 3
            
            if relevance_score > 0:
                results.append({
                    "item_id": item.item_id,
                    "title": item.title,
                    "category": item.category,
                    "subcategory": item.subcategory,
                    "difficulty_level": item.difficulty_level,
                    "estimated_read_time": item.estimated_read_time,
                    "views": item.views,
                    "upvotes": item.upvotes,
                    "relevance_score": relevance_score,
                    "created_by": item.created_by,
                    "created_at": item.created_at.isoformat()
                })
        
        # Sort by relevance score
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return results

    async def _notify_channel_subscribers(self, channel_id: str, message: Message):
        """Notify channel subscribers of new message (placeholder for real-time notifications)"""
        subscribers = self.channel_subscribers.get(channel_id, set())
        
        # In a real implementation, this would send WebSocket/push notifications
        notification_data = {
            "type": "new_message",
            "channel_id": channel_id,
            "message_id": message.message_id,
            "sender_id": message.sender_id,
            "content_preview": message.content[:100] if len(message.content) > 100 else message.content,
            "timestamp": message.timestamp.isoformat()
        }
        
        logger.debug(f"Would notify {len(subscribers)} subscribers for channel {channel_id}")

    async def _notify_document_subscribers(self, document_id: str, event_type: str, data: Dict[str, Any]):
        """Notify document subscribers of changes"""
        subscribers = self.document_subscribers.get(document_id, set())
        
        notification_data = {
            "type": event_type,
            "document_id": document_id,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        logger.debug(f"Would notify {len(subscribers)} subscribers for document {document_id}")

    async def _log_activity(self, activity_type: str, data: Dict[str, Any]):
        """Log collaboration activity"""
        activity = {
            "timestamp": datetime.now(),
            "activity_type": activity_type,
            "syndicate_id": self.syndicate_id,
            "data": data
        }
        
        self.activity_feed.append(activity)
        
        # Keep only last 1000 activities
        if len(self.activity_feed) > 1000:
            self.activity_feed = self.activity_feed[-1000:]

    def get_channel_messages(self, channel_id: str, limit: int = 50,
                           before: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get messages from a channel"""
        if channel_id not in self.channels:
            raise ValueError(f"Channel {channel_id} not found")
        
        channel = self.channels[channel_id]
        messages = channel.messages
        
        # Filter by timestamp if provided
        if before:
            messages = [m for m in messages if m.timestamp < before]
        
        # Sort by timestamp (newest first) and limit
        messages.sort(key=lambda x: x.timestamp, reverse=True)
        messages = messages[:limit]
        
        return [
            {
                "message_id": m.message_id,
                "sender_id": m.sender_id,
                "content": m.content,
                "message_type": m.message_type.value,
                "timestamp": m.timestamp.isoformat(),
                "edited_at": m.edited_at.isoformat() if m.edited_at else None,
                "reply_to": m.reply_to,
                "attachments": m.attachments,
                "mentions": list(m.mentions),
                "reactions": {emoji: list(users) for emoji, users in m.reactions.items()},
                "thread_replies_count": len(m.thread_replies),
                "pinned": m.pinned
            }
            for m in messages if not m.deleted
        ]

    def get_collaboration_analytics(self) -> Dict[str, Any]:
        """Get comprehensive collaboration analytics"""
        # Channel statistics
        total_channels = len(self.channels)
        total_messages = sum(c.message_count for c in self.channels.values())
        active_channels = len([c for c in self.channels.values() 
                             if c.last_activity >= datetime.now() - timedelta(days=7)])
        
        # Document statistics
        total_documents = len(self.documents)
        documents_by_type = {}
        for doc_type in DocumentType:
            documents_by_type[doc_type.value] = len([d for d in self.documents.values() 
                                                   if d.document_type == doc_type])
        
        # Project statistics
        total_projects = len(self.projects)
        projects_by_status = {}
        for status in ProjectStatus:
            projects_by_status[status.value] = len([p for p in self.projects.values() 
                                                  if p.status == status])
        
        # Knowledge base statistics
        total_kb_items = len(self.knowledge_base)
        kb_by_category = {}
        for item in self.knowledge_base.values():
            category = item.category
            kb_by_category[category] = kb_by_category.get(category, 0) + 1
        
        # Activity statistics
        recent_activity = len([a for a in self.activity_feed 
                             if a["timestamp"] >= datetime.now() - timedelta(days=7)])
        
        return {
            "syndicate_id": self.syndicate_id,
            "communication": {
                "total_channels": total_channels,
                "active_channels": active_channels,
                "total_messages": total_messages,
                "messages_last_week": len([m for c in self.channels.values() 
                                         for m in c.messages 
                                         if m.timestamp >= datetime.now() - timedelta(days=7)])
            },
            "documents": {
                "total_documents": total_documents,
                "documents_by_type": documents_by_type,
                "total_views": sum(d.view_count for d in self.documents.values()),
                "collaborative_documents": len([d for d in self.documents.values() 
                                              if len(d.collaborators) > 1])
            },
            "projects": {
                "total_projects": total_projects,
                "projects_by_status": projects_by_status,
                "average_completion": sum(p.completion_percentage for p in self.projects.values()) / len(self.projects) if self.projects else 0,
                "total_tasks": sum(len(p.tasks) for p in self.projects.values())
            },
            "knowledge_base": {
                "total_items": total_kb_items,
                "items_by_category": kb_by_category,
                "total_views": sum(item.views for item in self.knowledge_base.values()),
                "average_rating": sum(item.upvotes - item.downvotes for item in self.knowledge_base.values()) / len(self.knowledge_base) if self.knowledge_base else 0
            },
            "activity": {
                "total_activities": len(self.activity_feed),
                "recent_activity": recent_activity,
                "activity_types": {}
            }
        }