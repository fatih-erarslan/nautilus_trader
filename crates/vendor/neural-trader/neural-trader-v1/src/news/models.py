"""
News data models for the trading platform
"""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any


@dataclass
class NewsItem:
    """Data model for a news article"""
    id: str
    title: str
    content: str
    source: str
    timestamp: datetime
    url: str
    entities: List[str]
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate NewsItem data after initialization"""
        if not isinstance(self.timestamp, datetime):
            raise ValueError("timestamp must be a datetime object")
        
        if not isinstance(self.entities, list):
            raise ValueError("entities must be a list")
        
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")