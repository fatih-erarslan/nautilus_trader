"""Base classes for news parsing - GREEN phase"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

from .models import ParsedArticle

logger = logging.getLogger(__name__)


class NewsParser(ABC):
    """Abstract base class for news parsers"""
    
    def __init__(self):
        """Initialize parser with default configuration"""
        self.entity_extractors = []
        self.event_detector = None
        self.temporal_extractor = None
        
    @abstractmethod
    async def parse(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ParsedArticle:
        """
        Parse news content into structured format
        
        Args:
            content: Raw news content text
            metadata: Optional metadata about the article
            
        Returns:
            ParsedArticle with extracted information
        """
        pass
        
    @abstractmethod
    async def parse_batch(self, articles: List[str]) -> List[ParsedArticle]:
        """
        Parse multiple articles efficiently
        
        Args:
            articles: List of article contents
            
        Returns:
            List of ParsedArticle objects
        """
        pass
    
    def set_entity_extractors(self, extractors: List['EntityExtractor']) -> None:
        """Configure entity extractors"""
        self.entity_extractors = extractors
        logger.info(f"Configured {len(extractors)} entity extractors")
        
    def set_event_detector(self, detector: 'EventDetector') -> None:
        """Configure event detector"""
        self.event_detector = detector
        logger.info("Configured event detector")
    
    def set_temporal_extractor(self, extractor: 'TemporalExtractor') -> None:
        """Configure temporal extractor"""
        self.temporal_extractor = extractor
        logger.info("Configured temporal extractor")


class EntityExtractor(ABC):
    """Base class for entity extractors"""
    
    @abstractmethod
    def extract(self, text: str) -> List['Entity']:
        """
        Extract entities from text
        
        Args:
            text: Input text
            
        Returns:
            List of Entity objects
        """
        pass
        
    @abstractmethod
    def get_supported_types(self) -> List['EntityType']:
        """
        Get entity types this extractor supports
        
        Returns:
            List of supported EntityType values
        """
        pass