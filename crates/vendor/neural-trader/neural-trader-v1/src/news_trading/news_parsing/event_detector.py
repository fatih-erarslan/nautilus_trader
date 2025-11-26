"""Event detection for news parsing - GREEN phase"""

import re
from typing import List, Dict, Set, Optional
import logging

from .models import Event, EventType
from .extractors import UnifiedEntityExtractor

logger = logging.getLogger(__name__)


class EventDetector:
    """Detect events in news text"""
    
    def __init__(self):
        self.entity_extractor = UnifiedEntityExtractor()
        
        # Event patterns organized by type
        self.event_patterns = {
            EventType.PRICE_MOVEMENT: {
                'surge': r'\b(surg\w+|soar\w+|jump\w+|spike\w+|rally\w+|moon\w+)\b',
                'crash': r'\b(crash\w+|plummet\w+|plunge\w+|tumbl\w+|tank\w+|collapse\w+)\b',
                'rise': r'\b(ris\w+|climb\w+|gain\w+|advanc\w+|increas\w+)\b',
                'fall': r'\b(fall\w+|drop\w+|declin\w+|decreas\w+|slip\w+)\b',
                'break': r'\b(break\w+|breach\w+|cross\w+)\b.{0,20}(resistance|support|level)',
                'hit': r'\b(hit\w+|reach\w+|touch\w+)\b.{0,20}(high|low|ATH|ATL)',
            },
            EventType.REGULATORY: {
                'approval': r'\b(approv\w+|authoriz\w+|permit\w+|clear\w+|green.?light)\b',
                'ban': r'\b(ban\w+|prohibit\w+|outlaw\w+|block\w+|restrict\w+)\b',
                'regulation': r'\b(regulat\w+|SEC|CFTC|compliance|oversight|framework|Federal Reserve.*meeting)\b',
                'investigation': r'\b(investigat\w+|probe|inquiry|scrutin\w+)\b',
                'lawsuit': r'\b(lawsuit|sue\w+|litigation|legal action)\b',
            },
            EventType.PARTNERSHIP: {
                'partner': r'\b(partner\w+|collaborat\w+|team\w* up|join\w* forces)\b',
                'integration': r'\b(integrat\w+|implement\w+|adopt\w+)\b',
                'agreement': r'\b(agreement|deal|contract|arrangement)\b',
            },
            EventType.PRODUCT_LAUNCH: {
                'launch': r'\b(launch\w+|releas\w+|unveil\w+|introduc\w+|roll\w* out)\b',
                'announce': r'\b(announc\w+|reveal\w+)\b.{0,20}(product|platform|feature|service)',
                'upgrade': r'\b(upgrad\w+|updat\w+|enhanc\w+|improv\w+)\b',
            },
            EventType.SECURITY_BREACH: {
                'hack': r'\b(hack\w+|breach\w+|exploit\w+|attack\w+)\b',
                'theft': r'\b(stole\w+|theft|rob\w+)\b',
                'vulnerability': r'\b(vulnerabilit\w+|flaw|bug|weakness)\b',
                'security': r'\b(security.{0,20}(breach|incident|issue))\b',
            }
        }
        
    def detect(self, text: str) -> List[Event]:
        """Detect events in text"""
        events = []
        
        # Extract entities first for context
        entities = self.entity_extractor.extract(text)
        entity_names = [e.text for e in entities]
        
        for event_type, patterns in self.event_patterns.items():
            for description, pattern in patterns.items():
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                
                if matches:
                    # Calculate confidence based on match strength
                    confidence = self._calculate_confidence(text, pattern, matches)
                    
                    # Find entities involved
                    entities_involved = self._find_entities_involved(
                        text, matches[0], entity_names
                    )
                    
                    events.append(Event(
                        type=event_type,
                        description=description,
                        confidence=confidence,
                        entities_involved=entities_involved,
                        metadata={
                            "match_count": len(matches),
                            "match_text": matches[0].group()
                        }
                    ))
        
        # Remove duplicate events
        return self._deduplicate_events(events)
    
    def _calculate_confidence(self, text: str, pattern: str, matches: List) -> float:
        """Calculate confidence score for an event"""
        base_confidence = 0.7
        
        # Boost for multiple matches
        if len(matches) > 1:
            base_confidence += 0.1
        
        # Boost for strong signal words
        strong_signals = ['definitely', 'certainly', 'massive', 'significant', 'major']
        if any(signal in text.lower() for signal in strong_signals):
            base_confidence += 0.1
        
        # Reduce for hedge words
        hedge_words = ['might', 'could', 'possibly', 'perhaps', 'maybe']
        if any(hedge in text.lower() for hedge in hedge_words):
            base_confidence -= 0.2
        
        return min(0.95, max(0.3, base_confidence))
    
    def _find_entities_involved(self, text: str, match: re.Match, 
                               entity_names: List[str]) -> List[str]:
        """Find entities involved in an event"""
        entities_involved = []
        
        # Look for entities near the event match (within 50 characters)
        start = max(0, match.start() - 50)
        end = min(len(text), match.end() + 50)
        context = text[start:end]
        
        for entity in entity_names:
            if entity in context:
                entities_involved.append(entity)
        
        return entities_involved
    
    def _deduplicate_events(self, events: List[Event]) -> List[Event]:
        """Remove duplicate events"""
        unique_events = {}
        
        for event in events:
            key = (event.type, event.description)
            if key not in unique_events or event.confidence > unique_events[key].confidence:
                unique_events[key] = event
        
        return list(unique_events.values())