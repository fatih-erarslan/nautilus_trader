"""Temporal reference extraction and normalization - GREEN phase"""

import re
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class TemporalExtractor:
    """Extract temporal references from text"""
    
    def __init__(self):
        # Patterns for different types of temporal references
        self.temporal_patterns = [
            # Relative time
            r'\b(yesterday|today|tomorrow)\b',
            r'\b(last|next|this)\s+(week|month|year|quarter|decade)\b',
            r'\b(\d+|[Tt]wo|[Tt]hree|[Ff]our|[Ff]ive)\s+(hours?|days?|weeks?|months?|years?)\s+(ago|from now)\b',
            r'\b(in the )?(past|next|last)\s+(\d+)?\s*(hours?|days?|weeks?|months?|years?)\b',
            
            # Specific dates
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(,?\s+\d{4})?\b',
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b',  # Month alone
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b',
            
            # Quarters and fiscal periods
            r'\b(Q[1-4])\s*\d{4}\b',
            r'\b(Q[1-4])\b',  # Quarter without year
            r'\b(H[12])\s*\d{4}\b',  # Half year
            r'\b(FY|fiscal year)\s*\d{4}\b',
            
            # Day references
            r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
            
            # Week references
            r'\b(best|worst|this|that)\s+week\b',
            
            # Time of day
            r'\b(morning|afternoon|evening|night|overnight)\b',
            
            # Relative positions
            r'\b(beginning|start|end|middle)\s+of\s+(the\s+)?(week|month|year|quarter)\b',
        ]
        
    def extract(self, text: str) -> List[str]:
        """Extract temporal references from text"""
        references = []
        seen = set()
        
        for pattern in self.temporal_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                ref = match.group().strip()
                if ref.lower() not in seen:
                    seen.add(ref.lower())
                    references.append(ref)
        
        return references


def normalize_temporal(reference: str, base_time: datetime) -> datetime:
    """
    Normalize temporal reference to absolute datetime
    
    Args:
        reference: Temporal reference string
        base_time: Base time for relative references
        
    Returns:
        Normalized datetime
    """
    ref_lower = reference.lower().strip()
    
    # Simple relative dates
    if ref_lower == "yesterday":
        return base_time - timedelta(days=1)
    elif ref_lower == "today":
        return base_time
    elif ref_lower == "tomorrow":
        return base_time + timedelta(days=1)
    
    # Last/next period
    if "last week" in ref_lower:
        return base_time - timedelta(days=7)
    elif "next week" in ref_lower:
        return base_time + timedelta(days=7)
    elif "last month" in ref_lower:
        # Approximate - go back to same day previous month
        if base_time.month == 1:
            return base_time.replace(year=base_time.year - 1, month=12)
        else:
            return base_time.replace(month=base_time.month - 1)
    elif "next month" in ref_lower:
        # Approximate - go forward to same day next month
        if base_time.month == 12:
            return base_time.replace(year=base_time.year + 1, month=1)
        else:
            return base_time.replace(month=base_time.month + 1)
    
    # N days/weeks/months ago
    ago_match = re.match(r'(\d+)\s+(hours?|days?|weeks?|months?|years?)\s+ago', ref_lower)
    if ago_match:
        num = int(ago_match.group(1))
        unit = ago_match.group(2).rstrip('s')
        
        if unit == "hour":
            return base_time - timedelta(hours=num)
        elif unit == "day":
            return base_time - timedelta(days=num)
        elif unit == "week":
            return base_time - timedelta(weeks=num)
        elif unit == "month":
            # Approximate
            return base_time - timedelta(days=num * 30)
        elif unit == "year":
            return base_time - timedelta(days=num * 365)
    
    # Quarters
    quarter_match = re.match(r'Q(\d)\s+(\d{4})', reference)
    if quarter_match:
        quarter = int(quarter_match.group(1))
        year = int(quarter_match.group(2))
        
        # Map quarter to month
        quarter_starts = {1: 1, 2: 4, 3: 7, 4: 10}
        month = quarter_starts.get(quarter, 1)
        
        return datetime(year, month, 1)
    
    # If we can't parse it, return base_time
    logger.debug(f"Could not normalize temporal reference: {reference}")
    return base_time


class TemporalNormalizer:
    """Advanced temporal normalization with context awareness"""
    
    def __init__(self):
        self.extractor = TemporalExtractor()
        
    def extract_and_normalize(self, text: str, base_time: Optional[datetime] = None) -> Dict[str, datetime]:
        """
        Extract and normalize all temporal references
        
        Args:
            text: Input text
            base_time: Base time for normalization (default: now)
            
        Returns:
            Dictionary mapping reference text to normalized datetime
        """
        if base_time is None:
            base_time = datetime.now()
            
        references = self.extractor.extract(text)
        normalized = {}
        
        for ref in references:
            try:
                normalized[ref] = normalize_temporal(ref, base_time)
            except Exception as e:
                logger.debug(f"Failed to normalize '{ref}': {e}")
                normalized[ref] = base_time
                
        return normalized