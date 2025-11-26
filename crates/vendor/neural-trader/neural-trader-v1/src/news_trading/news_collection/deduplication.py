"""News deduplication using content similarity - GREEN phase"""

from typing import List, Dict, Set, Optional
import logging
from difflib import SequenceMatcher
import hashlib

from news.models import NewsItem

logger = logging.getLogger(__name__)


def deduplicate_news(items: List[NewsItem], 
                    threshold: float = 0.8,
                    merge_metadata: bool = False) -> List[NewsItem]:
    """
    Deduplicate news items based on content similarity
    
    Args:
        items: List of news items to deduplicate
        threshold: Similarity threshold (0-1) for considering duplicates
        merge_metadata: Whether to merge metadata from duplicates
        
    Returns:
        List of unique news items
    """
    if not items:
        return []
    
    if len(items) == 1:
        return items
    
    # Sort by timestamp to keep earliest
    sorted_items = sorted(items, key=lambda x: x.timestamp)
    
    # Track which items are duplicates
    seen_indices = set()
    unique_items = []
    duplicate_groups = {}  # Maps unique item index to list of duplicate indices
    
    for i, item1 in enumerate(sorted_items):
        if i in seen_indices:
            continue
            
        # This item is unique (so far)
        unique_idx = len(unique_items)
        unique_items.append(item1)
        duplicate_groups[unique_idx] = [i]
        
        # Check against remaining items
        for j in range(i + 1, len(sorted_items)):
            if j in seen_indices:
                continue
                
            item2 = sorted_items[j]
            
            # Calculate similarity
            similarity = calculate_similarity(item1, item2)
            
            if similarity >= threshold:
                # Mark as duplicate
                seen_indices.add(j)
                duplicate_groups[unique_idx].append(j)
                logger.debug(f"Found duplicate: '{item2.title}' similar to '{item1.title}' (similarity: {similarity:.2f})")
    
    # Merge metadata if requested
    if merge_metadata:
        for unique_idx, duplicate_indices in duplicate_groups.items():
            if len(duplicate_indices) > 1:
                unique_items[unique_idx] = merge_duplicate_metadata(
                    unique_items[unique_idx],
                    [sorted_items[idx] for idx in duplicate_indices]
                )
    
    logger.info(f"Deduplicated {len(items)} items to {len(unique_items)} unique items")
    return unique_items


def calculate_similarity(item1: NewsItem, item2: NewsItem) -> float:
    """
    Calculate similarity between two news items
    
    Args:
        item1: First news item
        item2: Second news item
        
    Returns:
        Similarity score between 0 and 1
    """
    # Weight different components
    title_weight = 0.3
    content_weight = 0.5
    entity_weight = 0.2
    
    # Calculate title similarity
    title_sim = SequenceMatcher(None, item1.title.lower(), item2.title.lower()).ratio()
    
    # Calculate content similarity
    content_sim = SequenceMatcher(None, item1.content.lower(), item2.content.lower()).ratio()
    
    # Calculate entity overlap
    entities1 = set(item1.entities)
    entities2 = set(item2.entities)
    if entities1 or entities2:
        entity_sim = len(entities1 & entities2) / len(entities1 | entities2)
    else:
        entity_sim = 1.0 if not entities1 and not entities2 else 0.0
    
    # Weighted average
    similarity = (
        title_weight * title_sim +
        content_weight * content_sim +
        entity_weight * entity_sim
    )
    
    return similarity


def merge_duplicate_metadata(primary: NewsItem, all_items: List[NewsItem]) -> NewsItem:
    """
    Merge metadata from duplicate items
    
    Args:
        primary: The primary item to keep
        all_items: All duplicate items including primary
        
    Returns:
        Primary item with merged metadata
    """
    # Merge entities
    all_entities = set()
    for item in all_items:
        all_entities.update(item.entities)
    primary.entities = sorted(list(all_entities))
    
    # Merge metadata
    merged_metadata = primary.metadata.copy()
    
    # Track duplicate sources
    duplicate_sources = []
    for item in all_items:
        duplicate_sources.append({
            "source": item.source,
            "url": item.url,
            "timestamp": item.timestamp.isoformat()
        })
        
        # Merge other metadata
        for key, value in item.metadata.items():
            if key not in merged_metadata and value is not None:
                merged_metadata[key] = value
    
    merged_metadata["duplicate_sources"] = duplicate_sources
    merged_metadata["duplicate_count"] = len(all_items)
    
    primary.metadata = merged_metadata
    
    return primary


def hash_content(content: str) -> str:
    """Generate hash of content for exact duplicate detection"""
    return hashlib.md5(content.lower().encode()).hexdigest()


def deduplicate_exact(items: List[NewsItem]) -> List[NewsItem]:
    """
    Remove exact duplicates based on content hash
    
    Args:
        items: List of news items
        
    Returns:
        List with exact duplicates removed
    """
    seen_hashes = set()
    unique_items = []
    
    for item in items:
        content_hash = hash_content(item.title + item.content)
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_items.append(item)
        else:
            logger.debug(f"Removed exact duplicate: {item.title}")
    
    return unique_items