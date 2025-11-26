"""News Collection module for aggregating real-time news from multiple sources"""

from .base import NewsSource
from .aggregator import NewsAggregator
from .deduplication import deduplicate_news, deduplicate_exact

__all__ = [
    'NewsSource',
    'NewsAggregator', 
    'deduplicate_news',
    'deduplicate_exact'
]