"""News sources for collection module"""

from .yahoo_finance_enhanced import YahooFinanceEnhancedSource
from .sec_filings import SECFilingsSource
from .treasury_enhanced import TreasuryEnhancedSource
from .federal_reserve_enhanced import FederalReserveEnhancedSource
from .technical_news import TechnicalNewsSource
from .yield_monitor import YieldCurveMonitor

__all__ = [
    'YahooFinanceEnhancedSource',
    'SECFilingsSource',
    'TreasuryEnhancedSource',
    'FederalReserveEnhancedSource',
    'TechnicalNewsSource',
    'YieldCurveMonitor'
]