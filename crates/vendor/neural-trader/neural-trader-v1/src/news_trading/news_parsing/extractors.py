"""Entity extractors for news parsing - GREEN phase"""

import re
from typing import List, Dict, Set, Tuple
import logging

from .base import EntityExtractor
from .models import Entity, EntityType

logger = logging.getLogger(__name__)


class CryptoEntityExtractor(EntityExtractor):
    """Extract cryptocurrency entities"""
    
    def __init__(self):
        # Major cryptocurrencies with their tickers
        self.crypto_patterns = {
            r'\b(Bitcoin|BTC)\b': ('Bitcoin', 'BTC'),
            r'\b(Ethereum|ETH)\b': ('Ethereum', 'ETH'),
            r'\b(Binance Coin|BNB)\b': ('Binance Coin', 'BNB'),
            r'\b(Cardano|ADA)\b': ('Cardano', 'ADA'),
            r'\b(Solana|SOL)\b': ('Solana', 'SOL'),
            r'\b(XRP|Ripple)\b': ('XRP', 'XRP'),
            r'\b(Polkadot|DOT)\b': ('Polkadot', 'DOT'),
            r'\b(Dogecoin|DOGE)\b': ('Dogecoin', 'DOGE'),
            r'\b(Avalanche|AVAX)\b': ('Avalanche', 'AVAX'),
            r'\b(Chainlink|LINK)\b': ('Chainlink', 'LINK'),
            r'\b(Polygon|MATIC)\b': ('Polygon', 'MATIC'),
            r'\b(Uniswap|UNI)\b': ('Uniswap', 'UNI'),
        }
        
    def extract(self, text: str) -> List[Entity]:
        """Extract cryptocurrency entities from text"""
        entities = []
        seen = set()
        
        for pattern, (name, ticker) in self.crypto_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if ticker not in seen:
                    seen.add(ticker)
                    
                    # Higher confidence if ticker is explicitly mentioned
                    confidence = 0.95 if f"({ticker})" in text or f" {ticker} " in text else 0.85
                    
                    entities.append(Entity(
                        text=name,
                        type=EntityType.CRYPTO,
                        ticker=ticker,
                        confidence=confidence,
                        metadata={"match_text": match.group()}
                    ))
        
        return self._deduplicate_entities(entities)
    
    def get_supported_types(self) -> List[EntityType]:
        """Get supported entity types"""
        return [EntityType.CRYPTO]
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities, keeping highest confidence"""
        unique = {}
        for entity in entities:
            key = entity.ticker
            if key not in unique or entity.confidence > unique[key].confidence:
                unique[key] = entity
        return list(unique.values())


class CompanyEntityExtractor(EntityExtractor):
    """Extract company entities"""
    
    def __init__(self):
        # Major companies with their tickers
        self.company_patterns = {
            r'\b(Apple|Apple Inc\.?)\b': ('Apple', 'AAPL'),
            r'\b(Microsoft|Microsoft Corp\.?)\b': ('Microsoft', 'MSFT'),
            r'\b(Google|Alphabet)\b': ('Alphabet', 'GOOGL'),
            r'\b(Amazon|Amazon\.com)\b': ('Amazon', 'AMZN'),
            r'\b(Tesla|Tesla Inc\.?)\b': ('Tesla', 'TSLA'),
            r'\b(Meta|Facebook)\b': ('Meta', 'META'),
            r'\b(NVIDIA|Nvidia)\b': ('NVIDIA', 'NVDA'),
            r'\b(Berkshire Hathaway)\b': ('Berkshire Hathaway', 'BRK.B'),
            r'\b(JPMorgan|JP Morgan)\b': ('JPMorgan', 'JPM'),
            r'\b(Johnson & Johnson|J&J)\b': ('Johnson & Johnson', 'JNJ'),
            r'\b(MicroStrategy)\b': ('MicroStrategy', 'MSTR'),
            r'\b(Coinbase)\b': ('Coinbase', 'COIN'),
            r'\b(PayPal)\b': ('PayPal', 'PYPL'),
            r'\b(Square|Block)\b': ('Block', 'SQ'),
            r'\b(SpaceX)\b': ('SpaceX', 'SPACE'),  # Private, but commonly mentioned
        }
    
    def extract(self, text: str) -> List[Entity]:
        """Extract company entities from text"""
        entities = []
        seen = set()
        
        for pattern, (name, ticker) in self.company_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                if ticker not in seen:
                    seen.add(ticker)
                    entities.append(Entity(
                        text=name,
                        type=EntityType.COMPANY,
                        ticker=ticker,
                        confidence=0.9
                    ))
        
        return entities
    
    def get_supported_types(self) -> List[EntityType]:
        """Get supported entity types"""
        return [EntityType.COMPANY]


class PersonEntityExtractor(EntityExtractor):
    """Extract person entities"""
    
    def __init__(self):
        # Notable people in finance/crypto
        self.person_patterns = {
            r'\b(Elon Musk)\b': 'Elon Musk',
            r'\b(Warren Buffett)\b': 'Warren Buffett',
            r'\b(Michael Saylor)\b': 'Michael Saylor',
            r'\b(Satoshi Nakamoto)\b': 'Satoshi Nakamoto',
            r'\b(Vitalik Buterin)\b': 'Vitalik Buterin',
            r'\b(Changpeng Zhao|CZ)\b': 'Changpeng Zhao',
            r'\b(Sam Bankman-Fried|SBF)\b': 'Sam Bankman-Fried',
            r'\b(Gary Gensler)\b': 'Gary Gensler',
            r'\b(Jerome Powell)\b': 'Jerome Powell',
            r'\b(Janet Yellen)\b': 'Janet Yellen',
            r'\b(Cathie Wood)\b': 'Cathie Wood',
            r'\b(Jack Dorsey)\b': 'Jack Dorsey',
            r'\b(Mark Zuckerberg)\b': 'Mark Zuckerberg',
            r'\b(Tim Cook)\b': 'Tim Cook',
        }
    
    def extract(self, text: str) -> List[Entity]:
        """Extract person entities from text"""
        entities = []
        seen = set()
        
        for pattern, name in self.person_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                if name not in seen:
                    seen.add(name)
                    entities.append(Entity(
                        text=name,
                        type=EntityType.PERSON,
                        confidence=0.9
                    ))
        
        return entities
    
    def get_supported_types(self) -> List[EntityType]:
        """Get supported entity types"""
        return [EntityType.PERSON]


class OrganizationEntityExtractor(EntityExtractor):
    """Extract organization entities"""
    
    def __init__(self):
        self.org_patterns = {
            r'\b(SEC|Securities and Exchange Commission)\b': 'SEC',
            r'\b(Federal Reserve|Fed)\b': 'Federal Reserve',
            r'\b(CFTC|Commodity Futures Trading Commission)\b': 'CFTC',
            r'\b(IMF|International Monetary Fund)\b': 'IMF',
            r'\b(World Bank)\b': 'World Bank',
            r'\b(European Central Bank|ECB)\b': 'ECB',
            r'\b(Bank of England|BoE)\b': 'Bank of England',
            r'\b(NYSE|New York Stock Exchange)\b': 'NYSE',
            r'\b(NASDAQ|Nasdaq)\b': 'NASDAQ',
        }
    
    def extract(self, text: str) -> List[Entity]:
        """Extract organization entities from text"""
        entities = []
        seen = set()
        
        for pattern, name in self.org_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                if name not in seen:
                    seen.add(name)
                    entities.append(Entity(
                        text=name,
                        type=EntityType.ORGANIZATION,
                        confidence=0.9
                    ))
        
        return entities
    
    def get_supported_types(self) -> List[EntityType]:
        """Get supported entity types"""
        return [EntityType.ORGANIZATION]


class LocationEntityExtractor(EntityExtractor):
    """Extract location entities"""
    
    def __init__(self):
        self.location_patterns = {
            r'\b(United States|US|USA|America)\b': 'United States',
            r'\b(China|Chinese)\b': 'China',
            r'\b(Europe|European Union|EU)\b': 'Europe',
            r'\b(United Kingdom|UK|Britain)\b': 'United Kingdom',
            r'\b(Japan|Japanese)\b': 'Japan',
            r'\b(Germany|German)\b': 'Germany',
            r'\b(France|French)\b': 'France',
            r'\b(New York)\b': 'New York',
            r'\b(Washington|Washington DC)\b': 'Washington',
            r'\b(Silicon Valley)\b': 'Silicon Valley',
            r'\b(Wall Street)\b': 'Wall Street',
        }
    
    def extract(self, text: str) -> List[Entity]:
        """Extract location entities from text"""
        entities = []
        seen = set()
        
        for pattern, name in self.location_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                if name not in seen:
                    seen.add(name)
                    entities.append(Entity(
                        text=name,
                        type=EntityType.LOCATION,
                        confidence=0.85
                    ))
        
        return entities
    
    def get_supported_types(self) -> List[EntityType]:
        """Get supported entity types"""
        return [EntityType.LOCATION]


class UnifiedEntityExtractor:
    """Unified extractor that combines all entity types"""
    
    def __init__(self):
        self.extractors = [
            CryptoEntityExtractor(),
            CompanyEntityExtractor(),
            PersonEntityExtractor(),
            OrganizationEntityExtractor(),
            LocationEntityExtractor()
        ]
    
    def extract(self, text: str) -> List[Entity]:
        """Extract all entity types from text"""
        all_entities = []
        
        for extractor in self.extractors:
            entities = extractor.extract(text)
            all_entities.extend(entities)
        
        return all_entities