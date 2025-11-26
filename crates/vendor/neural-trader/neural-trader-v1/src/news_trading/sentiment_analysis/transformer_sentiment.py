"""
Transformer-based sentiment analysis using FinBERT and similar models
"""
import re
import hashlib
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .base import SentimentAnalyzer
from .models import SentimentResult, MarketImpact, SentimentDirection, SentimentBreakdown

logger = logging.getLogger(__name__)


class TransformerSentiment(SentimentAnalyzer):
    """
    Sentiment analyzer using transformer models like FinBERT
    
    This implementation uses pre-trained transformer models specifically
    fine-tuned for financial sentiment analysis.
    """
    
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        model_config: Optional[Dict[str, Any]] = None,
        enable_cache: bool = False
    ):
        """
        Initialize transformer sentiment analyzer
        
        Args:
            model_name: HuggingFace model name
            model_config: Optional model configuration
            enable_cache: Enable result caching
        """
        self.model_name = model_name
        self.enable_cache = enable_cache
        self._cache = {} if enable_cache else None
        
        # Model configuration
        config = model_config or {}
        self.max_length = config.get("max_length", 512)
        self.batch_size = config.get("batch_size", 8)
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        logger.info(f"Loading transformer model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Label mapping for FinBERT
        self.label_map = {
            0: "negative",
            1: "neutral", 
            2: "positive"
        }
    
    def get_model_name(self) -> str:
        """Get the name of the sentiment model"""
        return self.model_name
    
    async def analyze(self, text: str, entities: List[str] = None) -> SentimentResult:
        """
        Analyze sentiment of the given text
        
        Args:
            text: The text to analyze
            entities: Optional list of entities mentioned in the text
            
        Returns:
            SentimentResult containing sentiment analysis
        """
        # Validate input
        if text is None:
            raise ValueError("Text cannot be None")
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Check cache if enabled
        if self.enable_cache:
            cache_key = self._get_cache_key(text, entities)
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        # Preprocess text
        cleaned_text = self._preprocess_text(text)
        
        # Tokenize and get model predictions
        start_time = datetime.now()
        inputs = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Calculate sentiment score and confidence
        sentiment_score, confidence = self._calculate_sentiment_metrics(predictions)
        
        # Entity-specific sentiment if entities provided
        entity_sentiment = 0.0
        if entities:
            entity_sentiment = await self._analyze_entity_sentiment(text, entities)
        
        # Determine market impact
        market_impact = self._predict_market_impact(
            sentiment_score,
            confidence,
            entities or [],
            text
        )
        
        # Create sentiment breakdown
        breakdown = SentimentBreakdown(
            headline=sentiment_score * 1.1 if "!" in text or text.isupper() else sentiment_score,
            content=sentiment_score,
            entities=entity_sentiment if entities else sentiment_score,
            tone=self._analyze_tone(text),
            language_intensity=self._analyze_intensity(text)
        )
        
        # Generate result
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = SentimentResult(
            article_id=self._generate_article_id(text),
            overall_sentiment=sentiment_score,
            confidence=confidence,
            market_impact=market_impact,
            sentiment_breakdown=breakdown,
            reasoning=self._generate_reasoning(sentiment_score, confidence, text),
            model_scores={
                "transformer": sentiment_score,
                "raw_positive": float(predictions[0][2]),
                "raw_neutral": float(predictions[0][1]),
                "raw_negative": float(predictions[0][0])
            },
            processing_time=processing_time
        )
        
        # Cache result if enabled
        if self.enable_cache:
            self._cache[cache_key] = result
        
        return result
    
    async def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """
        Analyze multiple texts efficiently
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of SentimentResult objects
        """
        # Process in batches for efficiency
        results = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_tasks = [self.analyze(text) for text in batch_texts]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Filter out exceptions
            for result in batch_results:
                if not isinstance(result, Exception):
                    results.append(result)
                else:
                    logger.error(f"Error in batch analysis: {str(result)}")
        
        return results
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\']', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove emoji and other unicode characters
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        return text.strip()
    
    def _calculate_sentiment_metrics(self, predictions: torch.Tensor) -> tuple[float, float]:
        """
        Calculate sentiment score and confidence from model predictions
        
        Args:
            predictions: Model output probabilities
            
        Returns:
            Tuple of (sentiment_score, confidence)
        """
        # Get the probabilities array
        probs = predictions[0]
        
        # If it's a tensor-like object, call cpu() and numpy()
        if hasattr(probs, 'cpu'):
            probs = probs.cpu()
        if hasattr(probs, 'numpy'):
            probs = probs.numpy()
        
        # Ensure we have a numpy array or list
        if isinstance(probs, (list, tuple)):
            probs = np.array(probs)
        elif hasattr(probs, '__array__'):
            probs = np.array(probs)
        elif isinstance(probs, np.ndarray):
            # Already a numpy array
            pass
        else:
            # For mocked tests, if we get here, just use it as is
            try:
                # Try to convert to numpy array
                probs = np.array(probs)
            except:
                # If all else fails, check if it's already array-like
                pass
        
        # Validate we have valid probabilities
        if not hasattr(probs, '__len__') or len(probs) == 0:
            # Debug information
            logger.error(f"Invalid predictions format: {type(probs)}, value: {probs}")
            raise ValueError(f"Invalid predictions format: {type(probs)}")
        
        # Calculate sentiment score (-1 to 1)
        # FinBERT: [negative, neutral, positive]
        sentiment_score = float(probs[2] - probs[0])  # positive - negative
        
        # Calculate confidence based on prediction certainty
        confidence = float(max(probs))
        
        # Adjust confidence based on how decisive the prediction is
        if abs(sentiment_score) < 0.2:  # Near neutral
            confidence *= 0.8
        
        return sentiment_score, confidence
    
    async def _analyze_entity_sentiment(self, text: str, entities: List[str]) -> float:
        """
        Analyze sentiment specifically around mentioned entities
        
        Args:
            text: Full text
            entities: List of entities to focus on
            
        Returns:
            Entity-specific sentiment score
        """
        entity_sentiments = []
        
        for entity in entities:
            # Find sentences containing the entity
            sentences = text.split('.')
            entity_sentences = [s for s in sentences if entity.lower() in s.lower()]
            
            if entity_sentences:
                # Analyze sentiment of entity-specific sentences
                entity_text = '. '.join(entity_sentences)
                
                inputs = self.tokenizer(
                    entity_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                probs = predictions[0].cpu().numpy()
                entity_sentiment = probs[2] - probs[0]
                entity_sentiments.append(entity_sentiment)
        
        return sum(entity_sentiments) / len(entity_sentiments) if entity_sentiments else 0.0
    
    def _predict_market_impact(
        self,
        sentiment_score: float,
        confidence: float,
        entities: List[str],
        text: str
    ) -> MarketImpact:
        """
        Predict market impact from sentiment analysis
        
        Args:
            sentiment_score: Overall sentiment score
            confidence: Confidence in the analysis
            entities: Affected entities
            text: Original text
            
        Returns:
            MarketImpact prediction
        """
        # Determine direction
        direction = SentimentDirection.from_score(sentiment_score)
        
        # Calculate magnitude based on sentiment strength and confidence
        magnitude = min(abs(sentiment_score) * confidence, 1.0)
        
        # Determine timeframe based on text patterns
        timeframe = self._determine_timeframe(text)
        
        # Identify catalysts
        catalysts = self._identify_catalysts(text)
        
        # Estimate volatility
        volatility = self._estimate_volatility(sentiment_score, text)
        
        return MarketImpact(
            direction=direction,
            magnitude=magnitude,
            timeframe=timeframe,
            affected_assets=entities,
            confidence=confidence,
            volatility_expected=volatility,
            catalysts=catalysts
        )
    
    def _determine_timeframe(self, text: str) -> str:
        """Determine impact timeframe from text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["breaking", "just", "now", "immediately"]):
            return "immediate"
        elif any(word in text_lower for word in ["upcoming", "next week", "soon"]):
            return "short-term"
        else:
            return "long-term"
    
    def _identify_catalysts(self, text: str) -> List[str]:
        """Identify key catalysts from text"""
        catalysts = []
        text_lower = text.lower()
        
        catalyst_patterns = {
            "regulatory": ["regulation", "sec", "government", "policy", "approved", "approval"],
            "institutional": ["institution", "fund", "investment", "adoption"],
            "technical": ["upgrade", "development", "launch", "release"],
            "market": ["crash", "surge", "rally", "selloff"]
        }
        
        for catalyst_type, keywords in catalyst_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                catalysts.append(catalyst_type)
        
        return catalysts[:3]  # Limit to top 3
    
    def _estimate_volatility(self, sentiment_score: float, text: str) -> str:
        """Estimate expected volatility"""
        text_lower = text.lower()
        
        # High volatility indicators
        if any(word in text_lower for word in ["crash", "surge", "plummet", "skyrocket"]):
            return "extreme"
        elif abs(sentiment_score) > 0.7:
            return "high"
        elif abs(sentiment_score) > 0.4:
            return "medium"
        else:
            return "low"
    
    def _analyze_tone(self, text: str) -> float:
        """Analyze the tone of the text"""
        # Simple tone analysis based on punctuation and capitalization
        exclamation_count = text.count('!')
        question_count = text.count('?')
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        tone_score = 0.0
        
        # Exclamations indicate stronger sentiment
        tone_score += min(exclamation_count * 0.1, 0.3)
        
        # Questions indicate uncertainty
        tone_score -= min(question_count * 0.05, 0.1)
        
        # High caps ratio indicates intensity
        if caps_ratio > 0.3:
            tone_score += 0.2
        
        return max(-1, min(1, tone_score))
    
    def _analyze_intensity(self, text: str) -> float:
        """Analyze language intensity"""
        text_lower = text.lower()
        
        # Intensity words
        high_intensity = ["extremely", "absolutely", "definitely", "massive", "huge"]
        medium_intensity = ["very", "quite", "significant", "major"]
        
        intensity_score = 0.5  # Base intensity
        
        for word in high_intensity:
            if word in text_lower:
                intensity_score += 0.1
        
        for word in medium_intensity:
            if word in text_lower:
                intensity_score += 0.05
        
        return min(1.0, intensity_score)
    
    def _generate_reasoning(self, sentiment_score: float, confidence: float, text: str) -> str:
        """Generate human-readable reasoning"""
        direction = "bullish" if sentiment_score > 0 else "bearish" if sentiment_score < 0 else "neutral"
        strength = "strong" if abs(sentiment_score) > 0.6 else "moderate" if abs(sentiment_score) > 0.3 else "weak"
        
        reasoning = f"Transformer model detected {strength} {direction} sentiment "
        reasoning += f"with {confidence:.0%} confidence. "
        
        if "!" in text or text.isupper():
            reasoning += "High emotional intensity detected. "
        
        return reasoning
    
    def _generate_article_id(self, text: str) -> str:
        """Generate unique article ID"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_cache_key(self, text: str, entities: Optional[List[str]]) -> str:
        """Generate cache key"""
        entity_str = ','.join(sorted(entities)) if entities else ''
        return f"{self._generate_article_id(text)}_{entity_str}"
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        # FinBERT primarily supports English
        # Some models may support additional languages
        if "xlm" in self.model_name.lower() or "mbert" in self.model_name.lower():
            return ["en", "es", "fr", "de", "it", "pt", "nl", "ru", "zh", "ja"]
        else:
            return ["en"]