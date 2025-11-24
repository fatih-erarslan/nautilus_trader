#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 10:42:34 2025
Updated on Mon Apr 28 10:15:22 2025

@author: ashina
"""

import numpy as np
import pandas as pd
import logging
import time
import asyncio
import json
import re
import os
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from enum import Enum, auto
from dataclasses import dataclass, field
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from collections import Counter 
# Optional imports with fallbacks

try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
    # Download necessary NLTK data if not already present
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
    # Load spaCy model if available, use simpler model as fallback
    try:
        nlp = spacy.load("en_core_web_trf")
    except OSError:
        try:
            nlp = spacy.load("en_core_web_md")
        except OSError:
            SPACY_AVAILABLE = False
except ImportError:
    SPACY_AVAILABLE = False

# Define LLM provider enum
class LLMProvider(Enum):
    """Enum for supported LLM providers"""
    OPENROUTER = auto()
    OLLAMA = auto()
    LMSTUDIO = auto()
    OPENAI = auto()
    
    @classmethod
    def from_string(cls, provider_name: str) -> 'LLMProvider':
        """Convert string to provider enum with case-insensitive matching"""
        provider_map = {
            'openrouter': cls.OPENROUTER,
            'ollama': cls.OLLAMA,
            'lmstudio': cls.LMSTUDIO,
            'openai': cls.OPENAI
        }
        lowered = provider_name.lower()
        if lowered in provider_map:
            return provider_map[lowered]
        # Partial matching as fallback
        for name, provider in provider_map.items():
            if name.startswith(lowered):
                return provider
        # Default to OpenRouter if no match
        return cls.LMSTUDIO

# Sentiment dimensions enum
class SentimentDimension(Enum):
    """Enum for sentiment analysis dimensions"""
    POLARITY = auto()      # Positive vs negative
    CONFIDENCE = auto()    # Confidence vs uncertainty
    FEAR = auto()          # Fear vs greed
    VOLATILITY = auto()    # Stable vs volatile expectations
    MOMENTUM = auto()      # Future trend expectations
    
    @classmethod
    def all_dimensions(cls) -> Dict[str, 'SentimentDimension']:
        """Return all sentiment dimensions as a dictionary"""
        return {
            'polarity': cls.POLARITY,
            'confidence': cls.CONFIDENCE,
            'fear': cls.FEAR,
            'volatility': cls.VOLATILITY,
            'momentum': cls.MOMENTUM
        }

@dataclass
class LLMConfig:
    """Configuration for LLM providers"""
    provider: LLMProvider = LLMProvider.OPENROUTER
    api_key: Optional[str] = None
    model: str = 'openai/gpt-4-turbo'
    base_url: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.4
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    def __post_init__(self):
        """Set base URL based on provider if not specified"""
        if self.base_url is None:
            if self.provider == LLMProvider.OPENROUTER:
                self.base_url = "https://api.openrouter.ai/api/v1/chat/completions"
            elif self.provider == LLMProvider.OLLAMA:
                self.base_url = "http://localhost:11434/api/chat"
            elif self.provider == LLMProvider.LMSTUDIO:
                self.base_url = "http://localhost:1234/v1/chat/completions"
            elif self.provider == LLMProvider.OPENAI:
                self.base_url = "https://api.openai.com/v1/chat/completions"

@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis"""
    enable_nltk: bool = True
    enable_spacy: bool = True
    enable_entity_level: bool = True
    enable_temporal_analysis: bool = True
    num_segments: int = 3                  # For temporal analysis
    custom_lexicon: Dict[str, float] = field(default_factory=dict)  # Custom sentiment lexicon
    entity_types: List[str] = field(default_factory=lambda: [
        "TICKER", "COMPANY", "INDUSTRY", "COMMODITY", "CURRENCY", "MARKET", "INDEX"
    ])
    
    # Configuration for the various sentiment dimensions
    dimension_weights: Dict[str, float] = field(default_factory=lambda: {
        'polarity': 1.0,
        'confidence': 0.8,
        'fear': 0.7,
        'volatility': 0.6,
        'momentum': 0.9
    })


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class NarrativeForecaster:
    """
    Narrative-based price forecasting using Large Language Models (LLMs).
    
    This module implements narrative-based forecasting techniques to enhance
    market predictions by combining technical analysis with narrative framing,
    leveraging "future retrospective" prompting for improved LLM predictions.
    
    Features fine-grained sentiment analysis and support for local LLM inference
    via Ollama and LMStudio.
    """
    
    def __init__(self, 
                 llm_config: Optional[Union[LLMConfig, Dict[str, Any]]] = None,
                 sentiment_config: Optional[Union[SentimentConfig, Dict[str, Any]]] = None,
                 cache_duration: int = 60, 
                 hardware_manager = None):
        """
        Initialize the Narrative Forecaster.
        
        Args:
            llm_config: Configuration for LLM provider
            sentiment_config: Configuration for sentiment analysis
            cache_duration: Cache duration in minutes
            hardware_manager: Hardware manager instance
        """
        self.logger = logging.getLogger(__name__)
        
        # Process configurations
        if isinstance(llm_config, dict):
            provider_str = llm_config.get('provider', 'openrouter')
            provider = LLMProvider.from_string(provider_str)
            llm_config['provider'] = provider
            self.llm_config = LLMConfig(**llm_config)
        else:
            self.llm_config = llm_config if llm_config is not None else LLMConfig()
            
        if isinstance(sentiment_config, dict):
            self.sentiment_config = SentimentConfig(**sentiment_config)
        else:
            self.sentiment_config = sentiment_config if sentiment_config is not None else SentimentConfig()
        
        # Initialize sentiment analyzers
        self._initialize_sentiment_analyzers()
            
        # Set up hardware manager and caching
        self.hardware_manager = hardware_manager
        self.cache_duration = cache_duration * 60  # Convert to seconds
        self.cache = {}
        
        # Request tracking
        self.last_request_time = 0
        self.request_interval = 1.0  # Minimum seconds between requests
        self.global_history = []
        
        # Performance tracking
        self.prediction_history = []
        self.accuracy_metrics = {'mape': None, 'rmse': None, 'accuracy': None}
        self.sentiment_history = []
        
        # API client session
        self.session = None
        
        self.logger.info(f"Narrative Forecaster initialized with provider: {self.llm_config.provider.name}")
        
    def _initialize_sentiment_analyzers(self):
        """Initialize sentiment analysis components based on configuration"""
        self.vader_analyzer = None
        if self.sentiment_config.enable_nltk and NLTK_AVAILABLE:
            try:
                self.vader_analyzer = SentimentIntensityAnalyzer()
                # Add custom lexicon to VADER if provided
                if self.sentiment_config.custom_lexicon:
                    self.vader_analyzer.lexicon.update(self.sentiment_config.custom_lexicon)
                self.logger.info("VADER sentiment analyzer initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize VADER: {e}")
        
        # Set up dimension-specific lexicons
        self._setup_dimension_lexicons()
    
    def _setup_dimension_lexicons(self):
        """Set up lexicons for different sentiment dimensions"""
        # Map of dimension-specific lexicons
        self.dimension_lexicons = {
            SentimentDimension.CONFIDENCE: {
                # Confidence words (positive score = confident)
                'confident': 0.8, 'certain': 0.8, 'definitely': 0.9, 'surely': 0.7,
                'undoubtedly': 0.9, 'convinced': 0.8, 'strong': 0.7, 'solid': 0.6,
                # Uncertainty words (negative score = uncertain)
                'uncertain': -0.7, 'unclear': -0.6, 'questionable': -0.7, 'doubtful': -0.8,
                'perhaps': -0.5, 'maybe': -0.5, 'might': -0.4, 'possibly': -0.5,
                'potentially': -0.4, 'unsure': -0.7, 'ambiguous': -0.6
            },
            SentimentDimension.FEAR: {
                # Fear words (positive score = fear)
                'fear': 0.8, 'panic': 0.9, 'worry': 0.7, 'concerned': 0.6,
                'anxious': 0.7, 'afraid': 0.8, 'risk': 0.6, 'danger': 0.8,
                'threat': 0.8, 'nervous': 0.7, 'crisis': 0.9, 'crash': 0.9,
                # Greed words (negative score = greed)
                'opportunity': -0.7, 'greedy': -0.8, 'optimistic': -0.6, 'bullish': -0.8,
                'confident': -0.6, 'excited': -0.7, 'fomo': -0.9, 'enthusiasm': -0.7
            },
            SentimentDimension.VOLATILITY: {
                # Volatility words (positive score = volatile)
                'volatile': 0.8, 'unstable': 0.7, 'turbulent': 0.8, 'erratic': 0.9,
                'fluctuating': 0.7, 'unpredictable': 0.8, 'wildly': 0.7, 'swing': 0.6,
                'rollercoaster': 0.9, 'chaotic': 0.9, 'sudden': 0.7, 'dramatic': 0.7,
                # Stability words (negative score = stable)
                'stable': -0.8, 'steady': -0.7, 'consistent': -0.7, 'calm': -0.8,
                'predictable': -0.7, 'balanced': -0.7, 'contained': -0.6, 'range-bound': -0.7
            },
            SentimentDimension.MOMENTUM: {
                # Upward momentum words (positive score = upward)
                'rising': 0.7, 'climbing': 0.7, 'surging': 0.9, 'rallying': 0.8,
                'uptrend': 0.8, 'breakout': 0.7, 'momentum': 0.6, 'outperform': 0.7,
                'higher': 0.6, 'gain': 0.6, 'increase': 0.6, 'growth': 0.6,
                # Downward momentum words (negative score = downward)
                'falling': -0.7, 'dropping': -0.7, 'declining': -0.7, 'slumping': -0.8,
                'downtrend': -0.8, 'breakdown': -0.7, 'lower': -0.6, 'loss': -0.6,
                'decrease': -0.6, 'downturn': -0.7, 'bearish': -0.7, 'retreat': -0.6
            }
        }
        
    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def _close_session(self):
        """Close aiohttp session if it exists"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
            
    def __del__(self):
        """Clean up resources on deletion"""
        if self.session:
            # Create a new event loop if necessary
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._close_session())
                else:
                    loop.run_until_complete(self._close_session())
            except Exception:
                pass  # Ignore errors during cleanup
    
    async def generate_narrative(self, symbol: str, current_price: float,
                               volume: float, support_level: float,
                               resistance_level: float, 
                               additional_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate a future retrospective narrative for price prediction.
        
        Args:
            symbol (str): Trading symbol/pair
            current_price (float): Current price
            volume (float): Current volume
            support_level (float): Identified support level
            resistance_level (float): Identified resistance level
            additional_context (Dict, optional): Additional market context
            
        Returns:
            Dict[str, Any]: Narrative forecast results
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"{symbol}_{current_price}_{volume}_{support_level}_{resistance_level}"
            if cache_key in self.cache:
                cache_entry = self.cache[cache_key]
                # If cache is still valid
                if time.time() - cache_entry['timestamp'] < self.cache_duration:
                    self.logger.debug(f"Using cached narrative for {symbol}")
                    return cache_entry['data']
            
            # Rate limiting
            time_since_last_request = time.time() - self.last_request_time
            if time_since_last_request < self.request_interval:
                await asyncio.sleep(self.request_interval - time_since_last_request)
            
            # Prepare narrative generation
            prompt = self._construct_future_prompt(
                symbol, current_price, volume, support_level, resistance_level, additional_context
            )
            
            # Generate narrative using LLM
            narrative_response = await self._generate_llm_response(prompt)
            
            # Extract key information from narrative
            extracted_data = self._extract_prediction_data(narrative_response, current_price)
            
            # Perform sentiment analysis
            sentiment_results = self._analyze_sentiment(narrative_response, symbol)
            
            # Add metadata
            result = {
                'symbol': symbol,
                'timestamp': time.time(),
                'current_price': current_price,
                'narrative': narrative_response,
                'price_prediction': extracted_data.get('price_prediction', current_price),
                'confidence_score': extracted_data.get('confidence_score', 0.5),
                'timeframe': extracted_data.get('timeframe', '24h'),
                'sentiment_analysis': sentiment_results,
                'key_factors': self.extract_key_factors(narrative_response),
                'execution_time_ms': (time.time() - start_time) * 1000
            }
            
            # Cache the result
            self.cache[cache_key] = {
                'timestamp': time.time(),
                'data': result
            }
            
            # Update last request time
            self.last_request_time = time.time()
            
            # Store prediction for later accuracy analysis
            self.prediction_history.append({
                'timestamp': time.time(),
                'symbol': symbol,
                'current_price': current_price,
                'predicted_price': result['price_prediction'],
                'confidence': result['confidence_score'],
                'sentiment': sentiment_results.get('overall', {}).get('polarity', 0),
                'actual_price': None  # To be filled later
            })
            
            # Store sentiment for historical analysis
            self.sentiment_history.append({
                'timestamp': time.time(),
                'symbol': symbol,
                'sentiment': sentiment_results
            })
            
            # Limit history sizes
            if len(self.prediction_history) > 100:
                self.prediction_history = self.prediction_history[-100:]
            if len(self.sentiment_history) > 100:
                self.sentiment_history = self.sentiment_history[-100:]
                
            # Clean old cache entries
            self._clean_cache()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating narrative forecast: {str(e)}", exc_info=True)
            return {
                'symbol': symbol,
                'timestamp': time.time(),
                'current_price': current_price,
                'narrative': "Error generating narrative forecast.",
                'price_prediction': current_price,  # Default to current price
                'confidence_score': 0.0,
                'timeframe': '24h',
                'sentiment_analysis': {'overall': {'polarity': 0.5}},
                'key_factors': [],
                'error': str(e)
            }
    
    def _construct_future_prompt(self, symbol: str, current_price: float,
                              volume: float, support_level: float,
                              resistance_level: float, 
                              additional_context: Optional[Dict] = None) -> str:
        """
        Construct future retrospective prompt for LLM.
        
        Args:
            symbol (str): Trading symbol/pair
            current_price (float): Current price
            volume (float): Current volume
            support_level (float): Identified support level
            resistance_level (float): Identified resistance level
            additional_context (Dict, optional): Additional market context
            
        Returns:
            str: Formatted prompt for LLM
        """
        # Format numbers for better readability
        formatted_price = f"${current_price:,.2f}"
        formatted_volume = f"{volume:,.2f}"
        formatted_support = f"${support_level:,.2f}"
        formatted_resistance = f"${resistance_level:,.2f}"
        
        # Include additional context if provided
        context_section = ""
        if additional_context:
            context_section = "Additional Market Context:\n"
            for key, value in additional_context.items():
                context_section += f"- {key}: {value}\n"
        
        # Construct the prompt using future retrospective technique
        prompt = f"""You are a financial analyst with expertise in cryptocurrency markets. Looking back from tomorrow, describe what happened to {symbol} given these current conditions:

Current Market Conditions:
- Current Price: {formatted_price}
- 24h Trading Volume: {formatted_volume}
- Support Level: {formatted_support}
- Resistance Level: {formatted_resistance}
{context_section}

Frame your response as a retrospective analysis explaining:
1. What price movements occurred in the next 24 hours
2. What key factors drove these movements
3. How trading volume changed
4. Which technical levels were tested or breached

Write in past tense as if you're looking back from tomorrow. Be specific about the price levels reached.

Also include your analysis of market sentiment, addressing:
1. The overall market mood (bullish/bearish/neutral)
2. Confidence level among traders (certain/uncertain)
3. Fear vs. greed balance
4. Expected volatility ahead
5. Momentum direction

End with a summary section that includes:
- Final price prediction (specific number)
- Confidence score (0.0-1.0)
- Key factors that influenced the movement (list 2-3 specific factors)

Format the summary as:
PRICE PREDICTION: [exact price]
CONFIDENCE: [score between 0-1]
KEY FACTORS: [factor 1], [factor 2], [factor 3]"""

        return prompt
    
    async def _generate_llm_response(self, prompt: str) -> str:
        """
        Generate response from LLM using the configured provider.
        
        Args:
            prompt (str): Input prompt
            
        Returns:
            str: LLM response text
        """
        # Check if API key is configured
        if not self.llm_config.api_key and self.llm_config.provider not in [LLMProvider.OLLAMA, LLMProvider.LMSTUDIO]:
            self.logger.error(f"API key not configured for provider: {self.llm_config.provider.name}")
            return "Error: API key not configured"
            
        # Ensure session exists
        session = await self._ensure_session()
        
        # Prepare for retry logic
        retry_count = 0
        max_retries = self.llm_config.retry_attempts
        
        while retry_count <= max_retries:
            try:
                # Choose correct request format based on provider
                if self.llm_config.provider == LLMProvider.OPENROUTER:
                    response = await self._call_openrouter_api(prompt, session)
                elif self.llm_config.provider == LLMProvider.OLLAMA:
                    response = await self._call_ollama_api(prompt, session)
                elif self.llm_config.provider == LLMProvider.LMSTUDIO:
                    response = await self._call_lmstudio_api(prompt, session)
                elif self.llm_config.provider == LLMProvider.OPENAI:
                    response = await self._call_openai_api(prompt, session)
                else:
                    self.logger.error(f"Unsupported provider: {self.llm_config.provider.name}")
                    return f"Error: Unsupported provider {self.llm_config.provider.name}"
                
                # If successful, return response
                return response
                
            except Exception as e:
                retry_count += 1
                if retry_count <= max_retries:
                    self.logger.warning(f"Error in LLM request (attempt {retry_count}/{max_retries}): {e}")
                    await asyncio.sleep(self.llm_config.retry_delay * retry_count)  # Exponential backoff
                else:
                    self.logger.error(f"Failed after {max_retries} attempts: {e}")
                    return f"Error generating narrative: {str(e)}"
    
    async def _call_openrouter_api(self, prompt: str, session: aiohttp.ClientSession) -> str:
        """Call OpenRouter API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.llm_config.api_key}"
        }
        
        data = {
            "model": self.llm_config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.llm_config.max_tokens,
            "temperature": self.llm_config.temperature,
        }
        
        async with session.post(
            self.llm_config.base_url, 
            headers=headers, 
            json=data,
            timeout=self.llm_config.timeout
        ) as response:
            if response.status == 200:
                json_response = await response.json()
                
                # Extract content from response
                if 'choices' in json_response and len(json_response['choices']) > 0:
                    if 'message' in json_response['choices'][0]:
                        return json_response['choices'][0]['message']['content']
                    else:
                        return json_response['choices'][0].get('text', '')
                else:
                    self.logger.error(f"Unexpected API response format: {json_response}")
                    return "Error: Unexpected API response format"
            else:
                error_text = await response.text()
                self.logger.error(f"API error: {response.status}, {error_text}")
                raise Exception(f"API request failed with status {response.status}: {error_text}")
    
    async def _call_ollama_api(self, prompt: str, session: aiohttp.ClientSession) -> str:
        """Call Ollama API"""
        data = {
            "model": self.llm_config.model.split('/')[-1] if '/' in self.llm_config.model else self.llm_config.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": self.llm_config.temperature,
                "num_predict": self.llm_config.max_tokens
            }
        }
        
        async with session.post(
            self.llm_config.base_url, 
            json=data,
            timeout=self.llm_config.timeout
        ) as response:
            if response.status == 200:
                json_response = await response.json()
                
                # Ollama format
                if 'message' in json_response and 'content' in json_response['message']:
                    return json_response['message']['content']
                else:
                    self.logger.error(f"Unexpected Ollama response format: {json_response}")
                    return "Error: Unexpected Ollama response format"
            else:
                error_text = await response.text()
                self.logger.error(f"Ollama API error: {response.status}, {error_text}")
                raise Exception(f"Ollama API request failed with status {response.status}: {error_text}")
    
    async def _call_lmstudio_api(self, prompt: str, session: aiohttp.ClientSession) -> str:
        """Call LMStudio API"""
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.llm_config.temperature,
            "max_tokens": self.llm_config.max_tokens,
            "stream": False
        }
        
        async with session.post(
            self.llm_config.base_url, 
            json=data,
            timeout=self.llm_config.timeout
        ) as response:
            if response.status == 200:
                json_response = await response.json()
                
                # LMStudio uses OpenAI-compatible format
                if 'choices' in json_response and len(json_response['choices']) > 0:
                    if 'message' in json_response['choices'][0]:
                        return json_response['choices'][0]['message']['content']
                    else:
                        return json_response['choices'][0].get('text', '')
                else:
                    self.logger.error(f"Unexpected LMStudio response format: {json_response}")
                    return "Error: Unexpected LMStudio response format"
            else:
                error_text = await response.text()
                self.logger.error(f"LMStudio API error: {response.status}, {error_text}")
                raise Exception(f"LMStudio API request failed with status {response.status}: {error_text}")
    
    async def _call_openai_api(self, prompt: str, session: aiohttp.ClientSession) -> str:
        """Call OpenAI API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.llm_config.api_key}"
        }
        
        data = {
            "model": self.llm_config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.llm_config.max_tokens,
            "temperature": self.llm_config.temperature,
        }
        
        async with session.post(
            self.llm_config.base_url, 
            headers=headers, 
            json=data,
            timeout=self.llm_config.timeout
        ) as response:
            if response.status == 200:
                json_response = await response.json()
                
                # Extract content from response
                if 'choices' in json_response and len(json_response['choices']) > 0:
                    if 'message' in json_response['choices'][0]:
                        return json_response['choices'][0]['message']['content']
                    else:
                        return json_response['choices'][0].get('text', '')
                else:
                    self.logger.error(f"Unexpected API response format: {json_response}")
                    return "Error: Unexpected OpenAI API response format"
            else:
                error_text = await response.text()
                self.logger.error(f"OpenAI API error: {response.status}, {error_text}")
                raise Exception(f"OpenAI API request failed with status {response.status}: {error_text}")
    
    def _extract_prediction_data(self, narrative: str, current_price: float) -> Dict[str, Any]:
        """
        Extract prediction data from narrative response.
        ENHANCED: Adds timeframe extraction.
        
        Args:
            narrative (str): Generated narrative
            current_price (float): Current price for fallback
            
        Returns:
            Dict[str, Any]: Extracted prediction data
        """
        self.logger.debug("Extracting prediction data from narrative...")
        try:
            # Initialize result with defaults
            result = {
                'price_prediction': current_price, # Default to current
                'confidence_score': 0.5,         # Default neutral
                'timeframe': '24h'               # Default timeframe
            }

            # --- Price Prediction Logic (Keep Existing) ---
            price_pattern = r"PRICE PREDICTION:\s*\$?([\d,]*\.?\d+)"
            price_match = re.search(price_pattern, narrative, re.IGNORECASE)
            if price_match:
                price_str = price_match.group(1).replace(',', '')
                result['price_prediction'] = float(price_str)
                self.logger.debug(f"Extracted price via PRICE PREDICTION tag: {result['price_prediction']}")
            else:
                # Fallback: Find numbers near current price
                self.logger.debug("PRICE PREDICTION tag not found, using fallback number search.")
                price_mentions = re.findall(r"\$?(\d+[,.]?\d*)", narrative)
                if price_mentions:
                    price_values = []
                    for p in price_mentions:
                        try: price_values.append(float(p.replace(',', '')))
                        except ValueError: pass
                    # Filter reasonable prices
                    reasonable_prices = [p for p in price_values if 0.7 * current_price <= p <= 1.3 * current_price]
                    if reasonable_prices:
                        price_counter = Counter(reasonable_prices)
                        # Select most frequent reasonable price
                        result['price_prediction'] = price_counter.most_common(1)[0][0]
                        self.logger.debug(f"Extracted price via fallback frequency: {result['price_prediction']}")
                    else:
                         self.logger.debug("Fallback price search found no reasonable price numbers.")
                else:
                     self.logger.debug("Fallback price search found no numbers.")
            # --- End Price Logic ---


            # --- Confidence Score Logic (Keep Existing) ---
            confidence_pattern = r"CONFIDENCE:\s*([0-1]?\.?\d+)"
            confidence_match = re.search(confidence_pattern, narrative, re.IGNORECASE)
            if confidence_match:
                confidence = float(confidence_match.group(1))
                result['confidence_score'] = max(0.0, min(1.0, confidence))
                self.logger.debug(f"Extracted confidence via CONFIDENCE tag: {result['confidence_score']}")
            else:
                # Fallback: Keyword search
                self.logger.debug("CONFIDENCE tag not found, using fallback keyword search.")
                confidence_terms = {'very confident': 0.9, 'confident': 0.8, 'likely': 0.7, 'probable': 0.7, # Added probable
                                    'possible': 0.6, 'might': 0.5, 'could': 0.5, # Added could
                                    'uncertain': 0.4, 'unlikely': 0.3, 'doubtful': 0.2, 'very uncertain': 0.1}
                found_confidence_keyword = False
                narrative_lower = narrative.lower() # Lowercase once
                # Check longer phrases first
                for term, score in sorted(confidence_terms.items(), key=lambda item: len(item[0]), reverse=True):
                    if term in narrative_lower:
                        result['confidence_score'] = score
                        self.logger.debug(f"Extracted confidence via keyword fallback: '{term}' -> {score}")
                        found_confidence_keyword = True
                        break
                if not found_confidence_keyword:
                     self.logger.debug("Fallback confidence search found no keywords. Using default 0.5.")
            # --- End Confidence Logic ---


            # --- ADDED: Timeframe Extraction Logic ---
            self.logger.debug("Attempting to extract timeframe...")
            timeframe_extracted = None
            # More specific patterns first
            tf_patterns = {
                r"(\d+)\s*hours?": lambda m: f"{m.group(1)}h",
                r"(\d+)\s*days?": lambda m: f"{m.group(1)}d",
                r"(\d+)\s*weeks?": lambda m: f"{m.group(1)}w",
                r"(\d+)\s*months?": lambda m: f"{m.group(1)}M",
                r"end of (?:the )?day": lambda m: "eod",
                r"end of (?:the )?week": lambda m: "eow",
                r"short-term": lambda m: "short",
                r"medium-term": lambda m: "medium",
                r"long-term": lambda m: "long",
                r"next\s*session": lambda m: "next_session",
                r"tomorrow": lambda m: "1d", # Map tomorrow to 1d
                r"overnight": lambda m: "overnight",
            }

            # Search near prediction keywords for timeframe hints
            prediction_context_search = re.search(r".{0,50}PRICE PREDICTION:.{0,100}", narrative, re.IGNORECASE | re.DOTALL)
            search_text = prediction_context_search.group(0) if prediction_context_search else narrative[:300] # Search near prediction or start of text

            for pattern, formatter in tf_patterns.items():
                match = re.search(pattern, search_text, re.IGNORECASE)
                if match:
                    timeframe_extracted = formatter(match)
                    self.logger.info(f"Extracted timeframe: '{timeframe_extracted}' based on pattern '{pattern}'")
                    break # Use the first specific match found

            if timeframe_extracted:
                result['timeframe'] = timeframe_extracted
            else:
                self.logger.debug(f"No specific timeframe extracted. Using default: {result['timeframe']}")
            # --- End Timeframe Logic ---


            self.logger.debug(f"Final extracted data: {result}")
            return result

        except Exception as e:
            self.logger.error(f"Error extracting prediction data: {str(e)}", exc_info=True)
            # Return default structure on error
            return {
                'price_prediction': current_price,
                'confidence_score': 0.1, # Low confidence on error
                'timeframe': '24h'
            }
    
    def extract_key_factors(self, narrative: str) -> List[str]:
        """
        Extract key market factors from narrative.
        
        Args:
            narrative (str): Generated narrative
            
        Returns:
            List[str]: Extracted key factors
        """
        try:
            factors = []
            
            # Check for KEY FACTORS section
            key_factors_pattern = r"KEY FACTORS: (.*?)(?:\n|$)"
            key_factors_match = re.search(key_factors_pattern, narrative)
            
            if key_factors_match:
                # Extract factors list
                factors_text = key_factors_match.group(1)
                # Split by commas
                raw_factors = factors_text.split(',')
                # Clean up factors
                factors = [factor.strip() for factor in raw_factors if factor.strip()]
            else:
                # Extract key factors by analyzing frequent terms
                # Common market factors to look for
                common_factors = [
                    "institutional adoption",
                    "regulatory news",
                    "market sentiment",
                    "technical levels",
                    "trading volume",
                    "support level",
                    "resistance level",
                    "market momentum",
                    "volatility",
                    "liquidity",
                    "whale activity",
                    "short squeeze",
                    "profit taking",
                    "oversold conditions",
                    "overbought conditions"
                ]
                
                for factor in common_factors:
                    if factor.lower() in narrative.lower():
                        # Find the sentence containing this factor
                        pattern = r'[^.]*' + factor + '[^.]*\.'
                        matches = re.findall(pattern, narrative, re.IGNORECASE)
                        if matches:
                            # Use the shortest sentence that mentions the factor
                            shortest_match = min(matches, key=len)
                            factors.append(shortest_match.strip())
                        else:
                            factors.append(factor)
            
            # Limit to top 3 factors
            return factors[:3]
            
        except Exception as e:
            self.logger.error(f"Error extracting key factors: {str(e)}", exc_info=True)
            return []
    
    def _analyze_sentiment(self, narrative: str, symbol: str) -> Dict[str, Any]:
        """
        Perform comprehensive sentiment analysis on narrative text.
        
        Args:
            narrative (str): Generated narrative text
            symbol (str): Trading symbol/pair for entity detection
            
        Returns:
            Dict[str, Any]: Detailed sentiment analysis results
        """
        try:
            # Initialize result structure
            sentiment_results = {
                'overall': {},  # Overall sentiment scores
                'entities': {},  # Entity-specific sentiment
                'temporal': {},  # Temporal sentiment shifts
                'dimensions': {} # Multi-dimensional sentiment
            }
            
            # 1. Overall sentiment
            overall_sentiment = self._calculate_overall_sentiment(narrative)
            sentiment_results['overall'] = overall_sentiment
            
            # 2. Entity-level sentiment (if enabled)
            if self.sentiment_config.enable_entity_level:
                entity_sentiment = self._calculate_entity_sentiment(narrative, symbol)
                sentiment_results['entities'] = entity_sentiment
            
            # 3. Temporal sentiment shifts (if enabled)
            if self.sentiment_config.enable_temporal_analysis:
                temporal_sentiment = self._calculate_temporal_sentiment(narrative)
                sentiment_results['temporal'] = temporal_sentiment
            
            # 4. Multi-dimensional sentiment
            dimension_sentiment = self._calculate_dimension_sentiment(narrative)
            sentiment_results['dimensions'] = dimension_sentiment
            
            return sentiment_results
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}", exc_info=True)
            # Return minimal sentiment data on error
            return {
                'overall': {'polarity': 0.5, 'error': str(e)},
                'entities': {},
                'temporal': {},
                'dimensions': {}
            }
    
    def _calculate_overall_sentiment(self, text: str) -> Dict[str, float]:
        """Calculate overall sentiment of text using available analyzers"""
        result = {'polarity': 0.5, 'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
        
        # Try VADER sentiment if available
        if self.vader_analyzer is not None:
            try:
                vader_scores = self.vader_analyzer.polarity_scores(text)
                result['polarity'] = vader_scores['compound'] * 0.5 + 0.5  # Map from [-1,1] to [0,1]
                result['positive'] = vader_scores['pos']
                result['negative'] = vader_scores['neg'] 
                result['neutral'] = vader_scores['neu']
                return result
            except Exception as e:
                self.logger.warning(f"VADER sentiment analysis error: {e}")
        
        # Fallback: Basic lexicon approach
        if not NLTK_AVAILABLE:
            positive_words = [
                "bullish", "increase", "gain", "positive", "growth",
                "support", "strong", "confidence", "adoption", "momentum",
                "surge", "rally", "uptrend", "breakthrough", "outperform",
                "optimistic", "opportunity", "recover", "upside", "climb"
            ]
            
            negative_words = [
                "bearish", "decrease", "loss", "negative", "decline",
                "resistance", "weak", "uncertainty", "rejection", "selling",
                "dump", "downtrend", "correction", "underperform", "risk",
                "pessimistic", "concern", "volatile", "downside", "plunge"
            ]
            
            # Count word occurrences
            text_lower = text.lower()
            positive_count = sum(text_lower.count(word) for word in positive_words)
            negative_count = sum(text_lower.count(word) for word in negative_words)
            total_count = positive_count + negative_count
            
            if total_count > 0:
                result['polarity'] = positive_count / total_count
                result['positive'] = positive_count / total_count
                result['negative'] = negative_count / total_count
                result['neutral'] = 1.0 - result['positive'] - result['negative']
            
        return result
    
    def _calculate_entity_sentiment(self, text: str, symbol: str) -> Dict[str, Dict[str, float]]:
        """Calculate sentiment specific to market entities in the text"""
        entity_sentiment = {symbol: {'polarity': 0.5}}  # Always include the symbol
        
        # Use spaCy for NER if available
        if SPACY_AVAILABLE:
            try:
                doc = nlp(text)
                
                # Extract entities of interest
                for ent in doc.ents:
                    if ent.label_ in ['ORG', 'PRODUCT', 'MONEY', 'PERCENT']:
                        # Get the sentence containing this entity
                        entity_sentences = []
                        for sent in doc.sents:
                            if ent.start >= sent.start and ent.end <= sent.end:
                                entity_sentences.append(sent.text)
                        
                        # Calculate sentiment for entity context
                        if entity_sentences:
                            entity_context = " ".join(entity_sentences)
                            entity_sentiment[ent.text] = {
                                'polarity': self._calculate_overall_sentiment(entity_context)['polarity']
                            }
            except Exception as e:
                self.logger.warning(f"Entity extraction error with spaCy: {e}")
                
        # Fallback: Simple entity extraction
        if len(entity_sentiment) <= 1:  # Only has the symbol
            # Look for common market entities
            market_entities = [
                "bitcoin", "ethereum", "market", "investors", "traders",
                "bulls", "bears", "whales", "institutions", "retail",
                "exchanges", "liquidity", "volume", "price", "trend"
            ]
            
            for entity in market_entities:
                if entity.lower() in text.lower():
                    # Extract sentences containing the entity
                    sentences = re.split(r'(?<=[.!?])\s+', text)
                    entity_sentences = [s for s in sentences if entity.lower() in s.lower()]
                    
                    if entity_sentences:
                        entity_context = " ".join(entity_sentences)
                        entity_sentiment[entity] = {
                            'polarity': self._calculate_overall_sentiment(entity_context)['polarity']
                        }
        
        return entity_sentiment
    
    def _calculate_temporal_sentiment(self, text: str) -> Dict[str, Dict[str, float]]:
        """Analyze sentiment shifts across narrative sections"""
        temporal_sentiment = {}
        
        try:
            # Split text into approximately equal segments
            segments = self._split_text_into_segments(text, self.sentiment_config.num_segments)
            
            # Calculate sentiment for each segment
            for i, segment in enumerate(segments):
                segment_name = f"segment_{i+1}"
                temporal_sentiment[segment_name] = self._calculate_overall_sentiment(segment)
                
            # Calculate sentiment shift
            if len(segments) >= 2:
                first_segment = temporal_sentiment.get('segment_1', {}).get('polarity', 0.5)
                last_segment = temporal_sentiment.get(f'segment_{len(segments)}', {}).get('polarity', 0.5)
                temporal_sentiment['shift'] = last_segment - first_segment
                
            return temporal_sentiment
                
        except Exception as e:
            self.logger.warning(f"Temporal sentiment analysis error: {e}")
            return {'segment_1': {'polarity': 0.5}}
    
    def _split_text_into_segments(self, text: str, num_segments: int) -> List[str]:
        """Split text into roughly equal segments for temporal analysis"""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) <= num_segments:
            return sentences
            
        # Group sentences into segments
        segment_size = len(sentences) // num_segments
        segments = []
        
        for i in range(num_segments - 1):
            start = i * segment_size
            end = (i + 1) * segment_size
            segments.append(" ".join(sentences[start:end]))
            
        # Last segment gets any remaining sentences
        segments.append(" ".join(sentences[(num_segments - 1) * segment_size:]))
        
        return segments
    
    def _calculate_dimension_sentiment(self, text: str) -> Dict[str, float]:
        """Calculate sentiment across multiple dimensions"""
        dimension_sentiment = {}
        weighted_sentiment = {}
        
        # Calculate base scores
        for dimension_name, dimension in SentimentDimension.all_dimensions().items():
            lexicon = self.dimension_lexicons.get(dimension, {})
            score = 0.5 # Default neutral
            if lexicon:
                score = self._lexicon_based_sentiment(text, lexicon)
            dimension_sentiment[dimension_name] = score
    
        # Apply weights to the new dictionary
        for dimension_name, score in dimension_sentiment.items():
            weight = self.sentiment_config.dimension_weights.get(dimension_name, 1.0)
            # Store both the raw score and the weight if needed, or just the score
            weighted_sentiment[dimension_name] = score # Store raw score
            weighted_sentiment[f"{dimension_name}_weight"] = weight # Store weight separately
    
        return weighted_sentiment # Return the new dictionary
    
    def _lexicon_based_sentiment(self, text: str, lexicon: Dict[str, float]) -> float:
        """Calculate sentiment score using a specific lexicon"""
        text_lower = text.lower()
        total_score = 0.0
        matches = 0
        
        for term, score in lexicon.items():
            # Count number of occurrences of each term
            count = text_lower.count(term)
            if count > 0:
                total_score += score * count
                matches += count
        
        # Return normalized score (0-1 range)
        if matches > 0:
            # Score is centered at 0.5, with range 0-1
            normalized = 0.5 + (total_score / (2 * matches))
            return max(0.0, min(1.0, normalized))
        else:
            return 0.5  # Neutral if no matches
    
    def adjust_profit_threshold(self, base_threshold: float, confidence_score: float,
                             sentiment_score: float, volatility: float) -> float:
        """
        Dynamically adjust profit threshold based on narrative analysis.
        
        Args:
            base_threshold (float): Base profit threshold
            confidence_score (float): Prediction confidence (0-1)
            sentiment_score (float): Sentiment analysis score (0-1)
            volatility (float): Market volatility
            
        Returns:
            float: Adjusted profit threshold
        """
        try:
            # Start with base threshold
            adjusted_threshold = base_threshold
            
            # Adjust based on confidence
            confidence_factor = (confidence_score - 0.5) * 2  # Scale to [-1, 1]
            adjusted_threshold *= (1 + confidence_factor * 0.2)  # ±20% adjustment
            
            # Adjust based on sentiment
            sentiment_factor = (sentiment_score - 0.5) * 2  # Scale to [-1, 1]
            adjusted_threshold *= (1 + sentiment_factor * 0.15)  # ±15% adjustment
            
            # Adjust based on volatility
            volatility_factor = volatility / 0.02  # Normalize to typical volatility
            adjusted_threshold *= (1 + (volatility_factor - 1) * 0.1)  # ±10% adjustment
            
            # Ensure threshold stays within reasonable bounds
            min_threshold = base_threshold * 0.5
            max_threshold = base_threshold * 2.0
            return np.clip(adjusted_threshold, min_threshold, max_threshold)
            
        except Exception as e:
            self.logger.error(f"Error adjusting profit threshold: {str(e)}", exc_info=True)
            return base_threshold  # Return original threshold on error
    
    def update_prediction_accuracy(self, symbol: str, actual_price: float) -> Dict[str, Any]:
        """
        Update accuracy metrics for previous predictions.
        
        Args:
            symbol (str): Trading symbol/pair
            actual_price (float): Actual realized price
            
        Returns:
            Dict[str, Any]: Updated accuracy metrics
        """
        try:
            # Find relevant predictions for this symbol
            relevant_predictions = [p for p in self.prediction_history 
                                 if p['symbol'] == symbol and p['actual_price'] is None]
            
            if not relevant_predictions:
                return self.accuracy_metrics
                
            # Update predictions with actual price
            for prediction in relevant_predictions:
                prediction['actual_price'] = actual_price
            
            # Calculate accuracy metrics
            
            # Only use predictions with actual prices
            completed_predictions = [p for p in self.prediction_history if p['actual_price'] is not None]
            
            if not completed_predictions:
                return self.accuracy_metrics
                
            # Extract actual and predicted values
            actuals = np.array([p['actual_price'] for p in completed_predictions])
            predicted = np.array([p['predicted_price'] for p in completed_predictions])
            
            # MAPE - Mean Absolute Percentage Error
            mape = np.mean(np.abs((actuals - predicted) / actuals)) * 100
            
            # RMSE - Root Mean Square Error
            rmse = np.sqrt(np.mean((actuals - predicted) ** 2))
            
            # Directional Accuracy
            directions_actual = np.sign(actuals - [p['current_price'] for p in completed_predictions])
            directions_predicted = np.sign(predicted - [p['current_price'] for p in completed_predictions])
            accuracy = np.mean(directions_actual == directions_predicted) * 100
            
            # Update accuracy metrics
            self.accuracy_metrics = {
                'mape': mape,
                'rmse': rmse,
                'accuracy': accuracy,
                'sample_size': len(completed_predictions)
            }
            
            return self.accuracy_metrics
            
        except Exception as e:
            self.logger.error(f"Error updating prediction accuracy: {str(e)}", exc_info=True)
            return self.accuracy_metrics
    
    def correlate_sentiment_with_performance(self) -> Dict[str, Any]:
        """
        Analyze correlation between sentiment dimensions and prediction accuracy.
        
        Returns:
            Dict[str, Any]: Correlation analysis results
        """
        try:
            # Find completed predictions
            completed_predictions = [p for p in self.prediction_history if p['actual_price'] is not None]
            
            if len(completed_predictions) < 5:  # Need minimum sample size
                return {'error': 'Insufficient data for correlation analysis'}
            
            # Initialize results
            correlations = {}
            
            # Calculate prediction errors
            for p in completed_predictions:
                p['error_pct'] = abs((p['actual_price'] - p['predicted_price']) / p['actual_price'])
                p['direction_correct'] = np.sign(p['actual_price'] - p['current_price']) == np.sign(p['predicted_price'] - p['current_price'])
            
            # Basic correlation: sentiment vs error
            sentiments = np.array([p.get('sentiment', 0.5) for p in completed_predictions])
            errors = np.array([p['error_pct'] for p in completed_predictions])
            directions = np.array([p['direction_correct'] for p in completed_predictions])
            
            # Calculate correlation between sentiment and error
            if len(sentiments) > 0 and len(errors) > 0:
                # Pearson correlation
                corr_coef = np.corrcoef(sentiments, errors)[0, 1]
                correlations['sentiment_error_correlation'] = corr_coef
                
                # Direction accuracy correlation
                direction_corr = np.corrcoef(sentiments, directions)[0, 1]
                correlations['sentiment_direction_correlation'] = direction_corr
            
            # Analyze advanced sentiment dimensions if we have deeper data
            sentiment_history_filtered = [s for s in self.sentiment_history 
                                       if any(p['symbol'] == s['symbol'] for p in completed_predictions)]
            
            # If we have sufficient sentiment history with dimensions data
            if sentiment_history_filtered and 'dimensions' in sentiment_history_filtered[0].get('sentiment', {}):
                # Extract sentiment dimensions for each prediction
                dimension_values = {}
                
                # Collect dimension values
                for dimension in ['polarity', 'confidence', 'fear', 'volatility', 'momentum']:
                    dimension_values[dimension] = []
                    
                    for p in completed_predictions:
                        # Find matching sentiment entry
                        matching_sentiment = next((s for s in sentiment_history_filtered 
                                               if s['symbol'] == p['symbol'] and 
                                               abs(s['timestamp'] - p['timestamp']) < 5), None)
                        
                        if matching_sentiment and 'dimensions' in matching_sentiment.get('sentiment', {}):
                            dim_value = matching_sentiment['sentiment']['dimensions'].get(dimension, 0.5)
                            dimension_values[dimension].append(dim_value)
                        else:
                            dimension_values[dimension].append(0.5)  # Neutral if no data
                
                # Calculate correlation for each dimension
                dimension_correlations = {}
                for dimension, values in dimension_values.items():
                    if len(values) >= 5:  # Need minimum data points
                        # Error correlation
                        error_corr = np.corrcoef(values, errors[:len(values)])[0, 1]
                        # Direction correlation
                        dir_corr = np.corrcoef(values, directions[:len(values)])[0, 1]
                        
                        dimension_correlations[dimension] = {
                            'error_correlation': error_corr,
                            'direction_correlation': dir_corr
                        }
                
                correlations['dimension_correlations'] = dimension_correlations
            
            return correlations
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment-performance correlation: {str(e)}", exc_info=True)
            return {'error': str(e)}
    
    def _clean_cache(self) -> None:
        """Clean expired entries from cache."""
        current_time = time.time()
        expired_keys = [k for k, v in self.cache.items() 
                      if current_time - v['timestamp'] > self.cache_duration]
        
        for key in expired_keys:
            del self.cache[key]
            
        if expired_keys:
            self.logger.debug(f"Cleaned {len(expired_keys)} expired entries from cache")
    
    def get_sentiment_history(self, symbol: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get sentiment history for analysis.
        
        Args:
            symbol (str, optional): Filter by symbol
            limit (int): Maximum entries to return
            
        Returns:
            List[Dict[str, Any]]: Sentiment history entries
        """
        if symbol:
            filtered = [s for s in self.sentiment_history if s['symbol'] == symbol]
            return filtered[-limit:] if filtered else []
        else:
            return self.sentiment_history[-limit:]
    
    def get_available_local_models(self) -> Dict[str, List[str]]:
        """
        Check for available local models in Ollama and LMStudio.
        
        Returns:
            Dict[str, List[str]]: Available models by provider
        """
        available_models = {
            'ollama': [],
            'lmstudio': []
        }
        
        # Check Ollama models
        try:
            with ThreadPoolExecutor() as executor:
                future = executor.submit(self._check_ollama_models)
                available_models['ollama'] = future.result(timeout=5)
        except Exception as e:
            self.logger.warning(f"Failed to check Ollama models: {e}")
        
        # Check LMStudio models
        try:
            with ThreadPoolExecutor() as executor:
                future = executor.submit(self._check_lmstudio_models)
                available_models['lmstudio'] = future.result(timeout=5)
        except Exception as e:
            self.logger.warning(f"Failed to check LMStudio models: {e}")
        
        return available_models
    
    def _check_ollama_models(self) -> List[str]:
        """Check available models in Ollama"""
        import requests
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=3)
            if response.status_code == 200:
                models_data = response.json()
                if 'models' in models_data:
                    return [model['name'] for model in models_data['models']]
                return []
            return []
        except Exception:
            return []
    
    def _check_lmstudio_models(self) -> List[str]:
        """Check available models in LMStudio"""
        import requests
        try:
            response = requests.get("http://localhost:1234/v1/models", timeout=3)
            if response.status_code == 200:
                models_data = response.json()
                if 'data' in models_data:
                    return [model['id'] for model in models_data['data']]
                return []
            return []
        except Exception:
            return []
    
    def set_provider(self, provider: Union[str, LLMProvider], 
                    api_key: Optional[str] = None,
                    model: Optional[str] = None,
                    base_url: Optional[str] = None) -> bool:
        """
        Change the LLM provider configuration.
        
        Args:
            provider: Provider enum or string
            api_key: API key for provider
            model: Model name/ID
            base_url: Base URL for API
            
        Returns:
            bool: True if successful
        """
        try:
            # Convert string to enum if needed
            if isinstance(provider, str):
                provider = LLMProvider.from_string(provider)
            
            # Keep existing parameters if not provided
            if api_key is None and self.llm_config.provider != provider:
                # Don't reuse API key when changing providers
                api_key = None if provider not in [LLMProvider.OLLAMA, LLMProvider.LMSTUDIO] else ''
            elif api_key is None:
                api_key = self.llm_config.api_key
                
            if model is None:
                # Set appropriate default model for provider
                if provider == LLMProvider.OLLAMA:
                    model = "llama3"
                elif provider == LLMProvider.LMSTUDIO:
                    model = "default"
                else:
                    model = self.llm_config.model
            
            # Create new config
            self.llm_config = LLMConfig(
                provider=provider,
                api_key=api_key,
                model=model,
                base_url=base_url,
                max_tokens=self.llm_config.max_tokens,
                temperature=self.llm_config.temperature,
                timeout=self.llm_config.timeout,
                retry_attempts=self.llm_config.retry_attempts,
                retry_delay=self.llm_config.retry_delay
            )
            
            self.logger.info(f"Provider changed to {provider.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error changing provider: {str(e)}")
            return False