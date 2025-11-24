#!/usr/bin/env python3
"""
Tengri System Integration
========================

This script demonstrates the integration of all Tengri system components:
1. CDFA (Cognitive Diversity Fusion Analysis)
2. RL (Reinforcement Learning)
3. Decision app
4. Pairlist app
5. Optimization app

The script showcases a complete workflow:
- Fetching market data
- Generating optimal trading pairs
- Optimizing models for performance
- Analyzing market patterns
- Making trading decisions
- Executing simulated trades

Author: Tengri Development Team
"""

import logging
import os
import sys
import time
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import traceback
import concurrent.futures
import threading
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tengri_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("tengri_integration")

# Add parent directory to path to ensure imports work
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

# CDFA Components
# Add parent directory to path to ensure imports work for cdfa_extensions
import sys
from pathlib import Path
current_file_path = Path(__file__).parent.absolute()
if str(current_file_path) not in sys.path:
    sys.path.append(str(current_file_path))

# Import CDFA components with improved import pattern to avoid circular imports
# First import the package
import cdfa_extensions

# Import individual modules directly 
from cdfa_extensions.hw_acceleration import HardwareAccelerator as HardwareAcceleration
from cdfa_extensions.cdfa_optimizer import CDFAOptimizer
from cdfa_extensions.cross_asset_analyzer import CrossAssetAnalyzer
from cdfa_extensions.wavelet_processor import WaveletProcessor
from cdfa_extensions.neuromorphic_analyzer import NeuromorphicAnalyzer

# For AdvancedCDFA and CDFAIntegration, use approach that avoids circular imports
from cdfa_extensions.advanced_cdfa import AdvancedCDFA
# Get CDFAIntegration through lazy import function
CDFAIntegration = cdfa_extensions.get_cdfa_integration()

# Pairlist Components
from cdfa_pairlist import CdfaPairlistGenerator
from tengri.pairlist_app.core import PairlistCore
from tengri.pairlist_app.client import PairlistClient

# Optimization Components
from cdfa_optimizer import CDFAOptimizer as MainOptimizer
from tengri.optimization_app.core import OptimizationCore
from tengri.optimization_app.client import OptimizationClient

# RL Components
from q_star_learning import SophisticatedQLearningAgent, ExperienceBuffer, EnvironmentWrapper

# Decision Components - Import dynamically later as needed
# These imports might be complex and depend on specific environment setup

# Market data components
from adaptive_market_data_fetcher import AdaptiveMarketDataFetcher
from cdfa_extensions.crypto_data_fetcher import CryptoDataFetcher

# Utility imports
import pandas as pd
import numpy as np
import redis
from redis.exceptions import RedisError
try:
    import torch
except ImportError:
    logger.warning("PyTorch not available, some features will be disabled")
    torch = None

# Constants
DEFAULT_CONFIG_PATH = os.path.join(current_dir, "config", "tengri_integration.json")
DEFAULT_REDIS_HOST = "localhost"
DEFAULT_REDIS_PORT = 6379
DEFAULT_API_BASE = "http://localhost:8000"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


class TengriIntegration:
    """
    Main integration class for the Tengri trading system.
    
    This class orchestrates the interaction between all components:
    - CDFA for market analysis
    - Pairlist for trading pair selection
    - Optimization for model performance
    - RL for adaptive learning
    - Decision for trade execution
    """
    
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH, debug: bool = False):
        """
        Initialize the Tengri integration system.
        
        Args:
            config_path: Path to the configuration file
            debug: Enable debug mode with additional logging
        """
        self.debug = debug
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.setLevel(logging.DEBUG)
        
        # Load configuration
        self.config = self._load_config(config_path)
        logger.info(f"Tengri integration initialized with configuration from {config_path}")
        
        # Redis connection for communication between components
        self.redis_client = self._setup_redis()
        
        # Initialize subsystems
        self.is_initialized = False
        self.hardware_acceleration = None
        self.market_data_fetcher = None
        self.pairlist_core = None
        self.pairlist_client = None
        self.optimization_core = None
        self.optimization_client = None
        self.cdfa_analyzer = None
        self.rl_agent = None
        self.decision_engine = None
        
        # Thread management
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.config.get("max_workers", 4))
        self.running = False
        self.main_thread = None
        
        # Performance tracking
        self.performance_metrics = {
            "start_time": None,
            "processing_times": {},
            "error_counts": {},
            "success_counts": {}
        }

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file with fallback to defaults."""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    logger.debug(f"Configuration loaded from {config_path}")
                    return config
            else:
                logger.warning(f"Configuration file {config_path} not found, using defaults")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            logger.debug(f"Exception details: {traceback.format_exc()}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration settings."""
        return {
            "redis": {
                "host": DEFAULT_REDIS_HOST,
                "port": DEFAULT_REDIS_PORT,
                "db": 0,
                "channel_prefix": "tengri:"
            },
            "api": {
                "base_url": DEFAULT_API_BASE,
                "timeout": 30
            },
            "pairlist": {
                "update_interval": 3600,  # seconds
                "max_pairs": 20,
                "min_volume": 1000000,
                "min_volatility": 0.01,
                "exchange": "binance"
            },
            "optimization": {
                "use_gpu": True,
                "optimization_level": "balanced",  # one of: minimal, balanced, aggressive
                "enable_torchscript": True,
                "enable_neuromorphic": False
            },
            "cdfa": {
                "timeframes": ["1h", "4h", "1d"],
                "window_size": 200,
                "feature_engineering": "advanced"
            },
            "rl": {
                "learning_rate": 0.001,
                "discount_factor": 0.95,
                "exploration_rate": 0.1,
                "batch_size": 64
            },
            "decision": {
                "risk_tolerance": 0.2,
                "max_open_trades": 10,
                "stake_amount": 0.05  # as fraction of portfolio
            },
            "max_workers": 4,
            "market_data": {
                "timeframes": ["1h", "4h", "1d"],
                "pairs": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT"],
                "days": 30
            }
        }
    
    def _setup_redis(self) -> Optional[redis.Redis]:
        """Set up Redis connection for inter-component communication."""
        try:
            redis_config = self.config.get("redis", {})
            client = redis.Redis(
                host=redis_config.get("host", DEFAULT_REDIS_HOST),
                port=redis_config.get("port", DEFAULT_REDIS_PORT),
                db=redis_config.get("db", 0),
                decode_responses=True
            )
            # Test connection
            client.ping()
            logger.info("Redis connection established successfully")
            return client
        except RedisError as e:
            logger.warning(f"Redis connection failed: {str(e)}. Some features will be disabled.")
            return None
    
    @contextmanager
    def _performance_tracker(self, operation_name: str):
        """Context manager to track performance of operations."""
        if operation_name not in self.performance_metrics["processing_times"]:
            self.performance_metrics["processing_times"][operation_name] = []
            self.performance_metrics["error_counts"][operation_name] = 0
            self.performance_metrics["success_counts"][operation_name] = 0
        
        start_time = time.time()
        success = False
        try:
            yield
            success = True
        finally:
            elapsed = time.time() - start_time
            self.performance_metrics["processing_times"][operation_name].append(elapsed)
            if success:
                self.performance_metrics["success_counts"][operation_name] += 1
            else:
                self.performance_metrics["error_counts"][operation_name] += 1
            
            if self.debug:
                status = "succeeded" if success else "failed"
                logger.debug(f"Operation '{operation_name}' {status} in {elapsed:.4f} seconds")
    
    def initialize(self) -> bool:
        """
        Initialize all Tengri components in the correct sequence.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if self.is_initialized:
            logger.warning("System already initialized")
            return True
        
        try:
            with self._performance_tracker("system_initialization"):
                logger.info("Initializing Tengri integration system...")
                
                # Step 1: Set up hardware acceleration
                self._initialize_hardware_acceleration()
                
                # Step 2: Initialize market data fetcher
                self._initialize_market_data()
                
                # Step 3: Initialize optimization core and client
                self._initialize_optimization()
                
                # Step 4: Initialize pairlist core and client
                self._initialize_pairlist()
                
                # Step 5: Initialize CDFA analyzer
                self._initialize_cdfa()
                
                # Step 6: Initialize RL agent
                self._initialize_rl_agent()
                
                # Step 7: Initialize decision engine
                self._initialize_decision_engine()
                
                self.is_initialized = True
                logger.info("Tengri integration system initialized successfully")
                return True
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            logger.debug(f"Exception details: {traceback.format_exc()}")
            return False
    
    def _initialize_hardware_acceleration(self):
        """Initialize hardware acceleration for performance optimization."""
        logger.info("Initializing hardware acceleration...")
        try:
            self.hardware_acceleration = HardwareAcceleration(
                enable_gpu=self.config["optimization"].get("use_gpu", True),
                log_level="DEBUG" if self.debug else "INFO"
            )
            
            hw_info = self.hardware_acceleration.get_hardware_info()
            logger.info(f"Hardware acceleration initialized: GPU available: {hw_info['gpu_available']}")
            
            if hw_info["gpu_available"]:
                logger.info(f"GPU: {hw_info['gpu_name']} with {hw_info['gpu_memory']} MB memory")
            
            return True
        except Exception as e:
            logger.warning(f"Hardware acceleration initialization failed: {str(e)}. Continuing with CPU only.")
            return False
    
    def _initialize_market_data(self):
        """Initialize market data fetcher for retrieving trading data."""
        logger.info("Initializing market data fetcher...")
        try:
            market_config = self.config.get("market_data", {})
            
            self.market_data_fetcher = AdaptiveMarketDataFetcher(
                exchange=self.config["pairlist"].get("exchange", "binance"),
                timeframes=market_config.get("timeframes", ["1h", "4h", "1d"]),
                base_currencies=["USDT"],
                cache_path=os.path.join(current_dir, "data", "market_cache"),
                use_cached_data=True,
                hardware_acceleration=self.hardware_acceleration if hasattr(self, "hardware_acceleration") else None
            )
            
            # Preload some data to verify the fetcher works
            pairs = market_config.get("pairs", ["BTC/USDT"])
            timeframe = market_config.get("timeframes", ["1h"])[0]
            test_data = self.market_data_fetcher.fetch_data(
                pairs[0], 
                timeframe, 
                limit=10
            )
            
            if test_data is not None and len(test_data) > 0:
                logger.info(f"Market data fetcher initialized successfully. Retrieved {len(test_data)} records for {pairs[0]}.")
                return True
            else:
                logger.warning("Market data fetcher initialized but test data retrieval failed.")
                return False
                
        except Exception as e:
            logger.error(f"Market data fetcher initialization failed: {str(e)}")
            logger.debug(f"Exception details: {traceback.format_exc()}")
            return False
    
    def _initialize_optimization(self):
        """Initialize optimization core and client for model optimization."""
        logger.info("Initializing optimization system...")
        try:
            # Initialize the main optimizer
            optimizer_config = self.config.get("optimization", {})
            
            # First set up the core CDFAOptimizer
            main_optimizer = MainOptimizer(
                optimization_level=optimizer_config.get("optimization_level", "balanced"),
                enable_gpu=optimizer_config.get("use_gpu", True),
                enable_torchscript=optimizer_config.get("enable_torchscript", True),
                enable_neuromorphic=optimizer_config.get("enable_neuromorphic", False),
                hardware_acceleration=self.hardware_acceleration
            )
            
            # Then create the optimization core service
            self.optimization_core = OptimizationCore(
                optimizer=main_optimizer,
                redis_client=self.redis_client,
                redis_prefix=self.config["redis"].get("channel_prefix", "tengri:") + "optimization:",
                log_level="DEBUG" if self.debug else "INFO"
            )
            
            # Initialize the client to interact with the optimization service
            self.optimization_client = OptimizationClient(
                api_base_url=self.config["api"].get("base_url", DEFAULT_API_BASE) + "/optimization",
                redis_client=self.redis_client,
                redis_prefix=self.config["redis"].get("channel_prefix", "tengri:") + "optimization:",
                timeout=self.config["api"].get("timeout", 30)
            )
            
            # Test the optimization setup
            hw_info = self.optimization_client.get_hardware_info()
            if hw_info:
                logger.info(f"Optimization system initialized successfully")
                if self.debug:
                    logger.debug(f"Hardware info: {json.dumps(hw_info, indent=2)}")
                return True
            else:
                logger.warning("Optimization system initialized but hardware info retrieval failed")
                return False
                
        except Exception as e:
            logger.error(f"Optimization system initialization failed: {str(e)}")
            logger.debug(f"Exception details: {traceback.format_exc()}")
            return False
    
    def _initialize_pairlist(self):
        """Initialize pairlist core and client for trading pair selection."""
        logger.info("Initializing pairlist system...")
        try:
            pairlist_config = self.config.get("pairlist", {})
            
            # First set up the CdfaPairlistGenerator
            pairlist_generator = CdfaPairlistGenerator(
                exchange=pairlist_config.get("exchange", "binance"),
                market_data_fetcher=self.market_data_fetcher,
                min_volume=pairlist_config.get("min_volume", 1000000),
                min_volatility=pairlist_config.get("min_volatility", 0.01),
                max_pairs=pairlist_config.get("max_pairs", 20),
                timeframe=self.config["cdfa"].get("timeframes", ["1h"])[0],
                window_size=self.config["cdfa"].get("window_size", 200)
            )
            
            # Then create the pairlist core service
            self.pairlist_core = PairlistCore(
                pairlist_generator=pairlist_generator,
                redis_client=self.redis_client,
                redis_prefix=self.config["redis"].get("channel_prefix", "tengri:") + "pairlist:",
                update_interval=pairlist_config.get("update_interval", 3600),
                log_level="DEBUG" if self.debug else "INFO"
            )
            
            # Initialize the client to interact with the pairlist service
            self.pairlist_client = PairlistClient(
                api_base_url=self.config["api"].get("base_url", DEFAULT_API_BASE) + "/pairlist",
                redis_client=self.redis_client,
                redis_prefix=self.config["redis"].get("channel_prefix", "tengri:") + "pairlist:",
                timeout=self.config["api"].get("timeout", 30)
            )
            
            # Test the pairlist setup
            pairlist_info = self.pairlist_client.get_info()
            if pairlist_info:
                logger.info(f"Pairlist system initialized successfully")
                if self.debug:
                    logger.debug(f"Pairlist info: {json.dumps(pairlist_info, indent=2)}")
                return True
            else:
                logger.warning("Pairlist system initialized but info retrieval failed")
                return False
                
        except Exception as e:
            logger.error(f"Pairlist system initialization failed: {str(e)}")
            logger.debug(f"Exception details: {traceback.format_exc()}")
            return False
    
    def _initialize_cdfa(self):
        """Initialize CDFA analyzer for market analysis."""
        logger.info("Initializing CDFA analyzer...")
        try:
            cdfa_config = self.config.get("cdfa", {})
            
            # Initialize components needed by CDFA
            wavelet_processor = WaveletProcessor(
                optimization_level=self.config["optimization"].get("optimization_level", "balanced")
            )
            
            cross_asset_analyzer = CrossAssetAnalyzer(
                timeframes=cdfa_config.get("timeframes", ["1h", "4h", "1d"]),
                window_size=cdfa_config.get("window_size", 200)
            )
            
            neuromorphic_analyzer = None
            if self.config["optimization"].get("enable_neuromorphic", False):
                neuromorphic_analyzer = NeuromorphicAnalyzer()
            
            # Create the full CDFA integration
            self.cdfa_analyzer = CDFAIntegration(
                market_data_fetcher=self.market_data_fetcher,
                wavelet_processor=wavelet_processor,
                cross_asset_analyzer=cross_asset_analyzer,
                neuromorphic_analyzer=neuromorphic_analyzer,
                hardware_acceleration=self.hardware_acceleration,
                feature_engineering=cdfa_config.get("feature_engineering", "advanced"),
                window_size=cdfa_config.get("window_size", 200)
            )
            
            # Test the CDFA setup with a single pair analysis
            market_config = self.config.get("market_data", {})
            pairs = market_config.get("pairs", ["BTC/USDT"])
            timeframe = cdfa_config.get("timeframes", ["1h"])[0]
            
            test_analysis = self.cdfa_analyzer.analyze_single_pair(
                pair=pairs[0],
                timeframe=timeframe,
                limit=cdfa_config.get("window_size", 200)
            )
            
            if test_analysis is not None:
                logger.info(f"CDFA analyzer initialized successfully. Test analysis completed for {pairs[0]}.")
                return True
            else:
                logger.warning("CDFA analyzer initialized but test analysis failed.")
                return False
                
        except Exception as e:
            logger.error(f"CDFA analyzer initialization failed: {str(e)}")
            logger.debug(f"Exception details: {traceback.format_exc()}")
            return False
    
    def _initialize_rl_agent(self):
        """Initialize Reinforcement Learning agent."""
        logger.info("Initializing RL agent...")
        try:
            rl_config = self.config.get("rl", {})
            
            # Set up environment wrapper for RL
            environment = EnvironmentWrapper(
                market_data_fetcher=self.market_data_fetcher,
                cdfa_analyzer=self.cdfa_analyzer
            )
            
            # Set up experience buffer
            experience_buffer = ExperienceBuffer(
                capacity=10000,
                state_dim=environment.state_dim,
                action_dim=environment.action_dim
            )
            
            # Create the RL agent
            self.rl_agent = SophisticatedQLearningAgent(
                state_dim=environment.state_dim,
                action_dim=environment.action_dim,
                learning_rate=rl_config.get("learning_rate", 0.001),
                discount_factor=rl_config.get("discount_factor", 0.95),
                exploration_rate=rl_config.get("exploration_rate", 0.1),
                batch_size=rl_config.get("batch_size", 64),
                experience_buffer=experience_buffer,
                hardware_acceleration=self.hardware_acceleration
            )
            
            # Try to load a saved model if available
            model_path = os.path.join(current_dir, "models", "rl_agent.pkl")
            if os.path.exists(model_path):
                self.rl_agent.load_model(model_path)
                logger.info(f"RL agent loaded from {model_path}")
            else:
                logger.info("No saved RL model found, initializing with default parameters")
            
            # Test the RL agent with a random state
            test_state = environment.get_random_state()
            test_action = self.rl_agent.select_action(test_state)
            
            logger.info(f"RL agent initialized successfully. Test action: {test_action}")
            return True
                
        except Exception as e:
            logger.error(f"RL agent initialization failed: {str(e)}")
            logger.debug(f"Exception details: {traceback.format_exc()}")
            return False
    
    def _initialize_decision_engine(self):
        """
        Initialize the decision engine.
        
        Note: This is a complex component that may require dynamic imports
        and specific environment setup. We'll implement a simplified version
        for integration demonstration.
        """
        logger.info("Initializing decision engine...")
        try:
            # For this integration example, we'll create a simple decision engine
            # that integrates the outputs from CDFA, RL, and other components
            from quantum_prospect_theory import QuantumProspectTheory
            from risk_manager import RiskManager
            
            risk_config = self.config.get("decision", {})
            
            # Initialize components needed by the decision engine
            risk_manager = RiskManager(
                risk_tolerance=risk_config.get("risk_tolerance", 0.2),
                max_open_trades=risk_config.get("max_open_trades", 10)
            )
            
            prospect_theory = QuantumProspectTheory()
            
            # Create a simplified decision engine
            self.decision_engine = {
                "risk_manager": risk_manager,
                "prospect_theory": prospect_theory,
                "config": risk_config
            }
            
            logger.info("Decision engine initialized successfully")
            return True
                
        except ImportError as e:
            logger.warning(f"Some decision engine components could not be imported: {str(e)}")
            logger.warning("Decision engine will run in limited mode")
            self.decision_engine = {"config": self.config.get("decision", {})}
            return False
        except Exception as e:
            logger.error(f"Decision engine initialization failed: {str(e)}")
            logger.debug(f"Exception details: {traceback.format_exc()}")
            return False
    
    def start(self) -> bool:
        """
        Start the Tengri integration system.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        if not self.is_initialized:
            logger.warning("Cannot start: System not initialized")
            if not self.initialize():
                return False
        
        if self.running:
            logger.warning("System already running")
            return True
        
        logger.info("Starting Tengri integration system...")
        self.running = True
        self.performance_metrics["start_time"] = time.time()
        
        # Start the main integration thread
        self.main_thread = threading.Thread(target=self._main_loop)
        self.main_thread.daemon = True
        self.main_thread.start()
        
        logger.info("Tengri integration system started")
        return True
    
    def stop(self) -> bool:
        """
        Stop the Tengri integration system.
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        if not self.running:
            logger.warning("System not running")
            return True
        
        logger.info("Stopping Tengri integration system...")
        self.running = False
        
        # Wait for main thread to complete
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=10)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Tengri integration system stopped")
        return True
    
    def _main_loop(self):
        """Main integration loop that coordinates all components."""
        try:
            logger.info("Main integration loop started")
            
            while self.running:
                try:
                    # Step 1: Generate trading pairs
                    pairs = self._generate_trading_pairs()
                    
                    # Step 2: Optimize models for these pairs
                    self._optimize_models_for_pairs(pairs)
                    
                    # Step 3: Analyze market data with CDFA
                    analysis_results = self._analyze_market_data(pairs)
                    
                    # Step 4: Feed data to RL agent for learning
                    self._update_rl_agent(analysis_results)
                    
                    # Step 5: Generate trading decisions
                    decisions = self._generate_trading_decisions(pairs, analysis_results)
                    
                    # Step 6: Simulate or execute trades
                    self._process_trading_decisions(decisions)
                    
                    # Log performance metrics periodically
                    if self.debug:
                        self._log_performance_metrics()
                    
                    # Sleep before next iteration to prevent excessive CPU usage
                    time.sleep(60)  # Adjust as needed
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}")
                    logger.debug(f"Exception details: {traceback.format_exc()}")
                    time.sleep(10)  # Shorter sleep on error
        
        except Exception as e:
            logger.error(f"Fatal error in main loop: {str(e)}")
            logger.debug(f"Exception details: {traceback.format_exc()}")
        finally:
            logger.info("Main integration loop ended")
    
    def _generate_trading_pairs(self) -> List[str]:
        """
        Generate trading pairs using the pairlist system.
        
        Returns:
            List[str]: List of trading pairs
        """
        with self._performance_tracker("generate_pairs"):
            try:
                logger.info("Generating trading pairs...")
                
                # Use the pairlist client to generate pairs
                pairs_response = self.pairlist_client.get_pairlist()
                
                if pairs_response and "pairs" in pairs_response:
                    pairs = pairs_response["pairs"]
                    logger.info(f"Generated {len(pairs)} trading pairs")
                    return pairs
                else:
                    # Fallback to default pairs if pairlist generation fails
                    logger.warning("Pairlist generation failed, using default pairs")
                    default_pairs = self.config.get("market_data", {}).get(
                        "pairs", ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
                    )
                    return default_pairs
                    
            except Exception as e:
                logger.error(f"Error generating trading pairs: {str(e)}")
                logger.debug(f"Exception details: {traceback.format_exc()}")
                
                # Return default pairs as fallback
                default_pairs = self.config.get("market_data", {}).get(
                    "pairs", ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
                )
                return default_pairs
    
    def _optimize_models_for_pairs(self, pairs: List[str]) -> bool:
        """
        Optimize models for the given trading pairs.
        
        Args:
            pairs: List of trading pairs to optimize for
            
        Returns:
            bool: True if optimization successful, False otherwise
        """
        with self._performance_tracker("optimize_models"):
            try:
                logger.info(f"Optimizing models for {len(pairs)} pairs...")
                
                # Create optimization tasks for each pair
                optimization_tasks = []
                
                for pair in pairs[:5]:  # Limit to first 5 pairs for demonstration
                    # Create a simplified model for optimization
                    model_config = {
                        "pair": pair,
                        "timeframes": self.config["cdfa"].get("timeframes", ["1h"]),
                        "window_size": self.config["cdfa"].get("window_size", 200),
                        "feature_engineering": self.config["cdfa"].get("feature_engineering", "advanced")
                    }
                    
                    # Submit optimization task
                    task_id = self.optimization_client.optimize_model(
                        model_config=model_config,
                        optimization_level=self.config["optimization"].get("optimization_level", "balanced")
                    )
                    
                    if task_id:
                        optimization_tasks.append(task_id)
                        logger.debug(f"Submitted optimization task {task_id} for {pair}")
                
                # Wait for optimization tasks to complete (with timeout)
                timeout = time.time() + 300  # 5 minutes timeout
                completed_tasks = 0
                
                while completed_tasks < len(optimization_tasks) and time.time() < timeout:
                    for task_id in optimization_tasks:
                        task_status = self.optimization_client.get_task_status(task_id)
                        if task_status and task_status.get("status") == "completed":
                            completed_tasks += 1
                    
                    if completed_tasks < len(optimization_tasks):
                        time.sleep(5)  # Wait before checking again
                
                success_rate = completed_tasks / len(optimization_tasks) if optimization_tasks else 0
                logger.info(f"Model optimization completed: {completed_tasks}/{len(optimization_tasks)} tasks successful ({success_rate:.0%})")
                return success_rate > 0.5  # Consider successful if more than 50% complete
                
            except Exception as e:
                logger.error(f"Error optimizing models: {str(e)}")
                logger.debug(f"Exception details: {traceback.format_exc()}")
                return False
    
    def _analyze_market_data(self, pairs: List[str]) -> Dict[str, Any]:
        """
        Analyze market data using CDFA.
        
        Args:
            pairs: List of trading pairs to analyze
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        with self._performance_tracker("analyze_market"):
            try:
                logger.info(f"Analyzing market data for {len(pairs)} pairs...")
                
                cdfa_config = self.config.get("cdfa", {})
                timeframe = cdfa_config.get("timeframes", ["1h"])[0]  # Use first timeframe
                window_size = cdfa_config.get("window_size", 200)
                
                # Analyze each pair and collect results
                analysis_results = {}
                
                for pair in pairs:
                    try:
                        pair_analysis = self.cdfa_analyzer.analyze_single_pair(
                            pair=pair,
                            timeframe=timeframe,
                            limit=window_size
                        )
                        
                        if pair_analysis is not None:
                            analysis_results[pair] = pair_analysis
                            logger.debug(f"Analysis completed for {pair}")
                        else:
                            logger.warning(f"Analysis failed for {pair}")
                            
                    except Exception as pair_e:
                        logger.error(f"Error analyzing {pair}: {str(pair_e)}")
                        continue
                
                logger.info(f"Market analysis completed for {len(analysis_results)}/{len(pairs)} pairs")
                return analysis_results
                
            except Exception as e:
                logger.error(f"Error analyzing market data: {str(e)}")
                logger.debug(f"Exception details: {traceback.format_exc()}")
                return {}
    
    def _update_rl_agent(self, analysis_results: Dict[str, Any]) -> bool:
        """
        Update the RL agent with new market data.
        
        Args:
            analysis_results: Market analysis results
            
        Returns:
            bool: True if update successful, False otherwise
        """
        with self._performance_tracker("update_rl"):
            try:
                if not analysis_results:
                    logger.warning("No analysis results available for RL update")
                    return False
                
                logger.info(f"Updating RL agent with data from {len(analysis_results)} pairs...")
                
                rl_config = self.config.get("rl", {})
                
                # For each pair, extract features and rewards to update the RL agent
                update_count = 0
                
                for pair, analysis in analysis_results.items():
                    # Extract state and action data
                    try:
                        # Convert analysis to state representation
                        state = self._extract_state_from_analysis(analysis)
                        
                        # Get historical action and reward
                        action = self.rl_agent.select_action(state)
                        reward = self._calculate_reward(pair, action, analysis)
                        
                        # Update agent with this experience
                        next_state = state  # Simplified - in real system this would be the next time step
                        self.rl_agent.update(state, action, reward, next_state)
                        
                        update_count += 1
                        
                    except Exception as pair_e:
                        logger.error(f"Error updating RL for {pair}: {str(pair_e)}")
                        continue
                
                # Periodically save the updated model
                if update_count > 0:
                    model_path = os.path.join(current_dir, "models", "rl_agent.pkl")
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    self.rl_agent.save_model(model_path)
                    logger.debug(f"RL model saved to {model_path}")
                
                logger.info(f"RL agent updated with {update_count}/{len(analysis_results)} pairs")
                return update_count > 0
                
            except Exception as e:
                logger.error(f"Error updating RL agent: {str(e)}")
                logger.debug(f"Exception details: {traceback.format_exc()}")
                return False
    
    def _extract_state_from_analysis(self, analysis: Dict[str, Any]) -> np.ndarray:
        """
        Extract state representation from analysis results.
        
        Args:
            analysis: Analysis results for a pair
            
        Returns:
            np.ndarray: State representation
        """
        # In a real implementation, this would extract meaningful features
        # For this integration example, we'll create a simplified state
        state_features = []
        
        # Extract trend features
        if "trend" in analysis:
            state_features.append(analysis["trend"].get("short_term", 0))
            state_features.append(analysis["trend"].get("medium_term", 0))
            state_features.append(analysis["trend"].get("long_term", 0))
        else:
            state_features.extend([0, 0, 0])
        
        # Extract volatility features
        if "volatility" in analysis:
            state_features.append(analysis["volatility"].get("current", 0))
            state_features.append(analysis["volatility"].get("change", 0))
        else:
            state_features.extend([0, 0])
        
        # Extract pattern features
        if "patterns" in analysis:
            patterns = analysis["patterns"]
            state_features.append(int(patterns.get("reversal", False)))
            state_features.append(int(patterns.get("continuation", False)))
            state_features.append(int(patterns.get("consolidation", False)))
        else:
            state_features.extend([0, 0, 0])
        
        # Add additional features if available
        if "prediction" in analysis:
            state_features.append(analysis["prediction"].get("price_change", 0))
            state_features.append(analysis["prediction"].get("confidence", 0))
        else:
            state_features.extend([0, 0])
        
        # Convert to numpy array and ensure proper shape
        state = np.array(state_features, dtype=np.float32)
        return state
    
    def _calculate_reward(self, pair: str, action: int, analysis: Dict[str, Any]) -> float:
        """
        Calculate reward for reinforcement learning.
        
        Args:
            pair: Trading pair
            action: Action taken
            analysis: Analysis results
            
        Returns:
            float: Reward value
        """
        # In a real implementation, this would calculate meaningful rewards
        # based on the action taken and the market outcome
        
        # For this integration example, we'll create a simplified reward
        # based on whether the action aligns with the predicted trend
        
        reward = 0.0
        
        if "prediction" in analysis:
            predicted_change = analysis["prediction"].get("price_change", 0)
            confidence = analysis["prediction"].get("confidence", 0.5)
            
            # Positive reward if action aligns with prediction
            if (action > 0 and predicted_change > 0) or (action < 0 and predicted_change < 0):
                reward = abs(predicted_change) * confidence
            # Negative reward if action contradicts prediction
            elif (action > 0 and predicted_change < 0) or (action < 0 and predicted_change > 0):
                reward = -abs(predicted_change) * confidence
        
        return reward
    
    def _generate_trading_decisions(self, pairs: List[str], analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate trading decisions based on analysis and RL.
        
        Args:
            pairs: List of trading pairs
            analysis_results: Market analysis results
            
        Returns:
            List[Dict[str, Any]]: Trading decisions
        """
        with self._performance_tracker("generate_decisions"):
            try:
                logger.info(f"Generating trading decisions for {len(pairs)} pairs...")
                
                decisions = []
                
                for pair in pairs:
                    if pair not in analysis_results:
                        logger.debug(f"No analysis available for {pair}, skipping decision")
                        continue
                    
                    analysis = analysis_results[pair]
                    
                    # Extract state for RL agent
                    state = self._extract_state_from_analysis(analysis)
                    
                    # Get action recommendation from RL agent
                    rl_action = self.rl_agent.select_action(state)
                    
                    # Get prediction from CDFA analysis
                    cdfa_prediction = analysis.get("prediction", {}).get("signal", 0)
                    
                    # Combine RL and CDFA insights
                    combined_signal = 0.7 * rl_action + 0.3 * cdfa_prediction
                    
                    # Apply risk management if decision engine is available
                    if self.decision_engine and "risk_manager" in self.decision_engine:
                        risk_manager = self.decision_engine["risk_manager"]
                        max_position_size = risk_manager.get_position_size(pair, combined_signal)
                    else:
                        # Default position sizing
                        max_position_size = self.config["decision"].get("stake_amount", 0.05)
                    
                    # Create decision object
                    decision = {
                        "pair": pair,
                        "timestamp": datetime.now().isoformat(),
                        "action": "buy" if combined_signal > 0.2 else "sell" if combined_signal < -0.2 else "hold",
                        "confidence": abs(combined_signal),
                        "position_size": max_position_size,
                        "analysis": {
                            "rl_signal": float(rl_action),
                            "cdfa_signal": float(cdfa_prediction),
                            "combined_signal": float(combined_signal)
                        }
                    }
                    
                    decisions.append(decision)
                    logger.debug(f"Decision for {pair}: {decision['action']} with confidence {decision['confidence']:.2f}")
                
                logger.info(f"Generated {len(decisions)} trading decisions")
                return decisions
                
            except Exception as e:
                logger.error(f"Error generating trading decisions: {str(e)}")
                logger.debug(f"Exception details: {traceback.format_exc()}")
                return []
    
    def _process_trading_decisions(self, decisions: List[Dict[str, Any]]) -> bool:
        """
        Process trading decisions (simulate or execute trades).
        
        Args:
            decisions: List of trading decisions
            
        Returns:
            bool: True if processing successful, False otherwise
        """
        with self._performance_tracker("process_decisions"):
            try:
                if not decisions:
                    logger.warning("No decisions to process")
                    return False
                
                logger.info(f"Processing {len(decisions)} trading decisions...")
                
                # In a real implementation, this would connect to an exchange API
                # or trading system to execute or simulate trades
                
                # For this integration example, we'll just log the decisions
                # and simulate outcomes
                
                # Group decisions by action
                buys = [d for d in decisions if d["action"] == "buy"]
                sells = [d for d in decisions if d["action"] == "sell"]
                holds = [d for d in decisions if d["action"] == "hold"]
                
                logger.info(f"Decision summary: {len(buys)} buys, {len(sells)} sells, {len(holds)} holds")
                
                # Log decisions to Redis for monitoring (if available)
                if self.redis_client:
                    try:
                        channel = self.config["redis"].get("channel_prefix", "tengri:") + "decisions"
                        self.redis_client.publish(channel, json.dumps({
                            "timestamp": datetime.now().isoformat(),
                            "decisions": decisions
                        }))
                    except RedisError:
                        pass
                
                # Save decisions to file for record keeping
                decisions_path = os.path.join(current_dir, "data", "decisions")
                os.makedirs(decisions_path, exist_ok=True)
                
                decisions_file = os.path.join(
                    decisions_path, 
                    f"decisions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                
                with open(decisions_file, 'w') as f:
                    json.dump(decisions, f, indent=2)
                
                logger.info(f"Trading decisions processed and saved to {decisions_file}")
                return True
                
            except Exception as e:
                logger.error(f"Error processing trading decisions: {str(e)}")
                logger.debug(f"Exception details: {traceback.format_exc()}")
                return False
    
    def _log_performance_metrics(self):
        """Log performance metrics for monitoring."""
        if not self.performance_metrics["start_time"]:
            return
        
        runtime = time.time() - self.performance_metrics["start_time"]
        
        # Calculate average processing times
        avg_times = {}
        for operation, times in self.performance_metrics["processing_times"].items():
            if times:
                avg_times[operation] = sum(times) / len(times)
        
        # Log summary
        logger.info(f"Performance metrics after {runtime:.1f} seconds:")
        for operation, avg_time in avg_times.items():
            success_count = self.performance_metrics["success_counts"].get(operation, 0)
            error_count = self.performance_metrics["error_counts"].get(operation, 0)
            total = success_count + error_count
            success_rate = success_count / total if total > 0 else 0
            
            logger.info(f"  - {operation}: avg={avg_time:.3f}s, success={success_count}/{total} ({success_rate:.0%})")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the Tengri integration system.
        
        Returns:
            Dict[str, Any]: Status information
        """
        return {
            "is_initialized": self.is_initialized,
            "running": self.running,
            "uptime": time.time() - self.performance_metrics["start_time"] if self.performance_metrics["start_time"] else 0,
            "components": {
                "hardware_acceleration": self.hardware_acceleration is not None,
                "market_data_fetcher": self.market_data_fetcher is not None,
                "pairlist": self.pairlist_core is not None and self.pairlist_client is not None,
                "optimization": self.optimization_core is not None and self.optimization_client is not None,
                "cdfa": self.cdfa_analyzer is not None,
                "rl_agent": self.rl_agent is not None,
                "decision_engine": self.decision_engine is not None
            },
            "performance": {
                k: v for k, v in self.performance_metrics.items() if k != "processing_times"
            }
        }

# Example usage
if __name__ == "__main__":
    # Parse command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Tengri Trading System Integration")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Create and start the integration system
    tengri = TengriIntegration(
        config_path=args.config or DEFAULT_CONFIG_PATH,
        debug=args.debug
    )
    
    if tengri.initialize():
        tengri.start()
        
        try:
            # Run for a specified time or until interrupted
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, shutting down...")
        finally:
            tengri.stop()
    else:
        logger.error("Failed to initialize Tengri integration system")