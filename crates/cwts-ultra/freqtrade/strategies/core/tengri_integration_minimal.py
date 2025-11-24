#!/usr/bin/env python3
"""
Tengri System Integration - Minimal Version
===========================================

This script demonstrates a minimal initialization of the Tengri system components
for testing purposes. It doesn't attempt to run the full workflow but just initializes
and verifies that the core components can be loaded correctly.
"""

import logging
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tengri_minimal.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("tengri_minimal")

# Add parent directory to path to ensure imports work
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

# Check if tengri directory exists
tengri_path = current_dir / "tengri"
logger.info(f"Looking for Tengri apps in {tengri_path}")
if not tengri_path.exists():
    logger.error(f"Tengri directory not found at {tengri_path}")
    sys.exit(1)

# Add tengri to path for importing
sys.path.append(str(tengri_path))

try:
    # Import common utilities
    logger.info("Importing common utilities...")
    from tengri.common.environment import TengriEnvironment, get_environment
    from tengri.common.utils.logger import get_logger
    
    # Initialize environment
    logger.info("Initializing environment...")
    env = get_environment()
    logger.info(f"Environment: {env.env_type}")
    logger.info(f"CPU cores: {env.get_capability('cpu.cores')}")
    logger.info(f"GPU available: {env.get_capability('gpu.available')}")
    
    # Import CDFA Extensions with improved import pattern
    logger.info("Importing CDFA extensions...")
    
    # Import the main package
    import cdfa_extensions
    
    # Import directly from modules to avoid circular imports
    from cdfa_extensions.hw_acceleration import HardwareAccelerator
    from cdfa_extensions.wavelet_processor import WaveletProcessor
    from cdfa_extensions.cross_asset_analyzer import CrossAssetAnalyzer
    from cdfa_extensions.cdfa_optimizer import CDFAOptimizer
    
    # Import both advanced_cdfa and neuromorphic modules (now imports are fixed)
    try:
        from cdfa_extensions.advanced_cdfa import AdvancedCDFA
        logger.info("Successfully imported AdvancedCDFA")
        
        from cdfa_extensions.neuromorphic_analyzer import NeuromorphicAnalyzer
        logger.info("Successfully imported NeuromorphicAnalyzer")
    except Exception as e:
        logger.error(f"Error importing advanced modules: {e}")
    
    # Use the lazy import pattern to get CDFAIntegration
    try:
        # Get the CDFAIntegration class through the lazy import function
        CDFAIntegration = cdfa_extensions.get_cdfa_integration()
        logger.info("Successfully accessed CDFAIntegration via lazy import")
    except Exception as e:
        logger.error(f"Error accessing CDFAIntegration: {e}")
    
    logger.info("CDFA extensions import tests completed")
    
    # Try to import and initialize each app
    apps_initialized = 0
    
    # 1. Pairlist App
    try:
        logger.info("Initializing Pairlist app...")
        from tengri.pairlist_app.core import PairlistCore
        from tengri.pairlist_app.client import PairlistClient
        logger.info("Successfully imported Pairlist app")
        apps_initialized += 1
    except Exception as e:
        logger.error(f"Error importing Pairlist app: {e}")
    
    # 2. Optimization App
    try:
        logger.info("Initializing Optimization app...")
        from tengri.optimization_app.core import OptimizationCore
        from tengri.optimization_app.client import OptimizationClient
        logger.info("Successfully imported Optimization app")
        apps_initialized += 1
    except Exception as e:
        logger.error(f"Error importing Optimization app: {e}")
    
    # 3. RL App
    try:
        logger.info("Initializing RL app...")
        from tengri.rl_app.core import RLCore
        from tengri.rl_app.client import RLClient
        logger.info("Successfully imported RL app")
        apps_initialized += 1
    except Exception as e:
        logger.error(f"Error importing RL app: {e}")
    
    # 4. Decision App
    try:
        logger.info("Initializing Decision app...")
        from tengri.decision_app.core import DecisionCore
        from tengri.decision_app.client import DecisionClient
        logger.info("Successfully imported Decision app")
        apps_initialized += 1
    except Exception as e:
        logger.error(f"Error importing Decision app: {e}")
    
    # 5. CDFA App
    try:
        logger.info("Initializing CDFA app...")
        from tengri.cdfa_app.core import CDFACore
        from tengri.cdfa_app.client import CDFAClient
        logger.info("Successfully imported CDFA app")
        apps_initialized += 1
    except Exception as e:
        logger.error(f"Error importing CDFA app: {e}")
    
    # Try importing some CDFA analyzers/detectors to verify they work
    try:
        logger.info("Testing CDFA analyzers...")
        from cdfa_extensions.analyzers.antifragility_analyzer import AntifragilityAnalyzer
        from cdfa_extensions.analyzers.fibonacci_analyzer import FibonacciAnalyzer
        logger.info("Successfully imported CDFA analyzers")
        
        logger.info("Testing CDFA detectors...")
        from cdfa_extensions.detectors.black_swan_detector import BlackSwanDetector
        from cdfa_extensions.detectors.whale_detector import WhaleDetector
        logger.info("Successfully imported CDFA detectors")
    except Exception as e:
        logger.error(f"Error importing CDFA analyzers/detectors: {e}")
    
    # Print summary
    logger.info(f"Initialization test completed. Successfully initialized {apps_initialized}/5 apps.")
    
except Exception as e:
    logger.error(f"Error in Tengri minimal initialization: {e}")
    import traceback
    logger.error(traceback.format_exc())

logger.info("Tengri minimal integration test completed")