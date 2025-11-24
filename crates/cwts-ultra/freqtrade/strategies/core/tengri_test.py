#!/usr/bin/env python3
"""
Tengri Integration Test
======================

A minimal test script to verify that the basic Tengri integration components work.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("tengri_test")

# Check if tengri directory exists
tengri_path = Path(__file__).parent / "tengri"
if not tengri_path.exists():
    logger.error(f"Tengri directory not found at {tengri_path}")
    sys.exit(1)

# Check for apps
logger.info(f"Checking Tengri apps in {tengri_path}")
apps = ["cdfa_app", "pairlist_app", "optimization_app", "rl_app", "decision_app"]
all_apps_exist = True

for app in apps:
    app_path = tengri_path / app
    if not app_path.exists():
        logger.error(f"App directory {app} not found")
        all_apps_exist = False
    else:
        # Check for core files
        files = ["__init__.py", "core.py", "client.py", "server.py"]
        for file in files:
            file_path = app_path / file
            if not file_path.exists():
                logger.error(f"File {file} not found in {app}")
                all_apps_exist = False
            else:
                logger.info(f"Found {app}/{file}")

# Check for common utilities
common_path = tengri_path / "common"
if not common_path.exists():
    logger.error(f"Common directory not found at {common_path}")
    sys.exit(1)

# Check for environment config file
env_file = common_path / "environment.py"
if not env_file.exists():
    logger.error(f"Environment file not found at {env_file}")
else:
    logger.info(f"Found environment.py")

# Check for logger utility
logger_file = common_path / "utils" / "logger.py"
if not logger_file.exists():
    logger.error(f"Logger utility not found at {logger_file}")
else:
    logger.info(f"Found utils/logger.py")

# Try to import environment
try:
    sys.path.append(str(tengri_path))
    from common.environment import TengriEnvironment
    
    # Initialize environment
    env = TengriEnvironment()
    logger.info(f"Successfully initialized TengriEnvironment: {env}")
    
    # Check environment capabilities
    logger.info(f"Environment type: {env.env_type}")
    logger.info(f"CPU cores: {env.get_capability('cpu.cores')}")
    logger.info(f"GPU available: {env.get_capability('gpu.available')}")
    
    # Try to get configuration
    logger.info(f"API port: {env.get_config('services.api.port')}")
    logger.info(f"Redis host: {env.get_config('services.redis.host')}")
    
    logger.info("Environment test successful")
except Exception as e:
    logger.error(f"Error importing environment: {e}")
    import traceback
    logger.error(traceback.format_exc())

logger.info("Tengri integration test completed")