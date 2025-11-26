"""
Model Loader for Trading Strategies

Handles loading optimized trading models and their parameters
"""

import logging
import json
from typing import Dict, Any, Optional
from pathlib import Path
import importlib.util
import sys

logger = logging.getLogger(__name__)


class ModelLoader:
    """Loads and manages trading strategy models"""
    
    def __init__(self, gpu_enabled: bool = False):
        self.gpu_enabled = gpu_enabled
        self.base_path = Path("/workspaces/ai-news-trader")
        self.strategy_cache: Dict[str, Any] = {}
        self.parameter_cache: Dict[str, Dict] = {}
        
        # GPU setup if available
        if self.gpu_enabled:
            self._setup_gpu()
    
    def _setup_gpu(self):
        """Setup GPU acceleration if available"""
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
                logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            else:
                logger.warning("GPU requested but not available")
                self.gpu_enabled = False
        except ImportError:
            logger.warning("PyTorch not installed - GPU acceleration disabled")
            self.gpu_enabled = False
    
    async def load_strategy(self, strategy_name: str) -> Any:
        """Load a trading strategy with its optimized parameters"""
        if strategy_name in self.strategy_cache:
            return self.strategy_cache[strategy_name]
        
        try:
            # Map strategy names to their optimized implementations
            strategy_map = {
                'mirror_trader': 'mirror_trader_optimized',
                'momentum_trader': 'enhanced_momentum_trader',
                'swing_trader': 'swing_trader_optimized',
                'mean_reversion_trader': 'mean_reversion_optimized'
            }
            
            module_name = strategy_map.get(strategy_name, strategy_name)
            module_path = self.base_path / f"src/trading/strategies/{module_name}.py"
            
            if not module_path.exists():
                # Fall back to base implementation
                module_path = self.base_path / f"src/trading/strategies/{strategy_name}.py"
            
            if not module_path.exists():
                raise FileNotFoundError(f"Strategy module not found: {strategy_name}")
            
            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Find the strategy class
            strategy_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    attr_name.lower().replace('_', '') == strategy_name.replace('_', '')):
                    strategy_class = attr
                    break
            
            if not strategy_class:
                # Try common naming patterns
                class_names = [
                    strategy_name.replace('_', ' ').title().replace(' ', ''),
                    f"{strategy_name.replace('_', ' ').title().replace(' ', '')}Strategy",
                    f"Optimized{strategy_name.replace('_', ' ').title().replace(' ', '')}",
                    f"Enhanced{strategy_name.replace('_', ' ').title().replace(' ', '')}"
                ]
                
                for class_name in class_names:
                    if hasattr(module, class_name):
                        strategy_class = getattr(module, class_name)
                        break
            
            if not strategy_class:
                raise ValueError(f"Strategy class not found in module: {module_name}")
            
            # Load optimized parameters
            parameters = await self.get_optimized_parameters(strategy_name)
            
            # Instantiate strategy with parameters
            if parameters and 'best_parameters' in parameters:
                strategy_instance = strategy_class(**parameters['best_parameters'])
            else:
                strategy_instance = strategy_class()
            
            # Enable GPU if available and strategy supports it
            if self.gpu_enabled and hasattr(strategy_instance, 'enable_gpu'):
                strategy_instance.enable_gpu()
            
            # Cache the loaded strategy
            self.strategy_cache[strategy_name] = strategy_instance
            
            logger.info(f"Successfully loaded strategy: {strategy_name}")
            return strategy_instance
            
        except Exception as e:
            logger.error(f"Failed to load strategy {strategy_name}: {str(e)}")
            raise
    
    async def get_optimized_parameters(self, strategy_name: str) -> Dict:
        """Get optimized parameters for a strategy"""
        if strategy_name in self.parameter_cache:
            return self.parameter_cache[strategy_name]
        
        try:
            # Map to parameter files
            parameter_files = {
                'mirror_trader': 'optimization_results.json',
                'momentum_trader': 'momentum_optimization_results.json',
                'swing_trader': 'swing_optimization_results.json',
                'mean_reversion_trader': 'mean_reversion_optimization_results.json'
            }
            
            param_file = parameter_files.get(strategy_name)
            if not param_file:
                logger.warning(f"No parameter file mapping for {strategy_name}")
                return {}
            
            file_path = self.base_path / param_file
            
            if file_path.exists():
                with open(file_path, 'r') as f:
                    parameters = json.load(f)
                    self.parameter_cache[strategy_name] = parameters
                    logger.info(f"Loaded optimized parameters for {strategy_name}")
                    return parameters
            else:
                # Try alternative locations
                alt_paths = [
                    self.base_path / 'results' / param_file,
                    self.base_path / 'optimization' / param_file,
                    self.base_path / 'src' / 'optimization' / 'results' / param_file
                ]
                
                for alt_path in alt_paths:
                    if alt_path.exists():
                        with open(alt_path, 'r') as f:
                            parameters = json.load(f)
                            self.parameter_cache[strategy_name] = parameters
                            logger.info(f"Loaded optimized parameters for {strategy_name} from {alt_path}")
                            return parameters
                
                logger.warning(f"No optimized parameters found for {strategy_name}")
                return self._get_default_parameters(strategy_name)
                
        except Exception as e:
            logger.error(f"Error loading parameters for {strategy_name}: {str(e)}")
            return self._get_default_parameters(strategy_name)
    
    def _get_default_parameters(self, strategy_name: str) -> Dict:
        """Get default parameters for a strategy"""
        defaults = {
            'mirror_trader': {
                'best_parameters': {
                    'berkshire_confidence': 0.7,
                    'bridgewater_confidence': 0.8,
                    'renaissance_confidence': 0.9,
                    'soros_confidence': 0.75,
                    'tiger_confidence': 0.7,
                    'max_position_pct': 0.05,
                    'stop_loss_threshold': -0.1,
                    'take_profit_threshold': 0.2,
                    'immediate_entry_days': 2,
                    'prompt_entry_days': 7
                },
                'performance_metrics': {
                    'sharpe_ratio': 1.5,
                    'max_drawdown': -0.15,
                    'win_rate': 0.6
                }
            },
            'momentum_trader': {
                'best_parameters': {
                    'lookback_period': 20,
                    'momentum_threshold': 0.02,
                    'volume_filter': 1.5,
                    'rsi_period': 14,
                    'rsi_overbought': 70,
                    'rsi_oversold': 30,
                    'macd_fast': 12,
                    'macd_slow': 26,
                    'macd_signal': 9
                },
                'performance_metrics': {
                    'sharpe_ratio': 1.3,
                    'max_drawdown': -0.18,
                    'win_rate': 0.55
                }
            },
            'swing_trader': {
                'best_parameters': {
                    'swing_threshold': 0.03,
                    'support_resistance_lookback': 50,
                    'breakout_volume_multiplier': 2.0,
                    'atr_period': 14,
                    'atr_multiplier': 2.0,
                    'fibonacci_levels': [0.236, 0.382, 0.5, 0.618, 0.786]
                },
                'performance_metrics': {
                    'sharpe_ratio': 1.4,
                    'max_drawdown': -0.12,
                    'win_rate': 0.58
                }
            },
            'mean_reversion_trader': {
                'best_parameters': {
                    'lookback_period': 20,
                    'std_dev_threshold': 2.0,
                    'mean_reversion_speed': 0.5,
                    'min_volatility': 0.01,
                    'bollinger_period': 20,
                    'bollinger_std': 2.0,
                    'rsi_period': 14,
                    'rsi_oversold': 30,
                    'rsi_overbought': 70
                },
                'performance_metrics': {
                    'sharpe_ratio': 1.2,
                    'max_drawdown': -0.10,
                    'win_rate': 0.65
                }
            }
        }
        
        return defaults.get(strategy_name, {})
    
    async def save_parameters(self, strategy_name: str, parameters: Dict) -> bool:
        """Save updated parameters for a strategy"""
        try:
            # Determine file path
            parameter_files = {
                'mirror_trader': 'optimization_results.json',
                'momentum_trader': 'momentum_optimization_results.json',
                'swing_trader': 'swing_optimization_results.json',
                'mean_reversion_trader': 'mean_reversion_optimization_results.json'
            }
            
            param_file = parameter_files.get(strategy_name)
            if not param_file:
                param_file = f"{strategy_name}_parameters.json"
            
            file_path = self.base_path / param_file
            
            # Create backup of existing file
            if file_path.exists():
                backup_path = file_path.with_suffix('.backup.json')
                file_path.rename(backup_path)
                logger.info(f"Created backup: {backup_path}")
            
            # Save new parameters
            with open(file_path, 'w') as f:
                json.dump(parameters, f, indent=2)
            
            # Update cache
            self.parameter_cache[strategy_name] = parameters
            
            logger.info(f"Saved parameters for {strategy_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save parameters for {strategy_name}: {str(e)}")
            return False
    
    def clear_cache(self):
        """Clear all cached strategies and parameters"""
        self.strategy_cache.clear()
        self.parameter_cache.clear()
        logger.info("Cleared model loader cache")
    
    async def reload_strategy(self, strategy_name: str) -> Any:
        """Reload a strategy (useful after parameter updates)"""
        # Remove from cache
        if strategy_name in self.strategy_cache:
            del self.strategy_cache[strategy_name]
        if strategy_name in self.parameter_cache:
            del self.parameter_cache[strategy_name]
        
        # Reload
        return await self.load_strategy(strategy_name)