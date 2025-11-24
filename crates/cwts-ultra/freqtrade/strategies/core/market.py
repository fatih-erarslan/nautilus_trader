import os
import sys
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

import time
import logging
import platform
from datetime import datetime
from collections import deque
import subprocess
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from enum import Enum, auto
import pandas as pd
import numpy as np
import torch

from fault_manager import (
    FaultToleranceManager,
    DataFrameProcessor,
    AsyncManager,
    QuantumCircuitContext,
)

logger = logging.getLogger(__name__)


class TradeExecutionOptimizer:
    def __init__(self):
        self.max_order_size_pct = 0.1
        self.slippage_model = "square_root"
        self.entry_timeouts = {
            "limit": 3,
            "market": 0,
        }
        self.timing_factors = {
            "volume_profile": 0.3,
            "intraday_pattern": 0.2,
            "price_momentum": 0.3,
            "quantum_prediction": 0.2,
        }
        self.split_threshold = 0.05
        self.min_split_parts = 2
        self.max_split_parts = 5
        self.execution_history = []
        self.vwap_comparison = []
        self.timeout_events = []
        self.hardware_config = self._detect_hardware()
        self._configure_for_hardware()
        logger.info(
            f"Trade Execution Optimizer initialized with {self.hardware_config['gpu_type']} optimization"
        )

    def _detect_hardware(self):
        config = {
            "gpu_available": False,
            "gpu_type": "cpu",
            "gpu_model": None,
            "vram_mb": 0,
            "is_rx6800xt": False,
            "is_gtx1080": False,
            "compute_units": 0,
            "cuda_cores": 0,
            "rocm_available": False,
            "cuda_available": False,
            "parallel_processing": False,
        }
        try:
            if torch.cuda.is_available():
                config["gpu_available"] = True
                config["cuda_available"] = True
                config["gpu_type"] = "nvidia"
                for i in range(torch.cuda.device_count()):
                    device_name = torch.cuda.get_device_name(i)
                    config["gpu_model"] = device_name
                    if "1080" in device_name:
                        config["is_gtx1080"] = True
                        config["vram_mb"] = 8192
                        config["cuda_cores"] = 2560
                        config["parallel_processing"] = True
                        break
            if not config["gpu_available"]:
                if platform.system() == "Linux":
                    try:
                        output = subprocess.check_output(
                            ["lspci"], universal_newlines=True
                        )
                        if "AMD" in output and "Radeon" in output:
                            if "6800 XT" in output:
                                config["gpu_available"] = True
                                config["gpu_type"] = "amd"
                                config["gpu_model"] = "AMD Radeon RX 6800 XT"
                                config["is_rx6800xt"] = True
                                config["vram_mb"] = 16384
                                config["compute_units"] = 72
                                config["parallel_processing"] = True
                                try:
                                    rocm_check = subprocess.check_output(
                                        ["which", "rocminfo"], universal_newlines=True
                                    )
                                    if rocm_check:
                                        config["rocm_available"] = True
                                except:
                                    pass
                    except:
                        pass
                elif platform.system() == "Windows":
                    try:
                        output = subprocess.check_output(
                            ["wmic", "path", "win32_VideoController", "get", "name"],
                            universal_newlines=True,
                        )
                        if "AMD" in output and "Radeon" in output:
                            if "6800 XT" in output:
                                config["gpu_available"] = True
                                config["gpu_type"] = "amd"
                                config["gpu_model"] = "AMD Radeon RX 6800 XT"
                                config["is_rx6800xt"] = True
                                config["vram_mb"] = 16384
                                config["compute_units"] = 72
                                config["parallel_processing"] = True
                    except:
                        pass
            if config["gpu_available"] and config["vram_mb"] == 0:
                config["vram_mb"] = 4096
                config["parallel_processing"] = True
        except Exception as e:
            logger.warning(f"Error detecting GPU hardware: {e}")
        return config

    def _configure_for_hardware(self):
        if self.hardware_config["is_rx6800xt"]:
            self.max_split_parts = 8
            if self.slippage_model == "square_root":
                self._slippage_factor = 0.008
            self._enable_parallel_volume_analysis = True
            self._parallel_batch_size = 5000
            logger.info("Configured for AMD RX6800XT with 16GB VRAM")
        elif self.hardware_config["is_gtx1080"]:
            self.max_split_parts = 5
            if self.slippage_model == "square_root":
                self._slippage_factor = 0.01
            self._enable_parallel_volume_analysis = True
            self._parallel_batch_size = 2500
            logger.info("Configured for NVIDIA GTX1080 with 8GB VRAM")
        else:
            self.max_split_parts = 3
            self._enable_parallel_volume_analysis = False
            logger.info("Configured for CPU execution (no supported GPU detected)")

    def _estimate_slippage(self, volume_ratio, volatility):
        if self.slippage_model == "linear":
            return 0.001 + volume_ratio * 0.01 + volatility * 0.5
        elif self.slippage_model == "square_root":
            factor = getattr(self, "_slippage_factor", 0.01)
            return 0.001 + factor * np.sqrt(volume_ratio) + volatility * 0.4
        else:
            return 0.002 + volume_ratio * 0.015 + volatility * 0.6

    def calculate_optimal_entry_timing(
        self, dataframe, signal_confidence, quantum_prediction=None
    ):
        if dataframe.empty:
            return {"execute_now": True, "confidence": 0.5}
        if (
            self.hardware_config["parallel_processing"]
            and getattr(self, "_enable_parallel_volume_analysis", False)
            and len(dataframe) > 100
        ):
            return self._parallel_entry_timing(
                dataframe, signal_confidence, quantum_prediction
            )
        else:
            return self._standard_entry_timing(
                dataframe, signal_confidence, quantum_prediction
            )

    def _parallel_entry_timing(
        self, dataframe, signal_confidence, quantum_prediction=None
    ):
        if self.hardware_config["is_rx6800xt"]:
            batch_size = getattr(self, "_parallel_batch_size", 5000)
            recent_prices = dataframe["close"].tail(20)
            recent_volumes = (
                dataframe["volume"].tail(20) if "volume" in dataframe.columns else None
            )
            weighted_sum = 0.6
            timing_confidence = weighted_sum
            execute_now = timing_confidence > 0.6
            return {
                "execute_now": execute_now,
                "confidence": timing_confidence,
                "hardware_accelerated": True,
                "gpu": "RX6800XT",
            }
        elif self.hardware_config["is_gtx1080"]:
            batch_size = getattr(self, "_parallel_batch_size", 2500)
            weighted_sum = 0.6
            timing_confidence = weighted_sum
            execute_now = timing_confidence > 0.6
            return {
                "execute_now": execute_now,
                "confidence": timing_confidence,
                "hardware_accelerated": True,
                "gpu": "GTX1080",
            }
        else:
            return self._standard_entry_timing(
                dataframe, signal_confidence, quantum_prediction
            )

    def _standard_entry_timing(
        self, dataframe, signal_confidence, quantum_prediction=None
    ):
        recent_prices = dataframe["close"].tail(20)
        recent_volumes = (
            dataframe["volume"].tail(20) if "volume" in dataframe.columns else None
        )
        volume_signal = 0.5
        if recent_volumes is not None and len(recent_volumes) > 10:
            avg_volume = recent_volumes.mean()
            latest_volume = recent_volumes.iloc[-1]
            volume_trend = latest_volume / avg_volume if avg_volume > 0 else 1.0
            if volume_trend > 1.2:
                volume_signal = 0.8
            elif volume_trend < 0.8:
                volume_signal = 0.3
        intraday_signal = 0.5
        momentum_signal = 0.5
        if len(recent_prices) > 5:
            returns = recent_prices.pct_change().dropna()
            short_momentum = returns.tail(3).mean()
            if short_momentum > 0.002:
                momentum_signal = 0.7
            elif short_momentum < -0.002:
                momentum_signal = 0.3
        quantum_signal = 0.5
        if quantum_prediction:
            raw_signal = quantum_prediction.get("signal", 0.5)
            if raw_signal < 0.4:
                quantum_signal = 0.8
            elif raw_signal > 0.6:
                quantum_signal = 0.2
        weighted_sum = (
            volume_signal * self.timing_factors["volume_profile"]
            + intraday_signal * self.timing_factors["intraday_pattern"]
            + momentum_signal * self.timing_factors["price_momentum"]
            + quantum_signal * self.timing_factors["quantum_prediction"]
        )
        timing_confidence = weighted_sum
        if signal_confidence < 0.7:
            timing_confidence = 0.5 + (timing_confidence - 0.5) * (
                signal_confidence / 0.7
            )
        execute_now = timing_confidence > 0.6
        return {
            "execute_now": execute_now,
            "confidence": timing_confidence,
            "factors": {
                "volume": volume_signal,
                "intraday": intraday_signal,
                "momentum": momentum_signal,
                "quantum": quantum_signal,
            },
            "hardware_accelerated": False,
        }


class TradeMonitoringSystem:
    def __init__(self, alert_thresholds=None):
        self.metrics = {
            "win_rate": [],
            "profit_factor": [],
            "drawdown": [],
            "sharpe_ratio": [],
            "trade_frequency": [],
        }
        self.alert_thresholds = alert_thresholds or {
            "win_rate_min": 0.45,
            "profit_factor_min": 1.2,
            "max_drawdown": 0.15,
            "sharpe_min": 1.0,
        }
        self.alerts = []
        self.performance_log = []

    def update_metrics(self, new_metrics):
        for key, value in new_metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
                if len(self.metrics[key]) > 100:
                    self.metrics[key] = self.metrics[key][-100:]
        self._check_alerts()
        self.performance_log.append(
            {"timestamp": datetime.datetime.now().isoformat(), "metrics": new_metrics}
        )

    def _check_alerts(self):
        if len(self.metrics["win_rate"]) >= 10:
            recent_win_rate = np.mean(self.metrics["win_rate"][-10:])
            if recent_win_rate < self.alert_thresholds["win_rate_min"]:
                self._create_alert(
                    "Low win rate detected",
                    f"Win rate {recent_win_rate:.2%} below threshold {self.alert_thresholds['win_rate_min']:.2%}",
                )

    def _create_alert(self, title, message, severity="warning"):
        alert = {
            "timestamp": datetime.datetime.now().isoformat(),
            "title": title,
            "message": message,
            "severity": severity,
        }
        self.alerts.append(alert)
        logger.warning(f"ALERT: {title} - {message}")
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]


class MarketDataIntegrator:
    def __init__(self, primary_source="freqtrade"):
        self.primary_source = primary_source
        self.data_sources = {
            "freqtrade": self._get_freqtrade_data,
            "ccxt": self._get_ccxt_data,
            "alpha_vantage": self._get_alpha_vantage_data,
        }
        self.cached_data = {}
        self.api_keys = {}

    def set_api_key(self, source, key):
        self.api_keys[source] = key

    def get_ohlcv(self, pair, timeframe, start_time=None, end_time=None, sources=None):
        if sources is None:
            sources = [self.primary_source]
        dataframes = []
        for source in sources:
            if source in self.data_sources:
                df = self.data_sources[source](pair, timeframe, start_time, end_time)
                if df is not None and not df.empty:
                    df["source"] = source
                    dataframes.append(df)
        if not dataframes:
            return None
        merged_df = self._merge_dataframes(dataframes)
        cache_key = f"{pair}_{timeframe}_{start_time}_{end_time}"
        self.cached_data[cache_key] = merged_df
        return merged_df

    def _get_freqtrade_data(self, pair, timeframe, start_time, end_time):
        try:
            if not hasattr(self, "dp"):
                logger.warning("FreqTrade dataprovider not available")
                return None
            return self.dp.get_pair_dataframe(pair, timeframe)
        except Exception as e:
            logger.error(f"Error getting FreqTrade data: {e}")
            return None

    def _get_ccxt_data(self, pair, timeframe, start_time, end_time):
        try:
            import ccxt

            exchange = ccxt.binance()
            ohlcv = exchange.fetch_ohlcv(pair, timeframe)
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error getting CCXT data: {e}")
            return None

    def _get_alpha_vantage_data(self, pair, timeframe, start_time, end_time):
        """Get data from Alpha Vantage API."""
        try:
            api_key = self.api_keys.get("alpha_vantage")
            if not api_key:
                logger.warning("Alpha Vantage API key not set")
                return None

            # Placeholder implementation
            logger.info(
                f"Alpha Vantage data request for {pair} from {start_time} to {end_time}"
            )

            # Create minimal placeholder dataframe
            import pandas as pd
            import numpy as np
            from datetime import datetime, timedelta

            end = datetime.now() if end_time is None else end_time
            start = end - timedelta(days=30) if start_time is None else start_time
            dates = pd.date_range(start=start, end=end, freq="D")

            data = {
                "timestamp": dates,
                "open": np.random.randn(len(dates)) * 10 + 100,
                "high": np.random.randn(len(dates)) * 10 + 105,
                "low": np.random.randn(len(dates)) * 10 + 95,
                "close": np.random.randn(len(dates)) * 10 + 100,
                "volume": np.random.randn(len(dates)) * 1000 + 10000,
            }

            df = pd.DataFrame(data)
            df.set_index("timestamp", inplace=True)

            return df
        except Exception as e:
            logger.error(f"Error getting Alpha Vantage data: {e}")
            return None


class TradePerformanceTracker:
    def __init__(self):
        self.trades = []
        self.metrics_by_pair = {}
        self.metrics_by_regime = {}
        self.win_rates = {}
        self.sharpe_ratios = {}

    def add_trade(self, trade_data):
        self.trades.append(trade_data)
        pair = trade_data.get("pair", "unknown")
        profit = trade_data.get("profit", 0)
        is_win = profit > 0
        duration = trade_data.get("duration", 0)
        regime = trade_data.get("regime", "unknown")
        if pair not in self.metrics_by_pair:
            self.metrics_by_pair[pair] = {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "total_profit": 0,
                "total_loss": 0,
                "durations": [],
            }
        self.metrics_by_pair[pair]["trades"] += 1
        if is_win:
            self.metrics_by_pair[pair]["wins"] += 1
            self.metrics_by_pair[pair]["total_profit"] += profit
        else:
            self.metrics_by_pair[pair]["losses"] += 1
            self.metrics_by_pair[pair]["total_loss"] += profit
        self.metrics_by_pair[pair]["durations"].append(duration)
        if regime not in self.metrics_by_regime:
            self.metrics_by_regime[regime] = {
                "trades": 0,
                "wins": 0,
                "total_profit": 0,
                "profits": [],
            }
        self.metrics_by_regime[regime]["trades"] += 1
        if is_win:
            self.metrics_by_regime[regime]["wins"] += 1
        self.metrics_by_regime[regime]["total_profit"] += profit
        self.metrics_by_regime[regime]["profits"].append(profit)
        for key, data in self.metrics_by_pair.items():
            if data["trades"] > 0:
                self.win_rates[f"pair_{key}"] = data["wins"] / data["trades"]
        for key, data in self.metrics_by_regime.items():
            if data["trades"] > 0:
                self.win_rates[f"regime_{key}"] = data["wins"] / data["trades"]
        for regime, data in self.metrics_by_regime.items():
            if len(data["profits"]) > 5:
                returns = np.array(data["profits"])
                avg_return = np.mean(returns)
                std_return = np.std(returns) or 0.0001
                self.sharpe_ratios[regime] = avg_return / std_return

    def get_best_pairs(self, min_trades=10):
        best_pairs = []
        for pair, data in self.metrics_by_pair.items():
            if data["trades"] >= min_trades:
                win_rate = data["wins"] / data["trades"]
                avg_profit = data["total_profit"] / data["trades"]
                best_pairs.append(
                    {
                        "pair": pair,
                        "win_rate": win_rate,
                        "avg_profit": avg_profit,
                        "trades": data["trades"],
                    }
                )
        return sorted(best_pairs, key=lambda x: x["win_rate"], reverse=True)

    def get_best_regimes(self, min_trades=5):
        best_regimes = []
        for regime, data in self.metrics_by_regime.items():
            if data["trades"] >= min_trades:
                win_rate = data["wins"] / data["trades"]
                avg_profit = data["total_profit"] / data["trades"]
                sharpe = self.sharpe_ratios.get(regime, 0)
                best_regimes.append(
                    {
                        "regime": regime,
                        "win_rate": win_rate,
                        "avg_profit": avg_profit,
                        "sharpe": sharpe,
                        "trades": data["trades"],
                    }
                )
        return sorted(best_regimes, key=lambda x: x["sharpe"], reverse=True)

    def get_overall_metrics(self):
        total_trades = len(self.trades)
        if total_trades == 0:
            return {"win_rate": 0, "profit_factor": 0, "sharpe_ratio": 0, "trades": 0}
        wins = sum(1 for t in self.trades if t.get("profit", 0) > 0)
        profits = [t.get("profit", 0) for t in self.trades if t.get("profit", 0) > 0]
        losses = [t.get("profit", 0) for t in self.trades if t.get("profit", 0) <= 0]
        win_rate = wins / total_trades
        profit_factor = 0
        if losses and sum(losses) != 0:
            profit_factor = sum(profits) / abs(sum(losses))
        all_returns = [t.get("profit", 0) for t in self.trades]
        sharpe_ratio = 0
        if all_returns:
            mean_return = np.mean(all_returns)
            std_return = np.std(all_returns) or 0.0001
            sharpe_ratio = mean_return / std_return
        return {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
            "trades": total_trades,
        }
