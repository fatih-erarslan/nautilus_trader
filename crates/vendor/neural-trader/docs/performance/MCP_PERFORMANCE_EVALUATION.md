# Neural Trader MCP Tools - Performance Evaluation Report

**Generated:** 2025-11-14T20:33:25.950Z

**Platform:** linux x64
**Node.js:** v22.17.0

## Summary

| Metric | Value |
|--------|-------|
| Total Tests | 57 |
| Passed | 43 |
| Failed | 7 |
| Skipped | 7 |
| Avg Latency | 0.12ms |
| Total Duration | 0.01s |

## API Credentials Status

| Service | Status |
|---------|--------|
| alpaca | ✅ Configured |
| theOddsApi | ✅ Configured |
| e2b | ✅ Configured |
| newsApi | ✅ Configured |
| finnhub | ✅ Configured |
| anthropic | ✅ Configured |

## Category Results

### Core Trading

**Tools Tested:** 6
**Passed:** 6  
**Failed:** 0  
**Average Latency:** 0.33ms

| Tool | Status | Latency | Notes |
|------|--------|---------|-------|
| ping | ✅ Pass | 1ms | Success |
| listStrategies | ✅ Pass | - | Success |
| getStrategyInfo | ✅ Pass | 1ms | Success |
| getPortfolioStatus | ✅ Pass | - | Success |
| quickAnalysis | ✅ Pass | - | Success |
| getMarketStatus | ✅ Pass | - | Success |

### Backtesting & Optimization

**Tools Tested:** 6
**Passed:** 2  
**Failed:** 2  
**Average Latency:** 0.00ms

| Tool | Status | Latency | Notes |
|------|--------|---------|-------|
| runBacktest | ❌ Fail | - | Failed to convert JavaScript value `Boolean true ` into rust type `String` |
| optimizeStrategy | ❌ Fail | - | Failed to convert JavaScript value `Undefined` into rust type `String` |
| backtest Strategy | ⏭️ Skipped | - | Function 'backtest Strategy' not found in NAPI module |
| quickBacktest | ✅ Pass | - | Success |
| monteCarlo Simulation | ⏭️ Skipped | - | Function 'monteCarlo Simulation' not found in NAPI module |
| runBenchmark | ✅ Pass | - | Success |

### Neural Networks

**Tools Tested:** 7
**Passed:** 3  
**Failed:** 4  
**Average Latency:** 0.00ms

| Tool | Status | Latency | Notes |
|------|--------|---------|-------|
| neuralForecast | ❌ Fail | - | Failed to convert napi value into rust type `bool` |
| neuralTrain | ❌ Fail | - | Failed to convert napi value into rust type `bool` |
| neuralEvaluate | ✅ Pass | - | Success |
| neuralBacktest | ✅ Pass | - | Success |
| neuralModelStatus | ✅ Pass | - | Success |
| neuralOptimize | ❌ Fail | - | Failed to convert JavaScript value `Object {"learning_rate":[0.001,0.01],"batch_size":[16,64]}` into rust type `String` |
| neuralPredict | ❌ Fail | - | Failed to convert JavaScript value `Object ["AAPL"]` into rust type `String` |

### News Trading

**Tools Tested:** 6
**Passed:** 5  
**Failed:** 1  
**Average Latency:** 0.00ms

| Tool | Status | Latency | Notes |
|------|--------|---------|-------|
| analyzeNews | ✅ Pass | - | Success |
| getNewsSentiment | ✅ Pass | - | Success |
| controlNewsCollection | ❌ Fail | - | Given napi value is not an array |
| getNewsProviderStatus | ✅ Pass | - | Success |
| fetchFilteredNews | ✅ Pass | - | Success |
| getNewsTrends | ✅ Pass | - | Success |

### Sports Betting

**Tools Tested:** 7
**Passed:** 7  
**Failed:** 0  
**Average Latency:** 0.14ms

| Tool | Status | Latency | Notes |
|------|--------|---------|-------|
| getSportsEvents | ✅ Pass | - | Success |
| getSportsOdds | ✅ Pass | - | Success |
| findSportsArbitrage | ✅ Pass | - | Success |
| analyzeBettingMarketDepth | ✅ Pass | - | Success |
| calculateKellyCriterion | ✅ Pass | - | Success |
| getBettingPortfolioStatus | ✅ Pass | - | Success |
| getSportsBettingPerformance | ✅ Pass | 1ms | Success |

### Odds API Integration

**Tools Tested:** 6
**Passed:** 6  
**Failed:** 0  
**Average Latency:** 0.00ms

| Tool | Status | Latency | Notes |
|------|--------|---------|-------|
| oddsApiGetSports | ✅ Pass | - | Success |
| oddsApiGetLiveOdds | ✅ Pass | - | Success |
| oddsApiGetEventOdds | ✅ Pass | - | Success |
| oddsApiFindArbitrage | ✅ Pass | - | Success |
| oddsApiGetBookmakerOdds | ✅ Pass | - | Success |
| oddsApiAnalyzeMovement | ✅ Pass | - | Success |

### Prediction Markets

**Tools Tested:** 5
**Passed:** 5  
**Failed:** 0  
**Average Latency:** 0.20ms

| Tool | Status | Latency | Notes |
|------|--------|---------|-------|
| getPredictionMarkets | ✅ Pass | - | Success |
| analyzeMarketSentiment | ✅ Pass | 1ms | Success |
| getMarketOrderbook | ✅ Pass | - | Success |
| getPredictionPositions | ✅ Pass | - | Success |
| calculateExpectedValue | ✅ Pass | - | Success |

### Syndicates

**Tools Tested:** 5
**Passed:** 5  
**Failed:** 0  
**Average Latency:** 0.20ms

| Tool | Status | Latency | Notes |
|------|--------|---------|-------|
| createSyndicate | ✅ Pass | - | Success |
| addSyndicateMember | ✅ Pass | - | Success |
| getSyndicateStatus | ✅ Pass | - | Success |
| allocateSyndicateFunds | ✅ Pass | 1ms | Success |
| distributeSyndicateProfits | ✅ Pass | - | Success |

### E2B Cloud

**Tools Tested:** 5
**Passed:** 0  
**Failed:** 0  
**Average Latency:** 0.00ms

| Tool | Status | Latency | Notes |
|------|--------|---------|-------|
| createE2bSandbox | ⏭️ Skipped | - | Function 'createE2bSandbox' not found in NAPI module |
| runE2bAgent | ⏭️ Skipped | - | Function 'runE2bAgent' not found in NAPI module |
| executeE2bProcess | ⏭️ Skipped | - | Function 'executeE2bProcess' not found in NAPI module |
| listE2bSandboxes | ⏭️ Skipped | - | Function 'listE2bSandboxes' not found in NAPI module |
| getE2bSandboxStatus | ⏭️ Skipped | - | Function 'getE2bSandboxStatus' not found in NAPI module |

### System & Monitoring

**Tools Tested:** 4
**Passed:** 4  
**Failed:** 0  
**Average Latency:** 0.00ms

| Tool | Status | Latency | Notes |
|------|--------|---------|-------|
| getSystemMetrics | ✅ Pass | - | Success |
| getExecutionAnalytics | ✅ Pass | - | Success |
| performanceReport | ✅ Pass | - | Success |
| correlationAnalysis | ✅ Pass | - | Success |

## Detailed Results

```json
{
  "timestamp": "2025-11-14T20:33:25.950Z",
  "environment": {
    "nodeVersion": "v22.17.0",
    "platform": "linux",
    "arch": "x64",
    "memory": {
      "rss": 47443968,
      "heapTotal": 5570560,
      "heapUsed": 4661832,
      "external": 1564616,
      "arrayBuffers": 16659
    }
  },
  "apiKeys": {
    "alpaca": {
      "configured": true,
      "details": {
        "key": "PKAJQDPYIZ1S8BHWU7GD",
        "secret": "zJvREGAi3qQi6zdjhMuemKeUlWhDid78mPIGLkTw",
        "baseUrl": "https://paper-api.alpaca.markets/v2"
      }
    },
    "theOddsApi": {
      "configured": true,
      "value": "***"
    },
    "e2b": {
      "configured": true,
      "value": "***"
    },
    "newsApi": {
      "configured": true,
      "value": "***"
    },
    "finnhub": {
      "configured": true,
      "value": "***"
    },
    "anthropic": {
      "configured": true,
      "value": "***"
    }
  },
  "categories": {
    "Core Trading": {
      "name": "Core Trading",
      "tools": [
        {
          "tool": "ping",
          "params": [],
          "success": true,
          "latency": 1,
          "error": null,
          "response": {
            "capabilities": [
              "trading",
              "neural",
              "gpu",
              "multi-broker",
              "sports",
              "syndicates"
            ],
            "components": {
              "nt_core": true,
              "nt_execution": true,
              "nt_strategies": true
            },
            "server": "neural-trader-mcp-napi",
            "status": "healthy",
            "timestamp": "2025-11-14T20:33:25.958766193+00:00",
            "version": "2.0.0"
          }
        },
        {
          "tool": "listStrategies",
          "params": [],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "source": "nt-strategies crate",
            "strategies": [
              {
                "description": "Momentum-based trading with technical indicators",
                "gpu_capable": false,
                "name": "momentum",
                "requires_gpu": false,
                "risk_level": "medium",
                "sharpe_ratio": null,
                "status": "available"
              },
              {
                "description": "Statistical mean reversion strategy",
                "gpu_capable": false,
                "name": "mean_reversion",
                "requires_gpu": false,
                "risk_level": "low",
                "sharpe_ratio": null,
                "status": "available"
              },
              {
                "description": "Cointegration-based pairs trading",
                "gpu_capable": false,
                "name": "pairs",
                "requires_gpu": false,
                "risk_level": "low",
                "sharpe_ratio": null,
                "status": "available"
              },
              {
                "description": "High-frequency mirror trading with neural pattern matching",
                "gpu_capable": true,
                "name": "mirror",
                "requires_gpu": false,
                "risk_level": "high",
                "sharpe_ratio": null,
                "status": "available"
              },
              {
                "description": "Enhanced momentum with neural predictions",
                "gpu_capable": true,
                "name": "enhanced_momentum",
                "requires_gpu": false,
                "risk_level": "medium",
                "sharpe_ratio": null,
                "status": "available"
              },
              {
                "description": "Neural network trend following",
                "gpu_capable": true,
                "name": "neural_trend",
                "requires_gpu": true,
                "risk_level": "medium",
                "sharpe_ratio": null,
                "status": "available"
              },
              {
                "description": "Sentiment-based neural trading",
                "gpu_capable": true,
                "name": "neural_sentiment",
                "requires_gpu": true,
                "risk_level": "high",
                "sharpe_ratio": null,
                "status": "available"
              },
              {
                "description": "Neural arbitrage opportunity detection",
                "gpu_capable": true,
                "name": "neural_arbitrage",
                "requires_gpu": true,
                "risk_level": "low",
                "sharpe_ratio": null,
                "status": "available"
              },
              {
                "description": "Ensemble of multiple strategies",
                "gpu_capable": false,
                "name": "ensemble",
                "requires_gpu": false,
                "risk_level": "medium",
                "sharpe_ratio": null,
                "status": "available"
              }
            ],
            "timestamp": "2025-11-14T20:33:25.959824572+00:00",
            "total_count": 9
          }
        },
        {
          "tool": "getStrategyInfo",
          "params": [
            "momentum"
          ],
          "success": true,
          "latency": 1,
          "error": null,
          "response": {
            "description": "Momentum-based trading with technical indicators",
            "gpu_capable": false,
            "parameters": {
              "lookback_period": {
                "default": 20,
                "range": [
                  10,
                  50
                ]
              },
              "stop_loss": {
                "default": 0.03,
                "range": [
                  0.01,
                  0.05
                ]
              },
              "take_profit": {
                "default": 0.05,
                "range": [
                  0.02,
                  0.1
                ]
              },
              "threshold": {
                "default": 0.02,
                "range": [
                  0.01,
                  0.05
                ]
              }
            },
            "performance_metrics": {
              "backtest_required": true,
              "note": "Run backtest to get performance history"
            },
            "source": "nt-strategies crate",
            "status": "configured",
            "strategy": "momentum",
            "timestamp": "2025-11-14T20:33:25.960062997+00:00"
          }
        },
        {
          "tool": "getPortfolioStatus",
          "params": [
            true
          ],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "configuration_required": {
              "env_vars": [
                "BROKER_API_KEY",
                "BROKER_API_SECRET",
                "BROKER_TYPE"
              ],
              "supported_brokers": [
                "alpaca",
                "interactive_brokers",
                "questrade",
                "oanda",
                "polygon",
                "ccxt"
              ]
            },
            "message": "Portfolio data requires broker connection",
            "mock_data_available": false,
            "status": "no_broker_configured",
            "timestamp": "2025-11-14T20:33:25.960340445+00:00"
          }
        },
        {
          "tool": "quickAnalysis",
          "params": [
            "AAPL",
            false
          ],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "data_required": {
              "bars": "Minimum 50 bars for technical indicators",
              "news": "Optional for sentiment analysis",
              "volume": "Required for volume-based indicators"
            },
            "gpu_accelerated": false,
            "indicators_available": [
              "RSI",
              "MACD",
              "SMA",
              "EMA",
              "Bollinger Bands",
              "ATR",
              "ADX",
              "Stochastic",
              "Volume Profile"
            ],
            "next_steps": "Connect market data provider to enable real-time analysis",
            "note": "Real-time analysis requires market data feed",
            "source": "nt-features crate",
            "status": "analysis_ready",
            "symbol": "AAPL",
            "timestamp": "2025-11-14T20:33:25.960521107+00:00"
          }
        },
        {
          "tool": "getMarketStatus",
          "params": [],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "next_close": "2024-11-15T16:00:00-05:00",
            "next_open": "2024-11-15T09:30:00-05:00",
            "session": "regular_trading",
            "status": "open",
            "timestamp": "2025-11-14T20:33:25.960762051+00:00"
          }
        }
      ],
      "summary": {
        "total": 6,
        "passed": 6,
        "failed": 0,
        "skipped": 0,
        "avgLatency": 0.3333333333333333,
        "totalLatency": 2
      }
    },
    "Backtesting & Optimization": {
      "name": "Backtesting & Optimization",
      "tools": [
        {
          "tool": "runBacktest",
          "params": [
            "momentum",
            "AAPL",
            "2024-01-01",
            "2024-06-01",
            null,
            true,
            "sp500"
          ],
          "success": false,
          "latency": 0,
          "error": "Failed to convert JavaScript value `Boolean true ` into rust type `String`",
          "response": null
        },
        {
          "tool": "optimizeStrategy",
          "params": [],
          "success": false,
          "latency": 0,
          "error": "Failed to convert JavaScript value `Undefined` into rust type `String`",
          "response": null
        },
        {
          "tool": "backtest Strategy",
          "params": [],
          "success": false,
          "latency": 0,
          "error": "Function 'backtest Strategy' not found in NAPI module",
          "response": null,
          "skipped": true
        },
        {
          "tool": "quickBacktest",
          "params": [
            "momentum",
            "AAPL",
            30
          ],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "days": 30,
            "return": 0.089,
            "sharpe": 2.3,
            "strategy": "momentum",
            "symbol": "AAPL",
            "timestamp": "2025-11-14T20:33:25.961245966+00:00"
          }
        },
        {
          "tool": "monteCarlo Simulation",
          "params": [],
          "success": false,
          "latency": 0,
          "error": "Function 'monteCarlo Simulation' not found in NAPI module",
          "response": null,
          "skipped": true
        },
        {
          "tool": "runBenchmark",
          "params": [
            "momentum",
            "performance",
            true
          ],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "benchmark_type": "performance",
            "gpu_accelerated": true,
            "results": {
              "execution_time_ms": 245.3,
              "latency_p50": 8.2,
              "latency_p95": 23.4,
              "throughput": 1234.5
            },
            "strategy": "momentum"
          }
        }
      ],
      "summary": {
        "total": 6,
        "passed": 2,
        "failed": 2,
        "skipped": 2,
        "avgLatency": 0,
        "totalLatency": 0
      }
    },
    "Neural Networks": {
      "name": "Neural Networks",
      "tools": [
        {
          "tool": "neuralForecast",
          "params": [
            "AAPL",
            10,
            null,
            0.95,
            false
          ],
          "success": false,
          "latency": 0,
          "error": "Failed to convert napi value into rust type `bool`",
          "response": null
        },
        {
          "tool": "neuralTrain",
          "params": [
            "/tmp/test_data.csv",
            "lstm",
            null,
            100,
            32,
            0.001,
            0.2,
            true
          ],
          "success": false,
          "latency": 0,
          "error": "Failed to convert napi value into rust type `bool`",
          "response": null
        },
        {
          "tool": "neuralEvaluate",
          "params": [
            "test_model",
            "/tmp/test_data.csv",
            null,
            true
          ],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "computation_time_ms": 67.3,
            "evaluation_id": "eval_1763152405",
            "gpu_accelerated": true,
            "metrics": {
              "directional_accuracy": 0.87,
              "mae": 0.0198,
              "mape": 0.0112,
              "r2_score": 0.94,
              "rmse": 0.0267
            },
            "model_id": "test_model",
            "predictions_vs_actual": {
              "correlation": 0.96,
              "max_error": 0.0567,
              "mean_absolute_error": 0.0198,
              "samples_evaluated": 1000
            },
            "test_data": "/tmp/test_data.csv",
            "timestamp": "2025-11-14T20:33:25.961838234+00:00"
          }
        },
        {
          "tool": "neuralBacktest",
          "params": [
            "test_model",
            "2024-01-01",
            "2024-06-01",
            "sp500",
            "daily",
            true
          ],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "backtest_id": "nb_1763152405",
            "benchmark_comparison": {
              "alpha": 0.123,
              "benchmark": "sp500",
              "benchmark_return": 0.234,
              "beta": 0.88
            },
            "computation_time_ms": 189.4,
            "gpu_accelerated": true,
            "model_id": "test_model",
            "performance": {
              "annualized_return": 0.342,
              "max_drawdown": 0.09,
              "sharpe_ratio": 3.12,
              "total_return": 0.567,
              "win_rate": 0.72
            },
            "period": {
              "end": "2024-06-01",
              "start": "2024-01-01"
            },
            "timestamp": "2025-11-14T20:33:25.962153367+00:00"
          }
        },
        {
          "tool": "neuralModelStatus",
          "params": [
            null
          ],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "models": [
              {
                "accuracy": 0.92,
                "architecture": "LSTM-Attention",
                "model_id": "lstm_v1",
                "parameters": 45632,
                "status": "ready",
                "training_date": "2024-11-01"
              }
            ],
            "timestamp": "2025-11-14T20:33:25.962296464+00:00",
            "total_models": 1
          }
        },
        {
          "tool": "neuralOptimize",
          "params": [
            "test_model",
            {
              "learning_rate": [
                0.001,
                0.01
              ],
              "batch_size": [
                16,
                64
              ]
            },
            100,
            "mae",
            true
          ],
          "success": false,
          "latency": 0,
          "error": "Failed to convert JavaScript value `Object {\"learning_rate\":[0.001,0.01],\"batch_size\":[16,64]}` into rust type `String`",
          "response": null
        },
        {
          "tool": "neuralPredict",
          "params": [
            "test_model",
            [
              "AAPL"
            ],
            true
          ],
          "success": false,
          "latency": 0,
          "error": "Failed to convert JavaScript value `Object [\"AAPL\"]` into rust type `String`",
          "response": null
        }
      ],
      "summary": {
        "total": 7,
        "passed": 3,
        "failed": 4,
        "skipped": 0,
        "avgLatency": 0,
        "totalLatency": 0
      }
    },
    "News Trading": {
      "name": "News Trading",
      "tools": [
        {
          "tool": "analyzeNews",
          "params": [
            "AAPL",
            24,
            "enhanced",
            false
          ],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "articles_analyzed": 42,
            "gpu_accelerated": false,
            "lookback_hours": 24,
            "sentiment": {
              "negative": 0.22,
              "neutral": 0,
              "overall": 0.72,
              "positive": 0.78
            },
            "symbol": "AAPL",
            "timestamp": "2025-11-14T20:33:25.962630482+00:00"
          }
        },
        {
          "tool": "getNewsSentiment",
          "params": [
            "AAPL",
            null
          ],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "articles": 28,
            "sentiment_score": 0.65,
            "sources": [
              "reuters",
              "bloomberg"
            ],
            "symbol": "AAPL",
            "timestamp": "2025-11-14T20:33:25.962764513+00:00"
          }
        },
        {
          "tool": "controlNewsCollection",
          "params": [
            "status",
            null,
            null,
            300,
            24
          ],
          "success": false,
          "latency": 0,
          "error": "Given napi value is not an array",
          "response": null
        },
        {
          "tool": "getNewsProviderStatus",
          "params": [],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "providers": [
              {
                "name": "reuters",
                "rate_limit": 1000,
                "status": "active"
              },
              {
                "name": "bloomberg",
                "rate_limit": 500,
                "status": "active"
              }
            ],
            "timestamp": "2025-11-14T20:33:25.962939193+00:00"
          }
        },
        {
          "tool": "fetchFilteredNews",
          "params": [
            [
              "AAPL"
            ],
            50,
            0.5,
            null
          ],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "articles": [],
            "filtered_count": 0,
            "symbols": [
              "AAPL"
            ],
            "timestamp": "2025-11-14T20:33:25.963078569+00:00",
            "total_count": 0
          }
        },
        {
          "tool": "getNewsTrends",
          "params": [
            [
              "AAPL"
            ],
            [
              1,
              6,
              24
            ]
          ],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "symbols": [
              "AAPL"
            ],
            "timestamp": "2025-11-14T20:33:25.963211333+00:00",
            "trends": {
              "1h": {
                "sentiment": 0.65,
                "volume": 12
              },
              "24h": {
                "sentiment": 0.72,
                "volume": 124
              },
              "6h": {
                "sentiment": 0.58,
                "volume": 45
              }
            }
          }
        }
      ],
      "summary": {
        "total": 6,
        "passed": 5,
        "failed": 1,
        "skipped": 0,
        "avgLatency": 0,
        "totalLatency": 0
      }
    },
    "Sports Betting": {
      "name": "Sports Betting",
      "tools": [
        {
          "tool": "getSportsEvents",
          "params": [
            "americanfootball_nfl",
            7,
            false
          ],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "events": [],
            "sport": "americanfootball_nfl",
            "timestamp": "2025-11-14T20:33:25.963378049+00:00"
          }
        },
        {
          "tool": "getSportsOdds",
          "params": [
            "americanfootball_nfl",
            null,
            null,
            false
          ],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "odds": [],
            "sport": "americanfootball_nfl",
            "timestamp": "2025-11-14T20:33:25.963481685+00:00"
          }
        },
        {
          "tool": "findSportsArbitrage",
          "params": [
            "americanfootball_nfl",
            0.01,
            false
          ],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "opportunities": [],
            "sport": "americanfootball_nfl",
            "timestamp": "2025-11-14T20:33:25.963582551+00:00"
          }
        },
        {
          "tool": "analyzeBettingMarketDepth",
          "params": [
            "test_market",
            "americanfootball_nfl",
            false
          ],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "depth": {},
            "market_id": "test_market",
            "sport": "americanfootball_nfl",
            "timestamp": "2025-11-14T20:33:25.963690041+00:00"
          }
        },
        {
          "tool": "calculateKellyCriterion",
          "params": [
            0.55,
            2,
            10000,
            1
          ],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "kelly_fraction": 0.10000000000000009,
            "recommended_bet": 500.00000000000045,
            "timestamp": "2025-11-14T20:33:25.963809213+00:00"
          }
        },
        {
          "tool": "getBettingPortfolioStatus",
          "params": [
            true
          ],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "portfolio": {},
            "risk": {},
            "timestamp": "2025-11-14T20:33:25.963926653+00:00"
          }
        },
        {
          "tool": "getSportsBettingPerformance",
          "params": [
            30,
            true
          ],
          "success": true,
          "latency": 1,
          "error": null,
          "response": {
            "performance": {},
            "period_days": 30,
            "timestamp": "2025-11-14T20:33:25.964043531+00:00"
          }
        }
      ],
      "summary": {
        "total": 7,
        "passed": 7,
        "failed": 0,
        "skipped": 0,
        "avgLatency": 0.14285714285714285,
        "totalLatency": 1
      }
    },
    "Odds API Integration": {
      "name": "Odds API Integration",
      "tools": [
        {
          "tool": "oddsApiGetSports",
          "params": [],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "sports": [],
            "timestamp": "2025-11-14T20:33:25.964176021+00:00"
          }
        },
        {
          "tool": "oddsApiGetLiveOdds",
          "params": [
            "americanfootball_nfl",
            "us",
            "h2h",
            "decimal",
            null
          ],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "odds": [],
            "sport": "americanfootball_nfl",
            "timestamp": "2025-11-14T20:33:25.964291293+00:00"
          }
        },
        {
          "tool": "oddsApiGetEventOdds",
          "params": [
            "americanfootball_nfl",
            "test_event",
            "us",
            "h2h",
            null
          ],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "event_id": "test_event",
            "odds": {},
            "sport": "americanfootball_nfl",
            "timestamp": "2025-11-14T20:33:25.964392662+00:00"
          }
        },
        {
          "tool": "oddsApiFindArbitrage",
          "params": [
            "americanfootball_nfl",
            "us",
            "h2h",
            0.01
          ],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "arbitrage_opportunities": [],
            "sport": "americanfootball_nfl",
            "timestamp": "2025-11-14T20:33:25.964510089+00:00"
          }
        },
        {
          "tool": "oddsApiGetBookmakerOdds",
          "params": [
            "americanfootball_nfl",
            "fanduel",
            "us",
            "h2h"
          ],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "bookmaker": "fanduel",
            "odds": [],
            "sport": "americanfootball_nfl",
            "timestamp": "2025-11-14T20:33:25.964627067+00:00"
          }
        },
        {
          "tool": "oddsApiAnalyzeMovement",
          "params": [
            "americanfootball_nfl",
            "test_event",
            5
          ],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "event_id": "test_event",
            "movement": [],
            "sport": "americanfootball_nfl",
            "timestamp": "2025-11-14T20:33:25.964737974+00:00"
          }
        }
      ],
      "summary": {
        "total": 6,
        "passed": 6,
        "failed": 0,
        "skipped": 0,
        "avgLatency": 0,
        "totalLatency": 0
      }
    },
    "Prediction Markets": {
      "name": "Prediction Markets",
      "tools": [
        {
          "tool": "getPredictionMarkets",
          "params": [
            null,
            10,
            "volume"
          ],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "markets": [],
            "timestamp": "2025-11-14T20:33:25.964929794+00:00",
            "total_count": 0
          }
        },
        {
          "tool": "analyzeMarketSentiment",
          "params": [
            "test_market",
            "standard",
            true,
            false
          ],
          "success": true,
          "latency": 1,
          "error": null,
          "response": {
            "market_id": "test_market",
            "sentiment": {},
            "timestamp": "2025-11-14T20:33:25.965048996+00:00"
          }
        },
        {
          "tool": "getMarketOrderbook",
          "params": [
            "test_market",
            10
          ],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "market_id": "test_market",
            "orderbook": {
              "asks": [],
              "bids": []
            },
            "timestamp": "2025-11-14T20:33:25.965144785+00:00"
          }
        },
        {
          "tool": "getPredictionPositions",
          "params": [],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "positions": [],
            "timestamp": "2025-11-14T20:33:25.965251442+00:00",
            "total_value": 0
          }
        },
        {
          "tool": "calculateExpectedValue",
          "params": [
            "test_market",
            100,
            1,
            true,
            false
          ],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "ev_percentage": 0.15,
            "expected_value": 114.99999999999999,
            "investment_amount": 100,
            "market_id": "test_market",
            "timestamp": "2025-11-14T20:33:25.965356280+00:00"
          }
        }
      ],
      "summary": {
        "total": 5,
        "passed": 5,
        "failed": 0,
        "skipped": 0,
        "avgLatency": 0.2,
        "totalLatency": 1
      }
    },
    "Syndicates": {
      "name": "Syndicates",
      "tools": [
        {
          "tool": "createSyndicate",
          "params": [
            "test-syndicate-1763152405965",
            "Test Syndicate",
            "Performance test"
          ],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "name": "Test Syndicate",
            "status": "created",
            "syndicate_id": "test-syndicate-1763152405965",
            "timestamp": "2025-11-14T20:33:25.965497661+00:00"
          }
        },
        {
          "tool": "addSyndicateMember",
          "params": [
            "test-syndicate",
            "Test Member",
            "test@example.com",
            "member",
            1000
          ],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "member_id": "mem_1763152405",
            "status": "added",
            "syndicate_id": "test-syndicate",
            "timestamp": "2025-11-14T20:33:25.965591819+00:00"
          }
        },
        {
          "tool": "getSyndicateStatus",
          "params": [
            "test-syndicate"
          ],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "members": 5,
            "status": "active",
            "syndicate_id": "test-syndicate",
            "timestamp": "2025-11-14T20:33:25.965716420+00:00",
            "total_capital": 50000
          }
        },
        {
          "tool": "allocateSyndicateFunds",
          "params": [
            "test-syndicate",
            "[]",
            "kelly_criterion"
          ],
          "success": true,
          "latency": 1,
          "error": null,
          "response": {
            "allocations": [],
            "syndicate_id": "test-syndicate",
            "timestamp": "2025-11-14T20:33:25.965820596+00:00",
            "total_allocated": 0
          }
        },
        {
          "tool": "distributeSyndicateProfits",
          "params": [
            "test-syndicate",
            1000,
            "hybrid"
          ],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "distributions": [],
            "syndicate_id": "test-syndicate",
            "timestamp": "2025-11-14T20:33:25.966379418+00:00",
            "total_profit": 1000
          }
        }
      ],
      "summary": {
        "total": 5,
        "passed": 5,
        "failed": 0,
        "skipped": 0,
        "avgLatency": 0.2,
        "totalLatency": 1
      }
    },
    "E2B Cloud": {
      "name": "E2B Cloud",
      "tools": [
        {
          "tool": "createE2bSandbox",
          "params": [
            "test-sandbox-1763152405966",
            "base",
            300,
            512,
            1
          ],
          "success": false,
          "latency": 0,
          "error": "Function 'createE2bSandbox' not found in NAPI module",
          "response": null,
          "skipped": true
        },
        {
          "tool": "runE2bAgent",
          "params": [],
          "success": false,
          "latency": 0,
          "error": "Function 'runE2bAgent' not found in NAPI module",
          "response": null,
          "skipped": true
        },
        {
          "tool": "executeE2bProcess",
          "params": [],
          "success": false,
          "latency": 0,
          "error": "Function 'executeE2bProcess' not found in NAPI module",
          "response": null,
          "skipped": true
        },
        {
          "tool": "listE2bSandboxes",
          "params": [
            null
          ],
          "success": false,
          "latency": 0,
          "error": "Function 'listE2bSandboxes' not found in NAPI module",
          "response": null,
          "skipped": true
        },
        {
          "tool": "getE2bSandboxStatus",
          "params": [
            "test-sandbox"
          ],
          "success": false,
          "latency": 0,
          "error": "Function 'getE2bSandboxStatus' not found in NAPI module",
          "response": null,
          "skipped": true
        }
      ],
      "summary": {
        "total": 5,
        "passed": 0,
        "failed": 0,
        "skipped": 5,
        "avgLatency": 0,
        "totalLatency": 0
      }
    },
    "System & Monitoring": {
      "name": "System & Monitoring",
      "tools": [
        {
          "tool": "getSystemMetrics",
          "params": [
            [
              "cpu",
              "memory"
            ],
            60,
            false
          ],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "metrics": {
              "cpu_usage": 45.3,
              "gpu_utilization": 78.5,
              "memory_usage": 62.1,
              "network_latency_ms": 8.2
            },
            "time_range_minutes": 60,
            "timestamp": "2025-11-14T20:33:25.966599599+00:00"
          }
        },
        {
          "tool": "getExecutionAnalytics",
          "params": [
            "1h"
          ],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "avg_latency_ms": 12.5,
            "fill_rate": 0.98,
            "time_period": "1h",
            "timestamp": "2025-11-14T20:33:25.966716467+00:00",
            "total_executions": 156
          }
        },
        {
          "tool": "performanceReport",
          "params": [
            "momentum",
            30,
            false
          ],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "gpu_accelerated": false,
            "max_drawdown": 0.08,
            "period_days": 30,
            "sharpe_ratio": 2.84,
            "strategy": "momentum",
            "total_return": 0.089,
            "win_rate": 0.68
          }
        },
        {
          "tool": "correlationAnalysis",
          "params": [
            [
              "AAPL",
              "MSFT"
            ],
            30,
            false
          ],
          "success": true,
          "latency": 0,
          "error": null,
          "response": {
            "correlation_matrix": {
              "AAPL": {
                "AAPL": 1,
                "GOOGL": 0.78,
                "MSFT": 0.82
              },
              "GOOGL": {
                "AAPL": 0.78,
                "GOOGL": 1,
                "MSFT": 0.73
              },
              "MSFT": {
                "AAPL": 0.82,
                "GOOGL": 0.73,
                "MSFT": 1
              }
            },
            "gpu_accelerated": false,
            "period_days": 30,
            "symbols": [
              "AAPL",
              "MSFT"
            ],
            "timestamp": "2025-11-14T20:33:25.966932360+00:00"
          }
        }
      ],
      "summary": {
        "total": 4,
        "passed": 4,
        "failed": 0,
        "skipped": 0,
        "avgLatency": 0,
        "totalLatency": 0
      }
    }
  },
  "summary": {
    "total": 57,
    "passed": 43,
    "failed": 7,
    "skipped": 7,
    "avgLatency": 0.11627906976744186,
    "totalDuration": 0.014
  }
}
```
