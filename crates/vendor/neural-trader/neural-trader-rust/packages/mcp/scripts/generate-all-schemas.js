#!/usr/bin/env node
/**
 * Comprehensive JSON Schema 1.1 Generator for Neural Trader MCP Tools
 * Generates complete schemas for all 87+ tools following MCP 2025-11 specification
 *
 * Categories:
 * - Core Trading (23 tools)
 * - Neural Network (7 tools)
 * - News Trading (8 tools)
 * - Portfolio & Risk (5 tools)
 * - Sports Betting (22 tools: 13 core + 9 Odds API)
 * - Prediction Markets (6 tools)
 * - Syndicate Investment (17 tools)
 * - E2B Cloud (10 tools)
 *
 * Total: 87+ verified tools
 */

const fs = require('fs');
const path = require('path');

const TOOLS_DIR = path.join(__dirname, '../tools');
const SCHEMA_VERSION = 'https://json-schema.org/draft/2020-12/schema';

// Ensure tools directory exists
if (!fs.existsSync(TOOLS_DIR)) {
  fs.mkdirSync(TOOLS_DIR, { recursive: true });
}

/**
 * Tool definitions with complete input/output schemas
 */
const TOOL_DEFINITIONS = {
  // ============================================================================
  // CORE TRADING TOOLS (23 tools)
  // ============================================================================

  ping: {
    title: 'ping',
    description: 'Simple ping tool to verify server connectivity.',
    category: 'system',
    input_schema: {
      type: 'object',
      properties: {},
      required: []
    },
    output_schema: {
      type: 'object',
      properties: {
        status: { type: 'string', enum: ['ok'] },
        timestamp: { type: 'string', format: 'date-time' },
        server_version: { type: 'string' }
      },
      required: ['status', 'timestamp']
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  list_strategies: {
    title: 'list_strategies',
    description: 'List all available trading strategies with GPU capabilities.',
    category: 'trading',
    input_schema: {
      type: 'object',
      properties: {},
      required: []
    },
    output_schema: {
      type: 'object',
      properties: {
        strategies: {
          type: 'array',
          items: { type: 'string' }
        },
        total_count: { type: 'integer' },
        gpu_enabled_count: { type: 'integer' },
        models: {
          type: 'object',
          additionalProperties: {
            type: 'object',
            properties: {
              sharpe_ratio: { type: 'number' },
              gpu_accelerated: { type: 'boolean' },
              status: { type: 'string' }
            }
          }
        }
      },
      required: ['strategies', 'total_count', 'models']
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  get_strategy_info: {
    title: 'get_strategy_info',
    description: 'Get detailed information about a trading strategy.',
    category: 'trading',
    input_schema: {
      type: 'object',
      properties: {
        strategy: {
          type: 'string',
          description: 'Strategy name (e.g., mirror_trading, momentum_trading)'
        }
      },
      required: ['strategy']
    },
    output_schema: {
      type: 'object',
      properties: {
        strategy: { type: 'string' },
        details: {
          type: 'object',
          properties: {
            performance_metrics: {
              type: 'object',
              properties: {
                sharpe_ratio: { type: 'number' },
                total_return: { type: 'number' },
                max_drawdown: { type: 'number' },
                win_rate: { type: 'number' },
                total_trades: { type: 'integer' }
              }
            },
            parameters: { type: 'object' },
            gpu_accelerated: { type: 'boolean' }
          }
        },
        status: { type: 'string' }
      },
      required: ['strategy', 'details', 'status']
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  quick_analysis: {
    title: 'quick_analysis',
    description: 'Get quick market analysis for a symbol with optional GPU acceleration.',
    category: 'analysis',
    input_schema: {
      type: 'object',
      properties: {
        symbol: {
          type: 'string',
          description: 'Trading symbol (e.g., AAPL, TSLA)'
        },
        use_gpu: {
          type: 'boolean',
          default: false,
          description: 'Enable GPU acceleration'
        }
      },
      required: ['symbol']
    },
    output_schema: {
      type: 'object',
      properties: {
        symbol: { type: 'string' },
        current_price: { type: 'number' },
        change_percent: { type: 'number' },
        volume: { type: 'integer' },
        technical_indicators: {
          type: 'object',
          properties: {
            rsi: { type: 'number' },
            macd: { type: 'number' },
            moving_avg_20: { type: 'number' },
            moving_avg_50: { type: 'number' }
          }
        },
        sentiment: {
          type: 'object',
          properties: {
            score: { type: 'number', minimum: -1, maximum: 1 },
            label: { type: 'string', enum: ['bullish', 'bearish', 'neutral'] }
          }
        },
        gpu_accelerated: { type: 'boolean' },
        execution_time: { type: 'string' }
      },
      required: ['symbol', 'current_price', 'technical_indicators']
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: true
    }
  },

  simulate_trade: {
    title: 'simulate_trade',
    description: 'Simulate a trading operation with performance tracking.',
    category: 'trading',
    input_schema: {
      type: 'object',
      properties: {
        strategy: { type: 'string', description: 'Trading strategy name' },
        symbol: { type: 'string', description: 'Trading symbol' },
        action: { type: 'string', enum: ['buy', 'sell'], description: 'Trade action' },
        use_gpu: { type: 'boolean', default: false, description: 'Enable GPU acceleration' }
      },
      required: ['strategy', 'symbol', 'action']
    },
    output_schema: {
      type: 'object',
      properties: {
        simulation_id: { type: 'string' },
        strategy: { type: 'string' },
        symbol: { type: 'string' },
        action: { type: 'string' },
        entry_price: { type: 'number' },
        exit_price: { type: 'number' },
        profit_loss: { type: 'number' },
        profit_loss_percent: { type: 'number' },
        confidence: { type: 'number', minimum: 0, maximum: 1 },
        risk_metrics: {
          type: 'object',
          properties: {
            var_95: { type: 'number' },
            sharpe_ratio: { type: 'number' }
          }
        },
        execution_time: { type: 'string' },
        gpu_accelerated: { type: 'boolean' }
      },
      required: ['simulation_id', 'strategy', 'symbol', 'profit_loss']
    },
    metadata: {
      cost: 'medium',
      latency: 'medium',
      gpu_capable: true
    }
  },

  run_backtest: {
    title: 'run_backtest',
    description: 'Run comprehensive historical backtest with GPU acceleration.',
    category: 'analysis',
    input_schema: {
      type: 'object',
      properties: {
        strategy: { type: 'string', description: 'Strategy name' },
        symbol: { type: 'string', description: 'Trading symbol' },
        start_date: { type: 'string', format: 'date', description: 'Start date (YYYY-MM-DD)' },
        end_date: { type: 'string', format: 'date', description: 'End date (YYYY-MM-DD)' },
        benchmark: { type: 'string', default: 'sp500', description: 'Benchmark index' },
        include_costs: { type: 'boolean', default: true, description: 'Include transaction costs' },
        use_gpu: { type: 'boolean', default: true, description: 'Enable GPU acceleration' }
      },
      required: ['strategy', 'symbol', 'start_date', 'end_date']
    },
    output_schema: {
      type: 'object',
      properties: {
        backtest_id: { type: 'string' },
        strategy: { type: 'string' },
        symbol: { type: 'string' },
        period: { type: 'string' },
        performance: {
          type: 'object',
          properties: {
            total_return: { type: 'number' },
            annual_return: { type: 'number' },
            sharpe_ratio: { type: 'number' },
            sortino_ratio: { type: 'number' },
            max_drawdown: { type: 'number' },
            win_rate: { type: 'number' },
            profit_factor: { type: 'number' },
            total_trades: { type: 'integer' }
          }
        },
        benchmark_comparison: {
          type: 'object',
          properties: {
            alpha: { type: 'number' },
            beta: { type: 'number' },
            outperformance: { type: 'number' }
          }
        },
        gpu_metrics: {
          type: 'object',
          properties: {
            enabled: { type: 'boolean' },
            speedup: { type: 'string' },
            execution_time: { type: 'string' }
          }
        }
      },
      required: ['backtest_id', 'strategy', 'performance']
    },
    metadata: {
      cost: 'high',
      latency: 'medium',
      gpu_capable: true
    }
  },

  optimize_strategy: {
    title: 'optimize_strategy',
    description: 'Optimize strategy parameters using GPU acceleration.',
    category: 'optimization',
    input_schema: {
      type: 'object',
      properties: {
        strategy: { type: 'string', description: 'Strategy name' },
        symbol: { type: 'string', description: 'Trading symbol' },
        parameter_ranges: {
          type: 'object',
          description: 'Parameter ranges to optimize',
          additionalProperties: true
        },
        max_iterations: { type: 'integer', default: 1000, minimum: 1, maximum: 10000 },
        optimization_metric: { type: 'string', default: 'sharpe_ratio', enum: ['sharpe_ratio', 'total_return', 'profit_factor'] },
        use_gpu: { type: 'boolean', default: true, description: 'Enable GPU acceleration' }
      },
      required: ['strategy', 'symbol', 'parameter_ranges']
    },
    output_schema: {
      type: 'object',
      properties: {
        optimization_id: { type: 'string' },
        strategy: { type: 'string' },
        best_parameters: { type: 'object' },
        best_score: { type: 'number' },
        iterations_completed: { type: 'integer' },
        improvement: { type: 'number' },
        gpu_metrics: {
          type: 'object',
          properties: {
            speedup: { type: 'string' },
            execution_time: { type: 'string' }
          }
        }
      },
      required: ['optimization_id', 'best_parameters', 'best_score']
    },
    metadata: {
      cost: 'very_high',
      latency: 'slow',
      gpu_capable: true
    }
  },

  risk_analysis: {
    title: 'risk_analysis',
    description: 'Comprehensive portfolio risk analysis with GPU acceleration.',
    category: 'risk',
    input_schema: {
      type: 'object',
      properties: {
        portfolio: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              symbol: { type: 'string' },
              quantity: { type: 'number' },
              entry_price: { type: 'number' }
            },
            required: ['symbol', 'quantity']
          }
        },
        time_horizon: { type: 'integer', default: 1, description: 'Time horizon in years' },
        var_confidence: { type: 'number', default: 0.05, minimum: 0.01, maximum: 0.1 },
        use_monte_carlo: { type: 'boolean', default: true, description: 'Use Monte Carlo simulation' },
        use_gpu: { type: 'boolean', default: true, description: 'Enable GPU acceleration' }
      },
      required: ['portfolio']
    },
    output_schema: {
      type: 'object',
      properties: {
        analysis_id: { type: 'string' },
        portfolio_value: { type: 'number' },
        risk_metrics: {
          type: 'object',
          properties: {
            var: { type: 'number' },
            cvar: { type: 'number' },
            portfolio_volatility: { type: 'number' },
            sharpe_ratio: { type: 'number' },
            beta: { type: 'number' }
          }
        },
        stress_tests: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              scenario: { type: 'string' },
              loss: { type: 'number' },
              loss_percent: { type: 'number' }
            }
          }
        },
        monte_carlo_results: {
          type: 'object',
          properties: {
            simulations: { type: 'integer' },
            mean_return: { type: 'number' },
            percentile_5: { type: 'number' },
            percentile_95: { type: 'number' }
          }
        },
        gpu_metrics: {
          type: 'object',
          properties: {
            enabled: { type: 'boolean' },
            speedup: { type: 'string' }
          }
        }
      },
      required: ['analysis_id', 'risk_metrics']
    },
    metadata: {
      cost: 'very_high',
      latency: 'medium',
      gpu_capable: true
    }
  },

  execute_trade: {
    title: 'execute_trade',
    description: 'Execute live trade with advanced order management.',
    category: 'trading',
    input_schema: {
      type: 'object',
      properties: {
        strategy: { type: 'string', description: 'Strategy name' },
        symbol: { type: 'string', description: 'Trading symbol' },
        action: { type: 'string', enum: ['buy', 'sell'], description: 'Trade action' },
        quantity: { type: 'integer', minimum: 1, description: 'Number of shares' },
        order_type: { type: 'string', default: 'market', enum: ['market', 'limit', 'stop_loss'], description: 'Order type' },
        limit_price: { type: 'number', description: 'Limit price (for limit orders)' }
      },
      required: ['strategy', 'symbol', 'action', 'quantity']
    },
    output_schema: {
      type: 'object',
      properties: {
        trade_id: { type: 'string' },
        order_id: { type: 'string' },
        status: { type: 'string', enum: ['pending', 'filled', 'partial', 'cancelled', 'rejected'] },
        symbol: { type: 'string' },
        action: { type: 'string' },
        quantity: { type: 'integer' },
        filled_quantity: { type: 'integer' },
        average_price: { type: 'number' },
        commission: { type: 'number' },
        execution_time: { type: 'string', format: 'date-time' },
        message: { type: 'string' }
      },
      required: ['trade_id', 'status', 'symbol']
    },
    metadata: {
      cost: 'high',
      latency: 'medium',
      gpu_capable: false
    }
  },

  get_portfolio_status: {
    title: 'get_portfolio_status',
    description: 'Get current portfolio status with analytics.',
    category: 'portfolio',
    input_schema: {
      type: 'object',
      properties: {
        include_analytics: { type: 'boolean', default: true, description: 'Include advanced analytics' }
      }
    },
    output_schema: {
      type: 'object',
      properties: {
        portfolio_value: { type: 'number' },
        cash_balance: { type: 'number' },
        positions: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              symbol: { type: 'string' },
              quantity: { type: 'number' },
              avg_entry_price: { type: 'number' },
              current_price: { type: 'number' },
              market_value: { type: 'number' },
              unrealized_pl: { type: 'number' },
              unrealized_pl_percent: { type: 'number' }
            }
          }
        },
        analytics: {
          type: 'object',
          properties: {
            total_return: { type: 'number' },
            daily_return: { type: 'number' },
            sharpe_ratio: { type: 'number' },
            volatility: { type: 'number' }
          }
        }
      },
      required: ['portfolio_value', 'positions']
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  performance_report: {
    title: 'performance_report',
    description: 'Generate detailed performance analytics report.',
    category: 'analytics',
    input_schema: {
      type: 'object',
      properties: {
        strategy: { type: 'string', description: 'Strategy name' },
        period_days: { type: 'integer', default: 30, minimum: 1, maximum: 365 },
        include_benchmark: { type: 'boolean', default: true },
        use_gpu: { type: 'boolean', default: false }
      },
      required: ['strategy']
    },
    output_schema: {
      type: 'object',
      properties: {
        strategy: { type: 'string' },
        period: { type: 'string' },
        performance_metrics: {
          type: 'object',
          properties: {
            total_return: { type: 'number' },
            sharpe_ratio: { type: 'number' },
            max_drawdown: { type: 'number' },
            win_rate: { type: 'number' }
          }
        },
        trades: {
          type: 'object',
          properties: {
            total: { type: 'integer' },
            winning: { type: 'integer' },
            losing: { type: 'integer' }
          }
        }
      }
    },
    metadata: {
      cost: 'medium',
      latency: 'medium',
      gpu_capable: true
    }
  },

  correlation_analysis: {
    title: 'correlation_analysis',
    description: 'Analyze asset correlations with GPU acceleration.',
    category: 'analysis',
    input_schema: {
      type: 'object',
      properties: {
        symbols: {
          type: 'array',
          items: { type: 'string' },
          minItems: 2,
          description: 'List of symbols to analyze'
        },
        period_days: { type: 'integer', default: 90, minimum: 30, maximum: 365 },
        use_gpu: { type: 'boolean', default: true }
      },
      required: ['symbols']
    },
    output_schema: {
      type: 'object',
      properties: {
        correlation_matrix: {
          type: 'array',
          items: {
            type: 'array',
            items: { type: 'number' }
          }
        },
        symbols: { type: 'array', items: { type: 'string' } },
        period: { type: 'string' },
        highest_correlation: {
          type: 'object',
          properties: {
            pair: { type: 'array', items: { type: 'string' } },
            coefficient: { type: 'number' }
          }
        },
        gpu_accelerated: { type: 'boolean' }
      }
    },
    metadata: {
      cost: 'medium',
      latency: 'medium',
      gpu_capable: true
    }
  },

  run_benchmark: {
    title: 'run_benchmark',
    description: 'Run comprehensive performance benchmarks.',
    category: 'system',
    input_schema: {
      type: 'object',
      properties: {
        strategy: { type: 'string', description: 'Strategy to benchmark' },
        benchmark_type: { type: 'string', default: 'performance', enum: ['performance', 'throughput', 'latency', 'accuracy'] },
        use_gpu: { type: 'boolean', default: true }
      },
      required: ['strategy']
    },
    output_schema: {
      type: 'object',
      properties: {
        benchmark_id: { type: 'string' },
        strategy: { type: 'string' },
        results: {
          type: 'object',
          properties: {
            execution_time_cpu: { type: 'string' },
            execution_time_gpu: { type: 'string' },
            speedup: { type: 'number' },
            throughput: { type: 'number' },
            accuracy: { type: 'number' }
          }
        }
      }
    },
    metadata: {
      cost: 'high',
      latency: 'slow',
      gpu_capable: true
    }
  },

  // Additional trading tools
  recommend_strategy: {
    title: 'recommend_strategy',
    description: 'Recommend best strategy based on market conditions.',
    category: 'strategy',
    input_schema: {
      type: 'object',
      properties: {
        market_conditions: {
          type: 'object',
          description: 'Current market conditions',
          additionalProperties: true
        },
        risk_tolerance: {
          type: 'string',
          default: 'moderate',
          enum: ['conservative', 'moderate', 'aggressive']
        },
        objectives: {
          type: 'array',
          items: { type: 'string' },
          default: ['profit', 'stability']
        }
      },
      required: ['market_conditions']
    },
    output_schema: {
      type: 'object',
      properties: {
        recommended_strategy: { type: 'string' },
        confidence: { type: 'number', minimum: 0, maximum: 1 },
        reasoning: { type: 'string' },
        alternatives: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              strategy: { type: 'string' },
              score: { type: 'number' }
            }
          }
        }
      }
    },
    metadata: {
      cost: 'medium',
      latency: 'fast',
      gpu_capable: false
    }
  },

  switch_active_strategy: {
    title: 'switch_active_strategy',
    description: 'Switch from one strategy to another.',
    category: 'strategy',
    input_schema: {
      type: 'object',
      properties: {
        from_strategy: { type: 'string' },
        to_strategy: { type: 'string' },
        close_positions: { type: 'boolean', default: false }
      },
      required: ['from_strategy', 'to_strategy']
    },
    output_schema: {
      type: 'object',
      properties: {
        status: { type: 'string', enum: ['success', 'partial', 'failed'] },
        from_strategy: { type: 'string' },
        to_strategy: { type: 'string' },
        positions_closed: { type: 'integer' },
        message: { type: 'string' }
      }
    },
    metadata: {
      cost: 'medium',
      latency: 'medium',
      gpu_capable: false
    }
  },

  get_strategy_comparison: {
    title: 'get_strategy_comparison',
    description: 'Compare multiple strategies across metrics.',
    category: 'strategy',
    input_schema: {
      type: 'object',
      properties: {
        strategies: {
          type: 'array',
          items: { type: 'string' },
          minItems: 2
        },
        metrics: {
          type: 'array',
          items: { type: 'string' },
          default: ['sharpe_ratio', 'total_return', 'max_drawdown']
        }
      },
      required: ['strategies']
    },
    output_schema: {
      type: 'object',
      properties: {
        comparison: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              strategy: { type: 'string' },
              metrics: { type: 'object' },
              rank: { type: 'integer' }
            }
          }
        },
        best_overall: { type: 'string' }
      }
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  adaptive_strategy_selection: {
    title: 'adaptive_strategy_selection',
    description: 'Automatically select best strategy for current conditions.',
    category: 'strategy',
    input_schema: {
      type: 'object',
      properties: {
        symbol: { type: 'string' },
        auto_switch: { type: 'boolean', default: false }
      },
      required: ['symbol']
    },
    output_schema: {
      type: 'object',
      properties: {
        selected_strategy: { type: 'string' },
        confidence: { type: 'number' },
        switched: { type: 'boolean' },
        reasoning: { type: 'string' }
      }
    },
    metadata: {
      cost: 'medium',
      latency: 'fast',
      gpu_capable: false
    }
  },

  get_system_metrics: {
    title: 'get_system_metrics',
    description: 'Get system performance metrics.',
    category: 'system',
    input_schema: {
      type: 'object',
      properties: {
        metrics: {
          type: 'array',
          items: { type: 'string' },
          default: ['cpu', 'memory', 'latency', 'throughput']
        },
        time_range_minutes: { type: 'integer', default: 60 },
        include_history: { type: 'boolean', default: false }
      }
    },
    output_schema: {
      type: 'object',
      properties: {
        cpu_usage: { type: 'number' },
        memory_usage: { type: 'number' },
        gpu_usage: { type: 'number' },
        latency: { type: 'number' },
        throughput: { type: 'number' },
        history: { type: 'array' }
      }
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  monitor_strategy_health: {
    title: 'monitor_strategy_health',
    description: 'Monitor strategy health and performance.',
    category: 'monitoring',
    input_schema: {
      type: 'object',
      properties: {
        strategy: { type: 'string' }
      },
      required: ['strategy']
    },
    output_schema: {
      type: 'object',
      properties: {
        strategy: { type: 'string' },
        health_score: { type: 'number', minimum: 0, maximum: 100 },
        status: { type: 'string', enum: ['healthy', 'degraded', 'critical'] },
        issues: { type: 'array', items: { type: 'string' } },
        recommendations: { type: 'array', items: { type: 'string' } }
      }
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  get_execution_analytics: {
    title: 'get_execution_analytics',
    description: 'Get trade execution analytics.',
    category: 'analytics',
    input_schema: {
      type: 'object',
      properties: {
        time_period: { type: 'string', default: '1h', enum: ['1h', '24h', '7d', '30d'] }
      }
    },
    output_schema: {
      type: 'object',
      properties: {
        total_executions: { type: 'integer' },
        average_latency: { type: 'number' },
        success_rate: { type: 'number' },
        slippage: {
          type: 'object',
          properties: {
            average: { type: 'number' },
            max: { type: 'number' }
          }
        }
      }
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  execute_multi_asset_trade: {
    title: 'execute_multi_asset_trade',
    description: 'Execute trades across multiple assets.',
    category: 'trading',
    input_schema: {
      type: 'object',
      properties: {
        trades: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              symbol: { type: 'string' },
              action: { type: 'string', enum: ['buy', 'sell'] },
              quantity: { type: 'integer' }
            },
            required: ['symbol', 'action', 'quantity']
          }
        },
        strategy: { type: 'string' },
        execute_parallel: { type: 'boolean', default: true },
        risk_limit: { type: 'number' }
      },
      required: ['trades', 'strategy']
    },
    output_schema: {
      type: 'object',
      properties: {
        batch_id: { type: 'string' },
        executions: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              symbol: { type: 'string' },
              status: { type: 'string' },
              trade_id: { type: 'string' }
            }
          }
        },
        summary: {
          type: 'object',
          properties: {
            successful: { type: 'integer' },
            failed: { type: 'integer' },
            total: { type: 'integer' }
          }
        }
      }
    },
    metadata: {
      cost: 'high',
      latency: 'medium',
      gpu_capable: false
    }
  },

  portfolio_rebalance: {
    title: 'portfolio_rebalance',
    description: 'Calculate portfolio rebalancing strategy.',
    category: 'portfolio',
    input_schema: {
      type: 'object',
      properties: {
        target_allocations: {
          type: 'object',
          additionalProperties: { type: 'number' },
          description: 'Target allocation percentages by symbol'
        },
        current_portfolio: { type: 'object' },
        rebalance_threshold: { type: 'number', default: 0.05 }
      },
      required: ['target_allocations']
    },
    output_schema: {
      type: 'object',
      properties: {
        rebalance_needed: { type: 'boolean' },
        trades: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              symbol: { type: 'string' },
              action: { type: 'string' },
              quantity: { type: 'number' }
            }
          }
        },
        estimated_cost: { type: 'number' }
      }
    },
    metadata: {
      cost: 'medium',
      latency: 'fast',
      gpu_capable: false
    }
  },

  cross_asset_correlation_matrix: {
    title: 'cross_asset_correlation_matrix',
    description: 'Generate correlation matrix for multiple assets.',
    category: 'analysis',
    input_schema: {
      type: 'object',
      properties: {
        assets: {
          type: 'array',
          items: { type: 'string' },
          minItems: 2
        },
        lookback_days: { type: 'integer', default: 90 },
        include_prediction_confidence: { type: 'boolean', default: true }
      },
      required: ['assets']
    },
    output_schema: {
      type: 'object',
      properties: {
        correlation_matrix: {
          type: 'array',
          items: {
            type: 'array',
            items: { type: 'number' }
          }
        },
        assets: { type: 'array', items: { type: 'string' } }
      }
    },
    metadata: {
      cost: 'medium',
      latency: 'medium',
      gpu_capable: true
    }
  },

  // ============================================================================
  // NEURAL NETWORK TOOLS (7 tools)
  // ============================================================================

  neural_forecast: {
    title: 'neural_forecast',
    description: 'Generate neural network forecasts with confidence intervals.',
    category: 'neural',
    input_schema: {
      type: 'object',
      properties: {
        symbol: { type: 'string', description: 'Trading symbol' },
        horizon: { type: 'integer', description: 'Forecast horizon in days', minimum: 1, maximum: 365 },
        model_id: { type: 'string', description: 'Neural model ID (optional)' },
        confidence_level: { type: 'number', default: 0.95, minimum: 0.5, maximum: 0.99 },
        use_gpu: { type: 'boolean', default: true }
      },
      required: ['symbol', 'horizon']
    },
    output_schema: {
      type: 'object',
      properties: {
        forecast_id: { type: 'string' },
        symbol: { type: 'string' },
        horizon: { type: 'integer' },
        predictions: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              day: { type: 'integer' },
              predicted_price: { type: 'number' },
              lower_bound: { type: 'number' },
              upper_bound: { type: 'number' },
              confidence: { type: 'number' }
            }
          }
        },
        model_info: {
          type: 'object',
          properties: {
            model_id: { type: 'string' },
            architecture: { type: 'string' },
            accuracy_metrics: { type: 'object' }
          }
        },
        gpu_accelerated: { type: 'boolean' }
      },
      required: ['forecast_id', 'predictions']
    },
    metadata: {
      cost: 'high',
      latency: 'medium',
      gpu_capable: true
    }
  },

  neural_train: {
    title: 'neural_train',
    description: 'Train a neural forecasting model with custom configuration.',
    category: 'neural',
    input_schema: {
      type: 'object',
      properties: {
        data_path: { type: 'string', description: 'Path to training data' },
        model_type: { type: 'string', description: 'Model architecture', enum: ['lstm', 'gru', 'transformer', 'cnn_lstm'] },
        epochs: { type: 'integer', default: 100, minimum: 1, maximum: 1000 },
        batch_size: { type: 'integer', default: 32, minimum: 1, maximum: 512 },
        learning_rate: { type: 'number', default: 0.001, minimum: 0.00001, maximum: 1 },
        validation_split: { type: 'number', default: 0.2, minimum: 0.1, maximum: 0.4 },
        use_gpu: { type: 'boolean', default: true }
      },
      required: ['data_path', 'model_type']
    },
    output_schema: {
      type: 'object',
      properties: {
        model_id: { type: 'string' },
        training_metrics: {
          type: 'object',
          properties: {
            train_loss: { type: 'number' },
            val_loss: { type: 'number' },
            epochs_completed: { type: 'integer' },
            training_time: { type: 'string' }
          }
        },
        model_info: {
          type: 'object',
          properties: {
            architecture: { type: 'string' },
            parameters: { type: 'integer' },
            size_mb: { type: 'number' }
          }
        },
        gpu_accelerated: { type: 'boolean' }
      },
      required: ['model_id', 'training_metrics']
    },
    metadata: {
      cost: 'very_high',
      latency: 'slow',
      gpu_capable: true
    }
  },

  neural_evaluate: {
    title: 'neural_evaluate',
    description: 'Evaluate trained neural model on test data.',
    category: 'neural',
    input_schema: {
      type: 'object',
      properties: {
        model_id: { type: 'string', description: 'Model ID to evaluate' },
        test_data: { type: 'string', description: 'Path to test data' },
        metrics: {
          type: 'array',
          items: { type: 'string' },
          default: ['mae', 'rmse', 'mape', 'r2_score']
        },
        use_gpu: { type: 'boolean', default: true }
      },
      required: ['model_id', 'test_data']
    },
    output_schema: {
      type: 'object',
      properties: {
        model_id: { type: 'string' },
        evaluation_metrics: {
          type: 'object',
          properties: {
            mae: { type: 'number' },
            rmse: { type: 'number' },
            mape: { type: 'number' },
            r2_score: { type: 'number' }
          }
        },
        predictions_vs_actual: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              actual: { type: 'number' },
              predicted: { type: 'number' },
              error: { type: 'number' }
            }
          }
        }
      }
    },
    metadata: {
      cost: 'high',
      latency: 'medium',
      gpu_capable: true
    }
  },

  neural_backtest: {
    title: 'neural_backtest',
    description: 'Run historical backtest of neural model predictions.',
    category: 'neural',
    input_schema: {
      type: 'object',
      properties: {
        model_id: { type: 'string' },
        start_date: { type: 'string', format: 'date' },
        end_date: { type: 'string', format: 'date' },
        benchmark: { type: 'string', default: 'sp500' },
        rebalance_frequency: { type: 'string', default: 'daily', enum: ['daily', 'weekly', 'monthly'] },
        use_gpu: { type: 'boolean', default: true }
      },
      required: ['model_id', 'start_date', 'end_date']
    },
    output_schema: {
      type: 'object',
      properties: {
        backtest_id: { type: 'string' },
        performance: {
          type: 'object',
          properties: {
            total_return: { type: 'number' },
            sharpe_ratio: { type: 'number' },
            max_drawdown: { type: 'number' }
          }
        }
      }
    },
    metadata: {
      cost: 'very_high',
      latency: 'slow',
      gpu_capable: true
    }
  },

  neural_model_status: {
    title: 'neural_model_status',
    description: 'Get status and info about neural models.',
    category: 'neural',
    input_schema: {
      type: 'object',
      properties: {
        model_id: { type: 'string', description: 'Specific model ID (optional)' }
      }
    },
    output_schema: {
      type: 'object',
      properties: {
        models: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              model_id: { type: 'string' },
              status: { type: 'string' },
              architecture: { type: 'string' },
              created_at: { type: 'string' }
            }
          }
        }
      }
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  neural_optimize: {
    title: 'neural_optimize',
    description: 'Optimize neural model hyperparameters.',
    category: 'neural',
    input_schema: {
      type: 'object',
      properties: {
        model_id: { type: 'string' },
        parameter_ranges: {
          type: 'object',
          additionalProperties: true
        },
        trials: { type: 'integer', default: 100, minimum: 1, maximum: 1000 },
        optimization_metric: { type: 'string', default: 'mae' },
        use_gpu: { type: 'boolean', default: true }
      },
      required: ['model_id', 'parameter_ranges']
    },
    output_schema: {
      type: 'object',
      properties: {
        optimization_id: { type: 'string' },
        best_parameters: { type: 'object' },
        best_score: { type: 'number' },
        trials_completed: { type: 'integer' }
      }
    },
    metadata: {
      cost: 'very_high',
      latency: 'slow',
      gpu_capable: true
    }
  },

};

// Load additional tool definitions from other parts
const toolDefinitionsPart2 = require('./tool-definitions-part2');
const toolDefinitionsPart3 = require('./tool-definitions-part3');

// Merge all tool definitions
Object.assign(TOOL_DEFINITIONS, toolDefinitionsPart2, toolDefinitionsPart3);

/**
 * Generate JSON Schema for a tool
 */
function generateSchema(toolName, toolDef) {
  return {
    $schema: SCHEMA_VERSION,
    $id: `/tools/${toolName}.json`,
    title: toolDef.title,
    description: toolDef.description,
    category: toolDef.category,
    type: 'object',
    properties: {
      input_schema: toolDef.input_schema,
      output_schema: toolDef.output_schema
    },
    metadata: {
      ...toolDef.metadata,
      version: '2.0.0'
    }
  };
}

/**
 * Main execution
 */
async function main() {
  console.log('🚀 Generating JSON Schema 1.1 definitions for Neural Trader MCP tools...\n');

  let successCount = 0;
  let errorCount = 0;
  const errors = [];

  for (const [toolName, toolDef] of Object.entries(TOOL_DEFINITIONS)) {
    try {
      const schema = generateSchema(toolName, toolDef);
      const filePath = path.join(TOOLS_DIR, `${toolName}.json`);

      fs.writeFileSync(filePath, JSON.stringify(schema, null, 2) + '\n');
      successCount++;
      console.log(`✅ Generated: ${toolName}.json`);
    } catch (error) {
      errorCount++;
      errors.push({ tool: toolName, error: error.message });
      console.error(`❌ Failed: ${toolName} - ${error.message}`);
    }
  }

  console.log(`\n📊 Generation Summary:`);
  console.log(`   ✅ Success: ${successCount} schemas`);
  console.log(`   ❌ Errors: ${errorCount} schemas`);
  console.log(`   📁 Output directory: ${TOOLS_DIR}`);

  if (errors.length > 0) {
    console.log('\n⚠️  Errors:');
    errors.forEach(({ tool, error }) => {
      console.log(`   - ${tool}: ${error}`);
    });
  }

  return { successCount, errorCount, errors };
}

// Run if executed directly
if (require.main === module) {
  main().then(result => {
    process.exit(result.errorCount > 0 ? 1 : 0);
  }).catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
  });
}

module.exports = { generateSchema, TOOL_DEFINITIONS };
