/**
 * Neural Trader MCP Tool Definitions - Part 2
 * News Trading, Prediction Markets, Sports Betting, Syndicate, and E2B tools
 */

module.exports = {
  // ============================================================================
  // NEWS TRADING TOOLS (8 tools)
  // ============================================================================

  analyze_news: {
    title: 'analyze_news',
    description: 'AI sentiment analysis of market news for a symbol.',
    category: 'news',
    input_schema: {
      type: 'object',
      properties: {
        symbol: { type: 'string', description: 'Trading symbol' },
        lookback_hours: { type: 'integer', default: 24, minimum: 1, maximum: 168 },
        sentiment_model: { type: 'string', default: 'enhanced', enum: ['basic', 'enhanced', 'advanced'] },
        use_gpu: { type: 'boolean', default: false }
      },
      required: ['symbol']
    },
    output_schema: {
      type: 'object',
      properties: {
        symbol: { type: 'string' },
        sentiment_score: { type: 'number', minimum: -1, maximum: 1 },
        sentiment_label: { type: 'string', enum: ['very_bearish', 'bearish', 'neutral', 'bullish', 'very_bullish'] },
        articles_analyzed: { type: 'integer' },
        key_topics: { type: 'array', items: { type: 'string' } },
        impact_assessment: {
          type: 'object',
          properties: {
            short_term: { type: 'string' },
            medium_term: { type: 'string' },
            confidence: { type: 'number' }
          }
        }
      },
      required: ['symbol', 'sentiment_score', 'articles_analyzed']
    },
    metadata: {
      cost: 'medium',
      latency: 'medium',
      gpu_capable: true
    }
  },

  get_news_sentiment: {
    title: 'get_news_sentiment',
    description: 'Get real-time news sentiment for a symbol.',
    category: 'news',
    input_schema: {
      type: 'object',
      properties: {
        symbol: { type: 'string' },
        sources: { type: 'array', items: { type: 'string' }, description: 'News sources to query' }
      },
      required: ['symbol']
    },
    output_schema: {
      type: 'object',
      properties: {
        symbol: { type: 'string' },
        overall_sentiment: { type: 'number' },
        by_source: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              source: { type: 'string' },
              sentiment: { type: 'number' },
              article_count: { type: 'integer' }
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

  control_news_collection: {
    title: 'control_news_collection',
    description: 'Control news collection: start, stop, configure.',
    category: 'news',
    input_schema: {
      type: 'object',
      properties: {
        action: { type: 'string', enum: ['start', 'stop', 'pause', 'configure'] },
        symbols: { type: 'array', items: { type: 'string' } },
        sources: { type: 'array', items: { type: 'string' } },
        update_frequency: { type: 'integer', default: 300, description: 'Update frequency in seconds' },
        lookback_hours: { type: 'integer', default: 24 }
      },
      required: ['action']
    },
    output_schema: {
      type: 'object',
      properties: {
        status: { type: 'string' },
        active_symbols: { type: 'array', items: { type: 'string' } },
        active_sources: { type: 'array', items: { type: 'string' } },
        collection_rate: { type: 'string' }
      }
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  get_news_provider_status: {
    title: 'get_news_provider_status',
    description: 'Get current status of all news providers.',
    category: 'news',
    input_schema: {
      type: 'object',
      properties: {}
    },
    output_schema: {
      type: 'object',
      properties: {
        providers: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              name: { type: 'string' },
              status: { type: 'string', enum: ['active', 'inactive', 'error'] },
              articles_today: { type: 'integer' },
              last_update: { type: 'string', format: 'date-time' }
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

  fetch_filtered_news: {
    title: 'fetch_filtered_news',
    description: 'Fetch news with advanced filtering options.',
    category: 'news',
    input_schema: {
      type: 'object',
      properties: {
        symbols: { type: 'array', items: { type: 'string' } },
        sentiment_filter: { type: 'string', enum: ['positive', 'negative', 'neutral', 'all'] },
        relevance_threshold: { type: 'number', default: 0.5, minimum: 0, maximum: 1 },
        limit: { type: 'integer', default: 50, maximum: 500 }
      },
      required: ['symbols']
    },
    output_schema: {
      type: 'object',
      properties: {
        articles: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              title: { type: 'string' },
              source: { type: 'string' },
              sentiment: { type: 'number' },
              relevance: { type: 'number' },
              published_at: { type: 'string', format: 'date-time' }
            }
          }
        },
        total_found: { type: 'integer' }
      }
    },
    metadata: {
      cost: 'medium',
      latency: 'medium',
      gpu_capable: false
    }
  },

  get_news_trends: {
    title: 'get_news_trends',
    description: 'Analyze news trends over multiple time intervals.',
    category: 'news',
    input_schema: {
      type: 'object',
      properties: {
        symbols: { type: 'array', items: { type: 'string' } },
        time_intervals: { type: 'array', items: { type: 'integer' }, default: [1, 6, 24] }
      },
      required: ['symbols']
    },
    output_schema: {
      type: 'object',
      properties: {
        trends: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              symbol: { type: 'string' },
              intervals: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    hours: { type: 'integer' },
                    sentiment: { type: 'number' },
                    volume: { type: 'integer' },
                    momentum: { type: 'string' }
                  }
                }
              }
            }
          }
        }
      }
    },
    metadata: {
      cost: 'medium',
      latency: 'medium',
      gpu_capable: false
    }
  },

  // ============================================================================
  // PREDICTION MARKETS TOOLS (6 tools)
  // ============================================================================

  get_prediction_markets_tool: {
    title: 'get_prediction_markets_tool',
    description: 'List available prediction markets with filtering.',
    category: 'prediction',
    input_schema: {
      type: 'object',
      properties: {
        category: { type: 'string', description: 'Market category filter' },
        sort_by: { type: 'string', default: 'volume', enum: ['volume', 'liquidity', 'popularity'] },
        limit: { type: 'integer', default: 10, maximum: 100 }
      }
    },
    output_schema: {
      type: 'object',
      properties: {
        markets: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              market_id: { type: 'string' },
              title: { type: 'string' },
              category: { type: 'string' },
              volume_24h: { type: 'number' },
              liquidity: { type: 'number' },
              outcomes: { type: 'array', items: { type: 'string' } }
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

  analyze_market_sentiment_tool: {
    title: 'analyze_market_sentiment_tool',
    description: 'Analyze market probabilities and sentiment.',
    category: 'prediction',
    input_schema: {
      type: 'object',
      properties: {
        market_id: { type: 'string' },
        analysis_depth: { type: 'string', default: 'standard', enum: ['quick', 'standard', 'deep'] },
        include_correlations: { type: 'boolean', default: true },
        use_gpu: { type: 'boolean', default: false }
      },
      required: ['market_id']
    },
    output_schema: {
      type: 'object',
      properties: {
        market_id: { type: 'string' },
        probabilities: { type: 'object' },
        sentiment_score: { type: 'number' },
        confidence: { type: 'number' },
        correlations: { type: 'array' }
      }
    },
    metadata: {
      cost: 'medium',
      latency: 'medium',
      gpu_capable: true
    }
  },

  get_market_orderbook_tool: {
    title: 'get_market_orderbook_tool',
    description: 'Get market depth and orderbook data.',
    category: 'prediction',
    input_schema: {
      type: 'object',
      properties: {
        market_id: { type: 'string' },
        depth: { type: 'integer', default: 10, maximum: 50 }
      },
      required: ['market_id']
    },
    output_schema: {
      type: 'object',
      properties: {
        market_id: { type: 'string' },
        bids: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              price: { type: 'number' },
              size: { type: 'number' }
            }
          }
        },
        asks: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              price: { type: 'number' },
              size: { type: 'number' }
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

  place_prediction_order_tool: {
    title: 'place_prediction_order_tool',
    description: 'Place market orders (demo mode).',
    category: 'prediction',
    input_schema: {
      type: 'object',
      properties: {
        market_id: { type: 'string' },
        outcome: { type: 'string' },
        side: { type: 'string', enum: ['buy', 'sell'] },
        quantity: { type: 'integer', minimum: 1 },
        order_type: { type: 'string', default: 'market', enum: ['market', 'limit'] },
        limit_price: { type: 'number' }
      },
      required: ['market_id', 'outcome', 'side', 'quantity']
    },
    output_schema: {
      type: 'object',
      properties: {
        order_id: { type: 'string' },
        status: { type: 'string' },
        filled_quantity: { type: 'integer' },
        average_price: { type: 'number' }
      }
    },
    metadata: {
      cost: 'medium',
      latency: 'medium',
      gpu_capable: false
    }
  },

  get_prediction_positions_tool: {
    title: 'get_prediction_positions_tool',
    description: 'Get current prediction market positions.',
    category: 'prediction',
    input_schema: {
      type: 'object',
      properties: {}
    },
    output_schema: {
      type: 'object',
      properties: {
        positions: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              market_id: { type: 'string' },
              outcome: { type: 'string' },
              quantity: { type: 'number' },
              avg_entry_price: { type: 'number' },
              current_value: { type: 'number' },
              unrealized_pl: { type: 'number' }
            }
          }
        },
        total_value: { type: 'number' }
      }
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  calculate_expected_value_tool: {
    title: 'calculate_expected_value_tool',
    description: 'Calculate expected value for prediction markets.',
    category: 'prediction',
    input_schema: {
      type: 'object',
      properties: {
        market_id: { type: 'string' },
        investment_amount: { type: 'number', minimum: 0 },
        confidence_adjustment: { type: 'number', default: 1.0 },
        include_fees: { type: 'boolean', default: true },
        use_gpu: { type: 'boolean', default: false }
      },
      required: ['market_id', 'investment_amount']
    },
    output_schema: {
      type: 'object',
      properties: {
        expected_value: { type: 'number' },
        kelly_criterion: { type: 'number' },
        recommended_size: { type: 'number' },
        risk_reward_ratio: { type: 'number' }
      }
    },
    metadata: {
      cost: 'medium',
      latency: 'fast',
      gpu_capable: true
    }
  },

  // ============================================================================
  // SPORTS BETTING TOOLS (22 tools: 13 core + 9 Odds API)
  // ============================================================================

  get_sports_events: {
    title: 'get_sports_events',
    description: 'Get upcoming sports events with analysis.',
    category: 'sports',
    input_schema: {
      type: 'object',
      properties: {
        sport: { type: 'string', enum: ['basketball', 'football', 'baseball', 'hockey', 'soccer'] },
        days_ahead: { type: 'integer', default: 7, minimum: 1, maximum: 30 },
        use_gpu: { type: 'boolean', default: false }
      },
      required: ['sport']
    },
    output_schema: {
      type: 'object',
      properties: {
        events: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              event_id: { type: 'string' },
              sport: { type: 'string' },
              home_team: { type: 'string' },
              away_team: { type: 'string' },
              start_time: { type: 'string', format: 'date-time' },
              venue: { type: 'string' }
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

  get_sports_odds: {
    title: 'get_sports_odds',
    description: 'Get real-time sports betting odds with market analysis.',
    category: 'sports',
    input_schema: {
      type: 'object',
      properties: {
        sport: { type: 'string' },
        market_types: { type: 'array', items: { type: 'string' } },
        regions: { type: 'array', items: { type: 'string' } },
        use_gpu: { type: 'boolean', default: false }
      },
      required: ['sport']
    },
    output_schema: {
      type: 'object',
      properties: {
        odds: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              event_id: { type: 'string' },
              bookmaker: { type: 'string' },
              market_type: { type: 'string' },
              odds: { type: 'object' }
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

  find_sports_arbitrage: {
    title: 'find_sports_arbitrage',
    description: 'Find arbitrage opportunities in sports betting.',
    category: 'sports',
    input_schema: {
      type: 'object',
      properties: {
        sport: { type: 'string' },
        min_profit_margin: { type: 'number', default: 0.01, minimum: 0.001 },
        use_gpu: { type: 'boolean', default: false }
      },
      required: ['sport']
    },
    output_schema: {
      type: 'object',
      properties: {
        opportunities: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              event_id: { type: 'string' },
              profit_margin: { type: 'number' },
              bookmakers: { type: 'array' },
              stakes: { type: 'object' }
            }
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

  analyze_betting_market_depth: {
    title: 'analyze_betting_market_depth',
    description: 'Analyze betting market depth and liquidity.',
    category: 'sports',
    input_schema: {
      type: 'object',
      properties: {
        market_id: { type: 'string' },
        sport: { type: 'string' },
        use_gpu: { type: 'boolean', default: false }
      },
      required: ['market_id', 'sport']
    },
    output_schema: {
      type: 'object',
      properties: {
        market_id: { type: 'string' },
        liquidity_score: { type: 'number' },
        depth_analysis: { type: 'object' },
        recommendation: { type: 'string' }
      }
    },
    metadata: {
      cost: 'medium',
      latency: 'medium',
      gpu_capable: true
    }
  },

  calculate_kelly_criterion: {
    title: 'calculate_kelly_criterion',
    description: 'Calculate optimal bet size using Kelly Criterion.',
    category: 'sports',
    input_schema: {
      type: 'object',
      properties: {
        probability: { type: 'number', minimum: 0, maximum: 1 },
        odds: { type: 'number', minimum: 1 },
        bankroll: { type: 'number', minimum: 0 },
        confidence: { type: 'number', default: 1, minimum: 0, maximum: 1 }
      },
      required: ['probability', 'odds', 'bankroll']
    },
    output_schema: {
      type: 'object',
      properties: {
        kelly_fraction: { type: 'number' },
        recommended_stake: { type: 'number' },
        expected_value: { type: 'number' },
        risk_of_ruin: { type: 'number' }
      }
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  simulate_betting_strategy: {
    title: 'simulate_betting_strategy',
    description: 'Simulate betting strategy with Monte Carlo.',
    category: 'sports',
    input_schema: {
      type: 'object',
      properties: {
        strategy_config: { type: 'object', additionalProperties: true },
        num_simulations: { type: 'integer', default: 1000, maximum: 100000 },
        use_gpu: { type: 'boolean', default: false }
      },
      required: ['strategy_config']
    },
    output_schema: {
      type: 'object',
      properties: {
        simulation_id: { type: 'string' },
        results: {
          type: 'object',
          properties: {
            mean_return: { type: 'number' },
            median_return: { type: 'number' },
            win_probability: { type: 'number' },
            risk_of_ruin: { type: 'number' }
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

  get_betting_portfolio_status: {
    title: 'get_betting_portfolio_status',
    description: 'Get betting portfolio status and risk metrics.',
    category: 'sports',
    input_schema: {
      type: 'object',
      properties: {
        include_risk_analysis: { type: 'boolean', default: true }
      }
    },
    output_schema: {
      type: 'object',
      properties: {
        total_bankroll: { type: 'number' },
        active_bets: { type: 'integer' },
        pending_bets: { type: 'integer' },
        total_exposed: { type: 'number' },
        risk_metrics: { type: 'object' }
      }
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  execute_sports_bet: {
    title: 'execute_sports_bet',
    description: 'Execute sports bet with validation.',
    category: 'sports',
    input_schema: {
      type: 'object',
      properties: {
        market_id: { type: 'string' },
        selection: { type: 'string' },
        stake: { type: 'number', minimum: 0 },
        odds: { type: 'number' },
        bet_type: { type: 'string', default: 'back', enum: ['back', 'lay'] },
        validate_only: { type: 'boolean', default: true }
      },
      required: ['market_id', 'selection', 'stake', 'odds']
    },
    output_schema: {
      type: 'object',
      properties: {
        bet_id: { type: 'string' },
        status: { type: 'string' },
        validation: { type: 'object' }
      }
    },
    metadata: {
      cost: 'medium',
      latency: 'medium',
      gpu_capable: false
    }
  },

  get_sports_betting_performance: {
    title: 'get_sports_betting_performance',
    description: 'Get sports betting performance analytics.',
    category: 'sports',
    input_schema: {
      type: 'object',
      properties: {
        period_days: { type: 'integer', default: 30 },
        include_detailed_analysis: { type: 'boolean', default: true }
      }
    },
    output_schema: {
      type: 'object',
      properties: {
        total_bets: { type: 'integer' },
        win_rate: { type: 'number' },
        roi: { type: 'number' },
        profit_loss: { type: 'number' }
      }
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  compare_betting_providers: {
    title: 'compare_betting_providers',
    description: 'Compare odds across betting providers.',
    category: 'sports',
    input_schema: {
      type: 'object',
      properties: {
        sport: { type: 'string' },
        event_filter: { type: 'string' },
        use_gpu: { type: 'boolean', default: false }
      },
      required: ['sport']
    },
    output_schema: {
      type: 'object',
      properties: {
        comparisons: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              event: { type: 'string' },
              best_odds: { type: 'object' },
              providers: { type: 'array' }
            }
          }
        }
      }
    },
    metadata: {
      cost: 'medium',
      latency: 'medium',
      gpu_capable: false
    }
  },

  // Odds API tools (9 tools)
  odds_api_get_sports: {
    title: 'odds_api_get_sports',
    description: 'Get list of available sports from The Odds API.',
    category: 'odds_api',
    input_schema: {
      type: 'object',
      properties: {}
    },
    output_schema: {
      type: 'object',
      properties: {
        sports: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              key: { type: 'string' },
              group: { type: 'string' },
              title: { type: 'string' },
              description: { type: 'string' },
              active: { type: 'boolean' }
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

  odds_api_get_live_odds: {
    title: 'odds_api_get_live_odds',
    description: 'Get live odds for a specific sport.',
    category: 'odds_api',
    input_schema: {
      type: 'object',
      properties: {
        sport: { type: 'string' },
        regions: { type: 'string', default: 'us' },
        markets: { type: 'string', default: 'h2h' },
        odds_format: { type: 'string', default: 'decimal', enum: ['decimal', 'american'] },
        bookmakers: { type: 'string' }
      },
      required: ['sport']
    },
    output_schema: {
      type: 'object',
      properties: {
        events: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              id: { type: 'string' },
              sport_key: { type: 'string' },
              commence_time: { type: 'string' },
              home_team: { type: 'string' },
              away_team: { type: 'string' },
              bookmakers: { type: 'array' }
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

  odds_api_get_event_odds: {
    title: 'odds_api_get_event_odds',
    description: 'Get detailed odds for a specific event.',
    category: 'odds_api',
    input_schema: {
      type: 'object',
      properties: {
        sport: { type: 'string' },
        event_id: { type: 'string' },
        regions: { type: 'string', default: 'us' },
        markets: { type: 'string', default: 'h2h,spreads,totals' },
        bookmakers: { type: 'string' }
      },
      required: ['sport', 'event_id']
    },
    output_schema: {
      type: 'object',
      properties: {
        event: {
          type: 'object',
          properties: {
            id: { type: 'string' },
            bookmakers: { type: 'array' }
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

  odds_api_find_arbitrage: {
    title: 'odds_api_find_arbitrage',
    description: 'Find arbitrage opportunities across bookmakers.',
    category: 'odds_api',
    input_schema: {
      type: 'object',
      properties: {
        sport: { type: 'string' },
        markets: { type: 'string', default: 'h2h' },
        regions: { type: 'string', default: 'us,uk,au' },
        min_profit_margin: { type: 'number', default: 0.01 }
      },
      required: ['sport']
    },
    output_schema: {
      type: 'object',
      properties: {
        opportunities: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              event: { type: 'string' },
              profit_margin: { type: 'number' },
              bets: { type: 'array' }
            }
          }
        }
      }
    },
    metadata: {
      cost: 'medium',
      latency: 'medium',
      gpu_capable: false
    }
  },

  odds_api_get_bookmaker_odds: {
    title: 'odds_api_get_bookmaker_odds',
    description: 'Get odds from a specific bookmaker.',
    category: 'odds_api',
    input_schema: {
      type: 'object',
      properties: {
        sport: { type: 'string' },
        bookmaker: { type: 'string' },
        regions: { type: 'string', default: 'us' },
        markets: { type: 'string', default: 'h2h' }
      },
      required: ['sport', 'bookmaker']
    },
    output_schema: {
      type: 'object',
      properties: {
        events: { type: 'array' }
      }
    },
    metadata: {
      cost: 'medium',
      latency: 'fast',
      gpu_capable: false
    }
  },

  odds_api_analyze_movement: {
    title: 'odds_api_analyze_movement',
    description: 'Analyze odds movement over time.',
    category: 'odds_api',
    input_schema: {
      type: 'object',
      properties: {
        sport: { type: 'string' },
        event_id: { type: 'string' },
        intervals: { type: 'integer', default: 5 }
      },
      required: ['sport', 'event_id']
    },
    output_schema: {
      type: 'object',
      properties: {
        movement: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              timestamp: { type: 'string' },
              odds: { type: 'object' },
              change: { type: 'number' }
            }
          }
        }
      }
    },
    metadata: {
      cost: 'medium',
      latency: 'medium',
      gpu_capable: false
    }
  },

  odds_api_calculate_probability: {
    title: 'odds_api_calculate_probability',
    description: 'Calculate implied probability from odds.',
    category: 'odds_api',
    input_schema: {
      type: 'object',
      properties: {
        odds: { type: 'number' },
        odds_format: { type: 'string', default: 'decimal', enum: ['decimal', 'american'] }
      },
      required: ['odds']
    },
    output_schema: {
      type: 'object',
      properties: {
        implied_probability: { type: 'number' }
      }
    },
    metadata: {
      cost: 'low',
      latency: 'fast',
      gpu_capable: false
    }
  },

  odds_api_compare_margins: {
    title: 'odds_api_compare_margins',
    description: 'Compare bookmaker margins.',
    category: 'odds_api',
    input_schema: {
      type: 'object',
      properties: {
        sport: { type: 'string' },
        regions: { type: 'string', default: 'us' },
        markets: { type: 'string', default: 'h2h' }
      },
      required: ['sport']
    },
    output_schema: {
      type: 'object',
      properties: {
        margins: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              bookmaker: { type: 'string' },
              average_margin: { type: 'number' }
            }
          }
        }
      }
    },
    metadata: {
      cost: 'medium',
      latency: 'medium',
      gpu_capable: false
    }
  },

  odds_api_get_upcoming: {
    title: 'odds_api_get_upcoming',
    description: 'Get upcoming events with odds.',
    category: 'odds_api',
    input_schema: {
      type: 'object',
      properties: {
        sport: { type: 'string' },
        days_ahead: { type: 'integer', default: 7 },
        regions: { type: 'string', default: 'us' },
        markets: { type: 'string', default: 'h2h' }
      },
      required: ['sport']
    },
    output_schema: {
      type: 'object',
      properties: {
        events: { type: 'array' }
      }
    },
    metadata: {
      cost: 'medium',
      latency: 'fast',
      gpu_capable: false
    }
  }
};
