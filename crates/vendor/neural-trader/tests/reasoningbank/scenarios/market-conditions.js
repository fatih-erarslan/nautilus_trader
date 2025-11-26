/**
 * Market Condition Scenarios for Benchmarking
 * Different market states to test adaptive learning
 */

const MARKET_SCENARIOS = {
  // Bull Market - Strong uptrend
  bullMarket: {
    name: 'Bull Market',
    description: 'Strong upward trend with high momentum',
    pricePattern: (basePrice, episode) => {
      const trend = 1.0 + (episode * 0.002); // 0.2% growth per episode
      const volatility = (Math.random() - 0.3) * 0.01; // Slight upward bias
      return basePrice * trend * (1 + volatility);
    },
    indicators: {
      rsiRange: [60, 80],
      macdTrend: 'positive',
      volumePattern: 'increasing',
      volatility: 'moderate'
    },
    optimalStrategy: 'trend_following',
    expectedWinRate: 0.65
  },

  // Bear Market - Strong downtrend
  bearMarket: {
    name: 'Bear Market',
    description: 'Strong downward trend with fear',
    pricePattern: (basePrice, episode) => {
      const trend = 1.0 - (episode * 0.0015); // -0.15% decline per episode
      const volatility = (Math.random() - 0.7) * 0.015; // Downward bias
      return basePrice * trend * (1 + volatility);
    },
    indicators: {
      rsiRange: [20, 40],
      macdTrend: 'negative',
      volumePattern: 'panic_selling',
      volatility: 'high'
    },
    optimalStrategy: 'short_selling',
    expectedWinRate: 0.55
  },

  // Sideways Market - Range-bound
  sidewaysMarket: {
    name: 'Sideways Market',
    description: 'Range-bound market with no clear trend',
    pricePattern: (basePrice, episode) => {
      const oscillation = Math.sin(episode * 0.3) * 0.02; // ±2% oscillation
      const noise = (Math.random() - 0.5) * 0.01;
      return basePrice * (1 + oscillation + noise);
    },
    indicators: {
      rsiRange: [40, 60],
      macdTrend: 'neutral',
      volumePattern: 'low',
      volatility: 'low'
    },
    optimalStrategy: 'mean_reversion',
    expectedWinRate: 0.52
  },

  // High Volatility - Choppy market
  highVolatility: {
    name: 'High Volatility',
    description: 'Highly volatile with large swings',
    pricePattern: (basePrice, episode) => {
      const swing = (Math.random() - 0.5) * 0.05; // ±5% swings
      const momentum = Math.sin(episode * 0.5) * 0.03;
      return basePrice * (1 + swing + momentum);
    },
    indicators: {
      rsiRange: [30, 70],
      macdTrend: 'volatile',
      volumePattern: 'high',
      volatility: 'extreme'
    },
    optimalStrategy: 'breakout',
    expectedWinRate: 0.48
  },

  // Market Crash - Rapid decline
  marketCrash: {
    name: 'Market Crash',
    description: 'Sudden and severe market decline',
    pricePattern: (basePrice, episode) => {
      const crashPhase = episode < 10 ? -0.05 : -0.01; // 5% drop then 1%
      const panic = episode < 10 ? (Math.random() - 0.9) * 0.03 : 0;
      return basePrice * (1 + crashPhase + panic);
    },
    indicators: {
      rsiRange: [10, 30],
      macdTrend: 'extreme_negative',
      volumePattern: 'extreme_high',
      volatility: 'extreme'
    },
    optimalStrategy: 'defensive',
    expectedWinRate: 0.35
  },

  // Recovery Phase - Post-crash recovery
  recoveryPhase: {
    name: 'Recovery Phase',
    description: 'Market recovering from decline',
    pricePattern: (basePrice, episode) => {
      const recovery = Math.min(episode * 0.003, 0.15); // Gradual recovery
      const consolidation = (Math.random() - 0.5) * 0.015;
      return basePrice * (1 + recovery + consolidation);
    },
    indicators: {
      rsiRange: [45, 65],
      macdTrend: 'turning_positive',
      volumePattern: 'moderate',
      volatility: 'moderate'
    },
    optimalStrategy: 'value_buying',
    expectedWinRate: 0.60
  },

  // News Event - Sudden spike
  newsEvent: {
    name: 'News Event',
    description: 'Major news causing price spike',
    pricePattern: (basePrice, episode) => {
      const eventEpisode = 15;
      const beforeEvent = episode < eventEpisode ? 0 : 0.08; // 8% spike
      const afterEvent = episode > eventEpisode + 5 ? -0.02 : 0; // Correction
      const noise = (Math.random() - 0.5) * 0.01;
      return basePrice * (1 + beforeEvent + afterEvent + noise);
    },
    indicators: {
      rsiRange: episode => episode === 15 ? [80, 95] : [50, 70],
      macdTrend: 'spike',
      volumePattern: 'extreme_high',
      volatility: 'extreme'
    },
    optimalStrategy: 'event_trading',
    expectedWinRate: 0.55
  },

  // Sector Rotation - Changing leadership
  sectorRotation: {
    name: 'Sector Rotation',
    description: 'Different sectors performing at different times',
    pricePattern: (basePrice, episode, symbol) => {
      // Different symbols have different patterns
      const symbolPhase = {
        'AAPL': episode % 30,
        'GOOGL': (episode + 10) % 30,
        'MSFT': (episode + 20) % 30,
        'TSLA': (episode + 5) % 30,
        'NVDA': (episode + 15) % 30
      };

      const phase = symbolPhase[symbol] || 0;
      const performance = phase < 10 ? 0.015 : phase < 20 ? 0 : -0.01;
      return basePrice * (1 + performance + (Math.random() - 0.5) * 0.01);
    },
    indicators: {
      rsiRange: [40, 70],
      macdTrend: 'rotating',
      volumePattern: 'sector_specific',
      volatility: 'moderate'
    },
    optimalStrategy: 'sector_momentum',
    expectedWinRate: 0.58
  }
};

/**
 * Generate market data for a scenario
 */
function generateMarketData(scenario, symbols, episodes) {
  const data = [];
  const basePrice = 100;

  for (let episode = 0; episode < episodes; episode++) {
    const episodeData = {
      episode,
      timestamp: Date.now() + (episode * 86400000), // 1 day apart
      symbols: {}
    };

    for (const symbol of symbols) {
      const price = scenario.pricePattern(basePrice, episode, symbol);
      const indicators = generateIndicators(scenario, episode, price);

      episodeData.symbols[symbol] = {
        price,
        ...indicators
      };
    }

    data.push(episodeData);
  }

  return data;
}

/**
 * Generate technical indicators for scenario
 */
function generateIndicators(scenario, episode, price) {
  const rsiRange = typeof scenario.indicators.rsiRange === 'function' ?
    scenario.indicators.rsiRange(episode) :
    scenario.indicators.rsiRange;

  const rsi = rsiRange[0] + Math.random() * (rsiRange[1] - rsiRange[0]);

  let macd;
  switch (scenario.indicators.macdTrend) {
    case 'positive':
      macd = Math.random() * 2;
      break;
    case 'negative':
      macd = -Math.random() * 2;
      break;
    case 'extreme_negative':
      macd = -(2 + Math.random() * 3);
      break;
    case 'turning_positive':
      macd = -1 + (episode * 0.1);
      break;
    case 'spike':
      macd = episode === 15 ? 5 : Math.random() - 0.5;
      break;
    default:
      macd = (Math.random() - 0.5) * 2;
  }

  const volume = generateVolume(scenario.indicators.volumePattern, episode);
  const volatility = generateVolatility(scenario.indicators.volatility);

  return {
    rsi,
    macd,
    volume,
    volatility,
    shortMA: price * (1 - 0.02 + Math.random() * 0.04),
    longMA: price * (1 - 0.05 + Math.random() * 0.1)
  };
}

function generateVolume(pattern, episode) {
  const baseVolume = 1000000;

  switch (pattern) {
    case 'increasing':
      return baseVolume * (1 + episode * 0.02);
    case 'panic_selling':
      return baseVolume * (2 + Math.random());
    case 'low':
      return baseVolume * 0.7;
    case 'high':
      return baseVolume * 1.5;
    case 'extreme_high':
      return baseVolume * (3 + Math.random() * 2);
    case 'sector_specific':
      return baseVolume * (0.8 + Math.random() * 0.6);
    default:
      return baseVolume;
  }
}

function generateVolatility(level) {
  switch (level) {
    case 'low':
      return 0.01 + Math.random() * 0.01;
    case 'moderate':
      return 0.02 + Math.random() * 0.02;
    case 'high':
      return 0.04 + Math.random() * 0.03;
    case 'extreme':
      return 0.07 + Math.random() * 0.05;
    default:
      return 0.02;
  }
}

/**
 * Evaluate strategy performance against scenario
 */
function evaluateStrategyFit(strategy, scenario, actualWinRate) {
  const isOptimal = strategy === scenario.optimalStrategy;
  const expectedWinRate = scenario.expectedWinRate;
  const performanceGap = actualWinRate - expectedWinRate;

  return {
    isOptimal,
    expectedWinRate,
    actualWinRate,
    performanceGap,
    grade: performanceGap > 0.05 ? 'excellent' :
           performanceGap > 0 ? 'good' :
           performanceGap > -0.05 ? 'fair' : 'poor'
  };
}

module.exports = {
  MARKET_SCENARIOS,
  generateMarketData,
  generateIndicators,
  evaluateStrategyFit
};
