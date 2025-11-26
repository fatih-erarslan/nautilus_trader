/**
 * Test configuration fixtures for different CLI scenarios
 */

const tradingConfig = {
  trading: {
    provider: 'alpaca',
    symbols: ['AAPL', 'MSFT', 'GOOGL'],
    strategy: 'momentum',
    parameters: {
      threshold: 0.02,
      lookback: 20,
      stop_loss: 0.05
    }
  },
  risk: {
    max_position_size: 10000,
    max_portfolio_risk: 0.02,
    stop_loss_pct: 0.05
  }
};

const backtestingConfig = {
  backtesting: {
    start_date: '2024-01-01',
    end_date: '2024-12-31',
    initial_capital: 100000,
    commission: 0.001
  },
  strategy: 'mean-reversion',
  parameters: {
    window: 20,
    threshold: 2.0
  }
};

const accountingConfig = {
  accounting: {
    method: 'HIFO',
    currency: 'USD',
    tax_lots: {
      enabled: true,
      tracking: 'automated'
    }
  },
  reporting: {
    frequency: 'monthly',
    tax_year: 2024
  }
};

const predictorConfig = {
  predictor: {
    method: 'conformal',
    confidence: 0.95,
    backend: 'wasm'
  },
  training: {
    window_size: 100,
    update_frequency: 'daily'
  }
};

const pairsTradingConfig = {
  trading: {
    provider: 'alpaca',
    symbols: ['AAPL-MSFT', 'SPY-QQQ'],
    strategy: 'pairs-trading',
    parameters: {
      lookback: 60,
      entry_threshold: 2.0,
      exit_threshold: 0.5
    }
  }
};

const sportsBettingConfig = {
  betting: {
    kelly_fraction: 0.25,
    min_odds: 1.5,
    max_exposure: 0.1
  },
  bookmakers: ['pinnacle', 'betfair', 'fanduel']
};

module.exports = {
  tradingConfig,
  backtestingConfig,
  accountingConfig,
  predictorConfig,
  pairsTradingConfig,
  sportsBettingConfig
};
