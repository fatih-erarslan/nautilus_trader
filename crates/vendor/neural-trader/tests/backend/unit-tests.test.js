/**
 * Comprehensive Unit Tests for Neural Trader Backend
 * Coverage Target: 95%+
 *
 * Test Categories:
 * - Trading Functions (listStrategies, quickAnalysis, simulateTrade, etc.)
 * - Neural Functions (neuralForecast, neuralTrain, neuralEvaluate, etc.)
 * - Sports Betting (getSportsEvents, findSportsArbitrage, calculateKellyCriterion, etc.)
 * - Syndicate Management (createSyndicate, allocateSyndicateFunds, etc.)
 * - E2B Swarm Operations (initE2bSwarm, deployTradingAgent, scaleSwarm, etc.)
 * - Security Features (initAuth, createApiKey, validateToken, etc.)
 * - Prediction Markets (getPredictionMarkets, analyzeMarketSentiment, etc.)
 */

const backend = require('../../neural-trader-rust/packages/neural-trader-backend');

describe('Neural Trader Backend - Unit Tests', () => {

  // ============================================================================
  // TRADING FUNCTIONS
  // ============================================================================

  describe('Trading Functions', () => {
    describe('listStrategies()', () => {
      it('should return array of available strategies', async () => {
        const strategies = await backend.listStrategies();

        expect(Array.isArray(strategies)).toBe(true);
        expect(strategies.length).toBeGreaterThan(0);

        strategies.forEach(strategy => {
          expect(strategy).toHaveProperty('name');
          expect(strategy).toHaveProperty('description');
          expect(strategy).toHaveProperty('gpuCapable');
          expect(typeof strategy.name).toBe('string');
          expect(typeof strategy.description).toBe('string');
          expect(typeof strategy.gpuCapable).toBe('boolean');
        });
      });

      it('should include common strategies', async () => {
        const strategies = await backend.listStrategies();
        const strategyNames = strategies.map(s => s.name.toLowerCase());

        expect(strategyNames.some(name => name.includes('momentum'))).toBe(true);
        expect(strategyNames.some(name => name.includes('mean') || name.includes('reversion'))).toBe(true);
      });

      it('should handle errors gracefully', async () => {
        // Test that function doesn't throw even under stress
        await expect(backend.listStrategies()).resolves.toBeDefined();
      });
    });

    describe('getStrategyInfo()', () => {
      it('should return detailed strategy information', async () => {
        const strategies = await backend.listStrategies();
        const strategy = strategies[0];

        const info = await backend.getStrategyInfo(strategy.name);

        expect(typeof info).toBe('string');
        expect(info.length).toBeGreaterThan(0);
      });

      it('should handle invalid strategy names', async () => {
        await expect(backend.getStrategyInfo('nonexistent_strategy'))
          .rejects.toThrow();
      });

      it('should handle empty string', async () => {
        await expect(backend.getStrategyInfo(''))
          .rejects.toThrow();
      });

      it('should handle special characters', async () => {
        await expect(backend.getStrategyInfo('strategy<script>alert(1)</script>'))
          .rejects.toThrow();
      });
    });

    describe('quickAnalysis()', () => {
      it('should analyze symbol with default options', async () => {
        const analysis = await backend.quickAnalysis('AAPL');

        expect(analysis).toHaveProperty('symbol', 'AAPL');
        expect(analysis).toHaveProperty('trend');
        expect(analysis).toHaveProperty('volatility');
        expect(analysis).toHaveProperty('volumeTrend');
        expect(analysis).toHaveProperty('recommendation');

        expect(typeof analysis.trend).toBe('string');
        expect(typeof analysis.volatility).toBe('number');
        expect(analysis.volatility).toBeGreaterThanOrEqual(0);
      });

      it('should support GPU acceleration flag', async () => {
        const analysisGpu = await backend.quickAnalysis('AAPL', true);
        const analysisCpu = await backend.quickAnalysis('AAPL', false);

        expect(analysisGpu).toHaveProperty('symbol', 'AAPL');
        expect(analysisCpu).toHaveProperty('symbol', 'AAPL');
      });

      it('should handle various stock symbols', async () => {
        const symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'];

        for (const symbol of symbols) {
          const analysis = await backend.quickAnalysis(symbol);
          expect(analysis.symbol).toBe(symbol);
        }
      });

      it('should reject invalid symbols', async () => {
        await expect(backend.quickAnalysis('')).rejects.toThrow();
        await expect(backend.quickAnalysis('!!!!')).rejects.toThrow();
      });

      it('should handle null/undefined GPU flag', async () => {
        const analysis1 = await backend.quickAnalysis('AAPL', null);
        const analysis2 = await backend.quickAnalysis('AAPL', undefined);

        expect(analysis1).toHaveProperty('symbol');
        expect(analysis2).toHaveProperty('symbol');
      });
    });

    describe('simulateTrade()', () => {
      it('should simulate trade with valid parameters', async () => {
        const simulation = await backend.simulateTrade('momentum', 'AAPL', 'buy', false);

        expect(simulation).toHaveProperty('strategy', 'momentum');
        expect(simulation).toHaveProperty('symbol', 'AAPL');
        expect(simulation).toHaveProperty('action', 'buy');
        expect(simulation).toHaveProperty('expectedReturn');
        expect(simulation).toHaveProperty('riskScore');
        expect(simulation).toHaveProperty('executionTimeMs');

        expect(typeof simulation.expectedReturn).toBe('number');
        expect(typeof simulation.riskScore).toBe('number');
        expect(simulation.riskScore).toBeGreaterThanOrEqual(0);
        expect(simulation.riskScore).toBeLessThanOrEqual(1);
        expect(simulation.executionTimeMs).toBeGreaterThan(0);
      });

      it('should support buy and sell actions', async () => {
        const buySimulation = await backend.simulateTrade('momentum', 'AAPL', 'buy');
        const sellSimulation = await backend.simulateTrade('momentum', 'AAPL', 'sell');

        expect(buySimulation.action).toBe('buy');
        expect(sellSimulation.action).toBe('sell');
      });

      it('should use GPU when specified', async () => {
        const gpuSim = await backend.simulateTrade('momentum', 'AAPL', 'buy', true);
        expect(gpuSim).toHaveProperty('executionTimeMs');
      });

      it('should reject invalid strategies', async () => {
        await expect(backend.simulateTrade('invalid_strategy', 'AAPL', 'buy'))
          .rejects.toThrow();
      });

      it('should reject invalid actions', async () => {
        await expect(backend.simulateTrade('momentum', 'AAPL', 'invalid_action'))
          .rejects.toThrow();
      });
    });

    describe('getPortfolioStatus()', () => {
      it('should return portfolio status with basic info', async () => {
        const status = await backend.getPortfolioStatus(false);

        expect(status).toHaveProperty('totalValue');
        expect(status).toHaveProperty('cash');
        expect(status).toHaveProperty('positions');
        expect(status).toHaveProperty('dailyPnl');
        expect(status).toHaveProperty('totalReturn');

        expect(typeof status.totalValue).toBe('number');
        expect(typeof status.cash).toBe('number');
        expect(typeof status.positions).toBe('number');
        expect(status.positions).toBeGreaterThanOrEqual(0);
      });

      it('should include analytics when requested', async () => {
        const statusWithAnalytics = await backend.getPortfolioStatus(true);
        expect(statusWithAnalytics).toHaveProperty('totalValue');
      });

      it('should work without parameters', async () => {
        const status = await backend.getPortfolioStatus();
        expect(status).toHaveProperty('totalValue');
      });
    });

    describe('executeTrade()', () => {
      it('should execute market order', async () => {
        const execution = await backend.executeTrade(
          'momentum',
          'AAPL',
          'buy',
          10,
          'market'
        );

        expect(execution).toHaveProperty('orderId');
        expect(execution).toHaveProperty('strategy', 'momentum');
        expect(execution).toHaveProperty('symbol', 'AAPL');
        expect(execution).toHaveProperty('action', 'buy');
        expect(execution).toHaveProperty('quantity', 10);
        expect(execution).toHaveProperty('status');
        expect(execution).toHaveProperty('fillPrice');

        expect(execution.orderId.length).toBeGreaterThan(0);
        expect(execution.fillPrice).toBeGreaterThan(0);
      });

      it('should execute limit order', async () => {
        const execution = await backend.executeTrade(
          'momentum',
          'AAPL',
          'buy',
          10,
          'limit',
          150.50
        );

        expect(execution).toHaveProperty('orderId');
        expect(execution.quantity).toBe(10);
      });

      it('should reject negative quantities', async () => {
        await expect(
          backend.executeTrade('momentum', 'AAPL', 'buy', -10)
        ).rejects.toThrow();
      });

      it('should reject zero quantities', async () => {
        await expect(
          backend.executeTrade('momentum', 'AAPL', 'buy', 0)
        ).rejects.toThrow();
      });

      it('should handle fractional quantities', async () => {
        const execution = await backend.executeTrade(
          'momentum',
          'AAPL',
          'buy',
          10.5
        );
        expect(execution).toHaveProperty('quantity');
      });
    });

    describe('runBacktest()', () => {
      it('should run backtest with valid parameters', async () => {
        const result = await backend.runBacktest(
          'momentum',
          'AAPL',
          '2023-01-01',
          '2023-12-31',
          false
        );

        expect(result).toHaveProperty('strategy', 'momentum');
        expect(result).toHaveProperty('symbol', 'AAPL');
        expect(result).toHaveProperty('startDate', '2023-01-01');
        expect(result).toHaveProperty('endDate', '2023-12-31');
        expect(result).toHaveProperty('totalReturn');
        expect(result).toHaveProperty('sharpeRatio');
        expect(result).toHaveProperty('maxDrawdown');
        expect(result).toHaveProperty('totalTrades');
        expect(result).toHaveProperty('winRate');

        expect(typeof result.totalReturn).toBe('number');
        expect(typeof result.sharpeRatio).toBe('number');
        expect(result.maxDrawdown).toBeLessThanOrEqual(0);
        expect(result.totalTrades).toBeGreaterThanOrEqual(0);
        expect(result.winRate).toBeGreaterThanOrEqual(0);
        expect(result.winRate).toBeLessThanOrEqual(1);
      });

      it('should use GPU when enabled', async () => {
        const result = await backend.runBacktest(
          'momentum',
          'AAPL',
          '2023-01-01',
          '2023-12-31',
          true
        );
        expect(result).toHaveProperty('totalReturn');
      });

      it('should reject invalid date ranges', async () => {
        await expect(
          backend.runBacktest('momentum', 'AAPL', '2023-12-31', '2023-01-01')
        ).rejects.toThrow();
      });

      it('should reject invalid date formats', async () => {
        await expect(
          backend.runBacktest('momentum', 'AAPL', 'invalid-date', '2023-12-31')
        ).rejects.toThrow();
      });
    });
  });

  // ============================================================================
  // NEURAL FUNCTIONS
  // ============================================================================

  describe('Neural Functions', () => {
    describe('neuralForecast()', () => {
      it('should generate forecasts with default parameters', async () => {
        const forecast = await backend.neuralForecast('AAPL', 30);

        expect(forecast).toHaveProperty('symbol', 'AAPL');
        expect(forecast).toHaveProperty('horizon', 30);
        expect(forecast).toHaveProperty('predictions');
        expect(forecast).toHaveProperty('confidenceIntervals');
        expect(forecast).toHaveProperty('modelAccuracy');

        expect(Array.isArray(forecast.predictions)).toBe(true);
        expect(forecast.predictions.length).toBe(30);
        expect(Array.isArray(forecast.confidenceIntervals)).toBe(true);
        expect(forecast.confidenceIntervals.length).toBe(30);

        forecast.confidenceIntervals.forEach(interval => {
          expect(interval).toHaveProperty('lower');
          expect(interval).toHaveProperty('upper');
          expect(interval.lower).toBeLessThanOrEqual(interval.upper);
        });

        expect(forecast.modelAccuracy).toBeGreaterThanOrEqual(0);
        expect(forecast.modelAccuracy).toBeLessThanOrEqual(1);
      });

      it('should support custom confidence levels', async () => {
        const forecast = await backend.neuralForecast('AAPL', 30, false, 0.99);
        expect(forecast.predictions.length).toBe(30);
      });

      it('should support GPU acceleration', async () => {
        const forecast = await backend.neuralForecast('AAPL', 30, true);
        expect(forecast).toHaveProperty('predictions');
      });

      it('should handle various horizon lengths', async () => {
        const horizons = [7, 30, 90, 365];

        for (const horizon of horizons) {
          const forecast = await backend.neuralForecast('AAPL', horizon);
          expect(forecast.predictions.length).toBe(horizon);
        }
      });

      it('should reject invalid horizons', async () => {
        await expect(backend.neuralForecast('AAPL', 0)).rejects.toThrow();
        await expect(backend.neuralForecast('AAPL', -1)).rejects.toThrow();
      });
    });

    describe('neuralTrain()', () => {
      it('should train model with valid parameters', async () => {
        const result = await backend.neuralTrain(
          '/tmp/training_data.csv',
          'lstm',
          100,
          false
        );

        expect(result).toHaveProperty('modelId');
        expect(result).toHaveProperty('modelType', 'lstm');
        expect(result).toHaveProperty('trainingTimeMs');
        expect(result).toHaveProperty('finalLoss');
        expect(result).toHaveProperty('validationAccuracy');

        expect(result.modelId.length).toBeGreaterThan(0);
        expect(result.trainingTimeMs).toBeGreaterThan(0);
        expect(result.finalLoss).toBeGreaterThanOrEqual(0);
        expect(result.validationAccuracy).toBeGreaterThanOrEqual(0);
        expect(result.validationAccuracy).toBeLessThanOrEqual(1);
      });

      it('should support different model types', async () => {
        const modelTypes = ['lstm', 'gru', 'transformer'];

        for (const modelType of modelTypes) {
          const result = await backend.neuralTrain('/tmp/data.csv', modelType);
          expect(result.modelType).toBe(modelType);
        }
      });

      it('should use GPU when enabled', async () => {
        const result = await backend.neuralTrain('/tmp/data.csv', 'lstm', 50, true);
        expect(result).toHaveProperty('modelId');
      });

      it('should reject invalid file paths', async () => {
        await expect(
          backend.neuralTrain('', 'lstm')
        ).rejects.toThrow();
      });
    });

    describe('neuralEvaluate()', () => {
      it('should evaluate trained model', async () => {
        const trainResult = await backend.neuralTrain('/tmp/data.csv', 'lstm');
        const evalResult = await backend.neuralEvaluate(
          trainResult.modelId,
          '/tmp/test_data.csv',
          false
        );

        expect(evalResult).toHaveProperty('modelId', trainResult.modelId);
        expect(evalResult).toHaveProperty('testSamples');
        expect(evalResult).toHaveProperty('mae');
        expect(evalResult).toHaveProperty('rmse');
        expect(evalResult).toHaveProperty('mape');
        expect(evalResult).toHaveProperty('r2Score');

        expect(evalResult.testSamples).toBeGreaterThan(0);
        expect(evalResult.mae).toBeGreaterThanOrEqual(0);
        expect(evalResult.rmse).toBeGreaterThanOrEqual(0);
        expect(evalResult.mape).toBeGreaterThanOrEqual(0);
      });

      it('should reject invalid model IDs', async () => {
        await expect(
          backend.neuralEvaluate('invalid_model_id', '/tmp/test.csv')
        ).rejects.toThrow();
      });
    });

    describe('neuralModelStatus()', () => {
      it('should list all models when no ID provided', async () => {
        const models = await backend.neuralModelStatus();

        expect(Array.isArray(models)).toBe(true);

        if (models.length > 0) {
          models.forEach(model => {
            expect(model).toHaveProperty('modelId');
            expect(model).toHaveProperty('modelType');
            expect(model).toHaveProperty('status');
            expect(model).toHaveProperty('createdAt');
            expect(model).toHaveProperty('accuracy');
          });
        }
      });

      it('should get specific model status', async () => {
        const trainResult = await backend.neuralTrain('/tmp/data.csv', 'lstm');
        const models = await backend.neuralModelStatus(trainResult.modelId);

        expect(Array.isArray(models)).toBe(true);
        expect(models.length).toBeGreaterThan(0);
        expect(models[0].modelId).toBe(trainResult.modelId);
      });
    });

    describe('neuralOptimize()', () => {
      it('should optimize model hyperparameters', async () => {
        const trainResult = await backend.neuralTrain('/tmp/data.csv', 'lstm');
        const paramRanges = JSON.stringify({
          learning_rate: [0.001, 0.01],
          hidden_units: [64, 128, 256]
        });

        const optResult = await backend.neuralOptimize(
          trainResult.modelId,
          paramRanges,
          false
        );

        expect(optResult).toHaveProperty('modelId', trainResult.modelId);
        expect(optResult).toHaveProperty('bestParams');
        expect(optResult).toHaveProperty('bestScore');
        expect(optResult).toHaveProperty('trialsCompleted');
        expect(optResult).toHaveProperty('optimizationTimeMs');

        expect(optResult.trialsCompleted).toBeGreaterThan(0);
        expect(optResult.optimizationTimeMs).toBeGreaterThan(0);
      });

      it('should use GPU when enabled', async () => {
        const trainResult = await backend.neuralTrain('/tmp/data.csv', 'lstm');
        const paramRanges = JSON.stringify({ learning_rate: [0.001, 0.01] });

        const optResult = await backend.neuralOptimize(
          trainResult.modelId,
          paramRanges,
          true
        );
        expect(optResult).toHaveProperty('bestParams');
      });
    });

    describe('neuralBacktest()', () => {
      it('should backtest neural model', async () => {
        const trainResult = await backend.neuralTrain('/tmp/data.csv', 'lstm');
        const backtest = await backend.neuralBacktest(
          trainResult.modelId,
          '2023-01-01',
          '2023-12-31',
          'sp500',
          false
        );

        expect(backtest).toHaveProperty('modelId', trainResult.modelId);
        expect(backtest).toHaveProperty('startDate', '2023-01-01');
        expect(backtest).toHaveProperty('endDate', '2023-12-31');
        expect(backtest).toHaveProperty('totalReturn');
        expect(backtest).toHaveProperty('sharpeRatio');
        expect(backtest).toHaveProperty('maxDrawdown');
        expect(backtest).toHaveProperty('winRate');
        expect(backtest).toHaveProperty('totalTrades');
      });
    });
  });

  // ============================================================================
  // SPORTS BETTING FUNCTIONS
  // ============================================================================

  describe('Sports Betting Functions', () => {
    describe('getSportsEvents()', () => {
      it('should get upcoming sports events', async () => {
        const events = await backend.getSportsEvents('nfl', 7);

        expect(Array.isArray(events)).toBe(true);

        if (events.length > 0) {
          events.forEach(event => {
            expect(event).toHaveProperty('eventId');
            expect(event).toHaveProperty('sport', 'nfl');
            expect(event).toHaveProperty('homeTeam');
            expect(event).toHaveProperty('awayTeam');
            expect(event).toHaveProperty('startTime');
            expect(event.eventId.length).toBeGreaterThan(0);
          });
        }
      });

      it('should support different sports', async () => {
        const sports = ['nfl', 'nba', 'mlb', 'nhl'];

        for (const sport of sports) {
          const events = await backend.getSportsEvents(sport);
          expect(Array.isArray(events)).toBe(true);
        }
      });

      it('should handle default days ahead parameter', async () => {
        const events = await backend.getSportsEvents('nfl');
        expect(Array.isArray(events)).toBe(true);
      });
    });

    describe('getSportsOdds()', () => {
      it('should get betting odds for sport', async () => {
        const odds = await backend.getSportsOdds('nfl');

        expect(Array.isArray(odds)).toBe(true);

        if (odds.length > 0) {
          odds.forEach(odd => {
            expect(odd).toHaveProperty('eventId');
            expect(odd).toHaveProperty('market');
            expect(odd).toHaveProperty('homeOdds');
            expect(odd).toHaveProperty('awayOdds');
            expect(odd).toHaveProperty('bookmaker');
            expect(odd.homeOdds).toBeGreaterThan(0);
            expect(odd.awayOdds).toBeGreaterThan(0);
          });
        }
      });
    });

    describe('findSportsArbitrage()', () => {
      it('should find arbitrage opportunities', async () => {
        const arbs = await backend.findSportsArbitrage('nfl', 0.01);

        expect(Array.isArray(arbs)).toBe(true);

        if (arbs.length > 0) {
          arbs.forEach(arb => {
            expect(arb).toHaveProperty('eventId');
            expect(arb).toHaveProperty('profitMargin');
            expect(arb).toHaveProperty('betHome');
            expect(arb).toHaveProperty('betAway');

            expect(arb.profitMargin).toBeGreaterThan(0);
            expect(arb.betHome).toHaveProperty('bookmaker');
            expect(arb.betHome).toHaveProperty('odds');
            expect(arb.betHome).toHaveProperty('stake');
            expect(arb.betAway).toHaveProperty('bookmaker');
            expect(arb.betAway).toHaveProperty('odds');
            expect(arb.betAway).toHaveProperty('stake');
          });
        }
      });

      it('should use default min profit margin', async () => {
        const arbs = await backend.findSportsArbitrage('nfl');
        expect(Array.isArray(arbs)).toBe(true);
      });
    });

    describe('calculateKellyCriterion()', () => {
      it('should calculate Kelly fraction', async () => {
        const kelly = await backend.calculateKellyCriterion(0.6, 2.0, 10000);

        expect(kelly).toHaveProperty('probability', 0.6);
        expect(kelly).toHaveProperty('odds', 2.0);
        expect(kelly).toHaveProperty('bankroll', 10000);
        expect(kelly).toHaveProperty('kellyFraction');
        expect(kelly).toHaveProperty('suggestedStake');

        expect(kelly.kellyFraction).toBeGreaterThanOrEqual(0);
        expect(kelly.suggestedStake).toBeGreaterThanOrEqual(0);
        expect(kelly.suggestedStake).toBeLessThanOrEqual(kelly.bankroll);
      });

      it('should handle edge cases', async () => {
        // Zero probability
        const kelly1 = await backend.calculateKellyCriterion(0, 2.0, 10000);
        expect(kelly1.kellyFraction).toBe(0);

        // Very low odds
        const kelly2 = await backend.calculateKellyCriterion(0.6, 1.1, 10000);
        expect(kelly2.suggestedStake).toBeLessThanOrEqual(10000);
      });

      it('should reject invalid probabilities', async () => {
        await expect(
          backend.calculateKellyCriterion(-0.1, 2.0, 10000)
        ).rejects.toThrow();

        await expect(
          backend.calculateKellyCriterion(1.1, 2.0, 10000)
        ).rejects.toThrow();
      });

      it('should reject invalid odds', async () => {
        await expect(
          backend.calculateKellyCriterion(0.6, 0, 10000)
        ).rejects.toThrow();

        await expect(
          backend.calculateKellyCriterion(0.6, -1, 10000)
        ).rejects.toThrow();
      });

      it('should reject negative bankroll', async () => {
        await expect(
          backend.calculateKellyCriterion(0.6, 2.0, -1000)
        ).rejects.toThrow();
      });
    });

    describe('executeSportsBet()', () => {
      it('should execute bet with validation', async () => {
        const bet = await backend.executeSportsBet(
          'market_123',
          'home_team',
          100,
          2.5,
          true
        );

        expect(bet).toHaveProperty('betId');
        expect(bet).toHaveProperty('marketId', 'market_123');
        expect(bet).toHaveProperty('selection', 'home_team');
        expect(bet).toHaveProperty('stake', 100);
        expect(bet).toHaveProperty('odds', 2.5);
        expect(bet).toHaveProperty('status');
        expect(bet).toHaveProperty('potentialReturn');

        expect(bet.potentialReturn).toBe(250); // 100 * 2.5
      });

      it('should execute bet without validation', async () => {
        const bet = await backend.executeSportsBet(
          'market_123',
          'home_team',
          100,
          2.5,
          false
        );
        expect(bet).toHaveProperty('betId');
      });

      it('should reject negative stakes', async () => {
        await expect(
          backend.executeSportsBet('market_123', 'home', -100, 2.5)
        ).rejects.toThrow();
      });
    });
  });

  // ============================================================================
  // SYNDICATE MANAGEMENT FUNCTIONS
  // ============================================================================

  describe('Syndicate Management Functions', () => {
    describe('createSyndicate()', () => {
      it('should create new syndicate', async () => {
        const syndicate = await backend.createSyndicate(
          'syn_test_001',
          'Test Syndicate',
          'Test description'
        );

        expect(syndicate).toHaveProperty('syndicateId', 'syn_test_001');
        expect(syndicate).toHaveProperty('name', 'Test Syndicate');
        expect(syndicate).toHaveProperty('description', 'Test description');
        expect(syndicate).toHaveProperty('totalCapital');
        expect(syndicate).toHaveProperty('memberCount');
        expect(syndicate).toHaveProperty('createdAt');

        expect(syndicate.totalCapital).toBe(0);
        expect(syndicate.memberCount).toBe(0);
      });

      it('should create syndicate without description', async () => {
        const syndicate = await backend.createSyndicate(
          'syn_test_002',
          'Test Syndicate 2'
        );
        expect(syndicate.syndicateId).toBe('syn_test_002');
      });

      it('should reject duplicate syndicate IDs', async () => {
        await backend.createSyndicate('syn_dup', 'Syndicate');
        await expect(
          backend.createSyndicate('syn_dup', 'Syndicate')
        ).rejects.toThrow();
      });

      it('should reject empty syndicate ID', async () => {
        await expect(
          backend.createSyndicate('', 'Syndicate')
        ).rejects.toThrow();
      });
    });

    describe('addSyndicateMember()', () => {
      beforeEach(async () => {
        await backend.createSyndicate('syn_members', 'Members Test');
      });

      it('should add member to syndicate', async () => {
        const member = await backend.addSyndicateMember(
          'syn_members',
          'John Doe',
          'john@example.com',
          'analyst',
          5000
        );

        expect(member).toHaveProperty('memberId');
        expect(member).toHaveProperty('syndicateId', 'syn_members');
        expect(member).toHaveProperty('name', 'John Doe');
        expect(member).toHaveProperty('email', 'john@example.com');
        expect(member).toHaveProperty('role', 'analyst');
        expect(member).toHaveProperty('contribution', 5000);
        expect(member).toHaveProperty('profitShare');
      });

      it('should reject invalid email formats', async () => {
        await expect(
          backend.addSyndicateMember('syn_members', 'John', 'invalid-email', 'analyst', 5000)
        ).rejects.toThrow();
      });

      it('should reject negative contributions', async () => {
        await expect(
          backend.addSyndicateMember('syn_members', 'John', 'john@example.com', 'analyst', -1000)
        ).rejects.toThrow();
      });
    });

    describe('getSyndicateStatus()', () => {
      it('should get syndicate status', async () => {
        await backend.createSyndicate('syn_status', 'Status Test');
        const status = await backend.getSyndicateStatus('syn_status');

        expect(status).toHaveProperty('syndicateId', 'syn_status');
        expect(status).toHaveProperty('totalCapital');
        expect(status).toHaveProperty('activeBets');
        expect(status).toHaveProperty('totalProfit');
        expect(status).toHaveProperty('roi');
        expect(status).toHaveProperty('memberCount');
      });

      it('should reject nonexistent syndicates', async () => {
        await expect(
          backend.getSyndicateStatus('nonexistent')
        ).rejects.toThrow();
      });
    });

    describe('allocateSyndicateFunds()', () => {
      it('should allocate funds to opportunities', async () => {
        await backend.createSyndicate('syn_alloc', 'Allocation Test');

        const opportunities = JSON.stringify([
          {
            id: 'opp1',
            expectedReturn: 0.15,
            riskScore: 0.3,
            amount: 1000
          }
        ]);

        const allocation = await backend.allocateSyndicateFunds(
          'syn_alloc',
          opportunities,
          'kelly_criterion'
        );

        expect(allocation).toHaveProperty('syndicateId', 'syn_alloc');
        expect(allocation).toHaveProperty('totalAllocated');
        expect(allocation).toHaveProperty('allocations');
        expect(allocation).toHaveProperty('expectedReturn');
        expect(allocation).toHaveProperty('riskScore');

        expect(Array.isArray(allocation.allocations)).toBe(true);
      });

      it('should use default strategy', async () => {
        await backend.createSyndicate('syn_alloc2', 'Allocation Test 2');
        const opportunities = JSON.stringify([{ id: 'opp1', amount: 1000 }]);

        const allocation = await backend.allocateSyndicateFunds(
          'syn_alloc2',
          opportunities
        );
        expect(allocation).toHaveProperty('totalAllocated');
      });
    });

    describe('distributeSyndicateProfits()', () => {
      it('should distribute profits to members', async () => {
        await backend.createSyndicate('syn_dist', 'Distribution Test');
        await backend.addSyndicateMember('syn_dist', 'Member1', 'm1@example.com', 'analyst', 10000);
        await backend.addSyndicateMember('syn_dist', 'Member2', 'm2@example.com', 'analyst', 5000);

        const distribution = await backend.distributeSyndicateProfits(
          'syn_dist',
          1000,
          'proportional'
        );

        expect(distribution).toHaveProperty('syndicateId', 'syn_dist');
        expect(distribution).toHaveProperty('totalProfit', 1000);
        expect(distribution).toHaveProperty('distributions');
        expect(distribution).toHaveProperty('distributionDate');

        expect(Array.isArray(distribution.distributions)).toBe(true);
        expect(distribution.distributions.length).toBeGreaterThan(0);

        distribution.distributions.forEach(dist => {
          expect(dist).toHaveProperty('memberId');
          expect(dist).toHaveProperty('amount');
          expect(dist).toHaveProperty('percentage');
        });
      });

      it('should support different distribution models', async () => {
        await backend.createSyndicate('syn_dist2', 'Distribution Test 2');
        await backend.addSyndicateMember('syn_dist2', 'M1', 'm@example.com', 'analyst', 10000);

        const models = ['proportional', 'hybrid', 'tiered'];

        for (const model of models) {
          const dist = await backend.distributeSyndicateProfits('syn_dist2', 1000, model);
          expect(dist).toHaveProperty('distributions');
        }
      });

      it('should reject negative profits', async () => {
        await backend.createSyndicate('syn_dist3', 'Test');
        await expect(
          backend.distributeSyndicateProfits('syn_dist3', -1000)
        ).rejects.toThrow();
      });
    });
  });

  // ============================================================================
  // PREDICTION MARKETS
  // ============================================================================

  describe('Prediction Markets', () => {
    describe('getPredictionMarkets()', () => {
      it('should get prediction markets', async () => {
        const markets = await backend.getPredictionMarkets();

        expect(Array.isArray(markets)).toBe(true);

        if (markets.length > 0) {
          markets.forEach(market => {
            expect(market).toHaveProperty('marketId');
            expect(market).toHaveProperty('question');
            expect(market).toHaveProperty('category');
            expect(market).toHaveProperty('volume');
            expect(market).toHaveProperty('endDate');
          });
        }
      });

      it('should filter by category', async () => {
        const markets = await backend.getPredictionMarkets('politics', 10);
        expect(Array.isArray(markets)).toBe(true);
      });

      it('should respect limit parameter', async () => {
        const markets = await backend.getPredictionMarkets(null, 5);
        expect(markets.length).toBeLessThanOrEqual(5);
      });
    });

    describe('analyzeMarketSentiment()', () => {
      it('should analyze market sentiment', async () => {
        const markets = await backend.getPredictionMarkets();
        if (markets.length > 0) {
          const sentiment = await backend.analyzeMarketSentiment(markets[0].marketId);

          expect(sentiment).toHaveProperty('marketId', markets[0].marketId);
          expect(sentiment).toHaveProperty('bullishProbability');
          expect(sentiment).toHaveProperty('bearishProbability');
          expect(sentiment).toHaveProperty('volumeTrend');
          expect(sentiment).toHaveProperty('sentimentScore');

          expect(sentiment.bullishProbability).toBeGreaterThanOrEqual(0);
          expect(sentiment.bullishProbability).toBeLessThanOrEqual(1);
          expect(sentiment.bearishProbability).toBeGreaterThanOrEqual(0);
          expect(sentiment.bearishProbability).toBeLessThanOrEqual(1);
        }
      });
    });
  });

  // ============================================================================
  // E2B SANDBOX & SWARM OPERATIONS
  // ============================================================================

  describe('E2B Sandbox Operations', () => {
    describe('createE2bSandbox()', () => {
      it('should create sandbox', async () => {
        const sandbox = await backend.createE2bSandbox('test-sandbox', 'nodejs');

        expect(sandbox).toHaveProperty('sandboxId');
        expect(sandbox).toHaveProperty('name', 'test-sandbox');
        expect(sandbox).toHaveProperty('template', 'nodejs');
        expect(sandbox).toHaveProperty('status');
        expect(sandbox).toHaveProperty('createdAt');
      });

      it('should use default template', async () => {
        const sandbox = await backend.createE2bSandbox('test-sandbox-2');
        expect(sandbox).toHaveProperty('sandboxId');
      });
    });

    describe('executeE2bProcess()', () => {
      let sandboxId;

      beforeEach(async () => {
        const sandbox = await backend.createE2bSandbox('exec-test');
        sandboxId = sandbox.sandboxId;
      });

      it('should execute process in sandbox', async () => {
        const result = await backend.executeE2bProcess(sandboxId, 'echo "hello"');

        expect(result).toHaveProperty('sandboxId', sandboxId);
        expect(result).toHaveProperty('command', 'echo "hello"');
        expect(result).toHaveProperty('exitCode');
        expect(result).toHaveProperty('stdout');
        expect(result).toHaveProperty('stderr');
      });
    });
  });

  describe('E2B Swarm Operations', () => {
    describe('initE2bSwarm()', () => {
      it('should initialize swarm with mesh topology', async () => {
        const config = JSON.stringify({
          maxAgents: 5,
          distributionStrategy: 0,
          enableGpu: false,
          autoScaling: false
        });

        const swarm = await backend.initE2bSwarm('mesh', config);

        expect(swarm).toHaveProperty('swarmId');
        expect(swarm).toHaveProperty('topology', 'mesh');
        expect(swarm).toHaveProperty('agentCount');
        expect(swarm).toHaveProperty('status');
        expect(swarm).toHaveProperty('createdAt');
      });

      it('should support different topologies', async () => {
        const topologies = ['mesh', 'hierarchical', 'ring', 'star'];
        const config = JSON.stringify({ maxAgents: 3 });

        for (const topology of topologies) {
          const swarm = await backend.initE2bSwarm(topology, config);
          expect(swarm.topology).toBe(topology);
        }
      });
    });

    describe('getSwarmStatus()', () => {
      it('should get swarm status', async () => {
        const config = JSON.stringify({ maxAgents: 3 });
        const swarm = await backend.initE2bSwarm('mesh', config);

        const status = await backend.getSwarmStatus(swarm.swarmId);

        expect(status).toHaveProperty('swarmId', swarm.swarmId);
        expect(status).toHaveProperty('status');
        expect(status).toHaveProperty('activeAgents');
        expect(status).toHaveProperty('idleAgents');
        expect(status).toHaveProperty('failedAgents');
        expect(status).toHaveProperty('totalTrades');
        expect(status).toHaveProperty('totalPnl');
        expect(status).toHaveProperty('uptimeSecs');
        expect(status).toHaveProperty('lastUpdate');
      });
    });

    describe('scaleSwarm()', () => {
      it('should scale swarm up', async () => {
        const config = JSON.stringify({ maxAgents: 10 });
        const swarm = await backend.initE2bSwarm('mesh', config);

        const scaleResult = await backend.scaleSwarm(swarm.swarmId, 7);

        expect(scaleResult).toHaveProperty('swarmId', swarm.swarmId);
        expect(scaleResult).toHaveProperty('previousCount');
        expect(scaleResult).toHaveProperty('newCount', 7);
        expect(scaleResult).toHaveProperty('agentsAdded');
        expect(scaleResult).toHaveProperty('agentsRemoved');
        expect(scaleResult).toHaveProperty('status');
        expect(scaleResult).toHaveProperty('scaledAt');
      });

      it('should scale swarm down', async () => {
        const config = JSON.stringify({ maxAgents: 10 });
        const swarm = await backend.initE2bSwarm('mesh', config);
        await backend.scaleSwarm(swarm.swarmId, 7);

        const scaleResult = await backend.scaleSwarm(swarm.swarmId, 3);
        expect(scaleResult.newCount).toBe(3);
      });
    });

    describe('shutdownSwarm()', () => {
      it('should shutdown swarm gracefully', async () => {
        const config = JSON.stringify({ maxAgents: 3 });
        const swarm = await backend.initE2bSwarm('mesh', config);

        const result = await backend.shutdownSwarm(swarm.swarmId);

        expect(typeof result).toBe('string');
        expect(result.length).toBeGreaterThan(0);
      });
    });

    describe('getSwarmMetrics()', () => {
      it('should get detailed swarm metrics', async () => {
        const config = JSON.stringify({ maxAgents: 3 });
        const swarm = await backend.initE2bSwarm('mesh', config);

        const metrics = await backend.getSwarmMetrics(swarm.swarmId);

        expect(metrics).toHaveProperty('swarmId', swarm.swarmId);
        expect(metrics).toHaveProperty('activeAgents');
        expect(metrics).toHaveProperty('throughput');
        expect(metrics).toHaveProperty('avgLatency');
        expect(metrics).toHaveProperty('successRate');
        expect(metrics).toHaveProperty('totalPnl');
        expect(metrics).toHaveProperty('resourceUtilization');
        expect(metrics).toHaveProperty('timestamp');
      });
    });

    describe('monitorSwarmHealth()', () => {
      it('should monitor overall swarm health', async () => {
        const health = await backend.monitorSwarmHealth();

        expect(health).toHaveProperty('status');
        expect(health).toHaveProperty('cpuUsage');
        expect(health).toHaveProperty('memoryUsage');
        expect(health).toHaveProperty('avgResponseTime');
        expect(health).toHaveProperty('healthyAgents');
        expect(health).toHaveProperty('degradedAgents');
        expect(health).toHaveProperty('errorRate');
        expect(health).toHaveProperty('lastCheck');
      });
    });
  });

  // ============================================================================
  // SECURITY FEATURES
  // ============================================================================

  describe('Security Features', () => {
    describe('initAuth()', () => {
      it('should initialize auth system', () => {
        const result = backend.initAuth('my-secret-key');
        expect(typeof result).toBe('string');
      });

      it('should work without custom secret', () => {
        const result = backend.initAuth();
        expect(typeof result).toBe('string');
      });
    });

    describe('createApiKey()', () => {
      beforeEach(() => {
        backend.initAuth();
      });

      it('should create API key with defaults', () => {
        const apiKey = backend.createApiKey('testuser', 'user');

        expect(typeof apiKey).toBe('string');
        expect(apiKey.length).toBeGreaterThan(0);
      });

      it('should create API key with custom rate limit', () => {
        const apiKey = backend.createApiKey('testuser', 'user', 1000);
        expect(typeof apiKey).toBe('string');
      });

      it('should create API key with expiration', () => {
        const apiKey = backend.createApiKey('testuser', 'user', 100, 30);
        expect(typeof apiKey).toBe('string');
      });

      it('should support different roles', () => {
        const roles = ['readonly', 'user', 'admin', 'service'];

        for (const role of roles) {
          const apiKey = backend.createApiKey(`user_${role}`, role);
          expect(typeof apiKey).toBe('string');
        }
      });
    });

    describe('validateApiKey()', () => {
      beforeEach(() => {
        backend.initAuth();
      });

      it('should validate valid API key', () => {
        const apiKey = backend.createApiKey('testuser', 'user');
        const user = backend.validateApiKey(apiKey);

        expect(user).toHaveProperty('userId');
        expect(user).toHaveProperty('username', 'testuser');
        expect(user).toHaveProperty('role');
        expect(user).toHaveProperty('apiKey', apiKey);
        expect(user).toHaveProperty('createdAt');
        expect(user).toHaveProperty('lastActivity');
      });

      it('should reject invalid API key', () => {
        expect(() => backend.validateApiKey('invalid-key')).toThrow();
      });

      it('should reject empty API key', () => {
        expect(() => backend.validateApiKey('')).toThrow();
      });
    });

    describe('generateToken()', () => {
      beforeEach(() => {
        backend.initAuth();
      });

      it('should generate JWT token', () => {
        const apiKey = backend.createApiKey('testuser', 'user');
        const token = backend.generateToken(apiKey);

        expect(typeof token).toBe('string');
        expect(token.length).toBeGreaterThan(0);
      });
    });

    describe('validateToken()', () => {
      beforeEach(() => {
        backend.initAuth();
      });

      it('should validate JWT token', () => {
        const apiKey = backend.createApiKey('testuser', 'user');
        const token = backend.generateToken(apiKey);
        const user = backend.validateToken(token);

        expect(user).toHaveProperty('username', 'testuser');
      });

      it('should reject invalid token', () => {
        expect(() => backend.validateToken('invalid.token.here')).toThrow();
      });
    });

    describe('checkAuthorization()', () => {
      beforeEach(() => {
        backend.initAuth();
      });

      it('should authorize valid operations', () => {
        const apiKey = backend.createApiKey('testuser', 'admin');
        const authorized = backend.checkAuthorization(apiKey, 'trade', 'user');

        expect(typeof authorized).toBe('boolean');
      });

      it('should deny unauthorized operations', () => {
        const apiKey = backend.createApiKey('testuser', 'readonly');
        const authorized = backend.checkAuthorization(apiKey, 'trade', 'admin');

        expect(authorized).toBe(false);
      });
    });

    describe('revokeApiKey()', () => {
      beforeEach(() => {
        backend.initAuth();
      });

      it('should revoke API key', () => {
        const apiKey = backend.createApiKey('testuser', 'user');
        const result = backend.revokeApiKey(apiKey);

        expect(typeof result).toBe('string');
        expect(() => backend.validateApiKey(apiKey)).toThrow();
      });
    });

    describe('Rate Limiting', () => {
      beforeEach(() => {
        backend.initRateLimiter();
      });

      it('should check rate limit', () => {
        const allowed = backend.checkRateLimit('user1', 1);
        expect(typeof allowed).toBe('boolean');
      });

      it('should get rate limit stats', () => {
        backend.checkRateLimit('user2', 5);
        const stats = backend.getRateLimitStats('user2');

        expect(stats).toHaveProperty('tokensAvailable');
        expect(stats).toHaveProperty('maxTokens');
        expect(stats).toHaveProperty('refillRate');
        expect(stats).toHaveProperty('totalRequests');
        expect(stats).toHaveProperty('blockedRequests');
        expect(stats).toHaveProperty('successRate');
      });

      it('should reset rate limit', () => {
        backend.checkRateLimit('user3', 100);
        const result = backend.resetRateLimit('user3');
        expect(typeof result).toBe('string');
      });
    });

    describe('Input Validation', () => {
      it('should sanitize input', () => {
        const dirty = '<script>alert("xss")</script>';
        const clean = backend.sanitizeInput(dirty);

        expect(typeof clean).toBe('string');
        expect(clean).not.toContain('<script>');
      });

      it('should validate trading params', () => {
        expect(backend.validateTradingParams('AAPL', 10, 150.50)).toBe(true);
        expect(backend.validateTradingParams('AAPL', -10, 150.50)).toBe(false);
      });

      it('should validate email format', () => {
        expect(backend.validateEmailFormat('test@example.com')).toBe(true);
        expect(backend.validateEmailFormat('invalid-email')).toBe(false);
      });

      it('should validate API key format', () => {
        backend.initAuth();
        const apiKey = backend.createApiKey('test', 'user');
        expect(backend.validateApiKeyFormat(apiKey)).toBe(true);
        expect(backend.validateApiKeyFormat('short')).toBe(false);
      });

      it('should check security threats', () => {
        const threats = backend.checkSecurityThreats('<script>alert(1)</script>');
        expect(Array.isArray(threats)).toBe(true);
      });
    });
  });

  // ============================================================================
  // NEWS & ANALYTICS
  // ============================================================================

  describe('News & Analytics', () => {
    describe('analyzeNews()', () => {
      it('should analyze news sentiment', async () => {
        const sentiment = await backend.analyzeNews('AAPL', 24);

        expect(sentiment).toHaveProperty('symbol', 'AAPL');
        expect(sentiment).toHaveProperty('sentimentScore');
        expect(sentiment).toHaveProperty('articleCount');
        expect(sentiment).toHaveProperty('positiveRatio');
        expect(sentiment).toHaveProperty('negativeRatio');

        expect(sentiment.positiveRatio).toBeGreaterThanOrEqual(0);
        expect(sentiment.positiveRatio).toBeLessThanOrEqual(1);
        expect(sentiment.negativeRatio).toBeGreaterThanOrEqual(0);
        expect(sentiment.negativeRatio).toBeLessThanOrEqual(1);
      });
    });

    describe('controlNewsCollection()', () => {
      it('should start news collection', async () => {
        const result = await backend.controlNewsCollection('start', ['AAPL', 'GOOGL']);
        expect(typeof result).toBe('string');
      });

      it('should stop news collection', async () => {
        const result = await backend.controlNewsCollection('stop');
        expect(typeof result).toBe('string');
      });
    });

    describe('riskAnalysis()', () => {
      it('should analyze portfolio risk', async () => {
        const portfolio = JSON.stringify([
          { symbol: 'AAPL', weight: 0.5 },
          { symbol: 'GOOGL', weight: 0.5 }
        ]);

        const risk = await backend.riskAnalysis(portfolio, false);

        expect(risk).toHaveProperty('var95');
        expect(risk).toHaveProperty('cvar95');
        expect(risk).toHaveProperty('sharpeRatio');
        expect(risk).toHaveProperty('maxDrawdown');
        expect(risk).toHaveProperty('beta');
      });
    });

    describe('optimizeStrategy()', () => {
      it('should optimize strategy parameters', async () => {
        const paramRanges = JSON.stringify({
          lookback: [10, 20, 30],
          threshold: [0.01, 0.02, 0.03]
        });

        const result = await backend.optimizeStrategy(
          'momentum',
          'AAPL',
          paramRanges,
          false
        );

        expect(result).toHaveProperty('strategy', 'momentum');
        expect(result).toHaveProperty('symbol', 'AAPL');
        expect(result).toHaveProperty('bestParams');
        expect(result).toHaveProperty('bestSharpe');
        expect(result).toHaveProperty('optimizationTimeMs');
      });
    });

    describe('correlationAnalysis()', () => {
      it('should analyze asset correlations', async () => {
        const symbols = ['AAPL', 'GOOGL', 'MSFT'];
        const correlation = await backend.correlationAnalysis(symbols, false);

        expect(correlation).toHaveProperty('symbols');
        expect(correlation).toHaveProperty('matrix');
        expect(correlation).toHaveProperty('analysisPeriod');

        expect(correlation.symbols).toEqual(symbols);
        expect(Array.isArray(correlation.matrix)).toBe(true);
        expect(correlation.matrix.length).toBe(symbols.length);
      });
    });
  });

  // ============================================================================
  // SYSTEM FUNCTIONS
  // ============================================================================

  describe('System Functions', () => {
    describe('getVersion()', () => {
      it('should return version string', () => {
        const version = backend.getVersion();
        expect(typeof version).toBe('string');
        expect(version.length).toBeGreaterThan(0);
      });
    });

    describe('initSyndicate()', () => {
      it('should initialize syndicate module', () => {
        const result = backend.initSyndicate();
        expect(typeof result).toBe('string');
      });
    });

    describe('getSystemInfo()', () => {
      it('should return system information', () => {
        const info = backend.getSystemInfo();

        expect(info).toHaveProperty('version');
        expect(info).toHaveProperty('rustVersion');
        expect(info).toHaveProperty('buildTimestamp');
        expect(info).toHaveProperty('features');
        expect(info).toHaveProperty('totalTools');

        expect(Array.isArray(info.features)).toBe(true);
        expect(info.totalTools).toBeGreaterThan(0);
      });
    });

    describe('healthCheck()', () => {
      it('should return health status', async () => {
        const health = await backend.healthCheck();

        expect(health).toHaveProperty('status');
        expect(health).toHaveProperty('timestamp');
        expect(health).toHaveProperty('uptimeSeconds');

        expect(health.uptimeSeconds).toBeGreaterThanOrEqual(0);
      });
    });
  });
});
