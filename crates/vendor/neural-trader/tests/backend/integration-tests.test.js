/**
 * Integration Tests for Neural Trader Backend
 * Coverage Target: 95%+
 *
 * End-to-End Workflows:
 * - Complete trading workflow
 * - Syndicate lifecycle from creation to profit distribution
 * - Swarm deployment and multi-agent coordination
 * - Authentication and authorization flows
 * - Neural model training to prediction pipeline
 */

const backend = require('../../neural-trader-rust/packages/neural-trader-backend');

describe('Neural Trader Backend - Integration Tests', () => {

  // ============================================================================
  // COMPLETE TRADING WORKFLOW
  // ============================================================================

  describe('Complete Trading Workflow', () => {
    it('should execute end-to-end trading flow', async () => {
      // 1. List available strategies
      const strategies = await backend.listStrategies();
      expect(strategies.length).toBeGreaterThan(0);
      const strategy = strategies[0];

      // 2. Analyze market
      const analysis = await backend.quickAnalysis('AAPL', false);
      expect(analysis.symbol).toBe('AAPL');

      // 3. Simulate trade
      const simulation = await backend.simulateTrade(
        strategy.name,
        'AAPL',
        'buy',
        false
      );
      expect(simulation.strategy).toBe(strategy.name);

      // 4. Check portfolio status
      const portfolioBefore = await backend.getPortfolioStatus(true);
      expect(portfolioBefore).toHaveProperty('totalValue');

      // 5. Execute trade
      const execution = await backend.executeTrade(
        strategy.name,
        'AAPL',
        'buy',
        10,
        'market'
      );
      expect(execution).toHaveProperty('orderId');

      // 6. Verify portfolio updated
      const portfolioAfter = await backend.getPortfolioStatus(true);
      expect(portfolioAfter.positions).toBeGreaterThanOrEqual(portfolioBefore.positions);
    });

    it('should perform backtest and optimize strategy', async () => {
      const strategies = await backend.listStrategies();
      const strategy = strategies[0];

      // 1. Run initial backtest
      const backtest = await backend.runBacktest(
        strategy.name,
        'AAPL',
        '2023-01-01',
        '2023-12-31',
        false
      );
      expect(backtest).toHaveProperty('sharpeRatio');

      // 2. Optimize strategy parameters
      const paramRanges = JSON.stringify({
        lookback: [10, 20, 30],
        threshold: [0.01, 0.02]
      });

      const optimization = await backend.optimizeStrategy(
        strategy.name,
        'AAPL',
        paramRanges,
        false
      );
      expect(optimization).toHaveProperty('bestParams');

      // 3. Run backtest with optimized parameters
      const optimizedBacktest = await backend.runBacktest(
        strategy.name,
        'AAPL',
        '2023-01-01',
        '2023-12-31',
        false
      );
      expect(optimizedBacktest).toHaveProperty('totalReturn');
    });

    it('should analyze risk and rebalance portfolio', async () => {
      // 1. Create portfolio
      const portfolio = JSON.stringify([
        { symbol: 'AAPL', weight: 0.4 },
        { symbol: 'GOOGL', weight: 0.3 },
        { symbol: 'MSFT', weight: 0.3 }
      ]);

      // 2. Analyze risk
      const riskAnalysis = await backend.riskAnalysis(portfolio, false);
      expect(riskAnalysis).toHaveProperty('var95');
      expect(riskAnalysis).toHaveProperty('sharpeRatio');

      // 3. Calculate correlations
      const correlation = await backend.correlationAnalysis(
        ['AAPL', 'GOOGL', 'MSFT'],
        false
      );
      expect(correlation.matrix).toBeDefined();

      // 4. Rebalance portfolio
      const targetAllocations = JSON.stringify({
        'AAPL': 0.33,
        'GOOGL': 0.33,
        'MSFT': 0.34
      });

      const rebalance = await backend.portfolioRebalance(
        targetAllocations,
        portfolio
      );
      expect(rebalance).toHaveProperty('tradesNeeded');
    });
  });

  // ============================================================================
  // COMPLETE SYNDICATE LIFECYCLE
  // ============================================================================

  describe('Complete Syndicate Lifecycle', () => {
    let syndicateId;
    let members = [];

    it('should create syndicate and add members', async () => {
      // 1. Create syndicate
      syndicateId = `syn_integration_${Date.now()}`;
      const syndicate = await backend.createSyndicate(
        syndicateId,
        'Integration Test Syndicate',
        'Testing complete lifecycle'
      );
      expect(syndicate.syndicateId).toBe(syndicateId);

      // 2. Add multiple members with different roles
      const member1 = await backend.addSyndicateMember(
        syndicateId,
        'Lead Investor',
        'lead@example.com',
        'lead_investor',
        50000
      );
      members.push(member1);

      const member2 = await backend.addSyndicateMember(
        syndicateId,
        'Senior Analyst',
        'analyst@example.com',
        'senior_analyst',
        30000
      );
      members.push(member2);

      const member3 = await backend.addSyndicateMember(
        syndicateId,
        'Contributing Member',
        'member@example.com',
        'contributing_member',
        20000
      );
      members.push(member3);

      // 3. Verify syndicate status
      const status = await backend.getSyndicateStatus(syndicateId);
      expect(status.memberCount).toBe(3);
      expect(status.totalCapital).toBe(100000);
    });

    it('should allocate funds and track bets', async () => {
      // 1. Get sports events
      const events = await backend.getSportsEvents('nfl', 7);
      expect(Array.isArray(events)).toBe(true);

      // 2. Get betting odds
      const odds = await backend.getSportsOdds('nfl');
      expect(Array.isArray(odds)).toBe(true);

      // 3. Find arbitrage opportunities
      const arbitrage = await backend.findSportsArbitrage('nfl', 0.01);
      expect(Array.isArray(arbitrage)).toBe(true);

      // 4. Allocate funds to opportunities
      const opportunities = JSON.stringify([
        {
          id: 'opp1',
          sport: 'nfl',
          event: 'Team A vs Team B',
          betType: 'moneyline',
          odds: 2.5,
          probability: 0.55,
          edge: 0.15,
          confidence: 0.8,
          expectedReturn: 0.12
        },
        {
          id: 'opp2',
          sport: 'nfl',
          event: 'Team C vs Team D',
          betType: 'spread',
          odds: 1.9,
          probability: 0.60,
          edge: 0.10,
          confidence: 0.75,
          expectedReturn: 0.08
        }
      ]);

      const allocation = await backend.allocateSyndicateFunds(
        syndicateId,
        opportunities,
        'kelly_criterion'
      );
      expect(allocation).toHaveProperty('totalAllocated');
      expect(allocation.allocations.length).toBeGreaterThan(0);
    });

    it('should execute bets with Kelly Criterion', async () => {
      // 1. Calculate Kelly bet size
      const kelly = await backend.calculateKellyCriterion(0.55, 2.5, 100000);
      expect(kelly).toHaveProperty('suggestedStake');

      // 2. Execute bet
      const bet = await backend.executeSportsBet(
        'market_001',
        'team_a',
        kelly.suggestedStake,
        2.5,
        true
      );
      expect(bet).toHaveProperty('betId');
    });

    it('should distribute profits to members', async () => {
      // 1. Simulate profit
      const totalProfit = 15000;

      // 2. Test proportional distribution
      const propDistribution = await backend.distributeSyndicateProfits(
        syndicateId,
        totalProfit,
        'proportional'
      );
      expect(propDistribution.distributions.length).toBe(3);

      // Verify proportional split based on capital
      const dist1 = propDistribution.distributions.find(d => d.memberId === members[0].memberId);
      const dist2 = propDistribution.distributions.find(d => d.memberId === members[1].memberId);
      const dist3 = propDistribution.distributions.find(d => d.memberId === members[2].memberId);

      expect(parseFloat(dist1.amount)).toBeGreaterThan(parseFloat(dist2.amount));
      expect(parseFloat(dist2.amount)).toBeGreaterThan(parseFloat(dist3.amount));

      // Total should equal profit
      const total = propDistribution.distributions.reduce(
        (sum, d) => sum + parseFloat(d.amount),
        0
      );
      expect(total).toBeCloseTo(totalProfit, 2);

      // 3. Test hybrid distribution
      const hybridDistribution = await backend.distributeSyndicateProfits(
        syndicateId,
        totalProfit,
        'hybrid'
      );
      expect(hybridDistribution.distributions.length).toBe(3);
    });

    it('should handle member withdrawals', async () => {
      // Use MemberManager for withdrawal testing
      const manager = new backend.MemberManager(syndicateId);

      // 1. Request normal withdrawal
      const withdrawal1 = manager.requestWithdrawal(
        members[0].memberId,
        '50000',
        '5000',
        false
      );
      const w1 = JSON.parse(withdrawal1);
      expect(w1).toHaveProperty('withdrawalId');
      expect(w1.amount).toBe('5000');

      // 2. Request emergency withdrawal
      const withdrawal2 = manager.requestWithdrawal(
        members[1].memberId,
        '30000',
        '3000',
        true
      );
      const w2 = JSON.parse(withdrawal2);
      expect(w2.isEmergency).toBe(true);

      // 3. Get withdrawal history
      const history = manager.getWithdrawalHistory();
      const withdrawals = JSON.parse(history);
      expect(withdrawals.length).toBeGreaterThanOrEqual(2);
    });

    it('should manage voting on syndicate decisions', async () => {
      const voting = new backend.VotingSystem(syndicateId);

      // 1. Create vote for strategy change
      const voteResult = voting.createVote(
        'strategy_change',
        'Switch to more aggressive Kelly Criterion (0.5x instead of 0.25x)',
        members[0].memberId,
        48
      );
      const vote = JSON.parse(voteResult);
      expect(vote).toHaveProperty('voteId');

      // 2. Members cast votes
      expect(voting.castVote(vote.voteId, members[0].memberId, 'approve', 2.0)).toBe(true);
      expect(voting.castVote(vote.voteId, members[1].memberId, 'approve', 1.5)).toBe(true);
      expect(voting.castVote(vote.voteId, members[2].memberId, 'reject', 1.0)).toBe(true);

      // 3. Get results
      const results = voting.getVoteResults(vote.voteId);
      const voteResults = JSON.parse(results);
      expect(voteResults.totalVotes).toBe(3);
      expect(voteResults.approveCount).toBeGreaterThan(voteResults.rejectCount);

      // 4. Finalize vote
      const finalized = voting.finalizeVote(vote.voteId);
      const finalResult = JSON.parse(finalized);
      expect(finalResult).toHaveProperty('outcome');
    });

    it('should track collaboration in channels', async () => {
      const hub = new backend.CollaborationHub(syndicateId);

      // 1. Create channels
      const generalChannel = hub.createChannel(
        'General',
        'General discussion',
        'public'
      );
      const general = JSON.parse(generalChannel);

      const strategyChannel = hub.createChannel(
        'Strategy Discussion',
        'Discuss betting strategies',
        'public'
      );
      const strategy = JSON.parse(strategyChannel);

      // 2. Add members to channels
      members.forEach(member => {
        hub.addMemberToChannel(general.channelId, member.memberId);
        hub.addMemberToChannel(strategy.channelId, member.memberId);
      });

      // 3. Post messages
      hub.postMessage(
        general.channelId,
        members[0].memberId,
        'Welcome to the syndicate!',
        'announcement',
        []
      );

      hub.postMessage(
        strategy.channelId,
        members[1].memberId,
        'I suggest we focus on NFL this week',
        'text',
        []
      );

      hub.postMessage(
        strategy.channelId,
        members[2].memberId,
        'Agreed, the lines look favorable',
        'text',
        []
      );

      // 4. Retrieve messages
      const messages = hub.getChannelMessages(strategy.channelId, 10);
      const messageList = JSON.parse(messages);
      expect(messageList.length).toBeGreaterThanOrEqual(2);

      // 5. List all channels
      const channels = hub.listChannels();
      const channelList = JSON.parse(channels);
      expect(channelList.length).toBeGreaterThanOrEqual(2);
    });
  });

  // ============================================================================
  // SWARM DEPLOYMENT AND COORDINATION
  // ============================================================================

  describe('Swarm Deployment and Multi-Agent Coordination', () => {
    let swarmId;
    let agents = [];

    it('should initialize swarm with topology', async () => {
      // 1. Initialize mesh swarm
      const config = JSON.stringify({
        maxAgents: 10,
        distributionStrategy: 0, // RoundRobin
        enableGpu: false,
        autoScaling: true,
        minAgents: 3,
        maxMemoryMb: 512,
        timeoutSecs: 300
      });

      const swarm = await backend.initE2bSwarm('mesh', config);
      swarmId = swarm.swarmId;

      expect(swarm).toHaveProperty('swarmId');
      expect(swarm.topology).toBe('mesh');
      expect(swarm.agentCount).toBeGreaterThanOrEqual(3);
      expect(swarm.status).toBe('active');
    });

    it('should deploy trading agents to sandboxes', async () => {
      // 1. Create sandboxes
      const sandbox1 = await backend.createE2bSandbox('agent-1', 'nodejs');
      const sandbox2 = await backend.createE2bSandbox('agent-2', 'nodejs');
      const sandbox3 = await backend.createE2bSandbox('agent-3', 'nodejs');

      // 2. Deploy different agent types
      const agent1 = await backend.deployTradingAgent(
        sandbox1.sandboxId,
        'momentum',
        ['AAPL', 'GOOGL'],
        JSON.stringify({ lookback: 20, threshold: 0.02 })
      );
      agents.push(agent1);

      const agent2 = await backend.deployTradingAgent(
        sandbox2.sandboxId,
        'mean_reversion',
        ['MSFT', 'AMZN'],
        JSON.stringify({ window: 30, bands: 2 })
      );
      agents.push(agent2);

      const agent3 = await backend.deployTradingAgent(
        sandbox3.sandboxId,
        'neural',
        ['TSLA', 'NVDA']
      );
      agents.push(agent3);

      expect(agents.length).toBe(3);
      agents.forEach(agent => {
        expect(agent).toHaveProperty('agentId');
        expect(agent).toHaveProperty('sandboxId');
        expect(agent.status).toBe('deployed');
      });
    });

    it('should execute strategy across swarm', async () => {
      const execution = await backend.executeSwarmStrategy(
        swarmId,
        'momentum',
        ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
      );

      expect(execution).toHaveProperty('executionId');
      expect(execution.swarmId).toBe(swarmId);
      expect(execution.strategy).toBe('momentum');
      expect(execution.agentsUsed).toBeGreaterThan(0);
      expect(execution.status).toBeDefined();
    });

    it('should monitor swarm health and performance', async () => {
      // 1. Get swarm status
      const status = await backend.getSwarmStatus(swarmId);
      expect(status.swarmId).toBe(swarmId);
      expect(status.activeAgents).toBeGreaterThanOrEqual(0);

      // 2. Get detailed metrics
      const metrics = await backend.getSwarmMetrics(swarmId);
      expect(metrics).toHaveProperty('throughput');
      expect(metrics).toHaveProperty('avgLatency');
      expect(metrics).toHaveProperty('successRate');

      // 3. Get performance analytics
      const performance = await backend.getSwarmPerformance(swarmId);
      expect(performance).toHaveProperty('totalReturn');
      expect(performance).toHaveProperty('sharpeRatio');
      expect(performance).toHaveProperty('winRate');

      // 4. Monitor overall health
      const health = await backend.monitorSwarmHealth();
      expect(health).toHaveProperty('status');
      expect(health).toHaveProperty('cpuUsage');
      expect(health).toHaveProperty('memoryUsage');
    });

    it('should scale swarm dynamically', async () => {
      // 1. Get current status
      const statusBefore = await backend.getSwarmStatus(swarmId);
      const initialCount = statusBefore.activeAgents;

      // 2. Scale up
      const scaleUp = await backend.scaleSwarm(swarmId, initialCount + 3);
      expect(scaleUp.newCount).toBe(initialCount + 3);
      expect(scaleUp.agentsAdded).toBe(3);

      // 3. Verify scaling
      const statusAfterUp = await backend.getSwarmStatus(swarmId);
      expect(statusAfterUp.activeAgents).toBe(initialCount + 3);

      // 4. Scale down
      const scaleDown = await backend.scaleSwarm(swarmId, initialCount);
      expect(scaleDown.newCount).toBe(initialCount);
      expect(scaleDown.agentsRemoved).toBe(3);

      // 5. Verify scaling down
      const statusAfterDown = await backend.getSwarmStatus(swarmId);
      expect(statusAfterDown.activeAgents).toBe(initialCount);
    });

    it('should rebalance swarm portfolio', async () => {
      const rebalance = await backend.rebalanceSwarm(swarmId);

      expect(rebalance).toHaveProperty('swarmId', swarmId);
      expect(rebalance).toHaveProperty('status');
      expect(rebalance).toHaveProperty('tradesExecuted');
      expect(rebalance).toHaveProperty('agentsRebalanced');
    });

    it('should list and manage individual agents', async () => {
      // 1. List all agents
      const agentList = await backend.listSwarmAgents(swarmId);
      expect(Array.isArray(agentList)).toBe(true);
      expect(agentList.length).toBeGreaterThan(0);

      // 2. Get individual agent status
      const agentStatus = await backend.getAgentStatus(agents[0].agentId);
      expect(agentStatus.agentId).toBe(agents[0].agentId);
      expect(agentStatus).toHaveProperty('activeTrades');
      expect(agentStatus).toHaveProperty('pnl');
      expect(agentStatus).toHaveProperty('cpuUsage');

      // 3. Stop an agent
      const stopResult = await backend.stopSwarmAgent(agents[0].agentId);
      expect(typeof stopResult).toBe('string');

      // 4. Restart the agent
      const restartedAgent = await backend.restartSwarmAgent(agents[0].agentId);
      expect(restartedAgent).toHaveProperty('agentId');
      expect(restartedAgent.status).toBe('deployed');
    });

    it('should shutdown swarm gracefully', async () => {
      const shutdownResult = await backend.shutdownSwarm(swarmId);

      expect(typeof shutdownResult).toBe('string');
      expect(shutdownResult.length).toBeGreaterThan(0);
    });
  });

  // ============================================================================
  // AUTHENTICATION AND AUTHORIZATION FLOW
  // ============================================================================

  describe('Authentication and Authorization Flow', () => {
    let apiKeys = {};

    beforeAll(() => {
      backend.initAuth('integration-test-secret');
      backend.initRateLimiter({
        maxRequestsPerMinute: 100,
        burstSize: 20,
        windowDurationSecs: 60
      });
    });

    it('should create API keys for different roles', () => {
      // 1. Create admin user
      apiKeys.admin = backend.createApiKey('admin_user', 'admin', 1000, 365);
      expect(apiKeys.admin.length).toBeGreaterThan(0);

      // 2. Create regular user
      apiKeys.user = backend.createApiKey('regular_user', 'user', 500, 90);
      expect(apiKeys.user.length).toBeGreaterThan(0);

      // 3. Create read-only user
      apiKeys.readonly = backend.createApiKey('readonly_user', 'readonly', 100, 30);
      expect(apiKeys.readonly.length).toBeGreaterThan(0);

      // 4. Create service account
      apiKeys.service = backend.createApiKey('service_account', 'service', 10000);
      expect(apiKeys.service.length).toBeGreaterThan(0);
    });

    it('should validate API keys and get user info', () => {
      // Validate each key
      Object.entries(apiKeys).forEach(([role, key]) => {
        const user = backend.validateApiKey(key);
        expect(user).toHaveProperty('userId');
        expect(user).toHaveProperty('username');
        expect(user).toHaveProperty('role');
        expect(user.apiKey).toBe(key);
      });
    });

    it('should generate and validate JWT tokens', () => {
      // 1. Generate tokens
      const adminToken = backend.generateToken(apiKeys.admin);
      const userToken = backend.generateToken(apiKeys.user);

      expect(adminToken.length).toBeGreaterThan(0);
      expect(userToken.length).toBeGreaterThan(0);

      // 2. Validate tokens
      const adminUser = backend.validateToken(adminToken);
      expect(adminUser.username).toBe('admin_user');

      const regularUser = backend.validateToken(userToken);
      expect(regularUser.username).toBe('regular_user');
    });

    it('should enforce role-based authorization', () => {
      // Admin can do everything
      expect(backend.checkAuthorization(apiKeys.admin, 'trade', 'admin')).toBe(true);
      expect(backend.checkAuthorization(apiKeys.admin, 'read', 'user')).toBe(true);

      // User can trade but not admin operations
      expect(backend.checkAuthorization(apiKeys.user, 'trade', 'user')).toBe(true);
      expect(backend.checkAuthorization(apiKeys.user, 'admin_operation', 'admin')).toBe(false);

      // Read-only cannot trade
      expect(backend.checkAuthorization(apiKeys.readonly, 'read', 'readonly')).toBe(true);
      expect(backend.checkAuthorization(apiKeys.readonly, 'trade', 'user')).toBe(false);
    });

    it('should enforce rate limiting', () => {
      const identifier = 'test_user_123';

      // First requests should succeed
      for (let i = 0; i < 10; i++) {
        expect(backend.checkRateLimit(identifier, 1)).toBe(true);
      }

      // Get stats
      const stats = backend.getRateLimitStats(identifier);
      expect(stats).toHaveProperty('totalRequests');
      expect(stats.totalRequests).toBeGreaterThanOrEqual(10);

      // Reset limit
      const resetResult = backend.resetRateLimit(identifier);
      expect(typeof resetResult).toBe('string');

      // Should work again after reset
      expect(backend.checkRateLimit(identifier, 1)).toBe(true);
    });

    it('should validate and sanitize inputs', () => {
      // 1. Sanitize malicious input
      const dirty = '<script>alert("xss")</script>';
      const clean = backend.sanitizeInput(dirty);
      expect(clean).not.toContain('<script>');

      // 2. Validate trading parameters
      expect(backend.validateTradingParams('AAPL', 10, 150.50)).toBe(true);
      expect(backend.validateTradingParams('AAPL', -10, 150.50)).toBe(false);
      expect(backend.validateTradingParams('', 10, 150.50)).toBe(false);

      // 3. Validate email
      expect(backend.validateEmailFormat('test@example.com')).toBe(true);
      expect(backend.validateEmailFormat('invalid')).toBe(false);

      // 4. Check security threats
      const threats = backend.checkSecurityThreats('<script>alert(1)</script>');
      expect(Array.isArray(threats)).toBe(true);
    });

    it('should revoke API keys', () => {
      // Create a temporary key
      const tempKey = backend.createApiKey('temp_user', 'user');

      // Verify it works
      const user = backend.validateApiKey(tempKey);
      expect(user.username).toBe('temp_user');

      // Revoke it
      const revokeResult = backend.revokeApiKey(tempKey);
      expect(typeof revokeResult).toBe('string');

      // Should no longer work
      expect(() => backend.validateApiKey(tempKey)).toThrow();
    });

    it('should log audit events', () => {
      backend.initAuditLogger(1000, false, false);

      // Log various events
      backend.logAuditEvent(
        'info',
        'authentication',
        'login',
        'success',
        'user_001',
        'testuser',
        '192.168.1.1'
      );

      backend.logAuditEvent(
        'security',
        'trading',
        'execute_trade',
        'success',
        'user_001',
        'testuser',
        '192.168.1.1',
        'AAPL',
        JSON.stringify({ quantity: 10, price: 150.50 })
      );

      // Get audit events
      const events = backend.getAuditEvents(10);
      expect(Array.isArray(events)).toBe(true);
      expect(events.length).toBeGreaterThanOrEqual(2);

      events.forEach(event => {
        expect(event).toHaveProperty('eventId');
        expect(event).toHaveProperty('timestamp');
        expect(event).toHaveProperty('level');
        expect(event).toHaveProperty('category');
        expect(event).toHaveProperty('action');
        expect(event).toHaveProperty('outcome');
      });
    });
  });

  // ============================================================================
  // NEURAL MODEL TRAINING TO PREDICTION PIPELINE
  // ============================================================================

  describe('Neural Model Training to Prediction Pipeline', () => {
    let modelId;

    it('should train neural model', async () => {
      // Train LSTM model
      const training = await backend.neuralTrain(
        '/tmp/training_data.csv',
        'lstm',
        50,
        false
      );

      modelId = training.modelId;

      expect(training).toHaveProperty('modelId');
      expect(training.modelType).toBe('lstm');
      expect(training.trainingTimeMs).toBeGreaterThan(0);
      expect(training.validationAccuracy).toBeGreaterThanOrEqual(0);
    });

    it('should evaluate model performance', async () => {
      const evaluation = await backend.neuralEvaluate(
        modelId,
        '/tmp/test_data.csv',
        false
      );

      expect(evaluation.modelId).toBe(modelId);
      expect(evaluation).toHaveProperty('mae');
      expect(evaluation).toHaveProperty('rmse');
      expect(evaluation).toHaveProperty('r2Score');
      expect(evaluation.testSamples).toBeGreaterThan(0);
    });

    it('should optimize model hyperparameters', async () => {
      const paramRanges = JSON.stringify({
        learning_rate: [0.001, 0.01, 0.1],
        hidden_units: [64, 128, 256],
        dropout: [0.1, 0.2, 0.3]
      });

      const optimization = await backend.neuralOptimize(
        modelId,
        paramRanges,
        false
      );

      expect(optimization.modelId).toBe(modelId);
      expect(optimization).toHaveProperty('bestParams');
      expect(optimization).toHaveProperty('bestScore');
      expect(optimization.trialsCompleted).toBeGreaterThan(0);
    });

    it('should generate forecasts with model', async () => {
      const forecast = await backend.neuralForecast('AAPL', 30, false, 0.95);

      expect(forecast.symbol).toBe('AAPL');
      expect(forecast.horizon).toBe(30);
      expect(forecast.predictions.length).toBe(30);
      expect(forecast.confidenceIntervals.length).toBe(30);

      forecast.confidenceIntervals.forEach(interval => {
        expect(interval.lower).toBeLessThanOrEqual(interval.upper);
      });
    });

    it('should backtest model performance', async () => {
      const backtest = await backend.neuralBacktest(
        modelId,
        '2023-01-01',
        '2023-12-31',
        'sp500',
        false
      );

      expect(backtest.modelId).toBe(modelId);
      expect(backtest).toHaveProperty('totalReturn');
      expect(backtest).toHaveProperty('sharpeRatio');
      expect(backtest).toHaveProperty('maxDrawdown');
      expect(backtest).toHaveProperty('winRate');
      expect(backtest).toHaveProperty('totalTrades');
    });

    it('should integrate model into trading workflow', async () => {
      // 1. Generate forecast
      const forecast = await backend.neuralForecast('AAPL', 5, false);

      // 2. Analyze market
      const analysis = await backend.quickAnalysis('AAPL', false);

      // 3. Make trading decision based on forecast and analysis
      const strategies = await backend.listStrategies();
      const neuralStrategy = strategies.find(s => s.name.toLowerCase().includes('neural'));

      if (neuralStrategy) {
        // 4. Simulate trade
        const simulation = await backend.simulateTrade(
          neuralStrategy.name,
          'AAPL',
          'buy',
          false
        );

        expect(simulation).toHaveProperty('expectedReturn');

        // 5. Execute if expected return is positive
        if (simulation.expectedReturn > 0) {
          const execution = await backend.executeTrade(
            neuralStrategy.name,
            'AAPL',
            'buy',
            10
          );

          expect(execution).toHaveProperty('orderId');
        }
      }
    });
  });

  // ============================================================================
  // SYSTEM HEALTH AND MONITORING
  // ============================================================================

  describe('System Health and Monitoring', () => {
    it('should check overall system health', async () => {
      const health = await backend.healthCheck();

      expect(health).toHaveProperty('status');
      expect(health).toHaveProperty('timestamp');
      expect(health).toHaveProperty('uptimeSeconds');
      expect(health.uptimeSeconds).toBeGreaterThanOrEqual(0);
    });

    it('should get system information', () => {
      const info = backend.getSystemInfo();

      expect(info).toHaveProperty('version');
      expect(info).toHaveProperty('rustVersion');
      expect(info).toHaveProperty('buildTimestamp');
      expect(info).toHaveProperty('features');
      expect(info).toHaveProperty('totalTools');

      expect(Array.isArray(info.features)).toBe(true);
      expect(info.totalTools).toBeGreaterThan(70);
    });

    it('should get module version', () => {
      const version = backend.getVersion();

      expect(typeof version).toBe('string');
      expect(version.length).toBeGreaterThan(0);
      expect(version).toMatch(/\d+\.\d+\.\d+/);
    });
  });
});
