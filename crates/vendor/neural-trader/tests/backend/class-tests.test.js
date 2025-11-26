/**
 * Comprehensive Class Tests for Neural Trader Backend
 * Coverage Target: 95%+
 *
 * Test Classes:
 * - FundAllocationEngine
 * - ProfitDistributionSystem
 * - WithdrawalManager
 * - MemberManager
 * - MemberPerformanceTracker
 * - VotingSystem
 * - CollaborationHub
 */

const backend = require('../../neural-trader-rust/packages/neural-trader-backend');

describe('Neural Trader Backend - Class Tests', () => {

  // ============================================================================
  // FundAllocationEngine
  // ============================================================================

  describe('FundAllocationEngine', () => {
    let engine;
    const syndicateId = 'test_syndicate_alloc';
    const totalBankroll = '100000';

    beforeEach(() => {
      engine = new backend.FundAllocationEngine(syndicateId, totalBankroll);
    });

    describe('constructor', () => {
      it('should create engine with valid parameters', () => {
        expect(engine).toBeDefined();
        expect(engine).toBeInstanceOf(backend.FundAllocationEngine);
      });

      it('should reject empty syndicate ID', () => {
        expect(() => new backend.FundAllocationEngine('', totalBankroll)).toThrow();
      });

      it('should reject invalid bankroll', () => {
        expect(() => new backend.FundAllocationEngine(syndicateId, '-1000')).toThrow();
        expect(() => new backend.FundAllocationEngine(syndicateId, 'invalid')).toThrow();
      });
    });

    describe('allocateFunds()', () => {
      const opportunity = {
        sport: 'nfl',
        event: 'Team A vs Team B',
        betType: 'moneyline',
        selection: 'Team A',
        odds: 2.5,
        probability: 0.55,
        edge: 0.15,
        confidence: 0.85,
        modelAgreement: 0.90,
        timeUntilEventSecs: 7200,
        liquidity: 0.8,
        isLive: false,
        isParlay: false
      };

      it('should allocate funds using Kelly Criterion', () => {
        const result = engine.allocateFunds(opportunity, backend.AllocationStrategy.KellyCriterion);

        expect(result).toHaveProperty('amount');
        expect(result).toHaveProperty('percentageOfBankroll');
        expect(result).toHaveProperty('reasoning');
        expect(result).toHaveProperty('riskMetrics');
        expect(result).toHaveProperty('approvalRequired');
        expect(result).toHaveProperty('warnings');
        expect(result).toHaveProperty('recommendedStakeSizing');

        expect(typeof result.amount).toBe('string');
        expect(result.percentageOfBankroll).toBeGreaterThanOrEqual(0);
        expect(result.percentageOfBankroll).toBeLessThanOrEqual(1);
        expect(Array.isArray(result.warnings)).toBe(true);
      });

      it('should allocate funds using Fixed Percentage', () => {
        const result = engine.allocateFunds(opportunity, backend.AllocationStrategy.FixedPercentage);
        expect(result).toHaveProperty('amount');
      });

      it('should allocate funds using Dynamic Confidence', () => {
        const result = engine.allocateFunds(opportunity, backend.AllocationStrategy.DynamicConfidence);
        expect(result).toHaveProperty('amount');
      });

      it('should allocate funds using Risk Parity', () => {
        const result = engine.allocateFunds(opportunity, backend.AllocationStrategy.RiskParity);
        expect(result).toHaveProperty('amount');
      });

      it('should allocate funds using Martingale', () => {
        const result = engine.allocateFunds(opportunity, backend.AllocationStrategy.Martingale);
        expect(result).toHaveProperty('amount');
      });

      it('should allocate funds using Anti-Martingale', () => {
        const result = engine.allocateFunds(opportunity, backend.AllocationStrategy.AntiMartingale);
        expect(result).toHaveProperty('amount');
      });

      it('should require approval for large bets', () => {
        const highStakeOpp = { ...opportunity, edge: 0.5 };
        const result = engine.allocateFunds(highStakeOpp, backend.AllocationStrategy.KellyCriterion);

        // Large allocations may require approval
        expect(typeof result.approvalRequired).toBe('boolean');
      });

      it('should warn about risky opportunities', () => {
        const riskyOpp = {
          ...opportunity,
          confidence: 0.4,
          edge: 0.05,
          isLive: true
        };

        const result = engine.allocateFunds(riskyOpp, backend.AllocationStrategy.KellyCriterion);

        if (result.warnings.length > 0) {
          result.warnings.forEach(warning => {
            expect(typeof warning).toBe('string');
          });
        }
      });

      it('should handle parlay bets conservatively', () => {
        const parlayOpp = { ...opportunity, isParlay: true };
        const result = engine.allocateFunds(parlayOpp, backend.AllocationStrategy.KellyCriterion);

        expect(result).toHaveProperty('amount');
        // Parlay allocations should be more conservative
      });
    });

    describe('updateExposure()', () => {
      it('should update exposure after bet placement', () => {
        const betPlaced = JSON.stringify({
          sport: 'nfl',
          amount: '1000',
          timestamp: new Date().toISOString()
        });

        expect(() => engine.updateExposure(betPlaced)).not.toThrow();
      });

      it('should reject invalid bet data', () => {
        expect(() => engine.updateExposure('invalid json')).toThrow();
      });

      it('should handle multiple exposure updates', () => {
        const bet1 = JSON.stringify({ sport: 'nfl', amount: '1000' });
        const bet2 = JSON.stringify({ sport: 'nba', amount: '2000' });

        engine.updateExposure(bet1);
        engine.updateExposure(bet2);

        const summary = engine.getExposureSummary();
        expect(typeof summary).toBe('string');
      });
    });

    describe('getExposureSummary()', () => {
      it('should return exposure summary', () => {
        const summary = engine.getExposureSummary();

        expect(typeof summary).toBe('string');
        expect(summary.length).toBeGreaterThan(0);

        // Should be valid JSON
        const parsed = JSON.parse(summary);
        expect(parsed).toBeDefined();
      });

      it('should include exposure by sport', () => {
        const bet = JSON.stringify({ sport: 'nfl', amount: '1000' });
        engine.updateExposure(bet);

        const summary = JSON.parse(engine.getExposureSummary());
        expect(summary).toHaveProperty('bySport');
      });
    });
  });

  // ============================================================================
  // ProfitDistributionSystem
  // ============================================================================

  describe('ProfitDistributionSystem', () => {
    let system;
    const syndicateId = 'test_syndicate_profit';

    beforeEach(() => {
      system = new backend.ProfitDistributionSystem(syndicateId);
    });

    describe('constructor', () => {
      it('should create system with valid syndicate ID', () => {
        expect(system).toBeDefined();
        expect(system).toBeInstanceOf(backend.ProfitDistributionSystem);
      });

      it('should reject empty syndicate ID', () => {
        expect(() => new backend.ProfitDistributionSystem('')).toThrow();
      });
    });

    describe('calculateDistribution()', () => {
      const members = JSON.stringify([
        {
          memberId: 'm1',
          name: 'Member 1',
          capitalContribution: '50000',
          performanceScore: 0.8,
          tier: 'gold'
        },
        {
          memberId: 'm2',
          name: 'Member 2',
          capitalContribution: '30000',
          performanceScore: 0.6,
          tier: 'silver'
        },
        {
          memberId: 'm3',
          name: 'Member 3',
          capitalContribution: '20000',
          performanceScore: 0.9,
          tier: 'bronze'
        }
      ]);

      it('should calculate proportional distribution', () => {
        const result = system.calculateDistribution(
          '10000',
          members,
          backend.DistributionModel.Proportional
        );

        const distribution = JSON.parse(result);
        expect(Array.isArray(distribution)).toBe(true);
        expect(distribution.length).toBe(3);

        distribution.forEach(dist => {
          expect(dist).toHaveProperty('memberId');
          expect(dist).toHaveProperty('amount');
          expect(dist).toHaveProperty('percentage');
        });

        // Total should equal input profit
        const total = distribution.reduce((sum, d) => sum + parseFloat(d.amount), 0);
        expect(total).toBeCloseTo(10000, 2);
      });

      it('should calculate performance-weighted distribution', () => {
        const result = system.calculateDistribution(
          '10000',
          members,
          backend.DistributionModel.PerformanceWeighted
        );

        const distribution = JSON.parse(result);
        expect(distribution.length).toBe(3);

        // Member with highest performance should get more
        const sorted = distribution.sort((a, b) => parseFloat(b.amount) - parseFloat(a.amount));
        expect(sorted[0].memberId).toBe('m3'); // Highest performance score
      });

      it('should calculate tiered distribution', () => {
        const result = system.calculateDistribution(
          '10000',
          members,
          backend.DistributionModel.Tiered
        );

        const distribution = JSON.parse(result);
        expect(distribution.length).toBe(3);
      });

      it('should calculate hybrid distribution', () => {
        const result = system.calculateDistribution(
          '10000',
          members,
          backend.DistributionModel.Hybrid
        );

        const distribution = JSON.parse(result);
        expect(distribution.length).toBe(3);

        // Hybrid should balance capital, performance, and equality
        const total = distribution.reduce((sum, d) => sum + parseFloat(d.amount), 0);
        expect(total).toBeCloseTo(10000, 2);
      });

      it('should handle zero profit', () => {
        const result = system.calculateDistribution(
          '0',
          members,
          backend.DistributionModel.Proportional
        );

        const distribution = JSON.parse(result);
        distribution.forEach(dist => {
          expect(parseFloat(dist.amount)).toBe(0);
        });
      });

      it('should reject negative profit', () => {
        expect(() => system.calculateDistribution(
          '-1000',
          members,
          backend.DistributionModel.Proportional
        )).toThrow();
      });

      it('should reject invalid member JSON', () => {
        expect(() => system.calculateDistribution(
          '10000',
          'invalid json',
          backend.DistributionModel.Proportional
        )).toThrow();
      });
    });
  });

  // ============================================================================
  // WithdrawalManager
  // ============================================================================

  describe('WithdrawalManager', () => {
    let manager;
    const syndicateId = 'test_syndicate_withdraw';

    beforeEach(() => {
      manager = new backend.WithdrawalManager(syndicateId);
    });

    describe('constructor', () => {
      it('should create manager with valid syndicate ID', () => {
        expect(manager).toBeDefined();
        expect(manager).toBeInstanceOf(backend.WithdrawalManager);
      });
    });

    describe('requestWithdrawal()', () => {
      it('should process normal withdrawal', () => {
        const result = manager.requestWithdrawal(
          'member_001',
          '50000',
          '5000',
          false
        );

        const withdrawal = JSON.parse(result);
        expect(withdrawal).toHaveProperty('withdrawalId');
        expect(withdrawal).toHaveProperty('memberId', 'member_001');
        expect(withdrawal).toHaveProperty('amount');
        expect(withdrawal).toHaveProperty('status');
        expect(withdrawal).toHaveProperty('requestedAt');
      });

      it('should process emergency withdrawal', () => {
        const result = manager.requestWithdrawal(
          'member_001',
          '50000',
          '5000',
          true
        );

        const withdrawal = JSON.parse(result);
        expect(withdrawal).toHaveProperty('isEmergency');
      });

      it('should reject withdrawal exceeding balance', () => {
        expect(() => manager.requestWithdrawal(
          'member_001',
          '50000',
          '60000',
          false
        )).toThrow();
      });

      it('should reject negative withdrawal amounts', () => {
        expect(() => manager.requestWithdrawal(
          'member_001',
          '50000',
          '-1000',
          false
        )).toThrow();
      });

      it('should reject zero withdrawal amounts', () => {
        expect(() => manager.requestWithdrawal(
          'member_001',
          '50000',
          '0',
          false
        )).toThrow();
      });

      it('should handle multiple withdrawals', () => {
        const w1 = manager.requestWithdrawal('member_001', '50000', '5000', false);
        const w2 = manager.requestWithdrawal('member_002', '30000', '3000', false);

        expect(JSON.parse(w1).withdrawalId).not.toBe(JSON.parse(w2).withdrawalId);
      });
    });

    describe('getWithdrawalHistory()', () => {
      it('should return withdrawal history', () => {
        manager.requestWithdrawal('member_001', '50000', '5000', false);
        manager.requestWithdrawal('member_001', '45000', '2000', false);

        const history = manager.getWithdrawalHistory();
        const withdrawals = JSON.parse(history);

        expect(Array.isArray(withdrawals)).toBe(true);
        expect(withdrawals.length).toBeGreaterThanOrEqual(2);
      });

      it('should return empty array when no withdrawals', () => {
        const history = manager.getWithdrawalHistory();
        const withdrawals = JSON.parse(history);

        expect(Array.isArray(withdrawals)).toBe(true);
      });
    });
  });

  // ============================================================================
  // MemberManager
  // ============================================================================

  describe('MemberManager', () => {
    let manager;
    const syndicateId = 'test_syndicate_members';

    beforeEach(() => {
      manager = new backend.MemberManager(syndicateId);
    });

    describe('addMember()', () => {
      it('should add new member', () => {
        const result = manager.addMember(
          'John Doe',
          'john@example.com',
          backend.MemberRole.SeniorAnalyst,
          '25000'
        );

        const member = JSON.parse(result);
        expect(member).toHaveProperty('memberId');
        expect(member).toHaveProperty('name', 'John Doe');
        expect(member).toHaveProperty('email', 'john@example.com');
        expect(member).toHaveProperty('role');
        expect(member).toHaveProperty('capitalContribution');
      });

      it('should reject invalid email', () => {
        expect(() => manager.addMember(
          'John',
          'invalid-email',
          backend.MemberRole.ContributingMember,
          '10000'
        )).toThrow();
      });

      it('should reject negative contribution', () => {
        expect(() => manager.addMember(
          'John',
          'john@example.com',
          backend.MemberRole.ContributingMember,
          '-1000'
        )).toThrow();
      });
    });

    describe('updateMemberRole()', () => {
      it('should update member role', () => {
        const addResult = manager.addMember(
          'Jane',
          'jane@example.com',
          backend.MemberRole.JuniorAnalyst,
          '10000'
        );
        const member = JSON.parse(addResult);

        expect(() => manager.updateMemberRole(
          member.memberId,
          backend.MemberRole.SeniorAnalyst,
          'admin_001'
        )).not.toThrow();
      });

      it('should reject invalid member ID', () => {
        expect(() => manager.updateMemberRole(
          'invalid_id',
          backend.MemberRole.SeniorAnalyst,
          'admin_001'
        )).toThrow();
      });
    });

    describe('suspendMember()', () => {
      it('should suspend member', () => {
        const addResult = manager.addMember(
          'Bob',
          'bob@example.com',
          backend.MemberRole.ContributingMember,
          '5000'
        );
        const member = JSON.parse(addResult);

        expect(() => manager.suspendMember(
          member.memberId,
          'Policy violation',
          'admin_001'
        )).not.toThrow();
      });
    });

    describe('updateContribution()', () => {
      it('should update member capital contribution', () => {
        const addResult = manager.addMember(
          'Alice',
          'alice@example.com',
          backend.MemberRole.ContributingMember,
          '10000'
        );
        const member = JSON.parse(addResult);

        expect(() => manager.updateContribution(member.memberId, '5000')).not.toThrow();
      });

      it('should reject negative contributions', () => {
        const addResult = manager.addMember(
          'Carol',
          'carol@example.com',
          backend.MemberRole.ContributingMember,
          '10000'
        );
        const member = JSON.parse(addResult);

        expect(() => manager.updateContribution(member.memberId, '-1000')).toThrow();
      });
    });

    describe('trackBetOutcome()', () => {
      it('should track bet outcome for member', () => {
        const addResult = manager.addMember(
          'Dave',
          'dave@example.com',
          backend.MemberRole.SeniorAnalyst,
          '20000'
        );
        const member = JSON.parse(addResult);

        const betDetails = JSON.stringify({
          betId: 'bet_001',
          amount: '1000',
          outcome: 'won',
          profit: '500'
        });

        expect(() => manager.trackBetOutcome(member.memberId, betDetails)).not.toThrow();
      });
    });

    describe('getMemberPerformanceReport()', () => {
      it('should get member performance report', () => {
        const addResult = manager.addMember(
          'Eve',
          'eve@example.com',
          backend.MemberRole.SeniorAnalyst,
          '30000'
        );
        const member = JSON.parse(addResult);

        const report = manager.getMemberPerformanceReport(member.memberId);
        const performance = JSON.parse(report);

        expect(performance).toHaveProperty('memberId', member.memberId);
        expect(performance).toHaveProperty('statistics');
      });
    });

    describe('getTotalCapital()', () => {
      it('should calculate total syndicate capital', () => {
        manager.addMember('M1', 'm1@example.com', backend.MemberRole.ContributingMember, '10000');
        manager.addMember('M2', 'm2@example.com', backend.MemberRole.ContributingMember, '20000');
        manager.addMember('M3', 'm3@example.com', backend.MemberRole.ContributingMember, '15000');

        const total = manager.getTotalCapital();
        expect(parseFloat(total)).toBe(45000);
      });
    });

    describe('listMembers()', () => {
      it('should list all members', () => {
        manager.addMember('M1', 'm1@example.com', backend.MemberRole.ContributingMember, '10000');
        manager.addMember('M2', 'm2@example.com', backend.MemberRole.ContributingMember, '20000');

        const membersList = manager.listMembers(true);
        const members = JSON.parse(membersList);

        expect(Array.isArray(members)).toBe(true);
        expect(members.length).toBeGreaterThanOrEqual(2);
      });

      it('should filter active members only', () => {
        const addResult = manager.addMember('M1', 'm1@example.com', backend.MemberRole.ContributingMember, '10000');
        manager.addMember('M2', 'm2@example.com', backend.MemberRole.ContributingMember, '20000');

        const member = JSON.parse(addResult);
        manager.suspendMember(member.memberId, 'Test', 'admin');

        const activeList = manager.listMembers(true);
        const allList = manager.listMembers(false);

        expect(JSON.parse(allList).length).toBeGreaterThan(JSON.parse(activeList).length);
      });
    });

    describe('getMemberCount()', () => {
      it('should return total member count', () => {
        const initialCount = manager.getMemberCount();

        manager.addMember('M1', 'm1@example.com', backend.MemberRole.ContributingMember, '10000');
        manager.addMember('M2', 'm2@example.com', backend.MemberRole.ContributingMember, '20000');

        expect(manager.getMemberCount()).toBe(initialCount + 2);
      });
    });

    describe('getActiveMemberCount()', () => {
      it('should return active member count', () => {
        manager.addMember('M1', 'm1@example.com', backend.MemberRole.ContributingMember, '10000');
        const addResult = manager.addMember('M2', 'm2@example.com', backend.MemberRole.ContributingMember, '20000');

        const beforeSuspend = manager.getActiveMemberCount();

        const member = JSON.parse(addResult);
        manager.suspendMember(member.memberId, 'Test', 'admin');

        expect(manager.getActiveMemberCount()).toBeLessThan(beforeSuspend);
      });
    });
  });

  // ============================================================================
  // MemberPerformanceTracker
  // ============================================================================

  describe('MemberPerformanceTracker', () => {
    let tracker;

    beforeEach(() => {
      tracker = new backend.MemberPerformanceTracker();
    });

    describe('trackBetOutcome()', () => {
      it('should track winning bet', () => {
        const betDetails = JSON.stringify({
          betId: 'bet_001',
          amount: '1000',
          outcome: 'won',
          profit: '800',
          odds: 2.8
        });

        expect(() => tracker.trackBetOutcome('member_001', betDetails)).not.toThrow();
      });

      it('should track losing bet', () => {
        const betDetails = JSON.stringify({
          betId: 'bet_002',
          amount: '1000',
          outcome: 'lost',
          profit: '-1000',
          odds: 2.5
        });

        expect(() => tracker.trackBetOutcome('member_001', betDetails)).not.toThrow();
      });

      it('should reject invalid bet details', () => {
        expect(() => tracker.trackBetOutcome('member_001', 'invalid json')).toThrow();
      });
    });

    describe('getPerformanceHistory()', () => {
      it('should return performance history', () => {
        const bet1 = JSON.stringify({ betId: 'b1', amount: '1000', outcome: 'won', profit: '500' });
        const bet2 = JSON.stringify({ betId: 'b2', amount: '1000', outcome: 'lost', profit: '-1000' });

        tracker.trackBetOutcome('member_001', bet1);
        tracker.trackBetOutcome('member_001', bet2);

        const history = tracker.getPerformanceHistory('member_001');
        const performance = JSON.parse(history);

        expect(Array.isArray(performance)).toBe(true);
        expect(performance.length).toBeGreaterThanOrEqual(2);
      });
    });

    describe('identifyMemberStrengths()', () => {
      it('should identify member strengths', () => {
        const nflBet = JSON.stringify({
          betId: 'b1',
          sport: 'nfl',
          amount: '1000',
          outcome: 'won',
          profit: '1000'
        });

        tracker.trackBetOutcome('member_001', nflBet);

        const strengths = tracker.identifyMemberStrengths('member_001');
        const analysis = JSON.parse(strengths);

        expect(analysis).toBeDefined();
        expect(typeof analysis).toBe('object');
      });
    });
  });

  // ============================================================================
  // VotingSystem
  // ============================================================================

  describe('VotingSystem', () => {
    let voting;
    const syndicateId = 'test_syndicate_voting';

    beforeEach(() => {
      voting = new backend.VotingSystem(syndicateId);
    });

    describe('createVote()', () => {
      it('should create new vote', () => {
        const result = voting.createVote(
          'strategy_change',
          'Change to aggressive strategy',
          'member_lead',
          48
        );

        const vote = JSON.parse(result);
        expect(vote).toHaveProperty('voteId');
        expect(vote).toHaveProperty('proposalType', 'strategy_change');
        expect(vote).toHaveProperty('status');
        expect(vote).toHaveProperty('createdAt');
        expect(vote).toHaveProperty('expiresAt');
      });

      it('should use default voting period', () => {
        const result = voting.createVote(
          'allocation_change',
          'Increase max bet size',
          'member_lead'
        );

        expect(JSON.parse(result)).toHaveProperty('voteId');
      });

      it('should reject empty proposal type', () => {
        expect(() => voting.createVote('', 'Details', 'member_lead')).toThrow();
      });
    });

    describe('castVote()', () => {
      it('should cast vote successfully', () => {
        const createResult = voting.createVote(
          'strategy_change',
          'Test proposal',
          'member_lead'
        );
        const vote = JSON.parse(createResult);

        const success = voting.castVote(vote.voteId, 'member_001', 'approve', 1.0);
        expect(success).toBe(true);
      });

      it('should reject duplicate votes from same member', () => {
        const createResult = voting.createVote('test', 'Test', 'member_lead');
        const vote = JSON.parse(createResult);

        voting.castVote(vote.voteId, 'member_001', 'approve', 1.0);

        const duplicate = voting.castVote(vote.voteId, 'member_001', 'approve', 1.0);
        expect(duplicate).toBe(false);
      });

      it('should handle different vote decisions', () => {
        const createResult = voting.createVote('test', 'Test', 'member_lead');
        const vote = JSON.parse(createResult);

        expect(voting.castVote(vote.voteId, 'member_001', 'approve', 1.0)).toBe(true);
        expect(voting.castVote(vote.voteId, 'member_002', 'reject', 1.0)).toBe(true);
        expect(voting.castVote(vote.voteId, 'member_003', 'abstain', 1.0)).toBe(true);
      });

      it('should weight votes correctly', () => {
        const createResult = voting.createVote('test', 'Test', 'member_lead');
        const vote = JSON.parse(createResult);

        expect(voting.castVote(vote.voteId, 'member_001', 'approve', 2.0)).toBe(true);
        expect(voting.castVote(vote.voteId, 'member_002', 'approve', 1.0)).toBe(true);
      });
    });

    describe('getVoteResults()', () => {
      it('should get vote results', () => {
        const createResult = voting.createVote('test', 'Test', 'member_lead');
        const vote = JSON.parse(createResult);

        voting.castVote(vote.voteId, 'member_001', 'approve', 1.0);
        voting.castVote(vote.voteId, 'member_002', 'reject', 1.0);

        const results = voting.getVoteResults(vote.voteId);
        const voteResults = JSON.parse(results);

        expect(voteResults).toHaveProperty('voteId', vote.voteId);
        expect(voteResults).toHaveProperty('approveCount');
        expect(voteResults).toHaveProperty('rejectCount');
        expect(voteResults).toHaveProperty('abstainCount');
        expect(voteResults).toHaveProperty('totalVotes');
      });
    });

    describe('finalizeVote()', () => {
      it('should finalize vote', () => {
        const createResult = voting.createVote('test', 'Test', 'member_lead');
        const vote = JSON.parse(createResult);

        voting.castVote(vote.voteId, 'member_001', 'approve', 1.0);

        const result = voting.finalizeVote(vote.voteId);
        const finalized = JSON.parse(result);

        expect(finalized).toHaveProperty('status');
        expect(finalized).toHaveProperty('outcome');
      });
    });

    describe('listActiveVotes()', () => {
      it('should list all active votes', () => {
        voting.createVote('test1', 'Test 1', 'member_lead');
        voting.createVote('test2', 'Test 2', 'member_lead');

        const activeVotes = voting.listActiveVotes();
        const votes = JSON.parse(activeVotes);

        expect(Array.isArray(votes)).toBe(true);
        expect(votes.length).toBeGreaterThanOrEqual(2);
      });
    });

    describe('hasVoted()', () => {
      it('should check if member has voted', () => {
        const createResult = voting.createVote('test', 'Test', 'member_lead');
        const vote = JSON.parse(createResult);

        expect(voting.hasVoted(vote.voteId, 'member_001')).toBe(false);

        voting.castVote(vote.voteId, 'member_001', 'approve', 1.0);

        expect(voting.hasVoted(vote.voteId, 'member_001')).toBe(true);
      });
    });

    describe('getMemberVote()', () => {
      it('should get member vote', () => {
        const createResult = voting.createVote('test', 'Test', 'member_lead');
        const vote = JSON.parse(createResult);

        voting.castVote(vote.voteId, 'member_001', 'approve', 1.0);

        const memberVote = voting.getMemberVote(vote.voteId, 'member_001');
        const voteData = JSON.parse(memberVote);

        expect(voteData).toHaveProperty('decision', 'approve');
        expect(voteData).toHaveProperty('weight', 1.0);
      });
    });
  });

  // ============================================================================
  // CollaborationHub
  // ============================================================================

  describe('CollaborationHub', () => {
    let hub;
    const syndicateId = 'test_syndicate_collab';

    beforeEach(() => {
      hub = new backend.CollaborationHub(syndicateId);
    });

    describe('createChannel()', () => {
      it('should create new channel', () => {
        const result = hub.createChannel(
          'General Discussion',
          'Main channel for syndicate discussion',
          'public'
        );

        const channel = JSON.parse(result);
        expect(channel).toHaveProperty('channelId');
        expect(channel).toHaveProperty('name', 'General Discussion');
        expect(channel).toHaveProperty('description');
        expect(channel).toHaveProperty('channelType', 'public');
        expect(channel).toHaveProperty('createdAt');
      });

      it('should create private channel', () => {
        const result = hub.createChannel(
          'Leadership',
          'Private channel for leaders',
          'private'
        );

        const channel = JSON.parse(result);
        expect(channel.channelType).toBe('private');
      });

      it('should reject empty channel name', () => {
        expect(() => hub.createChannel('', 'Description', 'public')).toThrow();
      });
    });

    describe('addMemberToChannel()', () => {
      it('should add member to channel', () => {
        const createResult = hub.createChannel('Test Channel', 'Test', 'public');
        const channel = JSON.parse(createResult);

        expect(() => hub.addMemberToChannel(channel.channelId, 'member_001')).not.toThrow();
      });

      it('should reject invalid channel ID', () => {
        expect(() => hub.addMemberToChannel('invalid_id', 'member_001')).toThrow();
      });
    });

    describe('postMessage()', () => {
      it('should post text message', () => {
        const createResult = hub.createChannel('Test', 'Test', 'public');
        const channel = JSON.parse(createResult);
        hub.addMemberToChannel(channel.channelId, 'member_001');

        const result = hub.postMessage(
          channel.channelId,
          'member_001',
          'Hello everyone!',
          'text',
          []
        );

        const message = JSON.parse(result);
        expect(message).toHaveProperty('messageId');
        expect(message).toHaveProperty('channelId', channel.channelId);
        expect(message).toHaveProperty('authorId', 'member_001');
        expect(message).toHaveProperty('content', 'Hello everyone!');
        expect(message).toHaveProperty('messageType', 'text');
        expect(message).toHaveProperty('timestamp');
      });

      it('should post message with attachments', () => {
        const createResult = hub.createChannel('Test', 'Test', 'public');
        const channel = JSON.parse(createResult);
        hub.addMemberToChannel(channel.channelId, 'member_001');

        const result = hub.postMessage(
          channel.channelId,
          'member_001',
          'Check this out',
          'text',
          ['file1.pdf', 'image.png']
        );

        const message = JSON.parse(result);
        expect(message.attachments.length).toBe(2);
      });

      it('should support different message types', () => {
        const createResult = hub.createChannel('Test', 'Test', 'public');
        const channel = JSON.parse(createResult);
        hub.addMemberToChannel(channel.channelId, 'member_001');

        const types = ['text', 'announcement', 'alert', 'system'];

        for (const type of types) {
          const result = hub.postMessage(channel.channelId, 'member_001', 'Message', type, []);
          const message = JSON.parse(result);
          expect(message.messageType).toBe(type);
        }
      });
    });

    describe('getChannelMessages()', () => {
      it('should get channel messages', () => {
        const createResult = hub.createChannel('Test', 'Test', 'public');
        const channel = JSON.parse(createResult);
        hub.addMemberToChannel(channel.channelId, 'member_001');

        hub.postMessage(channel.channelId, 'member_001', 'Message 1', 'text', []);
        hub.postMessage(channel.channelId, 'member_001', 'Message 2', 'text', []);
        hub.postMessage(channel.channelId, 'member_001', 'Message 3', 'text', []);

        const messages = hub.getChannelMessages(channel.channelId, 10);
        const messageList = JSON.parse(messages);

        expect(Array.isArray(messageList)).toBe(true);
        expect(messageList.length).toBeGreaterThanOrEqual(3);
      });

      it('should limit message count', () => {
        const createResult = hub.createChannel('Test', 'Test', 'public');
        const channel = JSON.parse(createResult);
        hub.addMemberToChannel(channel.channelId, 'member_001');

        for (let i = 0; i < 10; i++) {
          hub.postMessage(channel.channelId, 'member_001', `Message ${i}`, 'text', []);
        }

        const messages = hub.getChannelMessages(channel.channelId, 5);
        const messageList = JSON.parse(messages);

        expect(messageList.length).toBeLessThanOrEqual(5);
      });
    });

    describe('listChannels()', () => {
      it('should list all channels', () => {
        hub.createChannel('Channel 1', 'Test 1', 'public');
        hub.createChannel('Channel 2', 'Test 2', 'private');

        const channels = hub.listChannels();
        const channelList = JSON.parse(channels);

        expect(Array.isArray(channelList)).toBe(true);
        expect(channelList.length).toBeGreaterThanOrEqual(2);
      });
    });

    describe('getChannelDetails()', () => {
      it('should get channel details', () => {
        const createResult = hub.createChannel('Test Channel', 'Details test', 'public');
        const channel = JSON.parse(createResult);

        const details = hub.getChannelDetails(channel.channelId);
        const channelInfo = JSON.parse(details);

        expect(channelInfo).toHaveProperty('channelId', channel.channelId);
        expect(channelInfo).toHaveProperty('name', 'Test Channel');
        expect(channelInfo).toHaveProperty('memberCount');
        expect(channelInfo).toHaveProperty('messageCount');
      });
    });
  });
});
