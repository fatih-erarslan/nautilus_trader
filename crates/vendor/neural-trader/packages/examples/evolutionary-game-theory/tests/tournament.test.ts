/**
 * Tests for tournament system
 */

import { Tournament, quickTournament } from '../src/tournament.js';
import {
  ALWAYS_COOPERATE,
  ALWAYS_DEFECT,
  TIT_FOR_TAT,
  GRIM_TRIGGER,
  PAVLOV,
} from '../src/strategies.js';
import { PRISONERS_DILEMMA } from '../src/games.js';

describe('Tournament', () => {
  describe('Initialization', () => {
    it('should initialize with strategies', () => {
      const tournament = new Tournament({
        strategies: [ALWAYS_COOPERATE, ALWAYS_DEFECT],
      });

      expect(tournament).toBeDefined();
    });

    it('should use default configuration', () => {
      const tournament = new Tournament();
      const result = tournament.run();

      expect(result).toBeDefined();
      expect(result.rankings).toHaveLength(0); // No strategies
    });

    it('should accept custom configuration', () => {
      const tournament = new Tournament({
        game: PRISONERS_DILEMMA,
        strategies: [TIT_FOR_TAT],
        roundsPerMatch: 50,
        tournamentStyle: 'round-robin',
      });

      const result = tournament.run();
      expect(result.rankings).toHaveLength(1);
    });
  });

  describe('Adding Strategies', () => {
    it('should add new strategies', () => {
      const tournament = new Tournament({
        strategies: [ALWAYS_COOPERATE],
      });

      tournament.addStrategy(ALWAYS_DEFECT);
      const result = tournament.run();

      expect(result.rankings).toHaveLength(2);
    });

    it('should not duplicate strategies', () => {
      const tournament = new Tournament({
        strategies: [ALWAYS_COOPERATE],
      });

      tournament.addStrategy(ALWAYS_COOPERATE);
      const result = tournament.run();

      expect(result.rankings).toHaveLength(1);
    });
  });

  describe('Match Playing', () => {
    it('should play matches between strategies', () => {
      const tournament = new Tournament({
        strategies: [ALWAYS_COOPERATE, ALWAYS_DEFECT],
        roundsPerMatch: 10,
      });

      const result = tournament.run();

      expect(result.totalMatches).toBeGreaterThan(0);
      expect(result.rankings[0].matches).toBeGreaterThan(0);
    });

    it('should record scores correctly', () => {
      const tournament = new Tournament({
        strategies: [ALWAYS_COOPERATE, ALWAYS_DEFECT],
        roundsPerMatch: 100,
      });

      const result = tournament.run();

      // Always Defect should score higher
      const defectPlayer = result.rankings.find(
        (p) => p.id === ALWAYS_DEFECT.id
      );
      const cooperatePlayer = result.rankings.find(
        (p) => p.id === ALWAYS_COOPERATE.id
      );

      expect(defectPlayer!.score).toBeGreaterThan(cooperatePlayer!.score);
    });

    it('should handle noise in moves', () => {
      const tournament = new Tournament({
        strategies: [TIT_FOR_TAT, ALWAYS_COOPERATE],
        roundsPerMatch: 100,
        noiseProbability: 0.1,
      });

      const result = tournament.run();
      expect(result.rankings).toHaveLength(2);
    });
  });

  describe('Round-Robin Tournament', () => {
    it('should play all pairs', () => {
      const strategies = [
        ALWAYS_COOPERATE,
        ALWAYS_DEFECT,
        TIT_FOR_TAT,
      ];

      const tournament = new Tournament({
        strategies,
        roundsPerMatch: 50,
        tournamentStyle: 'round-robin',
      });

      const result = tournament.run();

      // Each strategy should play against all others + self
      expect(result.rankings).toHaveLength(3);
      expect(result.totalMatches).toBeGreaterThan(0);
    });

    it('should respect repeat matches', () => {
      const tournament = new Tournament({
        strategies: [ALWAYS_COOPERATE, ALWAYS_DEFECT],
        roundsPerMatch: 10,
        repeatMatches: 3,
      });

      const result = tournament.run();

      // More matches due to repeats
      expect(result.totalMatches).toBeGreaterThanOrEqual(6); // 3 repeats × 2 pairings
    });
  });

  describe('Elimination Tournament', () => {
    it('should run single elimination', () => {
      const strategies = [
        ALWAYS_COOPERATE,
        ALWAYS_DEFECT,
        TIT_FOR_TAT,
        GRIM_TRIGGER,
      ];

      const tournament = new Tournament({
        strategies,
        roundsPerMatch: 50,
        tournamentStyle: 'elimination',
      });

      const result = tournament.run();
      expect(result.rankings).toHaveLength(4);
    });

    it('should handle odd number of strategies', () => {
      const strategies = [
        ALWAYS_COOPERATE,
        ALWAYS_DEFECT,
        TIT_FOR_TAT,
      ];

      const tournament = new Tournament({
        strategies,
        tournamentStyle: 'elimination',
      });

      const result = tournament.run();
      expect(result.rankings).toHaveLength(3);
    });
  });

  describe('Swiss Tournament', () => {
    it('should run Swiss system', () => {
      const strategies = [
        ALWAYS_COOPERATE,
        ALWAYS_DEFECT,
        TIT_FOR_TAT,
        GRIM_TRIGGER,
      ];

      const tournament = new Tournament({
        strategies,
        tournamentStyle: 'swiss',
      });

      const result = tournament.run();
      expect(result.rankings).toHaveLength(4);
    });
  });

  describe('Results', () => {
    it('should rank strategies by score', () => {
      const tournament = new Tournament({
        strategies: [ALWAYS_COOPERATE, ALWAYS_DEFECT, TIT_FOR_TAT],
        roundsPerMatch: 100,
      });

      const result = tournament.run();

      // Rankings should be sorted by score
      for (let i = 0; i < result.rankings.length - 1; i++) {
        expect(result.rankings[i].score).toBeGreaterThanOrEqual(
          result.rankings[i + 1].score
        );
      }
    });

    it('should identify best strategy', () => {
      const tournament = new Tournament({
        strategies: [ALWAYS_COOPERATE, ALWAYS_DEFECT, TIT_FOR_TAT],
        roundsPerMatch: 100,
      });

      const result = tournament.run();

      expect(result.bestStrategy).toBeDefined();
      expect(result.bestStrategy.id).toBe(result.rankings[0].id);
    });

    it('should calculate average score', () => {
      const tournament = new Tournament({
        strategies: [ALWAYS_COOPERATE, ALWAYS_DEFECT],
        roundsPerMatch: 100,
      });

      const result = tournament.run();

      expect(result.averageScore).toBeGreaterThan(0);
      expect(typeof result.averageScore).toBe('number');
    });

    it('should calculate diversity index', () => {
      const tournament = new Tournament({
        strategies: [ALWAYS_COOPERATE, ALWAYS_DEFECT, TIT_FOR_TAT],
        roundsPerMatch: 100,
      });

      const result = tournament.run();

      expect(result.diversityIndex).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Strategy Analysis', () => {
    it('should analyze individual strategy performance', () => {
      const tournament = new Tournament({
        strategies: [ALWAYS_COOPERATE, ALWAYS_DEFECT, TIT_FOR_TAT],
        roundsPerMatch: 100,
      });

      tournament.run();
      const analysis = tournament.analyzeStrategy(TIT_FOR_TAT.id);

      expect(analysis.averageScore).toBeDefined();
      expect(analysis.winRate).toBeDefined();
      expect(analysis.cooperationRate).toBeDefined();
      expect(analysis.performanceByOpponent).toBeDefined();
    });

    it('should calculate cooperation rate', () => {
      const tournament = new Tournament({
        strategies: [ALWAYS_COOPERATE, ALWAYS_DEFECT],
        roundsPerMatch: 100,
      });

      tournament.run();
      const cooperateRate = tournament.getCooperationRate(ALWAYS_COOPERATE.id);

      expect(cooperateRate).toBeCloseTo(1.0);
    });

    it('should track performance by opponent', () => {
      const tournament = new Tournament({
        strategies: [TIT_FOR_TAT, ALWAYS_COOPERATE, ALWAYS_DEFECT],
        roundsPerMatch: 100,
      });

      tournament.run();
      const analysis = tournament.analyzeStrategy(TIT_FOR_TAT.id);

      expect(analysis.performanceByOpponent.size).toBeGreaterThan(0);
    });
  });

  describe('Match History', () => {
    it('should record match history', () => {
      const tournament = new Tournament({
        strategies: [ALWAYS_COOPERATE, ALWAYS_DEFECT],
        roundsPerMatch: 10,
        repeatMatches: 2,
      });

      tournament.run();
      const history = tournament.getMatchHistory(
        ALWAYS_COOPERATE.id,
        ALWAYS_DEFECT.id
      );

      expect(history.length).toBeGreaterThan(0);
      expect(history[0]).toHaveLength(2); // Two histories per match
    });
  });

  describe('Export Results', () => {
    it('should export complete tournament data', () => {
      const tournament = new Tournament({
        strategies: [TIT_FOR_TAT, PAVLOV],
        roundsPerMatch: 50,
      });

      const exported = tournament.exportResults();

      expect(exported.config).toBeDefined();
      expect(exported.results).toBeDefined();
      expect(exported.analyses).toBeDefined();
      expect(exported.analyses.size).toBe(2);
    });
  });
});

describe('quickTournament', () => {
  it('should run quick tournament', () => {
    const strategies = [
      ALWAYS_COOPERATE,
      ALWAYS_DEFECT,
      TIT_FOR_TAT,
    ];

    const result = quickTournament(strategies);

    expect(result.rankings).toHaveLength(3);
    expect(result.totalMatches).toBeGreaterThan(0);
  });

  it('should use custom game', () => {
    const result = quickTournament(
      [ALWAYS_COOPERATE, ALWAYS_DEFECT],
      PRISONERS_DILEMMA,
      50
    );

    expect(result.rankings).toHaveLength(2);
  });

  it('should respect rounds parameter', () => {
    const result = quickTournament([TIT_FOR_TAT, PAVLOV], undefined, 200);

    // Longer rounds should result in higher scores
    expect(result.rankings[0].score).toBeGreaterThan(0);
  });
});

describe('Tournament Dynamics', () => {
  it('should favor TFT in noisy environment', () => {
    const tournament = new Tournament({
      strategies: [TIT_FOR_TAT, GRIM_TRIGGER, PAVLOV],
      roundsPerMatch: 100,
      noiseProbability: 0.01,
    });

    const result = tournament.run();

    // TFT or Pavlov should do well with noise
    const winner = result.rankings[0];
    expect([TIT_FOR_TAT.id, PAVLOV.id]).toContain(winner.id);
  });

  it('should show cooperation emergence', () => {
    const tournament = new Tournament({
      strategies: [
        ALWAYS_COOPERATE,
        ALWAYS_DEFECT,
        TIT_FOR_TAT,
        PAVLOV,
      ],
      roundsPerMatch: 150,
    });

    const result = tournament.run();

    // Cooperative strategies should dominate
    const topTwo = result.rankings.slice(0, 2);
    const cooperativeCount = topTwo.filter((p) =>
      [TIT_FOR_TAT.id, PAVLOV.id, ALWAYS_COOPERATE.id].includes(p.id)
    ).length;

    expect(cooperativeCount).toBeGreaterThanOrEqual(1);
  });
});
