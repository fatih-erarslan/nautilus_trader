/**
 * Tournament system for iterated games
 *
 * Implements round-robin and elimination tournaments with memory-based strategies
 */

import type {
  Strategy,
  TournamentPlayer,
  TournamentResult,
  Game,
  GameHistory,
} from './types.js';
import { PRISONERS_DILEMMA } from './games.js';

/**
 * Tournament configuration
 */
export interface TournamentConfig {
  game: Game;
  strategies: Strategy[];
  roundsPerMatch: number;
  tournamentStyle: 'round-robin' | 'elimination' | 'swiss';
  repeatMatches: number; // Number of times to repeat each pairing
  noiseProbability: number; // Probability of move error
}

/**
 * Tournament manager
 */
export class Tournament {
  private config: TournamentConfig;
  private players: Map<string, TournamentPlayer>;
  private matchHistory: Map<string, GameHistory[]>;

  constructor(config: Partial<TournamentConfig> = {}) {
    this.config = {
      game: config.game || PRISONERS_DILEMMA,
      strategies: config.strategies || [],
      roundsPerMatch: config.roundsPerMatch || 100,
      tournamentStyle: config.tournamentStyle || 'round-robin',
      repeatMatches: config.repeatMatches || 1,
      noiseProbability: config.noiseProbability || 0.0,
    };

    this.players = new Map();
    this.matchHistory = new Map();

    // Initialize players
    for (const strategy of this.config.strategies) {
      this.players.set(strategy.id, {
        id: strategy.id,
        strategy,
        score: 0,
        wins: 0,
        losses: 0,
        draws: 0,
        matches: 0,
      });
    }
  }

  /**
   * Add a strategy to the tournament
   */
  addStrategy(strategy: Strategy): void {
    if (!this.players.has(strategy.id)) {
      this.players.set(strategy.id, {
        id: strategy.id,
        strategy,
        score: 0,
        wins: 0,
        losses: 0,
        draws: 0,
        matches: 0,
      });
      this.config.strategies.push(strategy);
    }
  }

  /**
   * Play a match between two strategies
   */
  private playMatch(
    strategy1: Strategy,
    strategy2: Strategy
  ): { score1: number; score2: number; history1: GameHistory; history2: GameHistory } {
    const history1: GameHistory = [];
    const history2: GameHistory = [];
    let score1 = 0;
    let score2 = 0;

    for (let round = 0; round < this.config.roundsPerMatch; round++) {
      // Get moves from strategies
      let move1 = strategy1.play(history1, history2);
      let move2 = strategy2.play(history2, history1);

      // Apply noise
      if (Math.random() < this.config.noiseProbability) {
        move1 = 1 - move1;
      }
      if (Math.random() < this.config.noiseProbability) {
        move2 = 1 - move2;
      }

      // Record moves
      history1.push(move1);
      history2.push(move2);

      // Calculate payoffs
      score1 += this.config.game.payoffMatrix[move1][move2];
      score2 += this.config.game.payoffMatrix[move2][move1];
    }

    return { score1, score2, history1, history2 };
  }

  /**
   * Run a round-robin tournament
   */
  private runRoundRobin(): void {
    const strategies = this.config.strategies;

    for (let i = 0; i < strategies.length; i++) {
      for (let j = i + 1; j < strategies.length; j++) {
        const strategy1 = strategies[i];
        const strategy2 = strategies[j];

        // Play multiple matches
        for (let repeat = 0; repeat < this.config.repeatMatches; repeat++) {
          const result = this.playMatch(strategy1, strategy2);

          // Update scores
          const player1 = this.players.get(strategy1.id)!;
          const player2 = this.players.get(strategy2.id)!;

          player1.score += result.score1;
          player2.score += result.score2;
          player1.matches++;
          player2.matches++;

          // Update win/loss/draw
          if (result.score1 > result.score2) {
            player1.wins++;
            player2.losses++;
          } else if (result.score2 > result.score1) {
            player2.wins++;
            player1.losses++;
          } else {
            player1.draws++;
            player2.draws++;
          }

          // Store history
          const key = `${strategy1.id}-${strategy2.id}-${repeat}`;
          this.matchHistory.set(key, [result.history1, result.history2]);
        }
      }
    }

    // Also play against self
    for (const strategy of strategies) {
      const result = this.playMatch(strategy, strategy);
      const player = this.players.get(strategy.id)!;
      player.score += result.score1;
      player.matches++;
      player.draws++;
    }
  }

  /**
   * Run elimination tournament (single elimination)
   */
  private runElimination(): void {
    let currentRound = [...this.config.strategies];

    while (currentRound.length > 1) {
      const nextRound: Strategy[] = [];

      // Pair up strategies
      for (let i = 0; i < currentRound.length; i += 2) {
        if (i + 1 >= currentRound.length) {
          // Odd number, last one gets bye
          nextRound.push(currentRound[i]);
          continue;
        }

        const strategy1 = currentRound[i];
        const strategy2 = currentRound[i + 1];
        const result = this.playMatch(strategy1, strategy2);

        // Winner advances
        const winner = result.score1 >= result.score2 ? strategy1 : strategy2;
        nextRound.push(winner);

        // Update records
        const player1 = this.players.get(strategy1.id)!;
        const player2 = this.players.get(strategy2.id)!;

        player1.score += result.score1;
        player2.score += result.score2;
        player1.matches++;
        player2.matches++;

        if (result.score1 > result.score2) {
          player1.wins++;
          player2.losses++;
        } else if (result.score2 > result.score1) {
          player2.wins++;
          player1.losses++;
        } else {
          // Tiebreaker
          if (Math.random() < 0.5) {
            player1.wins++;
            player2.losses++;
          } else {
            player2.wins++;
            player1.losses++;
          }
        }
      }

      currentRound = nextRound;
    }
  }

  /**
   * Run Swiss-system tournament
   */
  private runSwiss(rounds: number = 5): void {
    // Sort players by score for pairing
    for (let round = 0; round < rounds; round++) {
      const sortedPlayers = Array.from(this.players.values()).sort(
        (a, b) => b.score - a.score
      );

      // Pair consecutive players
      for (let i = 0; i < sortedPlayers.length; i += 2) {
        if (i + 1 >= sortedPlayers.length) break;

        const player1 = sortedPlayers[i];
        const player2 = sortedPlayers[i + 1];
        const result = this.playMatch(player1.strategy, player2.strategy);

        player1.score += result.score1;
        player2.score += result.score2;
        player1.matches++;
        player2.matches++;

        if (result.score1 > result.score2) {
          player1.wins++;
          player2.losses++;
        } else if (result.score2 > result.score1) {
          player2.wins++;
          player1.losses++;
        } else {
          player1.draws++;
          player2.draws++;
        }
      }
    }
  }

  /**
   * Run the tournament
   */
  run(): TournamentResult {
    // Reset scores
    for (const player of Array.from(this.players.values())) {
      player.score = 0;
      player.wins = 0;
      player.losses = 0;
      player.draws = 0;
      player.matches = 0;
    }

    // Run tournament based on style
    switch (this.config.tournamentStyle) {
      case 'round-robin':
        this.runRoundRobin();
        break;
      case 'elimination':
        this.runElimination();
        break;
      case 'swiss':
        this.runSwiss();
        break;
    }

    // Calculate results
    const rankings = Array.from(this.players.values()).sort(
      (a, b) => b.score - a.score
    );

    const totalScore = rankings.reduce((sum, p) => sum + p.score, 0);
    const averageScore = totalScore / rankings.length;

    const totalMatches = rankings.reduce((sum, p) => sum + p.matches, 0);

    // Calculate diversity (Shannon entropy of rankings)
    const diversityIndex = this.calculateDiversityIndex(rankings);

    return {
      rankings,
      totalMatches,
      generation: 0,
      bestStrategy: rankings[0].strategy,
      averageScore,
      diversityIndex,
    };
  }

  /**
   * Calculate diversity index (Shannon entropy)
   */
  private calculateDiversityIndex(players: TournamentPlayer[]): number {
    const totalScore = players.reduce((sum, p) => sum + p.score, 0);

    if (totalScore === 0) return 0;

    let entropy = 0;
    for (const player of players) {
      const proportion = player.score / totalScore;
      if (proportion > 0) {
        entropy -= proportion * Math.log2(proportion);
      }
    }

    return entropy;
  }

  /**
   * Get match history between two strategies
   */
  getMatchHistory(strategy1Id: string, strategy2Id: string): GameHistory[][] {
    const histories: GameHistory[][] = [];

    for (let i = 0; i < this.config.repeatMatches; i++) {
      const key = `${strategy1Id}-${strategy2Id}-${i}`;
      const history = this.matchHistory.get(key);
      if (history) {
        histories.push(history);
      }
    }

    return histories;
  }

  /**
   * Get cooperation rate for a strategy in tournament
   */
  getCooperationRate(strategyId: string): number {
    let totalMoves = 0;
    let cooperativeMoves = 0;

    for (const [key, histories] of Array.from(this.matchHistory.entries())) {
      if (key.includes(strategyId)) {
        const [hist1, hist2] = histories;
        const isPlayer1 = key.startsWith(strategyId);
        const relevantHistory = isPlayer1 ? hist1 : hist2;

        totalMoves += relevantHistory.length;
        cooperativeMoves += relevantHistory.filter((move) => move === 0).length;
      }
    }

    return totalMoves > 0 ? cooperativeMoves / totalMoves : 0;
  }

  /**
   * Analyze strategy performance
   */
  analyzeStrategy(strategyId: string): {
    averageScore: number;
    winRate: number;
    cooperationRate: number;
    performanceByOpponent: Map<string, number>;
  } {
    const player = this.players.get(strategyId);
    if (!player) {
      throw new Error(`Strategy ${strategyId} not found`);
    }

    const averageScore = player.matches > 0 ? player.score / player.matches : 0;
    const winRate = player.matches > 0
      ? (player.wins + 0.5 * player.draws) / player.matches
      : 0;
    const cooperationRate = this.getCooperationRate(strategyId);

    // Performance by opponent
    const performanceByOpponent = new Map<string, number>();
    for (const opponent of Array.from(this.config.strategies)) {
      if (opponent.id === strategyId) continue;

      const histories = this.getMatchHistory(strategyId, opponent.id);
      if (histories.length > 0) {
        let totalScore = 0;
        for (const [hist1, hist2] of histories) {
          for (let i = 0; i < hist1.length; i++) {
            totalScore += this.config.game.payoffMatrix[hist1[i]][hist2[i]];
          }
        }
        performanceByOpponent.set(
          opponent.id,
          totalScore / (histories.length * this.config.roundsPerMatch)
        );
      }
    }

    return {
      averageScore,
      winRate,
      cooperationRate,
      performanceByOpponent,
    };
  }

  /**
   * Export tournament results
   */
  exportResults(): {
    config: TournamentConfig;
    results: TournamentResult;
    analyses: Map<string, {
      averageScore: number;
      winRate: number;
      cooperationRate: number;
      performanceByOpponent: Map<string, number>;
    }>;
  } {
    const results = this.run();
    const analyses = new Map<string, {
      averageScore: number;
      winRate: number;
      cooperationRate: number;
      performanceByOpponent: Map<string, number>;
    }>();

    for (const strategy of this.config.strategies) {
      analyses.set(strategy.id, this.analyzeStrategy(strategy.id));
    }

    return {
      config: this.config,
      results,
      analyses,
    };
  }
}

/**
 * Run a quick tournament with default strategies
 */
export function quickTournament(
  strategies: Strategy[],
  game?: Game,
  rounds: number = 100
): TournamentResult {
  const tournament = new Tournament({
    game,
    strategies,
    roundsPerMatch: rounds,
    tournamentStyle: 'round-robin',
  });

  return tournament.run();
}
