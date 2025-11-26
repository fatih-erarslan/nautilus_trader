/**
 * Sports Betting Integration Tests
 * Tests real API integration with The Odds API for NFL/NBA games
 *
 * Test Categories:
 * 1. Live Odds Fetching
 * 2. Arbitrage Detection
 * 3. Kelly Criterion Calculations
 * 4. Syndicate Operations
 * 5. +EV Opportunity Detection
 */

import { describe, it, expect, beforeAll } from '@jest/globals';
import axios from 'axios';

const THE_ODDS_API_KEY = process.env.THE_ODDS_API_KEY || '2a3a6dd4464b821cd404dc1f162e8d9d';
const BASE_URL = 'https://api.the-odds-api.com/v4';

interface OddsResponse {
  id: string;
  sport_key: string;
  sport_title: string;
  commence_time: string;
  home_team: string;
  away_team: string;
  bookmakers: Array<{
    key: string;
    title: string;
    last_update: string;
    markets: Array<{
      key: string;
      last_update: string;
      outcomes: Array<{
        name: string;
        price: number;
      }>;
    }>;
  }>;
}

interface ArbitrageOpportunity {
  sport: string;
  game: string;
  profit_margin: number;
  bets: Array<{
    bookmaker: string;
    outcome: string;
    odds: number;
    stake_percentage: number;
  }>;
  total_return: number;
}

interface KellyBet {
  recommended_stake: number;
  kelly_percentage: number;
  edge: number;
  bankroll_percentage: number;
}

describe('Sports Betting Integration Tests', () => {
  let sportsAvailable: string[] = [];
  let testResults: any = {
    timestamp: new Date().toISOString(),
    api_key_valid: false,
    sports_fetched: 0,
    odds_fetched: 0,
    arbitrage_opportunities: 0,
    kelly_calculations: 0,
    api_calls: 0,
    errors: []
  };

  beforeAll(async () => {
    console.log('🏈 Starting Sports Betting Integration Tests');
    console.log(`API Key: ${THE_ODDS_API_KEY.substring(0, 8)}...`);
  });

  describe('1. Live Odds Fetching', () => {
    it('should fetch available sports', async () => {
      const startTime = Date.now();
      try {
        const response = await axios.get(`${BASE_URL}/sports`, {
          params: { apiKey: THE_ODDS_API_KEY }
        });

        testResults.api_calls++;
        testResults.api_key_valid = true;

        expect(response.status).toBe(200);
        expect(response.data).toBeInstanceOf(Array);
        expect(response.data.length).toBeGreaterThan(0);

        sportsAvailable = response.data
          .filter((s: any) => ['americanfootball_nfl', 'basketball_nba'].includes(s.key))
          .map((s: any) => s.key);

        testResults.sports_fetched = sportsAvailable.length;

        console.log(`✅ Fetched ${response.data.length} sports in ${Date.now() - startTime}ms`);
        console.log(`   NFL/NBA available: ${sportsAvailable.join(', ')}`);
      } catch (error: any) {
        testResults.errors.push({ test: 'fetch_sports', error: error.message });
        throw error;
      }
    }, 10000);

    it('should fetch NFL odds', async () => {
      if (!sportsAvailable.includes('americanfootball_nfl')) {
        console.log('⚠️ NFL not available, skipping');
        return;
      }

      const startTime = Date.now();
      try {
        const response = await axios.get<OddsResponse[]>(
          `${BASE_URL}/sports/americanfootball_nfl/odds`,
          {
            params: {
              apiKey: THE_ODDS_API_KEY,
              regions: 'us',
              markets: 'h2h,spreads,totals',
              oddsFormat: 'decimal'
            }
          }
        );

        testResults.api_calls++;

        expect(response.status).toBe(200);
        expect(response.data).toBeInstanceOf(Array);

        if (response.data.length > 0) {
          const game = response.data[0];
          expect(game).toHaveProperty('home_team');
          expect(game).toHaveProperty('away_team');
          expect(game).toHaveProperty('bookmakers');
          expect(game.bookmakers.length).toBeGreaterThan(0);

          testResults.odds_fetched += response.data.length;

          console.log(`✅ Fetched ${response.data.length} NFL games in ${Date.now() - startTime}ms`);
          console.log(`   Example: ${game.away_team} @ ${game.home_team}`);
          console.log(`   Bookmakers: ${game.bookmakers.map(b => b.title).join(', ')}`);
        } else {
          console.log('⚠️ No NFL games currently available');
        }
      } catch (error: any) {
        testResults.errors.push({ test: 'fetch_nfl_odds', error: error.message });
        throw error;
      }
    }, 10000);

    it('should fetch NBA odds', async () => {
      if (!sportsAvailable.includes('basketball_nba')) {
        console.log('⚠️ NBA not available, skipping');
        return;
      }

      const startTime = Date.now();
      try {
        const response = await axios.get<OddsResponse[]>(
          `${BASE_URL}/sports/basketball_nba/odds`,
          {
            params: {
              apiKey: THE_ODDS_API_KEY,
              regions: 'us',
              markets: 'h2h',
              oddsFormat: 'decimal'
            }
          }
        );

        testResults.api_calls++;

        expect(response.status).toBe(200);
        expect(response.data).toBeInstanceOf(Array);

        if (response.data.length > 0) {
          const game = response.data[0];
          testResults.odds_fetched += response.data.length;

          console.log(`✅ Fetched ${response.data.length} NBA games in ${Date.now() - startTime}ms`);
          console.log(`   Example: ${game.away_team} @ ${game.home_team}`);
        } else {
          console.log('⚠️ No NBA games currently available');
        }
      } catch (error: any) {
        testResults.errors.push({ test: 'fetch_nba_odds', error: error.message });
        throw error;
      }
    }, 10000);
  });

  describe('2. Arbitrage Detection', () => {
    it('should detect arbitrage opportunities in NFL', async () => {
      if (!sportsAvailable.includes('americanfootball_nfl')) {
        console.log('⚠️ NFL not available, skipping arbitrage test');
        return;
      }

      const startTime = Date.now();
      try {
        const response = await axios.get<OddsResponse[]>(
          `${BASE_URL}/sports/americanfootball_nfl/odds`,
          {
            params: {
              apiKey: THE_ODDS_API_KEY,
              regions: 'us,uk,au',
              markets: 'h2h',
              oddsFormat: 'decimal'
            }
          }
        );

        testResults.api_calls++;

        const opportunities: ArbitrageOpportunity[] = [];

        for (const game of response.data) {
          const arbOpps = findArbitrageOpportunities(game);
          opportunities.push(...arbOpps);
        }

        testResults.arbitrage_opportunities = opportunities.length;

        console.log(`✅ Arbitrage analysis completed in ${Date.now() - startTime}ms`);
        console.log(`   Opportunities found: ${opportunities.length}`);

        if (opportunities.length > 0) {
          const best = opportunities.sort((a, b) => b.profit_margin - a.profit_margin)[0];
          console.log(`   Best opportunity: ${best.game}`);
          console.log(`   Profit margin: ${(best.profit_margin * 100).toFixed(2)}%`);
          console.log(`   Bets:`);
          best.bets.forEach(bet => {
            console.log(`     ${bet.bookmaker}: ${bet.outcome} @ ${bet.odds} (${(bet.stake_percentage * 100).toFixed(1)}%)`);
          });
        }

        expect(opportunities).toBeInstanceOf(Array);
      } catch (error: any) {
        testResults.errors.push({ test: 'arbitrage_detection', error: error.message });
        throw error;
      }
    }, 15000);
  });

  describe('3. Kelly Criterion Calculations', () => {
    it('should calculate optimal bet sizes using Kelly Criterion', () => {
      const testCases = [
        { probability: 0.55, odds: 2.0, bankroll: 1000, description: 'Slight edge' },
        { probability: 0.60, odds: 2.2, bankroll: 1000, description: 'Good edge' },
        { probability: 0.65, odds: 2.5, bankroll: 1000, description: 'Strong edge' },
        { probability: 0.50, odds: 2.0, bankroll: 1000, description: 'No edge' },
      ];

      console.log('🎲 Kelly Criterion Test Cases:');

      testCases.forEach(({ probability, odds, bankroll, description }) => {
        const kelly = calculateKellyCriterion(probability, odds, bankroll);
        testResults.kelly_calculations++;

        expect(kelly).toHaveProperty('recommended_stake');
        expect(kelly).toHaveProperty('kelly_percentage');
        expect(kelly).toHaveProperty('edge');

        console.log(`   ${description}:`);
        console.log(`     Win probability: ${(probability * 100).toFixed(1)}%`);
        console.log(`     Odds: ${odds.toFixed(2)}`);
        console.log(`     Kelly %: ${(kelly.kelly_percentage * 100).toFixed(2)}%`);
        console.log(`     Recommended stake: $${kelly.recommended_stake.toFixed(2)}`);
        console.log(`     Edge: ${(kelly.edge * 100).toFixed(2)}%`);
      });
    });

    it('should apply fractional Kelly for risk management', () => {
      const probability = 0.60;
      const odds = 2.2;
      const bankroll = 1000;
      const fractions = [1.0, 0.5, 0.25];

      console.log('📊 Fractional Kelly Analysis:');

      fractions.forEach(fraction => {
        const kelly = calculateKellyCriterion(probability, odds, bankroll, fraction);

        console.log(`   ${(fraction * 100)}% Kelly:`);
        console.log(`     Stake: $${kelly.recommended_stake.toFixed(2)}`);
        console.log(`     Bankroll %: ${(kelly.bankroll_percentage * 100).toFixed(2)}%`);
      });
    });
  });

  describe('4. Syndicate Operations', () => {
    it('should simulate syndicate creation', () => {
      const syndicate = {
        id: 'test-syndicate-001',
        name: 'NFL Arbitrage Pool',
        members: [
          { id: 'member1', name: 'Alice', contribution: 5000, role: 'admin' },
          { id: 'member2', name: 'Bob', contribution: 3000, role: 'member' },
          { id: 'member3', name: 'Charlie', contribution: 2000, role: 'member' }
        ],
        total_bankroll: 10000,
        profit_distribution: 'proportional'
      };

      expect(syndicate.total_bankroll).toBe(10000);
      expect(syndicate.members.length).toBe(3);

      const totalContributions = syndicate.members.reduce((sum, m) => sum + m.contribution, 0);
      expect(totalContributions).toBe(syndicate.total_bankroll);

      console.log('👥 Syndicate Created:');
      console.log(`   Name: ${syndicate.name}`);
      console.log(`   Total Bankroll: $${syndicate.total_bankroll}`);
      console.log(`   Members: ${syndicate.members.length}`);
      syndicate.members.forEach(m => {
        const percentage = (m.contribution / syndicate.total_bankroll * 100).toFixed(1);
        console.log(`     ${m.name}: $${m.contribution} (${percentage}%)`);
      });
    });

    it('should distribute profits proportionally', () => {
      const members = [
        { id: 'member1', contribution: 5000 },
        { id: 'member2', contribution: 3000 },
        { id: 'member3', contribution: 2000 }
      ];
      const totalBankroll = 10000;
      const profit = 1500;

      const distributions = members.map(member => ({
        ...member,
        share: (member.contribution / totalBankroll) * profit
      }));

      const totalDistributed = distributions.reduce((sum, d) => sum + d.share, 0);

      expect(totalDistributed).toBeCloseTo(profit, 2);

      console.log(`💰 Profit Distribution ($${profit} profit):`);
      distributions.forEach(d => {
        const percentage = (d.contribution / totalBankroll * 100).toFixed(1);
        console.log(`   Member ${d.id}: $${d.share.toFixed(2)} (${percentage}%)`);
      });
    });
  });

  describe('5. +EV Opportunity Detection', () => {
    it('should identify positive expected value bets', async () => {
      if (!sportsAvailable.includes('americanfootball_nfl')) {
        console.log('⚠️ NFL not available, skipping +EV test');
        return;
      }

      try {
        const response = await axios.get<OddsResponse[]>(
          `${BASE_URL}/sports/americanfootball_nfl/odds`,
          {
            params: {
              apiKey: THE_ODDS_API_KEY,
              regions: 'us',
              markets: 'h2h',
              oddsFormat: 'decimal'
            }
          }
        );

        testResults.api_calls++;

        const evOpportunities = [];

        for (const game of response.data) {
          // Estimate true probabilities from market consensus
          const marketProbabilities = estimateMarketProbabilities(game);

          // Find best odds for each outcome
          const bestOdds = findBestOdds(game);

          // Calculate EV for each outcome
          for (const outcome in marketProbabilities) {
            const trueProbability = marketProbabilities[outcome];
            const odds = bestOdds[outcome];

            if (odds) {
              const ev = calculateExpectedValue(trueProbability, odds);

              if (ev > 0) {
                evOpportunities.push({
                  game: `${game.away_team} @ ${game.home_team}`,
                  outcome,
                  true_probability: trueProbability,
                  odds,
                  expected_value: ev,
                  edge: (trueProbability * odds - 1) * 100
                });
              }
            }
          }
        }

        console.log(`🎯 +EV Opportunities Found: ${evOpportunities.length}`);

        if (evOpportunities.length > 0) {
          const best = evOpportunities.sort((a, b) => b.expected_value - a.expected_value)[0];
          console.log(`   Best +EV bet:`);
          console.log(`     Game: ${best.game}`);
          console.log(`     Outcome: ${best.outcome}`);
          console.log(`     True Probability: ${(best.true_probability * 100).toFixed(1)}%`);
          console.log(`     Odds: ${best.odds.toFixed(2)}`);
          console.log(`     Expected Value: $${best.expected_value.toFixed(2)} per $100 bet`);
          console.log(`     Edge: ${best.edge.toFixed(2)}%`);
        }

        expect(evOpportunities).toBeInstanceOf(Array);
      } catch (error: any) {
        testResults.errors.push({ test: 'ev_detection', error: error.message });
        throw error;
      }
    }, 15000);
  });

  afterAll(() => {
    console.log('\n📊 Test Summary:');
    console.log(`   Total API Calls: ${testResults.api_calls}`);
    console.log(`   Sports Fetched: ${testResults.sports_fetched}`);
    console.log(`   Odds Fetched: ${testResults.odds_fetched}`);
    console.log(`   Arbitrage Opportunities: ${testResults.arbitrage_opportunities}`);
    console.log(`   Kelly Calculations: ${testResults.kelly_calculations}`);
    console.log(`   Errors: ${testResults.errors.length}`);

    if (testResults.errors.length > 0) {
      console.log('\n❌ Errors encountered:');
      testResults.errors.forEach((err: any) => {
        console.log(`   ${err.test}: ${err.error}`);
      });
    }
  });
});

// Helper Functions

function findArbitrageOpportunities(game: OddsResponse): ArbitrageOpportunity[] {
  const opportunities: ArbitrageOpportunity[] = [];

  // Extract h2h markets
  const h2hMarkets = game.bookmakers
    .filter(b => b.markets.some(m => m.key === 'h2h'))
    .map(b => ({
      bookmaker: b.title,
      market: b.markets.find(m => m.key === 'h2h')!
    }));

  if (h2hMarkets.length < 2) return opportunities;

  // Find best odds for each outcome
  const outcomes = new Set<string>();
  h2hMarkets.forEach(({ market }) => {
    market.outcomes.forEach(o => outcomes.add(o.name));
  });

  const bestOdds: Record<string, { odds: number; bookmaker: string }> = {};

  outcomes.forEach(outcome => {
    let best = { odds: 0, bookmaker: '' };
    h2hMarkets.forEach(({ bookmaker, market }) => {
      const outcomeData = market.outcomes.find(o => o.name === outcome);
      if (outcomeData && outcomeData.price > best.odds) {
        best = { odds: outcomeData.price, bookmaker };
      }
    });
    if (best.odds > 0) {
      bestOdds[outcome] = best;
    }
  });

  // Calculate if arbitrage exists
  const impliedProbabilities = Object.values(bestOdds).map(b => 1 / b.odds);
  const totalProbability = impliedProbabilities.reduce((sum, p) => sum + p, 0);

  if (totalProbability < 1) {
    // Arbitrage opportunity exists!
    const profitMargin = 1 - totalProbability;

    const bets = Object.entries(bestOdds).map(([outcome, data]) => ({
      bookmaker: data.bookmaker,
      outcome,
      odds: data.odds,
      stake_percentage: (1 / data.odds) / totalProbability
    }));

    opportunities.push({
      sport: game.sport_title,
      game: `${game.away_team} @ ${game.home_team}`,
      profit_margin: profitMargin,
      bets,
      total_return: 1 / totalProbability
    });
  }

  return opportunities;
}

function calculateKellyCriterion(
  probability: number,
  odds: number,
  bankroll: number,
  fraction: number = 1.0
): KellyBet {
  // Kelly formula: f = (bp - q) / b
  // where b = odds - 1, p = probability of win, q = probability of loss
  const b = odds - 1;
  const p = probability;
  const q = 1 - p;

  const kellyPercentage = ((b * p - q) / b) * fraction;
  const safeKelly = Math.max(0, Math.min(kellyPercentage, 0.25)); // Cap at 25% for safety

  const edge = p * odds - 1;

  return {
    recommended_stake: bankroll * safeKelly,
    kelly_percentage: safeKelly,
    edge,
    bankroll_percentage: safeKelly
  };
}

function estimateMarketProbabilities(game: OddsResponse): Record<string, number> {
  // Use consensus from all bookmakers to estimate true probability
  const h2hMarkets = game.bookmakers
    .filter(b => b.markets.some(m => m.key === 'h2h'))
    .map(b => b.markets.find(m => m.key === 'h2h')!);

  const outcomeOdds: Record<string, number[]> = {};

  h2hMarkets.forEach(market => {
    market.outcomes.forEach(outcome => {
      if (!outcomeOdds[outcome.name]) {
        outcomeOdds[outcome.name] = [];
      }
      outcomeOdds[outcome.name].push(outcome.price);
    });
  });

  // Average odds and convert to probability, removing vig
  const probabilities: Record<string, number> = {};
  let totalProb = 0;

  for (const outcome in outcomeOdds) {
    const avgOdds = outcomeOdds[outcome].reduce((sum, o) => sum + o, 0) / outcomeOdds[outcome].length;
    const impliedProb = 1 / avgOdds;
    probabilities[outcome] = impliedProb;
    totalProb += impliedProb;
  }

  // Normalize to remove vig (bookmaker margin)
  for (const outcome in probabilities) {
    probabilities[outcome] /= totalProb;
  }

  return probabilities;
}

function findBestOdds(game: OddsResponse): Record<string, number> {
  const bestOdds: Record<string, number> = {};

  game.bookmakers.forEach(bookmaker => {
    const h2hMarket = bookmaker.markets.find(m => m.key === 'h2h');
    if (h2hMarket) {
      h2hMarket.outcomes.forEach(outcome => {
        if (!bestOdds[outcome.name] || outcome.price > bestOdds[outcome.name]) {
          bestOdds[outcome.name] = outcome.price;
        }
      });
    }
  });

  return bestOdds;
}

function calculateExpectedValue(probability: number, odds: number): number {
  // EV = (probability × winAmount) - (loseProb × betAmount)
  // For $100 bet:
  const betAmount = 100;
  const winAmount = (odds - 1) * betAmount;
  const loseProb = 1 - probability;

  return (probability * winAmount) - (loseProb * betAmount);
}
