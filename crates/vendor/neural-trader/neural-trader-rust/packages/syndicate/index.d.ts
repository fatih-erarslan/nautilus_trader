/**
 * @neural-trader/syndicate
 *
 * Investment syndicate management with Kelly Criterion, profit distribution, and governance
 *
 * High-performance Rust NAPI bindings for collaborative betting and investment syndicates
 */

/** Allocation strategy for distributing syndicate capital */
export enum AllocationStrategy {
  /** Full Kelly Criterion - maximize log utility */
  KellyCriterion = 'kelly_criterion',
  /** Fixed percentage of bankroll per bet */
  FixedPercentage = 'fixed_percentage',
  /** Dynamic allocation based on confidence level */
  DynamicConfidence = 'dynamic_confidence',
  /** Risk parity across uncorrelated opportunities */
  RiskParity = 'risk_parity',
  /** Martingale - double bet after loss */
  Martingale = 'martingale',
  /** Anti-Martingale - double bet after win */
  AntiMartingale = 'anti_martingale'
}

/** Profit distribution models for syndicate members */
export enum DistributionModel {
  /** Distribute proportional to capital contribution */
  Proportional = 'proportional',
  /** Weight by member performance score */
  PerformanceWeighted = 'performance_weighted',
  /** Tiered distribution based on member tier */
  Tiered = 'tiered',
  /** Hybrid: base proportional + performance bonus */
  Hybrid = 'hybrid'
}

/** Member roles with different permissions */
export enum MemberRole {
  /** Full control over syndicate */
  LeadInvestor = 'lead_investor',
  /** Can approve large bets and manage strategy */
  SeniorAnalyst = 'senior_analyst',
  /** Can propose bets and analyze opportunities */
  JuniorAnalyst = 'junior_analyst',
  /** Capital contributor with voting rights */
  ContributingMember = 'contributing_member',
  /** View-only access to analytics */
  Observer = 'observer'
}

/** Member tier affecting profit distribution */
export enum MemberTier {
  Bronze = 'bronze',
  Silver = 'silver',
  Gold = 'gold',
  Platinum = 'platinum'
}

/** Vote types for syndicate governance */
export enum VoteType {
  /** Change allocation strategy */
  StrategyChange = 'strategy_change',
  /** Approve large bet (>threshold) */
  LargeBet = 'large_bet',
  /** Add new member */
  MemberAddition = 'member_addition',
  /** Remove existing member */
  MemberRemoval = 'member_removal',
  /** Emergency withdrawal */
  EmergencyWithdrawal = 'emergency_withdrawal',
  /** Modify bankroll rules */
  RuleChange = 'rule_change',
  /** Distribute profits */
  ProfitDistribution = 'profit_distribution'
}

/** Vote status */
export enum VoteStatus {
  Active = 'active',
  Passed = 'passed',
  Failed = 'failed',
  Expired = 'expired'
}

/** Withdrawal request status */
export enum WithdrawalStatus {
  Pending = 'pending',
  Approved = 'approved',
  Rejected = 'rejected',
  Completed = 'completed'
}

/** Bankroll management rules */
export interface BankrollRules {
  /** Maximum bet size as percentage of bankroll (0.0-1.0) */
  maxSingleBet: number;
  /** Maximum daily exposure as percentage (0.0-1.0) */
  maxDailyExposure: number;
  /** Maximum exposure to single sport as percentage (0.0-1.0) */
  maxSportConcentration: number;
  /** Minimum reserve to maintain (absolute value) */
  minimumReserve: number;
  /** Daily stop-loss threshold as percentage (0.0-1.0) */
  stopLossDaily: number;
  /** Weekly stop-loss threshold as percentage (0.0-1.0) */
  stopLossWeekly: number;
  /** Lock profits at threshold percentage (0.0-1.0) */
  profitLock: number;
  /** Maximum parlay allocation as percentage (0.0-1.0) */
  maxParlayPercentage: number;
  /** Maximum live betting allocation as percentage (0.0-1.0) */
  maxLiveBetting: number;
}

/** Member permissions matrix */
export interface MemberPermissions {
  createSyndicate: boolean;
  modifyStrategy: boolean;
  approveLargeBets: boolean;
  manageMembers: boolean;
  distributeProfits: boolean;
  accessAllAnalytics: boolean;
  vetoPower: boolean;
  proposeBets: boolean;
  viewOtherMembers: boolean;
  modifyBankrollRules: boolean;
  initiateVote: boolean;
  castVote: boolean;
  emergencyWithdraw: boolean;
  exportData: boolean;
  modifyOwnProfile: boolean;
  viewFinancials: boolean;
  accessHistoricalData: boolean;
  manageIntegrations: boolean;
}

/** Member statistics */
export interface MemberStatistics {
  totalBetsProposed: number;
  betsAccepted: number;
  winRate: number;
  averageOdds: number;
  totalProfitGenerated: number;
  totalLoss: number;
  sharpeRatio: number;
  roi: number;
  longestWinStreak: number;
  longestLoseStreak: number;
  averageBetSize: number;
  totalVotesCast: number;
  votesWithMajority: number;
  contributionScore: number;
  activityScore: number;
}

/** Syndicate member */
export interface Member {
  id: string;
  name: string;
  email: string;
  role: MemberRole;
  tier: MemberTier;
  permissions: MemberPermissions;
  capitalContribution: string;
  currentBalance: string;
  performanceScore: number;
  joinedAt: Date;
  lastActive: Date;
  statistics: MemberStatistics;
  isActive: boolean;
}

/** Betting opportunity details */
export interface BettingOpportunity {
  id: string;
  sport: string;
  event: string;
  betType: string;
  odds: number;
  probability: number;
  edge: number;
  confidence: number;
  stake?: number;
  proposedBy?: string;
  analysisNotes?: string;
  expiresAt?: Date;
  correlatedBets?: string[];
}

/** Allocation result from fund distribution */
export interface AllocationResult {
  opportunity: BettingOpportunity;
  allocatedAmount: string;
  strategy: AllocationStrategy;
  kellyPercentage: number;
  adjustedPercentage: number;
  expectedValue: number;
  riskScore: number;
  approved: boolean;
  approvalRequired: boolean;
  votingRequired: boolean;
  reasoning: string;
  warnings: string[];
}

/** Vote record */
export interface Vote {
  id: string;
  voteType: VoteType;
  proposer: string;
  proposal: string;
  details: Record<string, any>;
  votesFor: Map<string, boolean>;
  votesAgainst: Map<string, boolean>;
  abstentions: string[];
  status: VoteStatus;
  requiredMajority: number;
  createdAt: Date;
  expiresAt: Date;
  resolvedAt?: Date;
  result?: string;
}

/** Withdrawal request */
export interface WithdrawalRequest {
  id: string;
  memberId: string;
  amount: string;
  reason: string;
  status: WithdrawalStatus;
  requestedAt: Date;
  processedAt?: Date;
  approvedBy?: string[];
  rejectionReason?: string;
  isEmergency: boolean;
  penaltyApplied: number;
}

/** Risk metrics for portfolio */
export interface RiskMetrics {
  totalExposure: string;
  dailyExposure: string;
  weeklyExposure: string;
  sportConcentration: Map<string, number>;
  largestSingleBet: string;
  averageBetSize: string;
  kellyDeviation: number;
  portfolioVolatility: number;
  sharpeRatio: number;
  sortinoRatio: number;
  maxDrawdown: number;
  valueAtRisk95: number;
  valueAtRisk99: number;
  expectedShortfall: number;
  correlationRisk: number;
  liquidityRisk: number;
}

/** Performance report */
export interface PerformanceReport {
  syndicateId: string;
  period: {
    start: Date;
    end: Date;
  };
  bankroll: {
    starting: string;
    ending: string;
    peak: string;
    trough: string;
  };
  performance: {
    totalReturn: number;
    annualizedReturn: number;
    volatility: number;
    sharpeRatio: number;
    sortinoRatio: number;
    maxDrawdown: number;
    winRate: number;
    averageWin: number;
    averageLoss: number;
    profitFactor: number;
    expectancy: number;
  };
  betting: {
    totalBets: number;
    totalStake: string;
    totalProfitLoss: string;
    bySport: Map<string, {
      bets: number;
      stake: string;
      profitLoss: string;
      winRate: number;
    }>;
    byStrategy: Map<AllocationStrategy, {
      bets: number;
      stake: string;
      profitLoss: string;
      roi: number;
    }>;
  };
  members: {
    totalMembers: number;
    activeMembers: number;
    contributionsByMember: Map<string, {
      betsProposed: number;
      betsAccepted: number;
      profitGenerated: string;
      performanceScore: number;
    }>;
    topPerformers: Array<{
      memberId: string;
      metric: string;
      value: number;
    }>;
  };
  riskMetrics: RiskMetrics;
}

/** Syndicate state export */
export interface SyndicateState {
  syndicate: {
    id: string;
    totalBankroll: string;
    availableBankroll: string;
    rules: BankrollRules;
    createdAt: Date;
    version: string;
  };
  members: Member[];
  activeVotes: Vote[];
  pendingWithdrawals: WithdrawalRequest[];
  allocationHistory: AllocationResult[];
  performanceMetrics: PerformanceReport;
}

/**
 * Main syndicate manager class
 */
export class SyndicateManager {
  /**
   * Create a new syndicate manager instance
   * @param syndicateId Unique syndicate identifier
   * @param totalBankroll Initial bankroll amount
   */
  constructor(syndicateId: string, totalBankroll: string);

  /**
   * Allocate funds to betting opportunity using specified strategy
   * @param opportunity Betting opportunity details
   * @param strategy Allocation strategy to use
   * @returns Allocation result with approved amount and reasoning
   */
  allocateFunds(
    opportunity: BettingOpportunity,
    strategy: AllocationStrategy
  ): Promise<AllocationResult>;

  /**
   * Distribute profits among syndicate members
   * @param profit Total profit amount to distribute
   * @param model Distribution model to use
   * @returns Map of member ID to distributed amount
   */
  distributeProfits(
    profit: string,
    model: DistributionModel
  ): Promise<Map<string, string>>;

  /**
   * Add new member to syndicate
   * @param name Member name
   * @param email Member email
   * @param role Member role
   * @param capitalContribution Initial capital contribution
   * @returns New member ID
   */
  addMember(
    name: string,
    email: string,
    role: MemberRole,
    capitalContribution: string
  ): Promise<string>;

  /**
   * Remove member from syndicate
   * @param memberId Member ID to remove
   * @param requireVote Whether to require governance vote
   * @returns Success status
   */
  removeMember(memberId: string, requireVote: boolean): Promise<boolean>;

  /**
   * Update member role and permissions
   * @param memberId Member ID
   * @param newRole New role to assign
   * @returns Updated member
   */
  updateMemberRole(memberId: string, newRole: MemberRole): Promise<Member>;

  /**
   * Update member tier
   * @param memberId Member ID
   * @param newTier New tier to assign
   * @returns Updated member
   */
  updateMemberTier(memberId: string, newTier: MemberTier): Promise<Member>;

  /**
   * Request withdrawal of capital
   * @param memberId Member ID requesting withdrawal
   * @param amount Amount to withdraw
   * @param isEmergency Emergency withdrawal flag
   * @param reason Withdrawal reason
   * @returns Withdrawal request
   */
  requestWithdrawal(
    memberId: string,
    amount: string,
    isEmergency?: boolean,
    reason?: string
  ): Promise<WithdrawalRequest>;

  /**
   * Process pending withdrawal request
   * @param requestId Withdrawal request ID
   * @param approve Approval decision
   * @param approverId Member ID approving/rejecting
   * @returns Updated withdrawal request
   */
  processWithdrawal(
    requestId: string,
    approve: boolean,
    approverId: string
  ): Promise<WithdrawalRequest>;

  /**
   * Create governance vote
   * @param voteType Type of vote
   * @param proposer Member ID proposing
   * @param proposal Proposal description
   * @param details Additional vote details
   * @param durationHours Vote duration in hours
   * @returns Created vote
   */
  createVote(
    voteType: VoteType,
    proposer: string,
    proposal: string,
    details: Record<string, any>,
    durationHours?: number
  ): Promise<Vote>;

  /**
   * Cast vote on proposal
   * @param voteId Vote ID
   * @param memberId Member ID casting vote
   * @param inFavor Vote in favor (true) or against (false)
   * @returns Updated vote
   */
  castVote(voteId: string, memberId: string, inFavor: boolean): Promise<Vote>;

  /**
   * Get member by ID
   * @param memberId Member ID
   * @returns Member details
   */
  getMember(memberId: string): Promise<Member>;

  /**
   * Get all syndicate members
   * @param activeOnly Return only active members
   * @returns List of members
   */
  getMembers(activeOnly?: boolean): Promise<Member[]>;

  /**
   * Get current bankroll status
   * @returns Bankroll details
   */
  getBankrollStatus(): Promise<{
    total: string;
    available: string;
    allocated: string;
    reserve: string;
  }>;

  /**
   * Calculate current risk metrics
   * @returns Risk metrics
   */
  getRiskMetrics(): Promise<RiskMetrics>;

  /**
   * Get member performance statistics
   * @param memberId Member ID
   * @returns Member statistics
   */
  getMemberPerformance(memberId: string): Promise<MemberStatistics>;

  /**
   * Generate comprehensive performance report
   * @param startDate Report start date
   * @param endDate Report end date
   * @returns Performance report
   */
  generatePerformanceReport(
    startDate?: Date,
    endDate?: Date
  ): Promise<PerformanceReport>;

  /**
   * Update bankroll rules
   * @param newRules New bankroll rules
   * @param requireVote Whether to require governance vote
   * @returns Updated rules
   */
  updateBankrollRules(
    newRules: Partial<BankrollRules>,
    requireVote?: boolean
  ): Promise<BankrollRules>;

  /**
   * Get active votes
   * @returns List of active votes
   */
  getActiveVotes(): Promise<Vote[]>;

  /**
   * Get pending withdrawal requests
   * @param memberId Optional member ID filter
   * @returns List of pending withdrawals
   */
  getPendingWithdrawals(memberId?: string): Promise<WithdrawalRequest[]>;

  /**
   * Export complete syndicate state
   * @returns Syndicate state export
   */
  exportState(): Promise<SyndicateState>;

  /**
   * Import syndicate state
   * @param state Syndicate state to import
   * @returns Success status
   */
  importState(state: SyndicateState): Promise<boolean>;
}

/**
 * Create a new syndicate manager instance
 * @param id Unique syndicate identifier
 * @param totalBankroll Initial bankroll amount
 * @param rules Optional bankroll rules (uses defaults if not provided)
 * @returns New syndicate manager
 */
export function createSyndicate(
  id: string,
  totalBankroll: string,
  rules?: Partial<BankrollRules>
): Promise<SyndicateManager>;

/**
 * Calculate Kelly Criterion percentage
 * @param probability Win probability (0.0-1.0)
 * @param odds Decimal odds
 * @param edgePercentage Optional edge adjustment
 * @returns Kelly percentage (0.0-1.0)
 */
export function calculateKelly(
  probability: number,
  odds: number,
  edgePercentage?: number
): number;

/**
 * Calculate fractional Kelly (reduced risk)
 * @param probability Win probability
 * @param odds Decimal odds
 * @param fraction Kelly fraction (e.g., 0.25 for quarter Kelly)
 * @returns Fractional Kelly percentage
 */
export function calculateKellyFractional(
  probability: number,
  odds: number,
  fraction: number
): number;

/**
 * Calculate optimal bet size with constraints
 * @param bankroll Current bankroll
 * @param opportunity Betting opportunity
 * @param strategy Allocation strategy
 * @param rules Bankroll rules
 * @returns Optimal bet size
 */
export function calculateOptimalBetSize(
  bankroll: string,
  opportunity: BettingOpportunity,
  strategy: AllocationStrategy,
  rules: BankrollRules
): string;

/**
 * Validate bankroll rules for consistency
 * @param rules Bankroll rules to validate
 * @returns Validation result with errors if any
 */
export function validateBankrollRules(rules: BankrollRules): {
  valid: boolean;
  errors: string[];
};

/**
 * Calculate portfolio risk metrics
 * @param bankroll Current bankroll
 * @param allocations Active allocations
 * @param historicalReturns Historical return data
 * @returns Risk metrics
 */
export function calculateRiskMetrics(
  bankroll: string,
  allocations: AllocationResult[],
  historicalReturns: number[]
): Promise<RiskMetrics>;

/**
 * Simulate different allocation strategies
 * @param opportunities List of opportunities
 * @param bankroll Starting bankroll
 * @param strategies Strategies to simulate
 * @param iterations Number of simulation runs
 * @returns Simulation results by strategy
 */
export function simulateAllocationStrategies(
  opportunities: BettingOpportunity[],
  bankroll: string,
  strategies: AllocationStrategy[],
  iterations: number
): Promise<Map<AllocationStrategy, {
  finalBankroll: string;
  totalReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
}>>;

/**
 * Calculate tax liability for member earnings
 * @param memberId Member ID
 * @param earnings Total earnings amount
 * @param jurisdiction Tax jurisdiction
 * @returns Estimated tax liability
 */
export function calculateMemberTaxLiability(
  memberId: string,
  earnings: string,
  jurisdiction: string
): Promise<{
  grossEarnings: string;
  taxableAmount: string;
  estimatedTax: string;
  netEarnings: string;
  breakdown: Record<string, string>;
}>;

/**
 * Generate performance report for syndicate
 * @param syndicateId Syndicate ID
 * @param startDate Report start date
 * @param endDate Report end date
 * @returns Performance report
 */
export function generatePerformanceReport(
  syndicateId: string,
  startDate?: Date,
  endDate?: Date
): Promise<PerformanceReport>;

/**
 * Export syndicate state to JSON
 * @param syndicateId Syndicate ID
 * @returns Serialized state
 */
export function exportSyndicateState(syndicateId: string): Promise<string>;

/**
 * Import syndicate state from JSON
 * @param stateJson Serialized state
 * @returns Restored syndicate manager
 */
export function importSyndicateState(stateJson: string): Promise<SyndicateManager>;

/** Package version */
export const version: string;

/** Native binding filename */
export const nativeBinding: string;
