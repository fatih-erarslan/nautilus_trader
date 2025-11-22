import * as wasm from '../wasm/pkg/cwts_ultra_wasm';

export interface ProbabilisticRiskMetrics {
    confidence_interval_95: [number, number];
    confidence_interval_99: [number, number];
    expected_shortfall: number;
    tail_risk_probability: number;
    uncertainty_score: number;
    bayesian_var: number;
    heavy_tail_index: number;
    regime_probability: Record<string, number>;
    monte_carlo_iterations: number;
    timestamp: number;
}

export interface BayesianParameters {
    prior_alpha: number;
    prior_beta: number;
    volatility_prior_shape: number;
    volatility_prior_rate: number;
    learning_rate: number;
    evidence_weight: number;
}

export interface HeavyTailDistribution {
    degrees_of_freedom: number;
    location: number;
    scale: number;
    tail_index: number;
    kurtosis: number;
}

export interface MarketConditions {
    volatility: number;
    volume: number;
    spread: number;
    momentum: number;
    liquidity: number;
}

/**
 * Probabilistic Risk Management System - Web Interface
 * Provides real-time probabilistic risk assessment with uncertainty quantification
 */
export class ProbabilisticRiskManager {
    private engine: wasm.WasmProbabilisticRiskEngine;
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private animationFrameId: number | null = null;
    private historicalMetrics: ProbabilisticRiskMetrics[] = [];
    private realTimeData: number[] = [];
    private uncertaintyHistory: number[] = [];

    constructor(canvasId: string, bayesianParams?: Partial<BayesianParameters>) {
        // Initialize WASM engine
        const params = new wasm.WasmBayesianParameters();
        if (bayesianParams) {
            if (bayesianParams.prior_alpha !== undefined) params.prior_alpha = bayesianParams.prior_alpha;
            if (bayesianParams.prior_beta !== undefined) params.prior_beta = bayesianParams.prior_beta;
            if (bayesianParams.learning_rate !== undefined) params.learning_rate = bayesianParams.learning_rate;
        }

        this.engine = new wasm.WasmProbabilisticRiskEngine(params);

        // Initialize canvas
        this.canvas = document.getElementById(canvasId) as HTMLCanvasElement;
        if (!this.canvas) {
            throw new Error(`Canvas element with ID '${canvasId}' not found`);
        }

        const ctx = this.canvas.getContext('2d');
        if (!ctx) {
            throw new Error('Could not get 2D context from canvas');
        }
        this.ctx = ctx;

        // Set canvas size
        this.canvas.width = 1200;
        this.canvas.height = 800;

        console.log('🎯 Probabilistic Risk Manager initialized with advanced Bayesian inference');
    }

    /**
     * Process new market data and update risk metrics
     */
    async processMarketData(prices: number[], marketConditions: MarketConditions): Promise<ProbabilisticRiskMetrics> {
        try {
            // Calculate returns from prices
            const returns = wasm.calculate_returns_from_prices(new Float64Array(prices));
            
            // Update Bayesian parameters
            const bayesianResult = await this.engine.bayesian_parameter_estimation(returns);
            console.log('📊 Bayesian estimation completed:', bayesianResult);

            // Update regime probabilities
            await this.engine.update_regime_probabilities(marketConditions);

            // Generate comprehensive metrics
            const portfolio_value = 1000000; // $1M portfolio
            const metrics = await this.engine.generate_comprehensive_metrics(portfolio_value, marketConditions);

            // Store for historical analysis
            this.historicalMetrics.push(metrics);
            if (this.historicalMetrics.length > 1000) {
                this.historicalMetrics.shift(); // Keep only recent data
            }

            // Update real-time visualization
            this.updateVisualization(metrics, prices);

            return metrics;
        } catch (error) {
            console.error('❌ Error processing market data:', error);
            throw error;
        }
    }

    /**
     * Run Monte Carlo simulation for Value at Risk
     */
    async runMonteCarloVaR(portfolioValue: number, confidenceLevels: number[] = [0.95, 0.99], iterations: number = 10000): Promise<Record<string, number>> {
        try {
            console.log(`🎲 Running Monte Carlo simulation: ${iterations} iterations`);
            const startTime = performance.now();

            const results = await this.engine.monte_carlo_var(portfolioValue, confidenceLevels, iterations);
            
            const endTime = performance.now();
            console.log(`✅ Monte Carlo completed in ${(endTime - startTime).toFixed(2)}ms`);

            return results;
        } catch (error) {
            console.error('❌ Monte Carlo simulation failed:', error);
            throw error;
        }
    }

    /**
     * Model heavy-tail distribution characteristics
     */
    async analyzeHeavyTailDistribution(): Promise<HeavyTailDistribution> {
        try {
            const distribution = await this.engine.model_heavy_tail_distribution();
            console.log('📈 Heavy-tail distribution analysis:', distribution);
            return distribution;
        } catch (error) {
            console.error('❌ Heavy-tail analysis failed:', error);
            throw error;
        }
    }

    /**
     * Real-time uncertainty propagation
     */
    async propagateUncertainty(newPrice: number, previousUncertainty: number = 0.1): Promise<number> {
        try {
            const uncertainty = await this.engine.propagate_uncertainty(newPrice, previousUncertainty);
            
            // Update uncertainty history
            this.uncertaintyHistory.push(uncertainty);
            if (this.uncertaintyHistory.length > 500) {
                this.uncertaintyHistory.shift();
            }

            return uncertainty;
        } catch (error) {
            console.error('❌ Uncertainty propagation failed:', error);
            return previousUncertainty;
        }
    }

    /**
     * Start real-time monitoring and visualization
     */
    startRealTimeMonitoring(dataSource: () => Promise<{ prices: number[], conditions: MarketConditions }>, intervalMs: number = 1000): void {
        const monitor = async () => {
            try {
                const { prices, conditions } = await dataSource();
                await this.processMarketData(prices, conditions);
                
                // Schedule next update
                setTimeout(monitor, intervalMs);
            } catch (error) {
                console.error('❌ Real-time monitoring error:', error);
                setTimeout(monitor, intervalMs * 2); // Retry with backoff
            }
        };

        monitor();
        console.log('🔄 Real-time monitoring started');
    }

    /**
     * Generate performance benchmark report
     */
    async benchmarkPerformance(iterations: number = 5000): Promise<{ duration_ms: number, var_95: number, var_99: number }> {
        const benchmark = wasm.benchmark_monte_carlo(iterations);
        console.log('⚡ Performance benchmark:', benchmark);
        return benchmark;
    }

    /**
     * Update real-time visualization
     */
    private updateVisualization(metrics: ProbabilisticRiskMetrics, prices: number[]): void {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Set background
        this.ctx.fillStyle = '#0a0a0a';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw title
        this.ctx.fillStyle = '#00ff00';
        this.ctx.font = 'bold 24px Arial';
        this.ctx.fillText('CWTS Probabilistic Risk Assessment', 20, 40);

        // Draw VaR confidence intervals
        this.drawConfidenceIntervals(metrics, 80);

        // Draw regime probabilities
        this.drawRegimeProbabilities(metrics.regime_probability, 300);

        // Draw uncertainty evolution
        this.drawUncertaintyHistory(500);

        // Draw tail risk indicators
        this.drawTailRiskIndicators(metrics, 650);

        // Draw real-time price data
        this.drawPriceChart(prices, 80, 400);
    }

    private drawConfidenceIntervals(metrics: ProbabilisticRiskMetrics, y: number): void {
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = '16px Arial';
        this.ctx.fillText('Value at Risk Confidence Intervals', 20, y);

        // 95% CI
        this.ctx.fillStyle = '#ffaa00';
        this.ctx.fillText(`95% CI: [${metrics.confidence_interval_95[0].toFixed(0)}, ${metrics.confidence_interval_95[1].toFixed(0)}]`, 20, y + 30);

        // 99% CI  
        this.ctx.fillStyle = '#ff4444';
        this.ctx.fillText(`99% CI: [${metrics.confidence_interval_99[0].toFixed(0)}, ${metrics.confidence_interval_99[1].toFixed(0)}]`, 20, y + 55);

        // Expected Shortfall
        this.ctx.fillStyle = '#ff0000';
        this.ctx.fillText(`Expected Shortfall: ${metrics.expected_shortfall.toFixed(0)}`, 20, y + 80);

        // Uncertainty Score
        const uncertaintyColor = metrics.uncertainty_score > 0.7 ? '#ff0000' : metrics.uncertainty_score > 0.4 ? '#ffaa00' : '#00ff00';
        this.ctx.fillStyle = uncertaintyColor;
        this.ctx.fillText(`Uncertainty Score: ${(metrics.uncertainty_score * 100).toFixed(1)}%`, 20, y + 105);
    }

    private drawRegimeProbabilities(regimes: Record<string, number>, y: number): void {
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = '16px Arial';
        this.ctx.fillText('Market Regime Probabilities', 20, y);

        const regimeColors = {
            'low_volatility': '#00ff00',
            'medium_volatility': '#ffaa00',
            'high_volatility': '#ff4444',
            'crisis': '#ff0000'
        };

        let offsetY = 0;
        for (const [regime, probability] of Object.entries(regimes)) {
            const color = regimeColors[regime as keyof typeof regimeColors] || '#888888';
            this.ctx.fillStyle = color;
            
            // Draw probability bar
            const barWidth = probability * 300;
            this.ctx.fillRect(200, y + 25 + offsetY, barWidth, 15);
            
            // Draw label
            this.ctx.fillStyle = '#ffffff';
            this.ctx.font = '12px Arial';
            this.ctx.fillText(`${regime}: ${(probability * 100).toFixed(1)}%`, 20, y + 35 + offsetY);
            
            offsetY += 25;
        }
    }

    private drawUncertaintyHistory(y: number): void {
        if (this.uncertaintyHistory.length < 2) return;

        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = '16px Arial';
        this.ctx.fillText('Uncertainty Evolution', 20, y);

        this.ctx.strokeStyle = '#00aaff';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();

        const chartWidth = 400;
        const chartHeight = 100;
        const dataPoints = this.uncertaintyHistory.slice(-100); // Last 100 points

        for (let i = 0; i < dataPoints.length; i++) {
            const x = 20 + (i / (dataPoints.length - 1)) * chartWidth;
            const uncertaintyY = y + 30 + chartHeight - (dataPoints[i] * chartHeight);
            
            if (i === 0) {
                this.ctx.moveTo(x, uncertaintyY);
            } else {
                this.ctx.lineTo(x, uncertaintyY);
            }
        }

        this.ctx.stroke();

        // Draw uncertainty threshold lines
        this.ctx.strokeStyle = '#ff4444';
        this.ctx.lineWidth = 1;
        this.ctx.setLineDash([5, 5]);
        this.ctx.beginPath();
        const highThreshold = y + 30 + chartHeight - (0.7 * chartHeight);
        this.ctx.moveTo(20, highThreshold);
        this.ctx.lineTo(20 + chartWidth, highThreshold);
        this.ctx.stroke();
        this.ctx.setLineDash([]);
    }

    private drawTailRiskIndicators(metrics: ProbabilisticRiskMetrics, y: number): void {
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = '16px Arial';
        this.ctx.fillText('Tail Risk Analysis', 20, y);

        // Heavy tail index
        const tailRiskLevel = metrics.heavy_tail_index < 3 ? 'HIGH' : metrics.heavy_tail_index < 5 ? 'MEDIUM' : 'LOW';
        const tailColor = tailRiskLevel === 'HIGH' ? '#ff0000' : tailRiskLevel === 'MEDIUM' ? '#ffaa00' : '#00ff00';
        
        this.ctx.fillStyle = tailColor;
        this.ctx.fillText(`Tail Index: ${metrics.heavy_tail_index.toFixed(2)} (${tailRiskLevel} Risk)`, 20, y + 25);

        // Tail probability
        this.ctx.fillStyle = '#ffaa00';
        this.ctx.fillText(`Tail Event Probability: ${(metrics.tail_risk_probability * 100).toFixed(2)}%`, 20, y + 50);
    }

    private drawPriceChart(prices: number[], x: number, y: number): void {
        if (prices.length < 2) return;

        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = '16px Arial';
        this.ctx.fillText('Price Movement', x + 450, y);

        const chartWidth = 300;
        const chartHeight = 150;
        const minPrice = Math.min(...prices);
        const maxPrice = Math.max(...prices);
        const priceRange = maxPrice - minPrice;

        this.ctx.strokeStyle = '#00ff00';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();

        for (let i = 0; i < prices.length; i++) {
            const chartX = x + 450 + (i / (prices.length - 1)) * chartWidth;
            const chartY = y + 30 + chartHeight - ((prices[i] - minPrice) / priceRange) * chartHeight;
            
            if (i === 0) {
                this.ctx.moveTo(chartX, chartY);
            } else {
                this.ctx.lineTo(chartX, chartY);
            }
        }

        this.ctx.stroke();
    }

    /**
     * Export metrics to CSV for external analysis
     */
    exportMetricsToCSV(): string {
        const headers = [
            'timestamp', 'expected_shortfall', 'uncertainty_score', 'bayesian_var',
            'heavy_tail_index', 'tail_risk_probability', 'var_95_lower', 'var_95_upper',
            'var_99_lower', 'var_99_upper'
        ];

        const rows = this.historicalMetrics.map(m => [
            m.timestamp,
            m.expected_shortfall,
            m.uncertainty_score,
            m.bayesian_var,
            m.heavy_tail_index,
            m.tail_risk_probability,
            m.confidence_interval_95[0],
            m.confidence_interval_95[1],
            m.confidence_interval_99[0],
            m.confidence_interval_99[1]
        ]);

        const csvContent = [headers.join(','), ...rows.map(row => row.join(','))].join('\n');
        return csvContent;
    }

    /**
     * Generate statistical summary report
     */
    generateSummaryReport(): string {
        if (this.historicalMetrics.length === 0) {
            return 'No historical data available for summary';
        }

        const latest = this.historicalMetrics[this.historicalMetrics.length - 1];
        const avgUncertainty = this.uncertaintyHistory.reduce((a, b) => a + b, 0) / this.uncertaintyHistory.length;

        return `
PROBABILISTIC RISK ASSESSMENT SUMMARY
=====================================
Timestamp: ${new Date(latest.timestamp).toISOString()}
Portfolio Analysis Period: ${this.historicalMetrics.length} observations

RISK METRICS:
- Expected Shortfall (CVaR): ${latest.expected_shortfall.toFixed(2)}
- Bayesian VaR: ${latest.bayesian_var.toFixed(2)}
- 95% Confidence Interval: [${latest.confidence_interval_95[0].toFixed(2)}, ${latest.confidence_interval_95[1].toFixed(2)}]
- 99% Confidence Interval: [${latest.confidence_interval_99[0].toFixed(2)}, ${latest.confidence_interval_99[1].toFixed(2)}]

UNCERTAINTY ANALYSIS:
- Current Uncertainty Score: ${(latest.uncertainty_score * 100).toFixed(1)}%
- Average Uncertainty: ${(avgUncertainty * 100).toFixed(1)}%

TAIL RISK ANALYSIS:
- Heavy Tail Index: ${latest.heavy_tail_index.toFixed(2)}
- Tail Event Probability: ${(latest.tail_risk_probability * 100).toFixed(2)}%

MARKET REGIME PROBABILITIES:
${Object.entries(latest.regime_probability)
    .map(([regime, prob]) => `- ${regime}: ${(prob * 100).toFixed(1)}%`)
    .join('\n')}

SIMULATION DETAILS:
- Monte Carlo Iterations: ${latest.monte_carlo_iterations.toLocaleString()}
- Advanced Variance Reduction: Antithetic Variates + Control Variates
- Bayesian Parameter Estimation: Adaptive Learning Rate
`;
    }

    /**
     * Cleanup resources
     */
    destroy(): void {
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
        }
        console.log('🧹 Probabilistic Risk Manager destroyed');
    }
}

/**
 * Utility function to create sample market data for testing
 */
export function createSampleMarketData(volatility: number, trend: number, noiseLevel: number, length: number): number[] {
    return Array.from(wasm.create_sample_market_data(volatility, trend, noiseLevel, length));
}

/**
 * Utility function to calculate rolling volatility
 */
export function calculateRollingVolatility(returns: number[], windowSize: number): number[] {
    return Array.from(wasm.calculate_rolling_volatility(new Float64Array(returns), windowSize));
}