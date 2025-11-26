#!/usr/bin/env python3
"""
Energy Portfolio Stress Testing Engine
Comprehensive risk analysis with multiple scenarios and VaR calculations
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EnergyPortfolioStressTester:
    """
    Advanced stress testing engine for energy portfolios
    Supports multiple scenarios with VaR/CVaR calculations
    """

    def __init__(self, portfolio_composition=None):
        """
        Initialize stress tester with portfolio composition

        Args:
            portfolio_composition: Dict with asset weights
        """
        # Default energy portfolio composition
        self.portfolio = portfolio_composition or {
            'oil_majors': 0.30,      # XOM, CVX, BP
            'oil_services': 0.15,    # SLB, HAL, BKR
            'lng_companies': 0.20,   # LNG, TELL, CEI
            'refiners': 0.15,        # VLO, MPC, PSX
            'pipelines': 0.10,       # KMI, EPD, ET
            'renewables': 0.10       # NEE, ENPH, SEDG
        }

        # Historical correlations and volatilities
        self.correlations = self._build_correlation_matrix()
        self.volatilities = self._build_volatility_vector()

        # Scenario parameters
        self.scenarios = {}
        self.results = {}

    def _build_correlation_matrix(self):
        """Build historical correlation matrix for energy assets"""
        assets = list(self.portfolio.keys())
        n = len(assets)

        # Empirical correlations from energy sector analysis
        corr_matrix = np.array([
            [1.00, 0.75, 0.65, 0.70, 0.45, -0.25],  # oil_majors
            [0.75, 1.00, 0.60, 0.65, 0.40, -0.20],  # oil_services
            [0.65, 0.60, 1.00, 0.55, 0.50, -0.15],  # lng_companies
            [0.70, 0.65, 0.55, 1.00, 0.35, -0.10],  # refiners
            [0.45, 0.40, 0.50, 0.35, 1.00, 0.10],   # pipelines
            [-0.25, -0.20, -0.15, -0.10, 0.10, 1.00] # renewables
        ])

        return pd.DataFrame(corr_matrix, index=assets, columns=assets)

    def _build_volatility_vector(self):
        """Build volatility vector for each asset class"""
        return pd.Series({
            'oil_majors': 0.28,      # 28% annual volatility
            'oil_services': 0.35,    # 35% annual volatility
            'lng_companies': 0.32,   # 32% annual volatility
            'refiners': 0.30,       # 30% annual volatility
            'pipelines': 0.18,      # 18% annual volatility (MLPs)
            'renewables': 0.40      # 40% annual volatility
        })

    def model_oil_crash_scenario(self, oil_drop=-0.30, time_horizon=252):
        """
        Model oil price crash scenario (-30% WTI)

        Args:
            oil_drop: Percentage drop in oil prices
            time_horizon: Days to model (252 = 1 year)
        """
        print("Modeling Oil Price Crash Scenario...")

        # Direct impact factors by asset class
        impact_factors = {
            'oil_majors': oil_drop * 1.2,      # 1.2x beta to oil
            'oil_services': oil_drop * 1.8,    # 1.8x beta (high leverage)
            'lng_companies': oil_drop * 0.7,   # 0.7x beta (gas correlation)
            'refiners': oil_drop * -0.3,       # Negative beta (crack spreads)
            'pipelines': oil_drop * 0.4,       # 0.4x beta (fee-based)
            'renewables': oil_drop * -0.2      # Negative beta (substitution)
        }

        # Calculate portfolio impact
        portfolio_impact = sum(
            weight * impact_factors[asset]
            for asset, weight in self.portfolio.items()
        )

        # Monte Carlo simulation for path dependency
        n_sims = 10000
        daily_returns = np.zeros((n_sims, time_horizon))

        for i in range(n_sims):
            # Simulate correlated shocks
            shocks = np.random.multivariate_normal(
                mean=np.zeros(len(self.portfolio)),
                cov=self.correlations.values,
                size=time_horizon
            )

            # Apply volatility scaling and crash impact
            for day in range(time_horizon):
                if day == 0:  # Initial crash
                    daily_return = portfolio_impact
                else:  # Ongoing volatility
                    daily_return = np.sum([
                        self.portfolio[asset] *
                        self.volatilities[asset] *
                        shocks[day, i] / np.sqrt(252)
                        for i, asset in enumerate(self.portfolio.keys())
                    ])
                daily_returns[i, day] = daily_return

        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + daily_returns, axis=1) - 1
        final_returns = cumulative_returns[:, -1]

        # Calculate survival probability (positive return)
        survival_prob = np.mean(final_returns > 0)

        scenario_results = {
            'scenario': 'Oil Crash (-30%)',
            'immediate_impact': portfolio_impact,
            'final_returns_distribution': final_returns,
            'survival_probability': survival_prob,
            'var_95': np.percentile(final_returns, 5),
            'cvar_95': np.mean(final_returns[final_returns <= np.percentile(final_returns, 5)]),
            'expected_return': np.mean(final_returns),
            'volatility': np.std(final_returns)
        }

        self.scenarios['oil_crash'] = scenario_results
        return scenario_results

    def model_recession_scenario(self, demand_drop=-0.25, time_horizon=252):
        """
        Model recession scenario with energy demand destruction

        Args:
            demand_drop: Percentage drop in energy demand
            time_horizon: Days to model
        """
        print("Modeling Recession Scenario...")

        # Recession impact factors (demand destruction)
        impact_factors = {
            'oil_majors': demand_drop * 1.5,      # High leverage to demand
            'oil_services': demand_drop * 2.0,    # Highest leverage
            'lng_companies': demand_drop * 1.2,   # Moderate leverage
            'refiners': demand_drop * 1.3,        # Margin compression
            'pipelines': demand_drop * 0.8,       # Volume-based impact
            'renewables': demand_drop * 0.5       # Government support
        }

        # Economic indicators correlation
        gdp_correlation = -0.7  # Energy demand correlated with GDP

        portfolio_impact = sum(
            weight * impact_factors[asset]
            for asset, weight in self.portfolio.items()
        )

        # Monte Carlo with recession dynamics
        n_sims = 10000
        daily_returns = np.zeros((n_sims, time_horizon))

        for i in range(n_sims):
            # Model recession as persistent negative shock
            recession_intensity = np.random.exponential(0.3)  # Severity
            recession_duration = min(int(np.random.exponential(180)), time_horizon)

            for day in range(time_horizon):
                if day == 0:  # Initial shock
                    daily_return = portfolio_impact
                elif day < recession_duration:  # Recession period
                    base_return = np.sum([
                        self.portfolio[asset] *
                        self.volatilities[asset] *
                        np.random.normal(0, 1) / np.sqrt(252)
                        for asset in self.portfolio.keys()
                    ])
                    # Add recession drag
                    recession_drag = -recession_intensity * 0.001  # Daily drag
                    daily_return = base_return + recession_drag
                else:  # Recovery period
                    daily_return = np.sum([
                        self.portfolio[asset] *
                        self.volatilities[asset] *
                        np.random.normal(0.001, 1) / np.sqrt(252)  # Slight recovery bias
                        for asset in self.portfolio.keys()
                    ])

                daily_returns[i, day] = daily_return

        cumulative_returns = np.cumprod(1 + daily_returns, axis=1) - 1
        final_returns = cumulative_returns[:, -1]
        survival_prob = np.mean(final_returns > 0)

        scenario_results = {
            'scenario': 'Recession (Demand Destruction)',
            'immediate_impact': portfolio_impact,
            'final_returns_distribution': final_returns,
            'survival_probability': survival_prob,
            'var_95': np.percentile(final_returns, 5),
            'cvar_95': np.mean(final_returns[final_returns <= np.percentile(final_returns, 5)]),
            'expected_return': np.mean(final_returns),
            'volatility': np.std(final_returns)
        }

        self.scenarios['recession'] = scenario_results
        return scenario_results

    def model_opec_cut_scenario(self, oil_increase=0.20, time_horizon=252):
        """
        Model OPEC production cut scenario (+20% oil prices)

        Args:
            oil_increase: Percentage increase in oil prices
            time_horizon: Days to model
        """
        print("Modeling OPEC Production Cut Scenario...")

        # OPEC cut impact factors
        impact_factors = {
            'oil_majors': oil_increase * 1.1,      # Direct beneficiary
            'oil_services': oil_increase * 1.4,    # Increased drilling
            'lng_companies': oil_increase * 0.8,   # Moderate correlation
            'refiners': oil_increase * -0.2,       # Margin compression
            'pipelines': oil_increase * 0.3,       # Volume increase
            'renewables': oil_increase * -0.1      # Slight negative
        }

        portfolio_impact = sum(
            weight * impact_factors[asset]
            for asset, weight in self.portfolio.items()
        )

        # Monte Carlo with OPEC dynamics
        n_sims = 10000
        daily_returns = np.zeros((n_sims, time_horizon))

        for i in range(n_sims):
            # Model OPEC cut sustainability
            cut_duration = min(int(np.random.gamma(2, 90)), time_horizon)  # Gamma distribution

            for day in range(time_horizon):
                if day == 0:  # Initial announcement
                    daily_return = portfolio_impact
                elif day < cut_duration:  # Cut period
                    # Gradual price increase with volatility
                    trend = oil_increase * 0.5 / cut_duration  # Gradual increase
                    volatility_component = np.sum([
                        self.portfolio[asset] *
                        self.volatilities[asset] *
                        np.random.normal(0, 1) / np.sqrt(252)
                        for asset in self.portfolio.keys()
                    ])
                    daily_return = trend + volatility_component
                else:  # Post-cut period (price normalization)
                    normalization_factor = -oil_increase * 0.3 / (time_horizon - cut_duration)
                    volatility_component = np.sum([
                        self.portfolio[asset] *
                        self.volatilities[asset] *
                        np.random.normal(0, 1) / np.sqrt(252)
                        for asset in self.portfolio.keys()
                    ])
                    daily_return = normalization_factor + volatility_component

                daily_returns[i, day] = daily_return

        cumulative_returns = np.cumprod(1 + daily_returns, axis=1) - 1
        final_returns = cumulative_returns[:, -1]
        survival_prob = np.mean(final_returns > 0)

        scenario_results = {
            'scenario': 'OPEC Cut (+20%)',
            'immediate_impact': portfolio_impact,
            'final_returns_distribution': final_returns,
            'survival_probability': survival_prob,
            'var_95': np.percentile(final_returns, 5),
            'cvar_95': np.mean(final_returns[final_returns <= np.percentile(final_returns, 5)]),
            'expected_return': np.mean(final_returns),
            'volatility': np.std(final_returns)
        }

        self.scenarios['opec_cut'] = scenario_results
        return scenario_results

    def model_clean_energy_disruption(self, ev_adoption_rate=0.4, time_horizon=252):
        """
        Model clean energy disruption risk (accelerated EV adoption)

        Args:
            ev_adoption_rate: Accelerated EV adoption rate
            time_horizon: Days to model
        """
        print("Modeling Clean Energy Disruption Scenario...")

        # EV disruption impact factors
        impact_factors = {
            'oil_majors': -ev_adoption_rate * 0.8,     # Demand destruction
            'oil_services': -ev_adoption_rate * 1.2,   # Reduced drilling
            'lng_companies': ev_adoption_rate * 0.2,   # Gas for electricity
            'refiners': -ev_adoption_rate * 1.5,       # Direct gasoline impact
            'pipelines': -ev_adoption_rate * 0.6,      # Reduced transport
            'renewables': ev_adoption_rate * 2.0       # Direct beneficiary
        }

        portfolio_impact = sum(
            weight * impact_factors[asset]
            for asset, weight in self.portfolio.items()
        )

        # Monte Carlo with S-curve adoption
        n_sims = 10000
        daily_returns = np.zeros((n_sims, time_horizon))

        for i in range(n_sims):
            # Model S-curve adoption pattern
            adoption_speed = np.random.lognormal(0, 0.3)  # Variable adoption speed

            for day in range(time_horizon):
                if day == 0:  # Initial announcement/breakthrough
                    daily_return = portfolio_impact * 0.3  # Partial immediate impact
                else:
                    # S-curve progression
                    t = day / time_horizon
                    s_curve_factor = 1 / (1 + np.exp(-adoption_speed * (t - 0.5)))

                    disruption_impact = portfolio_impact * s_curve_factor * 0.002  # Daily progression

                    volatility_component = np.sum([
                        self.portfolio[asset] *
                        self.volatilities[asset] *
                        np.random.normal(0, 1) / np.sqrt(252)
                        for asset in self.portfolio.keys()
                    ])

                    daily_return = disruption_impact + volatility_component

                daily_returns[i, day] = daily_return

        cumulative_returns = np.cumprod(1 + daily_returns, axis=1) - 1
        final_returns = cumulative_returns[:, -1]
        survival_prob = np.mean(final_returns > 0)

        scenario_results = {
            'scenario': 'Clean Energy Disruption (EV)',
            'immediate_impact': portfolio_impact * 0.3,
            'final_returns_distribution': final_returns,
            'survival_probability': survival_prob,
            'var_95': np.percentile(final_returns, 5),
            'cvar_95': np.mean(final_returns[final_returns <= np.percentile(final_returns, 5)]),
            'expected_return': np.mean(final_returns),
            'volatility': np.std(final_returns)
        }

        self.scenarios['clean_disruption'] = scenario_results
        return scenario_results

    def calculate_portfolio_var_cvar(self, confidence_level=0.05):
        """
        Calculate portfolio VaR and CVaR under all scenarios

        Args:
            confidence_level: Confidence level for VaR (0.05 = 95% VaR)
        """
        print("Calculating Portfolio VaR and CVaR...")

        var_cvar_results = {}

        for scenario_name, scenario_data in self.scenarios.items():
            returns = scenario_data['final_returns_distribution']

            # Value at Risk (VaR)
            var = np.percentile(returns, confidence_level * 100)

            # Conditional Value at Risk (CVaR) - Expected Shortfall
            cvar = np.mean(returns[returns <= var])

            var_cvar_results[scenario_name] = {
                'var_95': var,
                'cvar_95': cvar,
                'worst_case': np.min(returns),
                'best_case': np.max(returns),
                'median': np.median(returns),
                'skewness': stats.skew(returns),
                'kurtosis': stats.kurtosis(returns)
            }

        self.results['var_cvar'] = var_cvar_results
        return var_cvar_results

    def generate_stress_test_report(self):
        """Generate comprehensive stress test report"""
        print("\nGenerating Comprehensive Stress Test Report...")

        report = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_composition': self.portfolio,
            'scenarios_tested': len(self.scenarios),
            'scenario_results': self.scenarios,
            'var_cvar_analysis': self.results.get('var_cvar', {}),
            'risk_optimization_results': self.results.get('risk_optimization', {}),
            'summary_statistics': self._generate_summary_stats()
        }

        return report

    def _generate_summary_stats(self):
        """Generate summary statistics across all scenarios"""
        if not self.scenarios:
            return {}

        survival_probs = [s['survival_probability'] for s in self.scenarios.values()]
        var_95_values = [s['var_95'] for s in self.scenarios.values()]
        cvar_95_values = [s['cvar_95'] for s in self.scenarios.values()]

        return {
            'average_survival_probability': np.mean(survival_probs),
            'worst_case_survival': np.min(survival_probs),
            'best_case_survival': np.max(survival_probs),
            'average_var_95': np.mean(var_95_values),
            'worst_var_95': np.min(var_95_values),
            'average_cvar_95': np.mean(cvar_95_values),
            'worst_cvar_95': np.min(cvar_95_values),
            'scenario_count': len(self.scenarios)
        }

if __name__ == "__main__":
    # Example usage
    tester = EnergyPortfolioStressTester()

    # Run all scenarios
    print("Starting Comprehensive Energy Portfolio Stress Testing...")
    tester.model_oil_crash_scenario()
    tester.model_recession_scenario()
    tester.model_opec_cut_scenario()
    tester.model_clean_energy_disruption()

    # Calculate VaR/CVaR
    tester.calculate_portfolio_var_cvar()

    # Generate report
    report = tester.generate_stress_test_report()

    print("\nStress Testing Complete!")
    print(f"Scenarios Tested: {report['scenarios_tested']}")
    print(f"Average Survival Probability: {report['summary_statistics']['average_survival_probability']:.2%}")
    print(f"Worst Case VaR (95%): {report['summary_statistics']['worst_var_95']:.2%}")