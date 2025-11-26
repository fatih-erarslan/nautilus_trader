"""
Sports Betting Validation Framework

Comprehensive validation system for sports betting operations including:
- Data quality checks for odds feeds
- Model prediction validation
- Risk calculation verification  
- Syndicate operation testing
- API integration validation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import requests
import json
import time

from ..risk_management import BetOpportunity, SyndicateMember, RiskFramework


class ValidationLevel(Enum):
    """Validation severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error" 
    CRITICAL = "critical"


class ValidationCategory(Enum):
    """Validation categories"""
    DATA_QUALITY = "data_quality"
    MODEL_PERFORMANCE = "model_performance"
    RISK_CALCULATION = "risk_calculation"
    SYNDICATE_OPERATION = "syndicate_operation"
    API_INTEGRATION = "api_integration"
    BUSINESS_LOGIC = "business_logic"


@dataclass
class ValidationResult:
    """Single validation result"""
    test_name: str
    category: ValidationCategory
    level: ValidationLevel
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    execution_time_ms: float = 0.0


@dataclass 
class ValidationReport:
    """Complete validation report"""
    report_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    warnings: int = 0
    errors: int = 0
    critical_issues: int = 0
    results: List[ValidationResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate overall success rate"""
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests
    
    @property
    def duration_seconds(self) -> float:
        """Calculate total execution time"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


class SportsValidationFramework:
    """Comprehensive validation framework for sports betting system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.validation_history: List[ValidationReport] = []
        
        # Default validation thresholds
        self.thresholds = {
            'min_accuracy': 0.85,
            'max_latency_ms': 100,
            'min_odds_value': 1.01,
            'max_odds_value': 50.0,
            'max_edge_threshold': 0.30,  # 30% edge is suspiciously high
            'min_confidence': 0.1,
            'max_kelly_fraction': 1.0,
            'min_bankroll': 1000,
            'max_portfolio_risk': 0.50,
            'api_timeout_seconds': 30
        }
        self.thresholds.update(self.config.get('thresholds', {}))
        
    def create_validation_report(self, report_name: str) -> ValidationReport:
        """Create new validation report"""
        report_id = f"{report_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return ValidationReport(
            report_id=report_id,
            start_time=datetime.now()
        )
    
    def add_result(self, report: ValidationReport, result: ValidationResult) -> None:
        """Add validation result to report"""
        report.results.append(result)
        report.total_tests += 1
        
        if result.passed:
            report.passed_tests += 1
        else:
            report.failed_tests += 1
            
        if result.level == ValidationLevel.WARNING:
            report.warnings += 1
        elif result.level == ValidationLevel.ERROR:
            report.errors += 1
        elif result.level == ValidationLevel.CRITICAL:
            report.critical_issues += 1
    
    def finalize_report(self, report: ValidationReport) -> ValidationReport:
        """Finalize validation report with summary"""
        report.end_time = datetime.now()
        
        # Generate summary
        report.summary = {
            'success_rate': report.success_rate,
            'duration_seconds': report.duration_seconds,
            'categories': {},
            'recommendations': []
        }
        
        # Categorize results
        for category in ValidationCategory:
            category_results = [r for r in report.results if r.category == category]
            if category_results:
                passed = sum(1 for r in category_results if r.passed)
                report.summary['categories'][category.value] = {
                    'total': len(category_results),
                    'passed': passed,
                    'success_rate': passed / len(category_results)
                }
        
        # Generate recommendations
        if report.critical_issues > 0:
            report.summary['recommendations'].append(
                "CRITICAL: Immediate attention required - system not ready for production"
            )
        elif report.errors > 0:
            report.summary['recommendations'].append(
                "ERROR: Resolve errors before proceeding to production"
            )
        elif report.warnings > 0:
            report.summary['recommendations'].append(
                "WARNING: Review warnings before production deployment"
            )
        else:
            report.summary['recommendations'].append(
                "SUCCESS: All validations passed - system ready for deployment"
            )
            
        self.validation_history.append(report)
        return report
        
    def validate_odds_data(self, odds_data: List[Dict[str, Any]], 
                          report: ValidationReport) -> None:
        """Validate odds feed data quality"""
        start_time = time.time()
        
        try:
            # Test 1: Check data completeness
            required_fields = ['odds', 'bookmaker', 'sport', 'event', 'timestamp']
            complete_records = 0
            
            for record in odds_data:
                if all(field in record and record[field] is not None for field in required_fields):
                    complete_records += 1
                    
            completeness_rate = complete_records / len(odds_data) if odds_data else 0
            
            result = ValidationResult(
                test_name="odds_data_completeness",
                category=ValidationCategory.DATA_QUALITY,
                level=ValidationLevel.ERROR if completeness_rate < 0.95 else ValidationLevel.INFO,
                passed=completeness_rate >= 0.95,
                message=f"Odds data completeness: {completeness_rate:.2%}",
                details={'completeness_rate': completeness_rate, 'total_records': len(odds_data)},
                execution_time_ms=(time.time() - start_time) * 1000
            )
            self.add_result(report, result)
            
            # Test 2: Validate odds values
            if odds_data:
                odds_values = [r.get('odds', 0) for r in odds_data if 'odds' in r]
                valid_odds = [o for o in odds_values if self.thresholds['min_odds_value'] <= o <= self.thresholds['max_odds_value']]
                odds_validity_rate = len(valid_odds) / len(odds_values) if odds_values else 0
                
                result = ValidationResult(
                    test_name="odds_value_validation",
                    category=ValidationCategory.DATA_QUALITY,
                    level=ValidationLevel.ERROR if odds_validity_rate < 0.99 else ValidationLevel.INFO,
                    passed=odds_validity_rate >= 0.99,
                    message=f"Odds value validity: {odds_validity_rate:.2%}",
                    details={
                        'validity_rate': odds_validity_rate,
                        'min_odds': min(odds_values) if odds_values else 0,
                        'max_odds': max(odds_values) if odds_values else 0,
                        'avg_odds': np.mean(odds_values) if odds_values else 0
                    },
                    execution_time_ms=(time.time() - start_time) * 1000
                )
                self.add_result(report, result)
                
            # Test 3: Check timestamp freshness
            if odds_data:
                now = datetime.now()
                fresh_records = 0
                
                for record in odds_data:
                    if 'timestamp' in record:
                        try:
                            ts = datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00'))
                            if (now - ts.replace(tzinfo=None)).total_seconds() < 3600:  # Within 1 hour
                                fresh_records += 1
                        except:
                            pass
                            
                freshness_rate = fresh_records / len(odds_data)
                
                result = ValidationResult(
                    test_name="odds_data_freshness",
                    category=ValidationCategory.DATA_QUALITY,
                    level=ValidationLevel.WARNING if freshness_rate < 0.80 else ValidationLevel.INFO,
                    passed=freshness_rate >= 0.80,
                    message=f"Odds data freshness: {freshness_rate:.2%}",
                    details={'freshness_rate': freshness_rate},
                    execution_time_ms=(time.time() - start_time) * 1000
                )
                self.add_result(report, result)
                
        except Exception as e:
            result = ValidationResult(
                test_name="odds_data_validation_error",
                category=ValidationCategory.DATA_QUALITY,
                level=ValidationLevel.CRITICAL,
                passed=False,
                message=f"Odds validation failed: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
            self.add_result(report, result)
    
    def validate_model_predictions(self, predictions: List[Dict[str, Any]], 
                                 historical_results: List[Dict[str, Any]],
                                 report: ValidationReport) -> None:
        """Validate ML model prediction quality"""
        start_time = time.time()
        
        try:
            # Test 1: Prediction format validation
            required_pred_fields = ['probability', 'confidence', 'model_id', 'timestamp']
            valid_predictions = 0
            
            for pred in predictions:
                if all(field in pred and pred[field] is not None for field in required_pred_fields):
                    prob = pred.get('probability', 0)
                    conf = pred.get('confidence', 0)
                    if 0 <= prob <= 1 and self.thresholds['min_confidence'] <= conf <= 1:
                        valid_predictions += 1
                        
            pred_validity_rate = valid_predictions / len(predictions) if predictions else 0
            
            result = ValidationResult(
                test_name="prediction_format_validation",
                category=ValidationCategory.MODEL_PERFORMANCE,
                level=ValidationLevel.ERROR if pred_validity_rate < 0.95 else ValidationLevel.INFO,
                passed=pred_validity_rate >= 0.95,
                message=f"Prediction format validity: {pred_validity_rate:.2%}",
                details={'validity_rate': pred_validity_rate, 'total_predictions': len(predictions)},
                execution_time_ms=(time.time() - start_time) * 1000
            )
            self.add_result(report, result)
            
            # Test 2: Historical accuracy assessment
            if historical_results and len(historical_results) >= 10:
                correct_predictions = 0
                total_evaluated = 0
                
                for result_data in historical_results:
                    if 'predicted_probability' in result_data and 'actual_outcome' in result_data:
                        pred_prob = result_data['predicted_probability']
                        actual = result_data['actual_outcome']  # 1 for win, 0 for loss
                        
                        # Simple accuracy check (prob > 0.5 means prediction of win)
                        predicted_outcome = 1 if pred_prob > 0.5 else 0
                        if predicted_outcome == actual:
                            correct_predictions += 1
                        total_evaluated += 1
                        
                accuracy = correct_predictions / total_evaluated if total_evaluated > 0 else 0
                
                result = ValidationResult(
                    test_name="model_historical_accuracy",
                    category=ValidationCategory.MODEL_PERFORMANCE,
                    level=ValidationLevel.WARNING if accuracy < self.thresholds['min_accuracy'] else ValidationLevel.INFO,
                    passed=accuracy >= self.thresholds['min_accuracy'],
                    message=f"Model accuracy: {accuracy:.2%}",
                    details={
                        'accuracy': accuracy,
                        'correct_predictions': correct_predictions,
                        'total_evaluated': total_evaluated
                    },
                    execution_time_ms=(time.time() - start_time) * 1000
                )
                self.add_result(report, result)
                
        except Exception as e:
            result = ValidationResult(
                test_name="model_validation_error",
                category=ValidationCategory.MODEL_PERFORMANCE,
                level=ValidationLevel.CRITICAL,
                passed=False,
                message=f"Model validation failed: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
            self.add_result(report, result)
    
    def validate_risk_calculations(self, bet_opportunities: List[BetOpportunity],
                                 risk_framework: RiskFramework,
                                 report: ValidationReport) -> None:
        """Validate risk calculation accuracy"""
        start_time = time.time()
        
        try:
            # Test 1: Kelly criterion calculations
            kelly_errors = 0
            edge_errors = 0
            
            for bet in bet_opportunities:
                # Validate edge calculation
                expected_edge = (bet.probability * bet.odds) - 1
                if abs(bet.edge - expected_edge) > 0.001:  # 0.1% tolerance
                    edge_errors += 1
                    
                # Validate Kelly fraction
                if bet.odds > 1 and bet.probability > 0:
                    expected_kelly = bet.edge / (bet.odds - 1) if bet.odds > 1 else 0
                    # Kelly calculation should be reasonable
                    if expected_kelly < 0 or expected_kelly > self.thresholds['max_kelly_fraction']:
                        kelly_errors += 1
                        
            edge_accuracy = 1 - (edge_errors / len(bet_opportunities)) if bet_opportunities else 1
            kelly_accuracy = 1 - (kelly_errors / len(bet_opportunities)) if bet_opportunities else 1
            
            result = ValidationResult(
                test_name="kelly_criterion_calculations",
                category=ValidationCategory.RISK_CALCULATION,
                level=ValidationLevel.ERROR if edge_accuracy < 0.95 or kelly_accuracy < 0.95 else ValidationLevel.INFO,
                passed=edge_accuracy >= 0.95 and kelly_accuracy >= 0.95,
                message=f"Kelly calculations - Edge accuracy: {edge_accuracy:.2%}, Kelly accuracy: {kelly_accuracy:.2%}",
                details={
                    'edge_accuracy': edge_accuracy,
                    'kelly_accuracy': kelly_accuracy,
                    'edge_errors': edge_errors,
                    'kelly_errors': kelly_errors
                },
                execution_time_ms=(time.time() - start_time) * 1000
            )
            self.add_result(report, result)
            
            # Test 2: Portfolio risk limits
            total_exposure = risk_framework.performance_monitor.get_total_exposure()
            current_bankroll = risk_framework.performance_monitor.get_current_bankroll()
            
            if current_bankroll > 0:
                exposure_ratio = total_exposure / current_bankroll
                risk_limit_check = exposure_ratio <= self.thresholds['max_portfolio_risk']
                
                result = ValidationResult(
                    test_name="portfolio_risk_limits",
                    category=ValidationCategory.RISK_CALCULATION,
                    level=ValidationLevel.WARNING if not risk_limit_check else ValidationLevel.INFO,
                    passed=risk_limit_check,
                    message=f"Portfolio exposure ratio: {exposure_ratio:.2%}",
                    details={
                        'exposure_ratio': exposure_ratio,
                        'total_exposure': total_exposure,
                        'current_bankroll': current_bankroll,
                        'risk_limit': self.thresholds['max_portfolio_risk']
                    },
                    execution_time_ms=(time.time() - start_time) * 1000
                )
                self.add_result(report, result)
                
        except Exception as e:
            result = ValidationResult(
                test_name="risk_calculation_error",
                category=ValidationCategory.RISK_CALCULATION,
                level=ValidationLevel.CRITICAL,
                passed=False,
                message=f"Risk calculation validation failed: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
            self.add_result(report, result)
    
    def validate_syndicate_operations(self, syndicate_members: List[SyndicateMember],
                                    risk_framework: RiskFramework,
                                    report: ValidationReport) -> None:
        """Validate syndicate operation logic"""
        start_time = time.time()
        
        try:
            # Test 1: Member permission validation
            permission_violations = 0
            total_members = len(syndicate_members)
            
            for member in syndicate_members:
                # Check betting limits are reasonable
                if member.betting_limit <= 0 or member.daily_limit <= 0:
                    permission_violations += 1
                if member.betting_limit > member.daily_limit:
                    permission_violations += 1
                    
            permission_validity = 1 - (permission_violations / total_members) if total_members > 0 else 1
            
            result = ValidationResult(
                test_name="member_permission_validation",
                category=ValidationCategory.SYNDICATE_OPERATION,
                level=ValidationLevel.ERROR if permission_validity < 0.90 else ValidationLevel.INFO,
                passed=permission_validity >= 0.90,
                message=f"Member permission validity: {permission_validity:.2%}",
                details={
                    'validity_rate': permission_validity,
                    'total_members': total_members,
                    'violations': permission_violations
                },
                execution_time_ms=(time.time() - start_time) * 1000
            )
            self.add_result(report, result)
            
            # Test 2: Consensus mechanism
            # Simulate consensus requirement for large bet
            large_bet = BetOpportunity(
                bet_id="TEST_CONSENSUS",
                sport="football",
                event="Test Event",
                selection="Test Selection",
                odds=2.0,
                probability=0.6,
                confidence=0.9
            )
            
            try:
                decision = risk_framework.evaluate_betting_opportunity(
                    bet_opportunity=large_bet,
                    bookmaker="TestBook",
                    jurisdiction="US", 
                    proposer_id=syndicate_members[0].member_id if syndicate_members else "TEST",
                    participating_members=[m.member_id for m in syndicate_members[:3]]
                )
                
                consensus_working = decision is not None
                
                result = ValidationResult(
                    test_name="consensus_mechanism_test",
                    category=ValidationCategory.SYNDICATE_OPERATION,
                    level=ValidationLevel.ERROR if not consensus_working else ValidationLevel.INFO,
                    passed=consensus_working,
                    message=f"Consensus mechanism: {'Working' if consensus_working else 'Failed'}",
                    details={'decision_generated': consensus_working},
                    execution_time_ms=(time.time() - start_time) * 1000
                )
                self.add_result(report, result)
                
            except Exception as consensus_error:
                result = ValidationResult(
                    test_name="consensus_mechanism_test",
                    category=ValidationCategory.SYNDICATE_OPERATION,
                    level=ValidationLevel.ERROR,
                    passed=False,
                    message=f"Consensus mechanism failed: {str(consensus_error)}",
                    execution_time_ms=(time.time() - start_time) * 1000
                )
                self.add_result(report, result)
                
        except Exception as e:
            result = ValidationResult(
                test_name="syndicate_validation_error",
                category=ValidationCategory.SYNDICATE_OPERATION,
                level=ValidationLevel.CRITICAL,
                passed=False,
                message=f"Syndicate validation failed: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
            self.add_result(report, result)
    
    def validate_api_integration(self, api_endpoints: List[Dict[str, str]],
                               report: ValidationReport) -> None:
        """Validate API integration health"""
        start_time = time.time()
        
        try:
            successful_endpoints = 0
            latency_issues = 0
            
            for endpoint in api_endpoints:
                endpoint_start = time.time()
                
                try:
                    # Test API connectivity
                    response = requests.get(
                        endpoint['url'],
                        headers=endpoint.get('headers', {}),
                        timeout=self.thresholds['api_timeout_seconds']
                    )
                    
                    latency_ms = (time.time() - endpoint_start) * 1000
                    
                    if response.status_code == 200:
                        successful_endpoints += 1
                        
                    if latency_ms > self.thresholds['max_latency_ms']:
                        latency_issues += 1
                        
                except requests.RequestException:
                    pass  # Count as failed
                    
            connectivity_rate = successful_endpoints / len(api_endpoints) if api_endpoints else 0
            latency_performance = 1 - (latency_issues / len(api_endpoints)) if api_endpoints else 1
            
            result = ValidationResult(
                test_name="api_connectivity_test",
                category=ValidationCategory.API_INTEGRATION,
                level=ValidationLevel.WARNING if connectivity_rate < 0.80 else ValidationLevel.INFO,
                passed=connectivity_rate >= 0.80,
                message=f"API connectivity: {connectivity_rate:.2%}, Latency performance: {latency_performance:.2%}",
                details={
                    'connectivity_rate': connectivity_rate,
                    'latency_performance': latency_performance,
                    'successful_endpoints': successful_endpoints,
                    'total_endpoints': len(api_endpoints),
                    'latency_issues': latency_issues
                },
                execution_time_ms=(time.time() - start_time) * 1000
            )
            self.add_result(report, result)
            
        except Exception as e:
            result = ValidationResult(
                test_name="api_integration_error",
                category=ValidationCategory.API_INTEGRATION,
                level=ValidationLevel.CRITICAL,
                passed=False,
                message=f"API integration validation failed: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
            self.add_result(report, result)
    
    def run_comprehensive_validation(self, 
                                   odds_data: Optional[List[Dict[str, Any]]] = None,
                                   predictions: Optional[List[Dict[str, Any]]] = None,
                                   historical_results: Optional[List[Dict[str, Any]]] = None,
                                   bet_opportunities: Optional[List[BetOpportunity]] = None,
                                   risk_framework: Optional[RiskFramework] = None,
                                   syndicate_members: Optional[List[SyndicateMember]] = None,
                                   api_endpoints: Optional[List[Dict[str, str]]] = None) -> ValidationReport:
        """Run comprehensive validation across all components"""
        
        report = self.create_validation_report("comprehensive_validation")
        
        self.logger.info(f"Starting comprehensive validation: {report.report_id}")
        
        # Run all validation categories
        if odds_data:
            self.validate_odds_data(odds_data, report)
            
        if predictions and historical_results:
            self.validate_model_predictions(predictions, historical_results, report)
            
        if bet_opportunities and risk_framework:
            self.validate_risk_calculations(bet_opportunities, risk_framework, report)
            
        if syndicate_members and risk_framework:
            self.validate_syndicate_operations(syndicate_members, risk_framework, report)
            
        if api_endpoints:
            self.validate_api_integration(api_endpoints, report)
        
        # Finalize report
        self.finalize_report(report)
        
        self.logger.info(f"Validation completed: {report.success_rate:.2%} success rate")
        
        return report
    
    def generate_validation_html_report(self, report: ValidationReport) -> str:
        """Generate HTML validation report"""
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sports Betting Validation Report - {report.report_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 5px; }}
                .summary {{ background-color: #f9f9f9; padding: 15px; margin: 20px 0; border-radius: 5px; }}
                .test-result {{ margin: 10px 0; padding: 10px; border-radius: 3px; }}
                .passed {{ background-color: #d4edda; border-left: 5px solid #28a745; }}
                .failed {{ background-color: #f8d7da; border-left: 5px solid #dc3545; }}
                .warning {{ background-color: #fff3cd; border-left: 5px solid #ffc107; }}
                .critical {{ background-color: #f8d7da; border-left: 5px solid #dc3545; font-weight: bold; }}
                .details {{ margin-top: 10px; font-size: 0.9em; color: #666; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Sports Betting Validation Report</h1>
                <p><strong>Report ID:</strong> {report.report_id}</p>
                <p><strong>Generated:</strong> {report.start_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Duration:</strong> {report.duration_seconds:.2f} seconds</p>
            </div>
            
            <div class="summary">
                <h2>Summary</h2>
                <table>
                    <tr><td><strong>Total Tests:</strong></td><td>{report.total_tests}</td></tr>
                    <tr><td><strong>Passed:</strong></td><td>{report.passed_tests}</td></tr>
                    <tr><td><strong>Failed:</strong></td><td>{report.failed_tests}</td></tr>
                    <tr><td><strong>Success Rate:</strong></td><td>{report.success_rate:.2%}</td></tr>
                    <tr><td><strong>Warnings:</strong></td><td>{report.warnings}</td></tr>
                    <tr><td><strong>Errors:</strong></td><td>{report.errors}</td></tr>
                    <tr><td><strong>Critical Issues:</strong></td><td>{report.critical_issues}</td></tr>
                </table>
                
                <h3>Recommendations</h3>
                <ul>
        """
        
        for rec in report.summary.get('recommendations', []):
            html_template += f"<li>{rec}</li>"
            
        html_template += "</ul></div><h2>Test Results</h2>"
        
        # Add test results
        for result in report.results:
            css_class = "passed" if result.passed else "failed"
            if result.level == ValidationLevel.WARNING:
                css_class = "warning"
            elif result.level == ValidationLevel.CRITICAL:
                css_class = "critical"
                
            html_template += f"""
            <div class="test-result {css_class}">
                <h3>{result.test_name} ({result.category.value})</h3>
                <p><strong>Status:</strong> {'PASSED' if result.passed else 'FAILED'} 
                   ({result.level.value.upper()})</p>
                <p><strong>Message:</strong> {result.message}</p>
                <p><strong>Execution Time:</strong> {result.execution_time_ms:.2f}ms</p>
                
                <div class="details">
                    <strong>Details:</strong><br>
            """
            
            for key, value in result.details.items():
                html_template += f"{key}: {value}<br>"
                
            html_template += "</div></div>"
            
        html_template += "</body></html>"
        
        return html_template


def create_test_data_samples():
    """Create sample test data for validation"""
    
    # Sample odds data
    odds_data = [
        {
            'odds': 1.91,
            'bookmaker': 'Pinnacle',
            'sport': 'football',
            'event': 'Chiefs vs Lions',
            'selection': 'Chiefs -3.5',
            'timestamp': datetime.now().isoformat()
        },
        {
            'odds': 2.10,
            'bookmaker': 'Bet365',
            'sport': 'basketball',
            'event': 'Lakers vs Warriors',
            'selection': 'Lakers ML',
            'timestamp': datetime.now().isoformat()
        }
    ]
    
    # Sample predictions
    predictions = [
        {
            'probability': 0.55,
            'confidence': 0.85,
            'model_id': 'neural_v1',
            'timestamp': datetime.now().isoformat()
        },
        {
            'probability': 0.62,
            'confidence': 0.92,
            'model_id': 'neural_v1',
            'timestamp': datetime.now().isoformat()
        }
    ]
    
    # Sample historical results
    historical_results = [
        {'predicted_probability': 0.60, 'actual_outcome': 1},
        {'predicted_probability': 0.45, 'actual_outcome': 0},
        {'predicted_probability': 0.75, 'actual_outcome': 1},
        {'predicted_probability': 0.30, 'actual_outcome': 0},
        {'predicted_probability': 0.65, 'actual_outcome': 1},
    ]
    
    # Sample API endpoints
    api_endpoints = [
        {
            'url': 'https://api.the-odds-api.com/v4/sports',
            'headers': {'X-RapidAPI-Key': 'test-key'}
        }
    ]
    
    return odds_data, predictions, historical_results, api_endpoints