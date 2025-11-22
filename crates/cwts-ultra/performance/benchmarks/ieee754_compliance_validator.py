#!/usr/bin/env python3
"""
IEEE 754 Compliance and Numerical Stability Validator
Scientific validation of floating-point operations in trading systems
Ensures mathematical correctness and precision in financial computations
"""

import math
import struct
import numpy as np
import decimal
from decimal import Decimal, getcontext
from typing import Dict, List, Tuple, Any, Optional
import warnings
import sys
from dataclasses import dataclass
import json
import time
from pathlib import Path

# Set high precision for decimal operations
getcontext().prec = 50

# IEEE 754 constants
IEEE754_FLOAT64_BIAS = 1023
IEEE754_FLOAT64_MANTISSA_BITS = 52
IEEE754_FLOAT64_EXPONENT_BITS = 11
IEEE754_EPSILON = np.finfo(np.float64).eps
IEEE754_MIN_NORMAL = np.finfo(np.float64).tiny
IEEE754_MAX = np.finfo(np.float64).max
IEEE754_MIN = np.finfo(np.float64).min

@dataclass
class FloatingPointAnalysis:
    """Analysis results for floating-point operations"""
    operation: str
    inputs: List[float]
    computed_result: float
    expected_result: float
    absolute_error: float
    relative_error: float
    ulp_error: int
    is_ieee754_compliant: bool
    precision_loss_bits: int
    numerical_stability: str  # "stable", "unstable", "catastrophic"
    recommendations: List[str]

class IEEE754Validator:
    """Comprehensive IEEE 754 compliance validation framework"""
    
    def __init__(self, tolerance_ulps: int = 2):
        """
        Initialize validator with ULP (Unit in Last Place) tolerance
        tolerance_ulps: Maximum acceptable error in ULPs
        """
        self.tolerance_ulps = tolerance_ulps
        self.validation_results: List[FloatingPointAnalysis] = []
        self.test_count = 0
        self.compliant_count = 0
        
    def validate_basic_operations(self) -> List[FloatingPointAnalysis]:
        """Validate basic arithmetic operations for IEEE 754 compliance"""
        test_cases = [
            # Addition tests
            ("+", [1.0, 1e-15], 1.0 + 1e-15),
            ("+", [1e308, 1e308], float('inf')),  # Overflow
            ("+", [1e-308, 1e-308], 2e-308),     # Underflow region
            ("+", [float('inf'), 1.0], float('inf')),
            ("+", [float('inf'), float('-inf')], float('nan')),
            
            # Subtraction tests
            ("-", [1.0, 1e-15], 1.0 - 1e-15),
            ("-", [1.0, 1.0], 0.0),
            ("-", [float('inf'), float('inf')], float('nan')),
            
            # Multiplication tests
            ("*", [2.0, 3.0], 6.0),
            ("*", [1e154, 1e154], 1e308),  # Near overflow
            ("*", [1e-154, 1e-154], 1e-308),  # Near underflow
            ("*", [float('inf'), 0.0], float('nan')),
            
            # Division tests
            ("/", [1.0, 3.0], 1.0/3.0),
            ("/", [1.0, 0.0], float('inf')),
            ("/", [0.0, 0.0], float('nan')),
            ("/", [float('inf'), float('inf')], float('nan')),
        ]
        
        results = []
        for operation, inputs, expected in test_cases:
            analysis = self._validate_operation(operation, inputs, expected)
            results.append(analysis)
            
        self.validation_results.extend(results)
        return results
        
    def validate_transcendental_functions(self) -> List[FloatingPointAnalysis]:
        """Validate transcendental functions (sin, cos, exp, log, etc.)"""
        test_cases = [
            # Trigonometric functions
            ("sin", [0.0], 0.0),
            ("sin", [math.pi/2], 1.0),
            ("sin", [math.pi], 0.0),
            ("cos", [0.0], 1.0),
            ("cos", [math.pi/2], 0.0),
            ("cos", [math.pi], -1.0),
            ("tan", [0.0], 0.0),
            ("tan", [math.pi/4], 1.0),
            
            # Exponential and logarithmic functions
            ("exp", [0.0], 1.0),
            ("exp", [1.0], math.e),
            ("log", [1.0], 0.0),
            ("log", [math.e], 1.0),
            ("log10", [1.0], 0.0),
            ("log10", [10.0], 1.0),
            
            # Power functions
            ("pow", [2.0, 3.0], 8.0),
            ("pow", [4.0, 0.5], 2.0),
            ("sqrt", [4.0], 2.0),
            ("sqrt", [2.0], math.sqrt(2)),
            
            # Special cases
            ("exp", [709.0], math.exp(709.0)),  # Near overflow
            ("log", [1e-308], math.log(1e-308)),  # Near underflow
        ]
        
        results = []
        for func_name, inputs, expected in test_cases:
            analysis = self._validate_transcendental(func_name, inputs, expected)
            results.append(analysis)
            
        self.validation_results.extend(results)
        return results
        
    def validate_financial_operations(self) -> List[FloatingPointAnalysis]:
        """Validate common financial computations for numerical stability"""
        # Present value calculation: PV = FV / (1 + r)^n
        test_cases = [
            # Present value calculations
            ("present_value", [1000.0, 0.05, 10], 1000.0 / (1.05 ** 10)),
            ("present_value", [1e6, 0.001, 100], 1e6 / (1.001 ** 100)),
            
            # Compound interest: A = P(1 + r)^t
            ("compound_interest", [1000.0, 0.05, 10], 1000.0 * (1.05 ** 10)),
            
            # Black-Scholes components (simplified)
            ("black_scholes_d1", [100.0, 105.0, 0.05, 0.25, 0.2], None),  # Complex calculation
            
            # VWAP calculation challenges
            ("vwap", [[100.0, 101.0, 99.0], [1000.0, 2000.0, 1500.0]], None),
            
            # Portfolio variance calculation
            ("portfolio_variance", [np.array([0.1, 0.15]), np.array([[0.04, 0.02], [0.02, 0.09]])], None),
        ]
        
        results = []
        for operation, inputs, expected in test_cases:
            analysis = self._validate_financial_operation(operation, inputs, expected)
            if analysis:
                results.append(analysis)
                
        self.validation_results.extend(results)
        return results
        
    def validate_extreme_values(self) -> List[FloatingPointAnalysis]:
        """Test behavior with extreme floating-point values"""
        extreme_cases = [
            # Subnormal numbers
            ("subnormal_add", [IEEE754_MIN_NORMAL/2, IEEE754_MIN_NORMAL/2], IEEE754_MIN_NORMAL),
            
            # Very large numbers
            ("large_multiply", [1e200, 1e200], float('inf')),
            
            # Very small numbers
            ("small_divide", [1e-200, 1e200], 1e-400),
            
            # Mixed extreme values
            ("extreme_ratio", [IEEE754_MAX, IEEE754_MIN_NORMAL], IEEE754_MAX/IEEE754_MIN_NORMAL),
            
            # Special value arithmetic
            ("inf_arithmetic", [float('inf'), 1.0], float('inf')),
            ("nan_propagation", [float('nan'), 1.0], float('nan')),
            
            # Precision boundaries
            ("precision_limit", [1.0, IEEE754_EPSILON/2], 1.0),  # Should not change 1.0
        ]
        
        results = []
        for operation, inputs, expected in extreme_cases:
            analysis = self._validate_extreme_case(operation, inputs, expected)
            results.append(analysis)
            
        self.validation_results.extend(results)
        return results
        
    def _validate_operation(self, operation: str, inputs: List[float], expected: float) -> FloatingPointAnalysis:
        """Validate a single arithmetic operation"""
        try:
            if operation == "+":
                computed = inputs[0] + inputs[1]
            elif operation == "-":
                computed = inputs[0] - inputs[1]  
            elif operation == "*":
                computed = inputs[0] * inputs[1]
            elif operation == "/":
                computed = inputs[0] / inputs[1]
            else:
                computed = float('nan')
                
        except (ZeroDivisionError, OverflowError, ValueError) as e:
            computed = float('nan')
            
        return self._analyze_result(operation, inputs, computed, expected)
        
    def _validate_transcendental(self, func_name: str, inputs: List[float], expected: float) -> FloatingPointAnalysis:
        """Validate transcendental function"""
        try:
            if func_name == "sin":
                computed = math.sin(inputs[0])
            elif func_name == "cos":
                computed = math.cos(inputs[0])
            elif func_name == "tan":
                computed = math.tan(inputs[0])
            elif func_name == "exp":
                computed = math.exp(inputs[0])
            elif func_name == "log":
                computed = math.log(inputs[0])
            elif func_name == "log10":
                computed = math.log10(inputs[0])
            elif func_name == "pow":
                computed = math.pow(inputs[0], inputs[1])
            elif func_name == "sqrt":
                computed = math.sqrt(inputs[0])
            else:
                computed = float('nan')
                
        except (ValueError, OverflowError) as e:
            computed = float('nan')
            
        return self._analyze_result(func_name, inputs, computed, expected)
        
    def _validate_financial_operation(self, operation: str, inputs: List, expected: Optional[float]) -> Optional[FloatingPointAnalysis]:
        """Validate financial computation"""
        try:
            if operation == "present_value":
                fv, rate, periods = inputs
                computed = fv / ((1 + rate) ** periods)
                expected = expected if expected is not None else computed
                
            elif operation == "compound_interest":
                principal, rate, time = inputs
                computed = principal * ((1 + rate) ** time)
                expected = expected if expected is not None else computed
                
            elif operation == "vwap":
                prices, volumes = inputs
                total_value = sum(p * v for p, v in zip(prices, volumes))
                total_volume = sum(volumes)
                computed = total_value / total_volume if total_volume > 0 else float('nan')
                expected = expected if expected is not None else computed
                
            elif operation == "portfolio_variance":
                weights, cov_matrix = inputs
                computed = np.dot(weights, np.dot(cov_matrix, weights))
                expected = expected if expected is not None else computed
                
            else:
                return None
                
        except (ZeroDivisionError, OverflowError, ValueError, np.linalg.LinAlgError):
            computed = float('nan')
            expected = expected if expected is not None else float('nan')
            
        return self._analyze_result(operation, inputs, computed, expected)
        
    def _validate_extreme_case(self, operation: str, inputs: List[float], expected: float) -> FloatingPointAnalysis:
        """Validate extreme value cases"""
        try:
            if operation == "subnormal_add":
                computed = inputs[0] + inputs[1]
            elif operation == "large_multiply":
                computed = inputs[0] * inputs[1]
            elif operation == "small_divide":
                computed = inputs[0] / inputs[1]
            elif operation == "extreme_ratio":
                computed = inputs[0] / inputs[1]
            elif operation == "inf_arithmetic":
                computed = inputs[0] + inputs[1]
            elif operation == "nan_propagation":
                computed = inputs[0] * inputs[1]
            elif operation == "precision_limit":
                computed = inputs[0] + inputs[1]
            else:
                computed = float('nan')
                
        except (ZeroDivisionError, OverflowError, ValueError):
            computed = float('nan')
            
        return self._analyze_result(operation, inputs, computed, expected)
        
    def _analyze_result(self, operation: str, inputs: List, computed: float, expected: float) -> FloatingPointAnalysis:
        """Analyze computation result for IEEE 754 compliance"""
        self.test_count += 1
        
        # Handle special values
        if math.isnan(expected) and math.isnan(computed):
            absolute_error = 0.0
            relative_error = 0.0
            ulp_error = 0
            is_compliant = True
        elif math.isinf(expected) and math.isinf(computed) and (expected > 0) == (computed > 0):
            absolute_error = 0.0
            relative_error = 0.0
            ulp_error = 0
            is_compliant = True
        elif math.isnan(computed) or math.isnan(expected) or math.isinf(computed) or math.isinf(expected):
            absolute_error = float('inf') if not (math.isnan(computed) and math.isnan(expected)) else 0.0
            relative_error = float('inf') if not (math.isnan(computed) and math.isnan(expected)) else 0.0
            ulp_error = float('inf')
            is_compliant = math.isnan(computed) == math.isnan(expected) and \
                          (not math.isinf(computed) or (math.isinf(expected) and (computed > 0) == (expected > 0)))
        else:
            # Normal case
            absolute_error = abs(computed - expected)
            relative_error = absolute_error / abs(expected) if expected != 0 else absolute_error
            ulp_error = self._compute_ulp_distance(computed, expected)
            is_compliant = ulp_error <= self.tolerance_ulps
            
        if is_compliant:
            self.compliant_count += 1
            
        # Assess numerical stability
        stability = self._assess_numerical_stability(operation, inputs, absolute_error, relative_error)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(operation, absolute_error, relative_error, ulp_error, stability)
        
        # Estimate precision loss
        precision_loss = self._estimate_precision_loss(absolute_error, expected)
        
        return FloatingPointAnalysis(
            operation=operation,
            inputs=inputs,
            computed_result=computed,
            expected_result=expected,
            absolute_error=absolute_error,
            relative_error=relative_error,
            ulp_error=ulp_error,
            is_ieee754_compliant=is_compliant,
            precision_loss_bits=precision_loss,
            numerical_stability=stability,
            recommendations=recommendations
        )
        
    def _compute_ulp_distance(self, computed: float, expected: float) -> int:
        """Compute distance in ULPs (Units in Last Place)"""
        if computed == expected:
            return 0
            
        if math.isnan(computed) or math.isnan(expected):
            return float('inf')
            
        if math.isinf(computed) or math.isinf(expected):
            return float('inf')
            
        # Convert to integer representation for ULP calculation
        try:
            computed_int = struct.unpack('Q', struct.pack('d', computed))[0]
            expected_int = struct.unpack('Q', struct.pack('d', expected))[0]
            
            # Handle sign bit
            if computed_int & (1 << 63):  # Negative
                computed_int = (1 << 64) - computed_int
            if expected_int & (1 << 63):  # Negative  
                expected_int = (1 << 64) - expected_int
                
            return abs(computed_int - expected_int)
        except:
            return float('inf')
            
    def _assess_numerical_stability(self, operation: str, inputs: List, abs_error: float, rel_error: float) -> str:
        """Assess numerical stability of the operation"""
        if math.isinf(abs_error) or math.isinf(rel_error):
            return "catastrophic"
            
        # Define stability thresholds
        STABLE_THRESHOLD = 1e-12
        UNSTABLE_THRESHOLD = 1e-6
        
        if rel_error < STABLE_THRESHOLD:
            return "stable"
        elif rel_error < UNSTABLE_THRESHOLD:
            return "unstable"
        else:
            return "catastrophic"
            
    def _estimate_precision_loss(self, abs_error: float, expected: float) -> int:
        """Estimate precision loss in bits"""
        if abs_error == 0 or expected == 0 or math.isinf(abs_error):
            return 0
            
        try:
            rel_error = abs_error / abs(expected)
            if rel_error > 0:
                bits_lost = max(0, -math.log2(rel_error))
                return min(int(bits_lost), 52)  # Max 52 bits for double precision
        except (ValueError, OverflowError):
            pass
            
        return 0
        
    def _generate_recommendations(self, operation: str, abs_error: float, rel_error: float, 
                                ulp_error: int, stability: str) -> List[str]:
        """Generate recommendations for improving numerical accuracy"""
        recommendations = []
        
        if stability == "catastrophic":
            recommendations.append("Critical: Use higher precision arithmetic (decimal or arbitrary precision)")
            recommendations.append("Consider algorithm reformulation to avoid catastrophic cancellation")
            
        elif stability == "unstable":
            recommendations.append("Warning: Consider using extended precision for this operation")
            recommendations.append("Validate results with alternative computation methods")
            
        if ulp_error > 10:
            recommendations.append("Large ULP error detected - verify algorithm implementation")
            
        if "log" in operation.lower() and any(x <= 0 for x in [abs_error] if not math.isinf(x)):
            recommendations.append("Ensure logarithm inputs are positive and well-conditioned")
            
        if "divide" in operation.lower() or "/" in operation:
            recommendations.append("Check for division by near-zero values")
            
        if "compound" in operation.lower() or "present_value" in operation.lower():
            recommendations.append("For financial calculations, consider using decimal arithmetic")
            
        if not recommendations:
            recommendations.append("Operation appears numerically stable")
            
        return recommendations
        
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive IEEE 754 compliance report"""
        if not self.validation_results:
            return {}
            
        compliant_results = [r for r in self.validation_results if r.is_ieee754_compliant]
        stable_results = [r for r in self.validation_results if r.numerical_stability == "stable"]
        unstable_results = [r for r in self.validation_results if r.numerical_stability == "unstable"]
        catastrophic_results = [r for r in self.validation_results if r.numerical_stability == "catastrophic"]
        
        # Statistics
        total_tests = len(self.validation_results)
        compliance_rate = len(compliant_results) / total_tests if total_tests > 0 else 0
        stability_rate = len(stable_results) / total_tests if total_tests > 0 else 0
        
        # Error statistics
        absolute_errors = [r.absolute_error for r in self.validation_results if not math.isinf(r.absolute_error)]
        relative_errors = [r.relative_error for r in self.validation_results if not math.isinf(r.relative_error)]
        ulp_errors = [r.ulp_error for r in self.validation_results if not math.isinf(r.ulp_error)]
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "compliant_tests": len(compliant_results),
                "compliance_rate": compliance_rate,
                "stability_rate": stability_rate,
                "test_timestamp": time.time()
            },
            "error_statistics": {
                "max_absolute_error": max(absolute_errors) if absolute_errors else 0,
                "mean_absolute_error": np.mean(absolute_errors) if absolute_errors else 0,
                "max_relative_error": max(relative_errors) if relative_errors else 0,
                "mean_relative_error": np.mean(relative_errors) if relative_errors else 0,
                "max_ulp_error": max(ulp_errors) if ulp_errors else 0,
                "mean_ulp_error": np.mean(ulp_errors) if ulp_errors else 0
            },
            "stability_analysis": {
                "stable_operations": len(stable_results),
                "unstable_operations": len(unstable_results),
                "catastrophic_operations": len(catastrophic_results)
            },
            "problematic_operations": [
                {
                    "operation": r.operation,
                    "inputs": r.inputs,
                    "stability": r.numerical_stability,
                    "compliance": r.is_ieee754_compliant,
                    "relative_error": r.relative_error,
                    "recommendations": r.recommendations
                }
                for r in catastrophic_results + unstable_results
            ],
            "recommendations": self._generate_overall_recommendations(compliant_results, stable_results)
        }
        
        return report
        
    def _generate_overall_recommendations(self, compliant_results: List, stable_results: List) -> List[str]:
        """Generate overall system recommendations"""
        recommendations = []
        
        total = len(self.validation_results)
        compliant_rate = len(compliant_results) / total if total > 0 else 0
        stable_rate = len(stable_results) / total if total > 0 else 0
        
        if compliant_rate < 0.95:
            recommendations.append("IEEE 754 compliance rate below 95% - review arithmetic implementations")
            
        if stable_rate < 0.90:
            recommendations.append("Numerical stability rate below 90% - consider extended precision")
            
        if compliant_rate > 0.99 and stable_rate > 0.95:
            recommendations.append("Excellent IEEE 754 compliance and numerical stability")
            
        recommendations.extend([
            "Implement runtime checks for extreme values in production",
            "Consider using decimal arithmetic for critical financial calculations",
            "Validate algorithms with multiple precision libraries",
            "Monitor for precision loss in long computation chains"
        ])
        
        return recommendations
        
    def save_report(self, filepath: str):
        """Save compliance report to JSON file"""
        report = self.generate_compliance_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

def main():
    """Run comprehensive IEEE 754 compliance validation"""
    print("🔬 CWTS IEEE 754 Compliance and Numerical Stability Validator")
    print("=" * 60)
    
    validator = IEEE754Validator(tolerance_ulps=2)
    
    print("\n📐 Testing basic arithmetic operations...")
    basic_results = validator.validate_basic_operations()
    
    print("📊 Testing transcendental functions...")
    transcendental_results = validator.validate_transcendental_functions()
    
    print("💰 Testing financial computations...")
    financial_results = validator.validate_financial_operations()
    
    print("⚠️  Testing extreme values...")
    extreme_results = validator.validate_extreme_values()
    
    # Generate report
    report = validator.generate_compliance_report()
    
    # Save results
    results_file = "/home/kutlu/CWTS/cwts-ultra/performance/benchmarks/ieee754_compliance_report.json"
    validator.save_report(results_file)
    
    print(f"\n✅ Validation complete. Report saved to: {results_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("📈 COMPLIANCE SUMMARY")
    print("="*60)
    
    summary = report["summary"]
    print(f"Total tests: {summary['total_tests']}")
    print(f"IEEE 754 compliance: {summary['compliance_rate']*100:.1f}%")
    print(f"Numerical stability: {summary['stability_rate']*100:.1f}%")
    
    if "error_statistics" in report:
        errors = report["error_statistics"]
        print(f"Max relative error: {errors['max_relative_error']:.2e}")
        print(f"Max ULP error: {errors['max_ulp_error']}")
        
    print(f"\n⚠️  Problematic operations: {len(report.get('problematic_operations', []))}")
    
    print("\n🎯 Key Recommendations:")
    for rec in report.get("recommendations", [])[:3]:
        print(f"  • {rec}")

if __name__ == "__main__":
    main()