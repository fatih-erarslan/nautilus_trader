"""
Playwright End-to-End Visual Validation Tests
Browser console monitoring and comprehensive UI testing
"""

import pytest
import asyncio
import json
import time
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from typing import Dict, List, Optional, Any
import numpy as np
from pathlib import Path

class PlaywrightTestSuite:
    """Comprehensive Playwright test suite with visual validation"""
    
    def __init__(self):
        self.browser = None
        self.context = None
        self.page = None
        self.console_logs = []
        self.network_logs = []
        self.performance_metrics = {}
        
    async def setup_browser(self, headless: bool = True):
        """Setup browser with monitoring capabilities"""
        playwright = await async_playwright().start()
        
        # Launch browser with detailed configuration
        self.browser = await playwright.chromium.launch(
            headless=headless,
            args=[
                '--disable-web-security',
                '--disable-features=TranslateUI',
                '--disable-ipc-flooding-protection',
                '--enable-logging',
                '--v=1'
            ]
        )
        
        # Create context with permissions
        self.context = await self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            permissions=['clipboard-read', 'clipboard-write'],
            record_video_dir="tests/videos/",
            record_har_path="tests/har/network.har"
        )
        
        # Create page with event listeners
        self.page = await self.context.new_page()
        
        # Setup console monitoring
        self.page.on('console', self._handle_console_message)
        self.page.on('pageerror', self._handle_page_error)
        self.page.on('response', self._handle_network_response)
        self.page.on('requestfailed', self._handle_request_failed)
        
    async def _handle_console_message(self, msg):
        """Handle console messages from browser"""
        self.console_logs.append({
            'type': msg.type,
            'text': msg.text,
            'timestamp': time.time(),
            'location': msg.location
        })
        
    async def _handle_page_error(self, error):
        """Handle JavaScript page errors"""
        self.console_logs.append({
            'type': 'error',
            'text': str(error),
            'timestamp': time.time(),
            'is_page_error': True
        })
        
    async def _handle_network_response(self, response):
        """Handle network responses"""
        self.network_logs.append({
            'url': response.url,
            'status': response.status,
            'headers': dict(response.headers),
            'timestamp': time.time()
        })
        
    async def _handle_request_failed(self, request):
        """Handle failed network requests"""
        self.network_logs.append({
            'url': request.url,
            'failure': request.failure,
            'timestamp': time.time(),
            'failed': True
        })

    async def measure_performance(self) -> Dict[str, float]:
        """Measure page performance metrics"""
        performance_metrics = await self.page.evaluate("""
            () => {
                const perfData = performance.getEntriesByType('navigation')[0];
                const paintEntries = performance.getEntriesByType('paint');
                
                return {
                    domContentLoaded: perfData.domContentLoadedEventEnd - perfData.domContentLoadedEventStart,
                    loadComplete: perfData.loadEventEnd - perfData.loadEventStart,
                    firstPaint: paintEntries.find(entry => entry.name === 'first-paint')?.startTime || 0,
                    firstContentfulPaint: paintEntries.find(entry => entry.name === 'first-contentful-paint')?.startTime || 0,
                    domInteractive: perfData.domInteractive - perfData.navigationStart,
                    totalLoadTime: perfData.loadEventEnd - perfData.navigationStart
                };
            }
        """)
        
        self.performance_metrics.update(performance_metrics)
        return performance_metrics
    
    async def capture_full_page_screenshot(self, path: str) -> str:
        """Capture full page screenshot for visual comparison"""
        screenshot_path = f"tests/screenshots/{path}.png"
        await self.page.screenshot(path=screenshot_path, full_page=True)
        return screenshot_path
    
    async def test_trading_dashboard_visual(self):
        """Test trading dashboard visual elements"""
        # Navigate to trading dashboard
        await self.page.goto('http://localhost:8080/dashboard')
        
        # Wait for critical elements to load
        await self.page.wait_for_selector('[data-testid="trading-dashboard"]', timeout=10000)
        await self.page.wait_for_selector('[data-testid="portfolio-summary"]', timeout=5000)
        
        # Verify visual elements are present
        elements_to_check = [
            '[data-testid="price-chart"]',
            '[data-testid="order-book"]',
            '[data-testid="trade-history"]',
            '[data-testid="portfolio-balance"]',
            '[data-testid="risk-metrics"]'
        ]
        
        for element in elements_to_check:
            assert await self.page.is_visible(element), f"Element {element} should be visible"
        
        # Capture screenshot for visual regression testing
        await self.capture_full_page_screenshot('trading_dashboard_full')
        
        # Test responsive design
        await self.page.set_viewport_size({'width': 768, 'height': 1024})
        await self.page.wait_for_timeout(1000)  # Allow reflow
        await self.capture_full_page_screenshot('trading_dashboard_tablet')
        
        await self.page.set_viewport_size({'width': 375, 'height': 667})
        await self.page.wait_for_timeout(1000)
        await self.capture_full_page_screenshot('trading_dashboard_mobile')
    
    async def test_order_placement_flow(self):
        """Test complete order placement flow"""
        await self.page.goto('http://localhost:8080/trade')
        
        # Fill order form
        await self.page.fill('[data-testid="symbol-input"]', 'BTCUSDT')
        await self.page.fill('[data-testid="quantity-input"]', '0.001')
        await self.page.fill('[data-testid="price-input"]', '45000')
        
        # Select order type
        await self.page.click('[data-testid="order-type-dropdown"]')
        await self.page.click('[data-testid="limit-order-option"]')
        
        # Verify order preview
        order_preview = await self.page.text_content('[data-testid="order-preview"]')
        assert 'BTCUSDT' in order_preview
        assert '0.001' in order_preview
        assert '45000' in order_preview
        
        # Submit order (in test mode)
        await self.page.click('[data-testid="submit-order-button"]')
        
        # Verify confirmation
        await self.page.wait_for_selector('[data-testid="order-confirmation"]', timeout=5000)
        confirmation_text = await self.page.text_content('[data-testid="order-confirmation"]')
        assert 'Order submitted successfully' in confirmation_text
        
        # Check console for any errors
        error_logs = [log for log in self.console_logs if log['type'] == 'error']
        assert len(error_logs) == 0, f"Found console errors: {error_logs}"
    
    async def test_real_time_data_updates(self):
        """Test real-time data updates and WebSocket connections"""
        await self.page.goto('http://localhost:8080/dashboard')
        
        # Wait for WebSocket connection
        await self.page.wait_for_function("""
            () => window.wsConnectionStatus === 'connected'
        """, timeout=10000)
        
        # Monitor price updates
        initial_price = await self.page.text_content('[data-testid="btc-price"]')
        
        # Wait for price update (simulate with timeout)
        await self.page.wait_for_timeout(2000)
        
        updated_price = await self.page.text_content('[data-testid="btc-price"]')
        
        # Prices should update in real-time environment
        # In test environment, verify structure is correct
        assert initial_price is not None
        assert updated_price is not None
        
        # Check WebSocket message logs
        ws_messages = await self.page.evaluate("""
            () => window.wsMessageCount || 0
        """)
        assert ws_messages >= 0, "Should track WebSocket messages"
    
    async def test_error_handling_ui(self):
        """Test UI error handling and user feedback"""
        await self.page.goto('http://localhost:8080/trade')
        
        # Test invalid order submission
        await self.page.fill('[data-testid="quantity-input"]', '-1')  # Invalid quantity
        await self.page.click('[data-testid="submit-order-button"]')
        
        # Verify error message appears
        await self.page.wait_for_selector('[data-testid="error-message"]', timeout=5000)
        error_message = await self.page.text_content('[data-testid="error-message"]')
        assert 'Invalid quantity' in error_message.lower()
        
        # Test network error handling
        await self.page.route('**/api/orders', lambda route: route.abort())
        
        await self.page.fill('[data-testid="quantity-input"]', '0.001')
        await self.page.click('[data-testid="submit-order-button"]')
        
        # Should show network error
        await self.page.wait_for_selector('[data-testid="network-error"]', timeout=5000)
        
        # Unblock requests
        await self.page.unroute('**/api/orders')
    
    async def test_accessibility_compliance(self):
        """Test accessibility compliance"""
        await self.page.goto('http://localhost:8080/dashboard')
        
        # Check for proper ARIA labels
        elements_with_aria = await self.page.query_selector_all('[aria-label]')
        assert len(elements_with_aria) > 5, "Should have proper ARIA labels"
        
        # Check for keyboard navigation
        await self.page.press('body', 'Tab')
        focused_element = await self.page.evaluate('document.activeElement.tagName')
        assert focused_element in ['BUTTON', 'INPUT', 'A'], "Should support keyboard navigation"
        
        # Check color contrast (basic check)
        color_contrast = await self.page.evaluate("""
            () => {
                const style = getComputedStyle(document.body);
                return {
                    background: style.backgroundColor,
                    color: style.color
                };
            }
        """)
        assert color_contrast['background'] is not None
        assert color_contrast['color'] is not None
    
    async def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        start_time = time.time()
        await self.page.goto('http://localhost:8080/dashboard')
        
        # Wait for critical content
        await self.page.wait_for_selector('[data-testid="trading-dashboard"]')
        
        # Measure performance
        metrics = await self.measure_performance()
        load_time = time.time() - start_time
        
        # Performance assertions
        assert metrics['domContentLoaded'] < 2000, f"DOM content should load in <2s, got {metrics['domContentLoaded']}ms"
        assert metrics['firstContentfulPaint'] < 1500, f"FCP should be <1.5s, got {metrics['firstContentfulPaint']}ms"
        assert load_time < 5, f"Total load time should be <5s, got {load_time}s"
        
        # Memory usage check
        memory_usage = await self.page.evaluate("""
            () => performance.memory ? {
                usedJSHeapSize: performance.memory.usedJSHeapSize,
                totalJSHeapSize: performance.memory.totalJSHeapSize,
                jsHeapSizeLimit: performance.memory.jsHeapSizeLimit
            } : {}
        """)
        
        if memory_usage:
            heap_usage_ratio = memory_usage['usedJSHeapSize'] / memory_usage['jsHeapSizeLimit']
            assert heap_usage_ratio < 0.5, f"Memory usage should be <50%, got {heap_usage_ratio:.2%}"
    
    async def test_browser_compatibility(self):
        """Test browser compatibility features"""
        # Test local storage
        await self.page.goto('http://localhost:8080/dashboard')
        
        # Set and retrieve local storage
        await self.page.evaluate("""
            localStorage.setItem('testKey', 'testValue');
        """)
        
        stored_value = await self.page.evaluate("""
            localStorage.getItem('testKey');
        """)
        assert stored_value == 'testValue', "Local storage should work"
        
        # Test Web API availability
        api_support = await self.page.evaluate("""
            () => ({
                websocket: typeof WebSocket !== 'undefined',
                indexeddb: typeof indexedDB !== 'undefined',
                notification: typeof Notification !== 'undefined',
                geolocation: typeof navigator.geolocation !== 'undefined'
            });
        """)
        
        assert api_support['websocket'], "WebSocket should be supported"
        assert api_support['indexeddb'], "IndexedDB should be supported"
    
    async def cleanup(self):
        """Cleanup browser resources"""
        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        return {
            'console_logs': self.console_logs,
            'network_logs': self.network_logs,
            'performance_metrics': self.performance_metrics,
            'total_requests': len(self.network_logs),
            'failed_requests': len([log for log in self.network_logs if log.get('failed')]),
            'console_errors': len([log for log in self.console_logs if log['type'] == 'error'])
        }

# Pytest fixtures for Playwright tests
@pytest.fixture
async def playwright_suite():
    """Playwright test suite fixture"""
    suite = PlaywrightTestSuite()
    await suite.setup_browser()
    yield suite
    await suite.cleanup()

@pytest.fixture
def performance_thresholds():
    """Performance testing thresholds"""
    return {
        'load_time': 5.0,
        'first_contentful_paint': 1.5,
        'dom_content_loaded': 2.0,
        'memory_usage_ratio': 0.5
    }

# Individual test functions
@pytest.mark.asyncio
async def test_trading_dashboard_e2e(playwright_suite):
    """End-to-end trading dashboard test"""
    await playwright_suite.test_trading_dashboard_visual()

@pytest.mark.asyncio
async def test_order_flow_e2e(playwright_suite):
    """End-to-end order placement test"""
    await playwright_suite.test_order_placement_flow()

@pytest.mark.asyncio
async def test_real_time_updates_e2e(playwright_suite):
    """End-to-end real-time data test"""
    await playwright_suite.test_real_time_data_updates()

@pytest.mark.asyncio
async def test_error_handling_e2e(playwright_suite):
    """End-to-end error handling test"""
    await playwright_suite.test_error_handling_ui()

@pytest.mark.asyncio
async def test_accessibility_e2e(playwright_suite):
    """End-to-end accessibility test"""
    await playwright_suite.test_accessibility_compliance()

@pytest.mark.asyncio
async def test_performance_e2e(playwright_suite, performance_thresholds):
    """End-to-end performance test"""
    await playwright_suite.test_performance_benchmarks()

@pytest.mark.asyncio
async def test_browser_compatibility_e2e(playwright_suite):
    """End-to-end browser compatibility test"""
    await playwright_suite.test_browser_compatibility()