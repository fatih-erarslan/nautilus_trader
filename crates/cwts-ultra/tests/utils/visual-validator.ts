import { chromium, firefox, webkit, Browser, Page, BrowserContext } from 'playwright';
import { PNG } from 'pngjs';
import * as pixelmatch from 'pixelmatch';
import * as fs from 'fs/promises';
import * as path from 'path';

/**
 * Visual Validator - Playwright-based visual validation with browser monitoring
 * Implements comprehensive visual regression testing and screenshot analysis
 */
export class VisualValidator {
  private browsers: Map<string, Browser> = new Map();
  private contexts: Map<string, BrowserContext> = new Map();
  private readonly screenshotDir: string;
  private readonly baselineDir: string;
  private readonly diffDir: string;
  
  constructor(rootDir: string = process.cwd()) {
    this.screenshotDir = path.join(rootDir, 'tests', 'visual', 'screenshots');
    this.baselineDir = path.join(rootDir, 'tests', 'visual', 'baselines');
    this.diffDir = path.join(rootDir, 'tests', 'visual', 'diffs');
  }

  async initialize(): Promise<void> {
    console.log('🎭 Initializing Visual Validator with Playwright...');
    
    // Create directories
    await Promise.all([
      fs.mkdir(this.screenshotDir, { recursive: true }),
      fs.mkdir(this.baselineDir, { recursive: true }),
      fs.mkdir(this.diffDir, { recursive: true })
    ]);

    // Launch all browsers
    const browserConfigs = [
      { name: 'chromium', launcher: chromium },
      { name: 'firefox', launcher: firefox },
      { name: 'webkit', launcher: webkit }
    ];

    for (const config of browserConfigs) {
      try {
        const browser = await config.launcher.launch({
          headless: true,
          args: [
            '--disable-web-security',
            '--disable-features=VizDisplayCompositor',
            '--disable-backgrounding-occluded-windows',
            '--disable-renderer-backgrounding',
            '--disable-field-trial-config',
            '--force-color-profile=srgb',
            '--disable-background-timer-throttling',
            '--disable-backgrounding-occluded-windows',
            '--disable-renderer-backgrounding'
          ]
        });
        
        this.browsers.set(config.name, browser);
        
        // Create browser context with monitoring
        const context = await browser.newContext({
          viewport: { width: 1280, height: 720 },
          deviceScaleFactor: 1,
          hasTouch: false,
          isMobile: false,
          locale: 'en-US',
          timezoneId: 'America/New_York',
          colorScheme: 'light',
          reducedMotion: 'reduce'
        });
        
        this.contexts.set(config.name, context);
        console.log(`✅ ${config.name} browser initialized`);
        
      } catch (error) {
        console.error(`❌ Failed to initialize ${config.name}:`, error);
      }
    }
    
    console.log('✅ Visual Validator initialized successfully');
  }

  async cleanup(): Promise<void> {
    console.log('🧹 Cleaning up Visual Validator...');
    
    // Close all contexts and browsers
    for (const [name, context] of this.contexts) {
      await context.close();
      console.log(`Closed ${name} context`);
    }
    
    for (const [name, browser] of this.browsers) {
      await browser.close();
      console.log(`Closed ${name} browser`);
    }
    
    this.contexts.clear();
    this.browsers.clear();
    
    console.log('✅ Visual Validator cleanup complete');
  }

  async validateAllBrowsers(): Promise<{
    chromium: BrowserValidationResult;
    firefox: BrowserValidationResult;
    webkit: BrowserValidationResult;
  }> {
    console.log('🌐 Validating UI components across all browsers...');

    const testUrls = [
      'http://localhost:3000',
      'http://localhost:3000/trading',
      'http://localhost:3000/dashboard',
      'http://localhost:3000/analytics'
    ];

    const results = await Promise.all([
      this.validateBrowser('chromium', testUrls),
      this.validateBrowser('firefox', testUrls),
      this.validateBrowser('webkit', testUrls)
    ]);

    return {
      chromium: results[0],
      firefox: results[1],
      webkit: results[2]
    };
  }

  async monitorConsoleErrors(): Promise<{
    errors: ConsoleMessage[];
    warnings: ConsoleMessage[];
    performance: PerformanceMetrics[];
  }> {
    console.log('📊 Monitoring browser console for errors...');

    const allErrors: ConsoleMessage[] = [];
    const allWarnings: ConsoleMessage[] = [];
    const allPerformanceMetrics: PerformanceMetrics[] = [];

    for (const [browserName, context] of this.contexts) {
      const page = await context.newPage();
      
      const errors: ConsoleMessage[] = [];
      const warnings: ConsoleMessage[] = [];
      
      // Monitor console messages
      page.on('console', (message) => {
        const consoleMessage: ConsoleMessage = {
          browser: browserName,
          type: message.type(),
          text: message.text(),
          location: message.location(),
          timestamp: new Date()
        };
        
        if (message.type() === 'error') {
          errors.push(consoleMessage);
        } else if (message.type() === 'warning') {
          warnings.push(consoleMessage);
        }
      });

      // Monitor page errors
      page.on('pageerror', (error) => {
        errors.push({
          browser: browserName,
          type: 'error',
          text: error.message,
          location: { url: page.url(), lineNumber: 0, columnNumber: 0 },
          timestamp: new Date()
        });
      });

      // Navigate to test pages and collect metrics
      try {
        await page.goto('http://localhost:3000');
        await page.waitForLoadState('networkidle');
        
        // Collect performance metrics
        const performanceMetrics = await this.collectPerformanceMetrics(page, browserName);
        allPerformanceMetrics.push(performanceMetrics);
        
      } catch (error) {
        errors.push({
          browser: browserName,
          type: 'error',
          text: `Navigation error: ${error.message}`,
          location: { url: 'http://localhost:3000', lineNumber: 0, columnNumber: 0 },
          timestamp: new Date()
        });
      }
      
      allErrors.push(...errors);
      allWarnings.push(...warnings);
      
      await page.close();
    }

    return {
      errors: allErrors,
      warnings: allWarnings,
      performance: allPerformanceMetrics
    };
  }

  async performRegressionTesting(): Promise<{
    regressionDetected: boolean;
    pixelDifferenceThreshold: number;
    comparisonResults: ScreenshotComparisonResult[];
  }> {
    console.log('📸 Performing screenshot-based regression testing...');

    const testScenarios = [
      { name: 'homepage', url: 'http://localhost:3000', selector: 'body' },
      { name: 'trading-view', url: 'http://localhost:3000/trading', selector: '.trading-interface' },
      { name: 'dashboard', url: 'http://localhost:3000/dashboard', selector: '.dashboard' },
      { name: 'modal-dialog', url: 'http://localhost:3000', selector: '.modal', action: 'openModal' }
    ];

    const comparisonResults: ScreenshotComparisonResult[] = [];
    let totalPixelDifference = 0;
    let maxPixelDifference = 0;

    for (const browserName of this.browsers.keys()) {
      const context = this.contexts.get(browserName);
      if (!context) continue;

      const page = await context.newPage();

      for (const scenario of testScenarios) {
        try {
          await page.goto(scenario.url);
          await page.waitForLoadState('networkidle');

          // Perform action if specified
          if (scenario.action === 'openModal') {
            await page.click('[data-testid="open-modal"]');
            await page.waitForSelector('.modal', { state: 'visible' });
          }

          // Take screenshot
          const screenshotPath = path.join(this.screenshotDir, `${browserName}-${scenario.name}.png`);
          await page.screenshot({
            path: screenshotPath,
            fullPage: true,
            animations: 'disabled'
          });

          // Compare with baseline
          const baselinePath = path.join(this.baselineDir, `${browserName}-${scenario.name}.png`);
          const comparisonResult = await this.compareScreenshots(screenshotPath, baselinePath, browserName, scenario.name);
          
          comparisonResults.push(comparisonResult);
          totalPixelDifference += comparisonResult.pixelDifference;
          maxPixelDifference = Math.max(maxPixelDifference, comparisonResult.pixelDifference);

        } catch (error) {
          console.error(`❌ Screenshot comparison failed for ${browserName}-${scenario.name}:`, error);
          comparisonResults.push({
            browser: browserName,
            scenario: scenario.name,
            pixelDifference: 1.0, // Mark as failed
            passed: false,
            diffImagePath: '',
            error: error.message
          });
        }
      }

      await page.close();
    }

    const avgPixelDifference = comparisonResults.length > 0 ? totalPixelDifference / comparisonResults.length : 0;
    const regressionThreshold = 0.01; // 1% pixel difference threshold

    return {
      regressionDetected: maxPixelDifference > regressionThreshold,
      pixelDifferenceThreshold: avgPixelDifference,
      comparisonResults
    };
  }

  async validateResponsiveDesign(): Promise<{
    desktop: ResponsiveTestResult;
    tablet: ResponsiveTestResult;
    mobile: ResponsiveTestResult;
  }> {
    console.log('📱 Validating responsive design across viewports...');

    const viewports = [
      { name: 'desktop', width: 1920, height: 1080 },
      { name: 'tablet', width: 768, height: 1024 },
      { name: 'mobile', width: 375, height: 667 }
    ];

    const results: { [key: string]: ResponsiveTestResult } = {};

    for (const viewport of viewports) {
      const context = this.contexts.get('chromium'); // Use Chromium for responsive tests
      if (!context) continue;

      const page = await context.newPage();
      await page.setViewportSize({ width: viewport.width, height: viewport.height });

      try {
        await page.goto('http://localhost:3000');
        await page.waitForLoadState('networkidle');

        // Test layout elements
        const layoutTests = await Promise.all([
          this.testElementVisibility(page, '[data-testid="navigation"]'),
          this.testElementVisibility(page, '[data-testid="main-content"]'),
          this.testElementVisibility(page, '[data-testid="footer"]'),
          this.testResponsiveBreakpoints(page, viewport.width),
          this.testTextReadability(page),
          this.testTouchTargetSize(page, viewport.name === 'mobile')
        ]);

        results[viewport.name] = {
          viewportSize: viewport,
          layoutValid: layoutTests.every(test => test.passed),
          elementTests: layoutTests,
          screenshotPath: path.join(this.screenshotDir, `responsive-${viewport.name}.png`)
        };

        // Take screenshot for documentation
        await page.screenshot({
          path: results[viewport.name].screenshotPath,
          fullPage: true
        });

      } catch (error) {
        results[viewport.name] = {
          viewportSize: viewport,
          layoutValid: false,
          elementTests: [],
          screenshotPath: '',
          error: error.message
        };
      }

      await page.close();
    }

    return {
      desktop: results.desktop,
      tablet: results.tablet,
      mobile: results.mobile
    };
  }

  private async validateBrowser(browserName: string, urls: string[]): Promise<BrowserValidationResult> {
    const context = this.contexts.get(browserName);
    if (!context) {
      return {
        passed: false,
        errors: [`Browser ${browserName} not initialized`],
        screenshots: [],
        performanceMetrics: null
      };
    }

    const page = await context.newPage();
    const errors: string[] = [];
    const screenshots: string[] = [];

    // Monitor console errors
    page.on('console', (message) => {
      if (message.type() === 'error') {
        errors.push(`Console error: ${message.text()}`);
      }
    });

    page.on('pageerror', (error) => {
      errors.push(`Page error: ${error.message}`);
    });

    // Test each URL
    for (const url of urls) {
      try {
        await page.goto(url, { waitUntil: 'networkidle' });
        
        // Take screenshot
        const screenshotPath = path.join(this.screenshotDir, `${browserName}-${url.replace(/[^a-zA-Z0-9]/g, '_')}.png`);
        await page.screenshot({ path: screenshotPath, fullPage: true });
        screenshots.push(screenshotPath);

        // Run basic accessibility checks
        await this.runAccessibilityChecks(page, errors);
        
      } catch (error) {
        errors.push(`Failed to load ${url}: ${error.message}`);
      }
    }

    // Collect performance metrics
    const performanceMetrics = await this.collectPerformanceMetrics(page, browserName);

    await page.close();

    return {
      passed: errors.length === 0,
      errors,
      screenshots,
      performanceMetrics
    };
  }

  private async compareScreenshots(currentPath: string, baselinePath: string, browser: string, scenario: string): Promise<ScreenshotComparisonResult> {
    try {
      // Check if baseline exists
      const baselineExists = await fs.access(baselinePath).then(() => true).catch(() => false);
      if (!baselineExists) {
        // If no baseline, copy current as baseline
        await fs.copyFile(currentPath, baselinePath);
        return {
          browser,
          scenario,
          pixelDifference: 0,
          passed: true,
          diffImagePath: '',
          note: 'Baseline created'
        };
      }

      // Load images
      const currentBuffer = await fs.readFile(currentPath);
      const baselineBuffer = await fs.readFile(baselinePath);
      
      const currentImage = PNG.sync.read(currentBuffer);
      const baselineImage = PNG.sync.read(baselineBuffer);

      // Create diff image
      const { width, height } = currentImage;
      const diffImage = new PNG({ width, height });

      // Compare images
      const numDiffPixels = pixelmatch(
        currentImage.data,
        baselineImage.data,
        diffImage.data,
        width,
        height,
        { threshold: 0.1 }
      );

      const totalPixels = width * height;
      const pixelDifference = numDiffPixels / totalPixels;

      // Save diff image if there are differences
      let diffImagePath = '';
      if (numDiffPixels > 0) {
        diffImagePath = path.join(this.diffDir, `${browser}-${scenario}-diff.png`);
        await fs.writeFile(diffImagePath, PNG.sync.write(diffImage));
      }

      return {
        browser,
        scenario,
        pixelDifference,
        passed: pixelDifference < 0.01, // 1% threshold
        diffImagePath
      };

    } catch (error) {
      return {
        browser,
        scenario,
        pixelDifference: 1.0,
        passed: false,
        diffImagePath: '',
        error: error.message
      };
    }
  }

  private async collectPerformanceMetrics(page: Page, browser: string): Promise<PerformanceMetrics> {
    try {
      const performanceData = await page.evaluate(() => {
        const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
        const paint = performance.getEntriesByType('paint');
        
        return {
          domContentLoaded: navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart,
          loadComplete: navigation.loadEventEnd - navigation.loadEventStart,
          firstPaint: paint.find(entry => entry.name === 'first-paint')?.startTime || 0,
          firstContentfulPaint: paint.find(entry => entry.name === 'first-contentful-paint')?.startTime || 0,
          memoryUsage: (performance as any).memory ? {
            used: (performance as any).memory.usedJSHeapSize,
            total: (performance as any).memory.totalJSHeapSize,
            limit: (performance as any).memory.jsHeapSizeLimit
          } : null
        };
      });

      return {
        browser,
        timestamp: new Date(),
        ...performanceData
      };

    } catch (error) {
      console.error(`Failed to collect performance metrics for ${browser}:`, error);
      return {
        browser,
        timestamp: new Date(),
        domContentLoaded: 0,
        loadComplete: 0,
        firstPaint: 0,
        firstContentfulPaint: 0,
        memoryUsage: null
      };
    }
  }

  private async runAccessibilityChecks(page: Page, errors: string[]): Promise<void> {
    try {
      // Check for missing alt attributes
      const missingAltImages = await page.$$eval('img:not([alt])', images => images.length);
      if (missingAltImages > 0) {
        errors.push(`${missingAltImages} images missing alt attributes`);
      }

      // Check for proper heading structure
      const headings = await page.$$eval('h1, h2, h3, h4, h5, h6', headings => 
        headings.map(h => ({ tag: h.tagName, text: h.textContent }))
      );
      
      const h1Count = headings.filter(h => h.tag === 'H1').length;
      if (h1Count !== 1) {
        errors.push(`Expected 1 H1 tag, found ${h1Count}`);
      }

      // Check for keyboard focus indicators
      await page.keyboard.press('Tab');
      const focusedElement = await page.evaluate(() => document.activeElement?.tagName);
      if (!focusedElement) {
        errors.push('No keyboard focus indicators detected');
      }

    } catch (error) {
      errors.push(`Accessibility check failed: ${error.message}`);
    }
  }

  private async testElementVisibility(page: Page, selector: string): Promise<ElementTestResult> {
    try {
      const element = await page.$(selector);
      if (!element) {
        return { selector, passed: false, error: 'Element not found' };
      }

      const isVisible = await element.isVisible();
      return { selector, passed: isVisible };

    } catch (error) {
      return { selector, passed: false, error: error.message };
    }
  }

  private async testResponsiveBreakpoints(page: Page, width: number): Promise<ElementTestResult> {
    // Test CSS media queries and responsive behavior
    try {
      const breakpointTest = await page.evaluate((viewportWidth) => {
        const computedStyle = getComputedStyle(document.body);
        const display = computedStyle.display;
        
        // Check if responsive styles are applied correctly
        if (viewportWidth < 768) {
          return document.querySelector('.mobile-hidden')?.style.display === 'none';
        } else if (viewportWidth < 1024) {
          return document.querySelector('.tablet-visible')?.style.display !== 'none';
        } else {
          return document.querySelector('.desktop-visible')?.style.display !== 'none';
        }
      }, width);

      return { 
        selector: 'responsive-breakpoints', 
        passed: breakpointTest !== false 
      };

    } catch (error) {
      return { 
        selector: 'responsive-breakpoints', 
        passed: false, 
        error: error.message 
      };
    }
  }

  private async testTextReadability(page: Page): Promise<ElementTestResult> {
    try {
      const readabilityTest = await page.evaluate(() => {
        const elements = document.querySelectorAll('p, span, div, h1, h2, h3, h4, h5, h6');
        let failedElements = 0;

        for (const element of elements) {
          const style = getComputedStyle(element);
          const fontSize = parseFloat(style.fontSize);
          const lineHeight = parseFloat(style.lineHeight);
          
          // Check minimum font size (14px) and line height
          if (fontSize < 14 || (lineHeight && lineHeight < fontSize * 1.2)) {
            failedElements++;
          }
        }

        return failedElements === 0;
      });

      return { 
        selector: 'text-readability', 
        passed: readabilityTest 
      };

    } catch (error) {
      return { 
        selector: 'text-readability', 
        passed: false, 
        error: error.message 
      };
    }
  }

  private async testTouchTargetSize(page: Page, isMobile: boolean): Promise<ElementTestResult> {
    if (!isMobile) {
      return { selector: 'touch-targets', passed: true, note: 'Not applicable for non-mobile' };
    }

    try {
      const touchTargetTest = await page.evaluate(() => {
        const interactiveElements = document.querySelectorAll('button, a, input, select, textarea');
        let failedElements = 0;

        for (const element of interactiveElements) {
          const rect = element.getBoundingClientRect();
          const minSize = 44; // 44px minimum touch target size
          
          if (rect.width < minSize || rect.height < minSize) {
            failedElements++;
          }
        }

        return failedElements === 0;
      });

      return { 
        selector: 'touch-targets', 
        passed: touchTargetTest 
      };

    } catch (error) {
      return { 
        selector: 'touch-targets', 
        passed: false, 
        error: error.message 
      };
    }
  }
}

// Type definitions for visual validation
interface BrowserValidationResult {
  passed: boolean;
  errors: string[];
  screenshots: string[];
  performanceMetrics: PerformanceMetrics | null;
}

interface ConsoleMessage {
  browser: string;
  type: string;
  text: string;
  location: {
    url: string;
    lineNumber: number;
    columnNumber: number;
  };
  timestamp: Date;
}

interface PerformanceMetrics {
  browser: string;
  timestamp: Date;
  domContentLoaded: number;
  loadComplete: number;
  firstPaint: number;
  firstContentfulPaint: number;
  memoryUsage: {
    used: number;
    total: number;
    limit: number;
  } | null;
}

interface ScreenshotComparisonResult {
  browser: string;
  scenario: string;
  pixelDifference: number;
  passed: boolean;
  diffImagePath: string;
  error?: string;
  note?: string;
}

interface ResponsiveTestResult {
  viewportSize: { name: string; width: number; height: number };
  layoutValid: boolean;
  elementTests: ElementTestResult[];
  screenshotPath: string;
  error?: string;
}

interface ElementTestResult {
  selector: string;
  passed: boolean;
  error?: string;
  note?: string;
}