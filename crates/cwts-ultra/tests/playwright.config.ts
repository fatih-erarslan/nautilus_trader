import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  outputDir: './reports/playwright-results',
  
  // Global test settings
  timeout: 60 * 1000,
  expect: {
    timeout: 10 * 1000,
    toHaveScreenshot: { 
      threshold: 0.2,
      mode: 'pixel',
      animations: 'disabled'
    },
    toMatchScreenshot: {
      threshold: 0.2,
      mode: 'pixel'
    }
  },
  
  // Test configuration
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 1,
  workers: process.env.CI ? 1 : undefined,
  
  // Reporter configuration
  reporter: [
    ['html', { 
      outputFolder: './reports/playwright-html',
      open: 'never'
    }],
    ['junit', { 
      outputFile: './reports/junit/playwright-results.xml' 
    }],
    ['json', {
      outputFile: './reports/json/playwright-results.json'
    }],
    ['./utils/mathematical-reporter.ts'],
    ['./utils/visual-validation-reporter.ts']
  ],
  
  use: {
    // Browser configuration
    headless: true,
    viewport: { width: 1280, height: 720 },
    ignoreHTTPSErrors: true,
    
    // Screenshot and video configuration
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    trace: 'retain-on-failure',
    
    // Console monitoring
    actionTimeout: 10 * 1000,
    navigationTimeout: 30 * 1000,
    
    // Browser context
    contextOptions: {
      recordVideo: {
        dir: './reports/videos/',
        size: { width: 1280, height: 720 }
      }
    }
  },

  // Browser projects for comprehensive testing
  projects: [
    // Desktop browsers
    {
      name: 'chromium-desktop',
      use: { 
        ...devices['Desktop Chrome'],
        channel: 'chrome'
      }
    },
    {
      name: 'firefox-desktop',
      use: { 
        ...devices['Desktop Firefox']
      }
    },
    {
      name: 'webkit-desktop',
      use: { 
        ...devices['Desktop Safari']
      }
    },
    
    // Mobile browsers
    {
      name: 'mobile-chrome',
      use: { 
        ...devices['Pixel 5']
      }
    },
    {
      name: 'mobile-safari',
      use: { 
        ...devices['iPhone 12']
      }
    },
    
    // Performance testing
    {
      name: 'performance-chrome',
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: {
          args: [
            '--enable-precise-memory-info',
            '--js-flags=--expose-gc',
            '--enable-logging',
            '--log-level=0',
            '--v=1',
            '--vmodule=*/blink/renderer/core/layout/*=2'
          ]
        }
      }
    }
  ],

  // Web server configuration
  webServer: {
    command: 'npm run start:test',
    port: 3000,
    reuseExistingServer: !process.env.CI,
    timeout: 120 * 1000
  },
  
  // Global setup and teardown
  globalSetup: require.resolve('./utils/global-setup.ts'),
  globalTeardown: require.resolve('./utils/global-teardown.ts')
});