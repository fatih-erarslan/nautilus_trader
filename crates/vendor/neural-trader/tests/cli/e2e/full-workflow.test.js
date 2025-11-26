/**
 * E2E tests for complete CLI workflows
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const os = require('os');

describe('CLI E2E Full Workflows', () => {
  const cliPath = path.join(__dirname, '../../../bin/cli.js');
  let tempDir;

  beforeEach(() => {
    tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'neural-trader-e2e-'));
    process.chdir(tempDir);
  });

  afterEach(() => {
    if (tempDir && fs.existsSync(tempDir)) {
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
  });

  describe('New user onboarding', () => {
    it('should guide user from zero to running project', () => {
      // User starts with nothing
      expect(fs.readdirSync(tempDir).length).toBe(0);

      // User runs help to understand commands
      const helpOutput = execSync(`node ${cliPath} help`, {
        encoding: 'utf8',
        cwd: tempDir
      });
      expect(helpOutput).toContain('Usage:');
      expect(helpOutput).toContain('init');

      // User lists available packages
      const listOutput = execSync(`node ${cliPath} list`, {
        encoding: 'utf8',
        cwd: tempDir
      });
      expect(listOutput).toContain('trading');

      // User gets info about trading package
      const infoOutput = execSync(`node ${cliPath} info trading`, {
        encoding: 'utf8',
        cwd: tempDir
      });
      expect(infoOutput).toContain('Trading Strategy System');
      expect(infoOutput).toContain('neural-trader init trading');

      // User initializes trading project
      const initOutput = execSync(`node ${cliPath} init trading`, {
        encoding: 'utf8',
        cwd: tempDir
      });
      expect(initOutput).toContain('✅ Project initialized!');
      expect(initOutput).toContain('npm install');

      // Verify project structure
      expect(fs.existsSync(path.join(tempDir, 'package.json'))).toBe(true);
      expect(fs.existsSync(path.join(tempDir, 'config.json'))).toBe(true);
      expect(fs.existsSync(path.join(tempDir, 'src/main.js'))).toBe(true);
      expect(fs.existsSync(path.join(tempDir, 'README.md'))).toBe(true);

      // User runs health check
      const doctorOutput = execSync(`node ${cliPath} doctor`, {
        encoding: 'utf8',
        cwd: tempDir
      });
      expect(doctorOutput).toContain('package.json');
      expect(doctorOutput).toContain('Found');

      // User tests installation
      const testOutput = execSync(`node ${cliPath} test`, {
        encoding: 'utf8',
        cwd: tempDir
      });
      expect(testOutput).toContain('Tests complete');
    });
  });

  describe('Trading project setup', () => {
    it('should create production-ready trading project', () => {
      // Initialize trading project
      execSync(`node ${cliPath} init trading`, { cwd: tempDir });

      // Verify package.json has correct dependencies
      const pkg = JSON.parse(
        fs.readFileSync(path.join(tempDir, 'package.json'), 'utf8')
      );
      expect(pkg.dependencies).toHaveProperty('neural-trader');
      expect(pkg.name).toContain('trading');

      // Verify config has trading settings
      const config = JSON.parse(
        fs.readFileSync(path.join(tempDir, 'config.json'), 'utf8')
      );
      expect(config.trading).toHaveProperty('provider');
      expect(config.trading).toHaveProperty('symbols');
      expect(config.trading).toHaveProperty('strategy');
      expect(config.risk).toHaveProperty('max_position_size');

      // Verify main.js has correct structure
      const mainContent = fs.readFileSync(
        path.join(tempDir, 'src/main.js'),
        'utf8'
      );
      expect(mainContent).toContain('require(');
      expect(mainContent).toContain('neural-trader');
      expect(mainContent).toContain('async function main');

      // Verify README has instructions
      const readme = fs.readFileSync(
        path.join(tempDir, 'README.md'),
        'utf8'
      );
      expect(readme).toContain('Neural Trader');
      expect(readme).toContain('npm install');
    });

    it('should create all necessary directories', () => {
      execSync(`node ${cliPath} init trading`, { cwd: tempDir });

      const expectedDirs = ['src', 'data', 'config', 'strategies', 'backtest-results'];
      expectedDirs.forEach(dir => {
        expect(fs.existsSync(path.join(tempDir, dir))).toBe(true);
      });
    });
  });

  describe('Accounting project setup', () => {
    it('should create accounting project with specialized config', () => {
      execSync(`node ${cliPath} init accounting`, { cwd: tempDir });

      // Verify accounting-specific directories
      expect(fs.existsSync(path.join(tempDir, 'reports'))).toBe(true);
      expect(fs.existsSync(path.join(tempDir, 'tax-lots'))).toBe(true);

      // Verify accounting config
      const config = JSON.parse(
        fs.readFileSync(path.join(tempDir, 'config.json'), 'utf8')
      );
      expect(config).toHaveProperty('accounting');
      expect(config.accounting.method).toBe('HIFO');
      expect(config.accounting.tax_lots.enabled).toBe(true);

      // Verify package dependencies
      const pkg = JSON.parse(
        fs.readFileSync(path.join(tempDir, 'package.json'), 'utf8')
      );
      expect(pkg.dependencies).toHaveProperty('@neural-trader/agentic-accounting-core');
    });
  });

  describe('Example project setup', () => {
    it('should create example project with proper structure', () => {
      execSync(`node ${cliPath} init example:portfolio-optimization`, { cwd: tempDir });

      // Verify example-specific structure
      expect(fs.existsSync(path.join(tempDir, 'src'))).toBe(true);
      expect(fs.existsSync(path.join(tempDir, 'data'))).toBe(true);
      expect(fs.existsSync(path.join(tempDir, 'output'))).toBe(true);

      // Verify package.json
      const pkg = JSON.parse(
        fs.readFileSync(path.join(tempDir, 'package.json'), 'utf8')
      );
      expect(pkg.name).toContain('portfolio-optimization');
    });
  });

  describe('Multi-project workflow', () => {
    it('should handle multiple project types in sequence', () => {
      const types = ['trading', 'backtesting', 'accounting'];

      types.forEach(type => {
        const projectDir = path.join(tempDir, type);
        fs.mkdirSync(projectDir);

        execSync(`node ${cliPath} init ${type}`, { cwd: projectDir });

        expect(fs.existsSync(path.join(projectDir, 'package.json'))).toBe(true);
        expect(fs.existsSync(path.join(projectDir, 'config.json'))).toBe(true);

        const doctorOutput = execSync(`node ${cliPath} doctor`, {
          encoding: 'utf8',
          cwd: projectDir
        });
        expect(doctorOutput).toContain('package.json');
      });
    });
  });

  describe('Error recovery scenarios', () => {
    it('should recover from missing files', () => {
      // Initialize project
      execSync(`node ${cliPath} init trading`, { cwd: tempDir });

      // Remove config.json
      fs.unlinkSync(path.join(tempDir, 'config.json'));

      // Doctor should report missing config
      const doctorOutput = execSync(`node ${cliPath} doctor`, {
        encoding: 'utf8',
        cwd: tempDir
      });
      expect(doctorOutput).toContain('config.json');

      // Re-initialize should fix it
      execSync(`node ${cliPath} init trading`, { cwd: tempDir });
      expect(fs.existsSync(path.join(tempDir, 'config.json'))).toBe(true);
    });
  });

  describe('Command chaining', () => {
    it('should handle multiple commands in succession', () => {
      const commands = [
        'version',
        'help',
        'list',
        'init trading',
        'test',
        'doctor'
      ];

      commands.forEach(cmd => {
        const output = execSync(`node ${cliPath} ${cmd}`, {
          encoding: 'utf8',
          cwd: tempDir
        });
        expect(output).toBeTruthy();
        expect(output.length).toBeGreaterThan(0);
      });
    });
  });

  describe('Cross-platform compatibility', () => {
    it('should work on current platform', () => {
      const platform = os.platform();

      // Test basic commands on all platforms
      const versionOutput = execSync(`node ${cliPath} version`, {
        encoding: 'utf8',
        cwd: tempDir
      });
      expect(versionOutput).toContain('2.3.15');

      // Test initialization
      execSync(`node ${cliPath} init trading`, { cwd: tempDir });
      expect(fs.existsSync(path.join(tempDir, 'package.json'))).toBe(true);

      // Platform-specific checks
      if (platform === 'win32') {
        // Windows-specific
        expect(tempDir).toMatch(/[A-Z]:\\/);
      } else {
        // Unix-like
        expect(tempDir).toMatch(/^\//);
      }
    });
  });

  describe('Complete lifecycle', () => {
    it('should support full project lifecycle', () => {
      // 1. Discovery
      const listOutput = execSync(`node ${cliPath} list`, {
        encoding: 'utf8',
        cwd: tempDir
      });
      expect(listOutput).toContain('Available');

      // 2. Research
      const infoOutput = execSync(`node ${cliPath} info trading`, {
        encoding: 'utf8',
        cwd: tempDir
      });
      expect(infoOutput).toContain('Features:');

      // 3. Initialize
      const initOutput = execSync(`node ${cliPath} init trading`, {
        encoding: 'utf8',
        cwd: tempDir
      });
      expect(initOutput).toContain('initialized');

      // 4. Validate
      const testOutput = execSync(`node ${cliPath} test`, {
        encoding: 'utf8',
        cwd: tempDir
      });
      expect(testOutput).toContain('complete');

      // 5. Health check
      const doctorOutput = execSync(`node ${cliPath} doctor`, {
        encoding: 'utf8',
        cwd: tempDir
      });
      expect(doctorOutput).toContain('Health Check');

      // 6. Verify all files present
      expect(fs.existsSync(path.join(tempDir, 'package.json'))).toBe(true);
      expect(fs.existsSync(path.join(tempDir, 'config.json'))).toBe(true);
      expect(fs.existsSync(path.join(tempDir, 'src/main.js'))).toBe(true);
      expect(fs.existsSync(path.join(tempDir, 'README.md'))).toBe(true);
    });
  });
});
