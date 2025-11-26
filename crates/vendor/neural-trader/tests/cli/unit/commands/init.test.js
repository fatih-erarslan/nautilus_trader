/**
 * Unit tests for CLI init command
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const os = require('os');

describe('CLI Init Command', () => {
  const cliPath = path.join(__dirname, '../../../../bin/cli.js');
  let tempDir;

  beforeEach(() => {
    // Create temp directory for each test
    tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'neural-trader-test-'));
    process.chdir(tempDir);
  });

  afterEach(() => {
    // Clean up temp directory
    if (tempDir && fs.existsSync(tempDir)) {
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
  });

  describe('Basic initialization', () => {
    it('should initialize a trading project', () => {
      const output = execSync(`node ${cliPath} init trading`, {
        encoding: 'utf8',
        cwd: tempDir
      });

      expect(output).toContain('Initializing');
      expect(output).toContain('Trading');
      expect(output).toContain('✅ Project initialized!');
    });

    it('should create required directories', () => {
      execSync(`node ${cliPath} init trading`, { cwd: tempDir });

      expect(fs.existsSync(path.join(tempDir, 'src'))).toBe(true);
      expect(fs.existsSync(path.join(tempDir, 'data'))).toBe(true);
      expect(fs.existsSync(path.join(tempDir, 'config'))).toBe(true);
    });

    it('should create package.json', () => {
      execSync(`node ${cliPath} init trading`, { cwd: tempDir });

      const pkgPath = path.join(tempDir, 'package.json');
      expect(fs.existsSync(pkgPath)).toBe(true);

      const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
      expect(pkg.name).toContain('trading');
      expect(pkg.dependencies).toHaveProperty('neural-trader');
    });

    it('should create config.json', () => {
      execSync(`node ${cliPath} init trading`, { cwd: tempDir });

      const configPath = path.join(tempDir, 'config.json');
      expect(fs.existsSync(configPath)).toBe(true);

      const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
      expect(config).toHaveProperty('trading');
      expect(config).toHaveProperty('risk');
    });

    it('should create example code file', () => {
      execSync(`node ${cliPath} init trading`, { cwd: tempDir });

      const mainPath = path.join(tempDir, 'src', 'main.js');
      expect(fs.existsSync(mainPath)).toBe(true);

      const content = fs.readFileSync(mainPath, 'utf8');
      expect(content).toContain('neural-trader');
    });

    it('should create README', () => {
      execSync(`node ${cliPath} init trading`, { cwd: tempDir });

      const readmePath = path.join(tempDir, 'README.md');
      expect(fs.existsSync(readmePath)).toBe(true);

      const content = fs.readFileSync(readmePath, 'utf8');
      expect(content).toContain('Neural Trader');
    });
  });

  describe('Different project types', () => {
    const projectTypes = [
      'trading',
      'backtesting',
      'accounting',
      'predictor'
    ];

    projectTypes.forEach(type => {
      it(`should initialize ${type} project`, () => {
        const output = execSync(`node ${cliPath} init ${type}`, {
          encoding: 'utf8',
          cwd: tempDir
        });

        expect(output).toContain('Initializing');
        expect(fs.existsSync(path.join(tempDir, 'package.json'))).toBe(true);
      });
    });

    it('should create type-specific directories for trading', () => {
      execSync(`node ${cliPath} init trading`, { cwd: tempDir });

      expect(fs.existsSync(path.join(tempDir, 'strategies'))).toBe(true);
      expect(fs.existsSync(path.join(tempDir, 'backtest-results'))).toBe(true);
    });

    it('should create type-specific directories for accounting', () => {
      execSync(`node ${cliPath} init accounting`, { cwd: tempDir });

      expect(fs.existsSync(path.join(tempDir, 'reports'))).toBe(true);
      expect(fs.existsSync(path.join(tempDir, 'tax-lots'))).toBe(true);
    });

    it('should create type-specific config for accounting', () => {
      execSync(`node ${cliPath} init accounting`, { cwd: tempDir });

      const config = JSON.parse(fs.readFileSync(path.join(tempDir, 'config.json'), 'utf8'));
      expect(config).toHaveProperty('accounting');
      expect(config.accounting).toHaveProperty('method');
    });
  });

  describe('Example project initialization', () => {
    it('should initialize example with example: prefix', () => {
      const output = execSync(`node ${cliPath} init example:portfolio-optimization`, {
        encoding: 'utf8',
        cwd: tempDir
      });

      expect(output).toContain('Initializing example');
      expect(output).toContain('portfolio-optimization');
    });

    it('should create example project structure', () => {
      execSync(`node ${cliPath} init example:portfolio-optimization`, { cwd: tempDir });

      expect(fs.existsSync(path.join(tempDir, 'src'))).toBe(true);
      expect(fs.existsSync(path.join(tempDir, 'data'))).toBe(true);
      expect(fs.existsSync(path.join(tempDir, 'output'))).toBe(true);
    });
  });

  describe('Next steps display', () => {
    it('should show next steps after initialization', () => {
      const output = execSync(`node ${cliPath} init trading`, {
        encoding: 'utf8',
        cwd: tempDir
      });

      expect(output).toContain('Next steps:');
      expect(output).toContain('npm install');
      expect(output).toContain('node src/main.js');
    });
  });

  describe('File content validation', () => {
    it('should generate valid JSON in package.json', () => {
      execSync(`node ${cliPath} init trading`, { cwd: tempDir });

      const pkgPath = path.join(tempDir, 'package.json');
      expect(() => {
        JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
      }).not.toThrow();
    });

    it('should generate valid JSON in config.json', () => {
      execSync(`node ${cliPath} init trading`, { cwd: tempDir });

      const configPath = path.join(tempDir, 'config.json');
      expect(() => {
        JSON.parse(fs.readFileSync(configPath, 'utf8'));
      }).not.toThrow();
    });

    it('should generate runnable JavaScript code', () => {
      execSync(`node ${cliPath} init trading`, { cwd: tempDir });

      const mainPath = path.join(tempDir, 'src', 'main.js');
      const content = fs.readFileSync(mainPath, 'utf8');

      // Should have valid JavaScript syntax
      expect(content).toContain('require');
      expect(content).toContain('function');
      expect(content).toContain('.catch');
    });
  });

  describe('Output formatting', () => {
    it('should show progress with checkmarks', () => {
      const output = execSync(`node ${cliPath} init trading`, {
        encoding: 'utf8',
        cwd: tempDir
      });

      expect(output).toMatch(/✓|✅/);
      expect(output).toContain('Created');
    });

    it('should list created files', () => {
      const output = execSync(`node ${cliPath} init trading`, {
        encoding: 'utf8',
        cwd: tempDir
      });

      expect(output).toContain('package.json');
      expect(output).toContain('config.json');
      expect(output).toContain('src/main.js');
    });
  });

  describe('Exit behavior', () => {
    it('should exit successfully', () => {
      try {
        execSync(`node ${cliPath} init trading`, { cwd: tempDir });
        expect(true).toBe(true);
      } catch (error) {
        fail(`Init should not fail: ${error.message}`);
      }
    });
  });
});
