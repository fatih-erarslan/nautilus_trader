/**
 * Integration tests for CLI workflows
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const os = require('os');

describe('CLI Integration Workflows', () => {
  const cliPath = path.join(__dirname, '../../../bin/cli.js');
  let tempDir;

  beforeEach(() => {
    tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'neural-trader-test-'));
    process.chdir(tempDir);
  });

  afterEach(() => {
    if (tempDir && fs.existsSync(tempDir)) {
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
  });

  describe('Complete project setup workflow', () => {
    it('should complete full project initialization', () => {
      // Step 1: Check version
      const versionOutput = execSync(`node ${cliPath} version`, {
        encoding: 'utf8',
        cwd: tempDir
      });
      expect(versionOutput).toContain('2.3.15');

      // Step 2: Initialize project
      const initOutput = execSync(`node ${cliPath} init trading`, {
        encoding: 'utf8',
        cwd: tempDir
      });
      expect(initOutput).toContain('✅ Project initialized!');

      // Step 3: Verify files were created
      expect(fs.existsSync(path.join(tempDir, 'package.json'))).toBe(true);
      expect(fs.existsSync(path.join(tempDir, 'config.json'))).toBe(true);
      expect(fs.existsSync(path.join(tempDir, 'src/main.js'))).toBe(true);

      // Step 4: Run test command
      const testOutput = execSync(`node ${cliPath} test`, {
        encoding: 'utf8',
        cwd: tempDir
      });
      expect(testOutput).toContain('Tests complete');

      // Step 5: Run doctor command
      const doctorOutput = execSync(`node ${cliPath} doctor`, {
        encoding: 'utf8',
        cwd: tempDir
      });
      expect(doctorOutput).toContain('Health Check Results');
    });

    it('should handle different project types', () => {
      const types = ['trading', 'backtesting', 'accounting'];

      types.forEach(type => {
        const testDir = fs.mkdtempSync(path.join(os.tmpdir(), `nt-${type}-`));
        try {
          execSync(`node ${cliPath} init ${type}`, { cwd: testDir });

          expect(fs.existsSync(path.join(testDir, 'package.json'))).toBe(true);
          expect(fs.existsSync(path.join(testDir, 'config.json'))).toBe(true);

          const config = JSON.parse(
            fs.readFileSync(path.join(testDir, 'config.json'), 'utf8')
          );

          if (type === 'accounting') {
            expect(config).toHaveProperty('accounting');
          } else {
            expect(config).toHaveProperty('trading');
          }
        } finally {
          fs.rmSync(testDir, { recursive: true, force: true });
        }
      });
    });
  });

  describe('Discovery workflow', () => {
    it('should help user discover available packages', () => {
      // Step 1: List packages
      const listOutput = execSync(`node ${cliPath} list`, {
        encoding: 'utf8',
        cwd: tempDir
      });
      expect(listOutput).toContain('Available Neural Trader Packages');

      // Step 2: Get info on specific package
      const infoOutput = execSync(`node ${cliPath} info trading`, {
        encoding: 'utf8',
        cwd: tempDir
      });
      expect(infoOutput).toContain('Trading Strategy System');
      expect(infoOutput).toContain('Features:');
      expect(infoOutput).toContain('Initialize:');

      // Step 3: Initialize based on info
      const initOutput = execSync(`node ${cliPath} init trading`, {
        encoding: 'utf8',
        cwd: tempDir
      });
      expect(initOutput).toContain('✅ Project initialized!');
    });
  });

  describe('Error recovery workflow', () => {
    it('should provide helpful errors and recovery steps', () => {
      // Try to install without package.json
      try {
        execSync(`node ${cliPath} install @neural-trader/core`, {
          cwd: tempDir,
          stdio: 'pipe'
        });
        fail('Should have thrown error');
      } catch (error) {
        expect(error.stderr.toString()).toContain('No package.json found');
        expect(error.stderr.toString()).toContain('neural-trader init');
      }

      // Initialize project to fix the issue
      execSync(`node ${cliPath} init trading`, { cwd: tempDir });

      // Now package.json exists, doctor should be happy
      const doctorOutput = execSync(`node ${cliPath} doctor`, {
        encoding: 'utf8',
        cwd: tempDir
      });
      expect(doctorOutput).toContain('Found');
    });
  });

  describe('Validation workflow', () => {
    it('should validate project health after setup', () => {
      // Initialize project
      execSync(`node ${cliPath} init trading`, { cwd: tempDir });

      // Run comprehensive checks
      const doctorOutput = execSync(`node ${cliPath} doctor`, {
        encoding: 'utf8',
        cwd: tempDir
      });

      expect(doctorOutput).toContain('package.json');
      expect(doctorOutput).toContain('Found');

      // Test NAPI bindings
      const testOutput = execSync(`node ${cliPath} test`, {
        encoding: 'utf8',
        cwd: tempDir
      });

      expect(testOutput).toContain('Installed Packages');
      expect(testOutput).toContain('neural-trader');
    });
  });

  describe('Multi-command sequences', () => {
    it('should handle rapid command execution', () => {
      const commands = [
        'version',
        'help',
        'list'
      ];

      commands.forEach(cmd => {
        const output = execSync(`node ${cliPath} ${cmd}`, {
          encoding: 'utf8',
          cwd: tempDir
        });
        expect(output).toBeTruthy();
      });
    });

    it('should maintain state across commands', () => {
      // Initialize
      execSync(`node ${cliPath} init trading`, { cwd: tempDir });

      // Multiple checks should all see the initialized state
      for (let i = 0; i < 3; i++) {
        const output = execSync(`node ${cliPath} doctor`, {
          encoding: 'utf8',
          cwd: tempDir
        });
        expect(output).toContain('package.json');
        expect(output).toContain('Found');
      }
    });
  });

  describe('Example project workflow', () => {
    it('should initialize example projects', () => {
      const initOutput = execSync(`node ${cliPath} init example:portfolio-optimization`, {
        encoding: 'utf8',
        cwd: tempDir
      });

      expect(initOutput).toContain('Initializing example');
      expect(initOutput).toContain('portfolio-optimization');

      // Verify structure
      expect(fs.existsSync(path.join(tempDir, 'package.json'))).toBe(true);
      expect(fs.existsSync(path.join(tempDir, 'src'))).toBe(true);
      expect(fs.existsSync(path.join(tempDir, 'output'))).toBe(true);

      // Run health check
      const doctorOutput = execSync(`node ${cliPath} doctor`, {
        encoding: 'utf8',
        cwd: tempDir
      });
      expect(doctorOutput).toContain('package.json');
    });
  });

  describe('File integrity validation', () => {
    it('should create valid and parseable files', () => {
      execSync(`node ${cliPath} init trading`, { cwd: tempDir });

      // Validate package.json
      const pkg = JSON.parse(
        fs.readFileSync(path.join(tempDir, 'package.json'), 'utf8')
      );
      expect(pkg).toHaveProperty('name');
      expect(pkg).toHaveProperty('version');
      expect(pkg).toHaveProperty('dependencies');

      // Validate config.json
      const config = JSON.parse(
        fs.readFileSync(path.join(tempDir, 'config.json'), 'utf8')
      );
      expect(config).toHaveProperty('trading');
      expect(config.trading).toHaveProperty('provider');
      expect(config.trading).toHaveProperty('symbols');

      // Validate main.js is syntactically correct
      const mainPath = path.join(tempDir, 'src/main.js');
      expect(() => {
        require(mainPath);
      }).not.toThrow(SyntaxError);
    });
  });
});
