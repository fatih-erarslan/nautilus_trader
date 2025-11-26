# Neural Trader CLI Test Suite

Comprehensive test suite for the Neural Trader CLI implementation.

## Test Structure

```
tests/cli/
├── unit/                    # Unit tests for individual commands
│   └── commands/
│       ├── version.test.js
│       ├── help.test.js
│       ├── info.test.js
│       ├── list.test.js
│       ├── init.test.js
│       ├── install.test.js
│       ├── test.test.js
│       └── doctor.test.js
├── integration/            # Integration tests for command flows
│   └── cli-workflow.test.js
├── e2e/                    # End-to-end tests for complete workflows
│   └── full-workflow.test.js
├── performance/            # Performance tests
│   └── startup-time.test.js
├── fixtures/               # Test data and configurations
│   ├── mock-data.js
│   └── test-configs.js
├── __mocks__/             # Mock implementations
│   ├── fs.js
│   └── child_process.js
├── jest.config.js         # Jest configuration
├── setup-cli-tests.js     # Test setup
└── test-sequencer.js      # Custom test ordering
```

## Running Tests

### Run all CLI tests
```bash
npm run test:cli
```

### Run specific test types
```bash
# Unit tests only
npm run test:cli:unit

# Integration tests
npm run test:cli:integration

# E2E tests
npm run test:cli:e2e

# Performance tests
npm run test:cli:performance
```

### Run with coverage
```bash
npm run test:cli:coverage
```

### Run in watch mode
```bash
npm run test:cli:watch
```

## Test Categories

### Unit Tests
Test individual CLI commands in isolation:
- **version** - Version display and system info
- **help** - Help text and documentation
- **info** - Package information display
- **list** - Package listing
- **init** - Project initialization
- **install** - Package installation
- **test** - Component testing
- **doctor** - Health checks

### Integration Tests
Test command flows and interactions:
- Complete project setup workflows
- Discovery and initialization flows
- Error recovery scenarios
- Multi-command sequences

### E2E Tests
Test complete user workflows:
- New user onboarding
- Trading project setup
- Accounting project setup
- Example project setup
- Cross-platform compatibility

### Performance Tests
Test CLI performance characteristics:
- Startup time measurements
- Cold vs warm start performance
- Memory efficiency
- Concurrent execution

## Test Coverage Goals

- **Statements**: >80%
- **Branches**: >75%
- **Functions**: >80%
- **Lines**: >80%

## Writing Tests

### Test Structure
Follow the Arrange-Act-Assert pattern:

```javascript
describe('Feature', () => {
  it('should do something', () => {
    // Arrange
    const input = 'test';

    // Act
    const result = someFunction(input);

    // Assert
    expect(result).toBe('expected');
  });
});
```

### Using Fixtures
```javascript
const { mockPackageJson, mockConfigJson } = require('../fixtures/mock-data');

it('should use mock data', () => {
  const pkg = mockPackageJson;
  expect(pkg).toHaveProperty('name');
});
```

### Testing CLI Commands
```javascript
const { execSync } = require('child_process');
const cliPath = path.join(__dirname, '../../../bin/cli.js');

it('should run command', () => {
  const output = execSync(`node ${cliPath} version`, { encoding: 'utf8' });
  expect(output).toContain('2.3.15');
});
```

### Using Temp Directories
```javascript
let tempDir;

beforeEach(() => {
  tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'neural-trader-test-'));
  process.chdir(tempDir);
});

afterEach(() => {
  fs.rmSync(tempDir, { recursive: true, force: true });
});
```

## Mocking

### Mock Filesystem
```javascript
jest.mock('fs');
const fs = require('fs');

fs.existsSync.mockReturnValue(true);
fs.readFileSync.mockReturnValue('{"name": "test"}');
```

### Mock Child Process
```javascript
jest.mock('child_process');
const { spawn } = require('child_process');

spawn.mockImplementation(() => ({
  on: jest.fn((event, cb) => {
    if (event === 'exit') cb(0);
  })
}));
```

## Best Practices

1. **Isolation**: Each test should be independent
2. **Cleanup**: Always clean up temp files and directories
3. **Speed**: Unit tests should be fast (<100ms)
4. **Clarity**: Test names should describe expected behavior
5. **Coverage**: Test both success and error paths
6. **Mocking**: Mock external dependencies appropriately
7. **Assertions**: Use specific assertions, not generic ones

## Debugging Tests

### Run specific test
```bash
npm run test:cli -- -t "should display version"
```

### Run with debug output
```bash
DEBUG=* npm run test:cli
```

### Run in Node debugger
```bash
node --inspect-brk node_modules/.bin/jest tests/cli/unit/commands/version.test.js
```

## Continuous Integration

Tests are automatically run on:
- Pull requests
- Commits to main branch
- Release tags

Coverage reports are uploaded to the coverage service.

## Contributing

When adding new CLI features:
1. Write tests first (TDD)
2. Ensure all tests pass
3. Maintain coverage above thresholds
4. Update documentation

## Troubleshooting

### Tests timeout
Increase timeout in jest.config.js or individual tests:
```javascript
jest.setTimeout(60000);
```

### Temp directory cleanup issues
Ensure proper cleanup in afterEach:
```javascript
afterEach(() => {
  if (tempDir && fs.existsSync(tempDir)) {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
});
```

### Mock not working
Check mock is defined before module import:
```javascript
jest.mock('module-name');
const module = require('module-name');
```

## Resources

- [Jest Documentation](https://jestjs.io/docs/getting-started)
- [Testing Best Practices](https://testingjavascript.com/)
- [CLI Testing Guide](https://github.com/avajs/ava/blob/main/docs/recipes/testing-cli-apps.md)
