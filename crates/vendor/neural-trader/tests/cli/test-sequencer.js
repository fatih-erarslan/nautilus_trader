/**
 * Custom test sequencer for CLI tests
 * Runs tests in a logical order: unit -> integration -> e2e -> performance
 */

const Sequencer = require('@jest/test-sequencer').default;

class CustomSequencer extends Sequencer {
  sort(tests) {
    // Copy the tests array
    const copyTests = Array.from(tests);

    // Define order priority
    const getOrder = (testPath) => {
      if (testPath.includes('/unit/')) return 1;
      if (testPath.includes('/integration/')) return 2;
      if (testPath.includes('/e2e/')) return 3;
      if (testPath.includes('/performance/')) return 4;
      return 5;
    };

    // Sort by order priority, then alphabetically
    return copyTests.sort((testA, testB) => {
      const orderA = getOrder(testA.path);
      const orderB = getOrder(testB.path);

      if (orderA !== orderB) {
        return orderA - orderB;
      }

      return testA.path.localeCompare(testB.path);
    });
  }
}

module.exports = CustomSequencer;
