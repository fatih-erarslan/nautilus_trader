/**
 * Environment Loader Helper
 *
 * Loads and validates E2B credentials from environment variables
 */

const path = require('path');
const fs = require('fs');

/**
 * Load environment variables from .env file
 */
function loadEnv() {
  const envPath = path.resolve(__dirname, '../../../.env');

  if (fs.existsSync(envPath)) {
    require('dotenv').config({ path: envPath });
    console.log('✅ Loaded .env from:', envPath);
  } else {
    console.warn('⚠️  .env file not found, using system environment');
  }
}

/**
 * Get E2B credentials from environment
 * @returns {Object} Credentials object
 */
function getE2BCredentials() {
  const credentials = {
    apiKey: process.env.E2B_API_KEY || '',
    accessToken: process.env.E2B_ACCESS_TOKEN || '',
  };

  if (!credentials.apiKey) {
    console.warn('⚠️  E2B_API_KEY not set in environment');
  }

  if (!credentials.accessToken) {
    console.warn('⚠️  E2B_ACCESS_TOKEN not set in environment');
  }

  return credentials;
}

/**
 * Validate E2B credentials
 * @param {Object} credentials - Credentials to validate
 * @returns {boolean} True if valid
 */
function validateCredentials(credentials) {
  if (!credentials.apiKey || credentials.apiKey.length < 10) {
    console.error('❌ Invalid E2B_API_KEY');
    return false;
  }

  return true;
}

/**
 * Get test configuration
 * @returns {Object} Test configuration
 */
function getTestConfig() {
  return {
    mockMode: !process.env.E2B_API_KEY,
    timeout: parseInt(process.env.TEST_TIMEOUT || '120000', 10),
    verbose: process.env.TEST_VERBOSE === 'true',
    cleanupEnabled: process.env.TEST_CLEANUP !== 'false',
  };
}

module.exports = {
  loadEnv,
  getE2BCredentials,
  validateCredentials,
  getTestConfig,
};
