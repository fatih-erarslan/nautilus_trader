const { existsSync, readFileSync } = require('fs');
const { join } = require('path');
const { platform, arch } = process;

const PLATFORM_TARGETS = {
  'win32-x64': 'syndicate.win32-x64-msvc.node',
  'win32-ia32': 'syndicate.win32-ia32-msvc.node',
  'win32-arm64': 'syndicate.win32-arm64-msvc.node',
  'darwin-x64': 'syndicate.darwin-x64.node',
  'darwin-arm64': 'syndicate.darwin-arm64.node',
  'linux-x64-gnu': 'syndicate.linux-x64-gnu.node',
  'linux-x64-musl': 'syndicate.linux-x64-musl.node',
  'linux-arm64-gnu': 'syndicate.linux-arm64-gnu.node',
  'linux-arm64-musl': 'syndicate.linux-arm64-musl.node',
  'linux-arm-gnueabihf': 'syndicate.linux-arm-gnueabihf.node'
};

function getTargetPlatform() {
  const platformKey = `${platform}-${arch}`;

  // For Linux, detect libc variant
  if (platform === 'linux') {
    const isMusl = existsSync('/etc/alpine-release') ||
                   (process.report?.getReport()?.header?.glibcVersionRuntime === undefined);
    const libc = isMusl ? 'musl' : 'gnu';
    return `${platform}-${arch}-${libc}`;
  }

  return platformKey;
}

function loadNativeModule() {
  const targetPlatform = getTargetPlatform();
  const nativeFile = PLATFORM_TARGETS[targetPlatform];

  if (!nativeFile) {
    throw new Error(
      `Unsupported platform: ${targetPlatform}\n` +
      `Supported platforms: ${Object.keys(PLATFORM_TARGETS).join(', ')}`
    );
  }

  const localPath = join(__dirname, nativeFile);
  const cargoTargetPath = join(__dirname, '../../target/release', nativeFile);

  // Try local path first (published package)
  if (existsSync(localPath)) {
    return require(localPath);
  }

  // Fall back to cargo build directory (development)
  if (existsSync(cargoTargetPath)) {
    return require(cargoTargetPath);
  }

  throw new Error(
    `Native module not found for ${targetPlatform}\n` +
    `Expected at: ${localPath}\n` +
    `Or at: ${cargoTargetPath}\n` +
    `Run 'npm run build' to compile the native module.`
  );
}

// Load and export native module
const nativeModule = loadNativeModule();

module.exports = {
  // Enums
  AllocationStrategy: nativeModule.AllocationStrategy,
  DistributionModel: nativeModule.DistributionModel,
  MemberRole: nativeModule.MemberRole,
  MemberTier: nativeModule.MemberTier,
  VoteType: nativeModule.VoteType,
  VoteStatus: nativeModule.VoteStatus,
  WithdrawalStatus: nativeModule.WithdrawalStatus,

  // Classes
  SyndicateManager: nativeModule.SyndicateManager,

  // Functions
  createSyndicate: nativeModule.createSyndicate,
  calculateKelly: nativeModule.calculateKelly,
  calculateKellyFractional: nativeModule.calculateKellyFractional,
  calculateOptimalBetSize: nativeModule.calculateOptimalBetSize,
  validateBankrollRules: nativeModule.validateBankrollRules,
  calculateRiskMetrics: nativeModule.calculateRiskMetrics,
  simulateAllocationStrategies: nativeModule.simulateAllocationStrategies,
  calculateMemberTaxLiability: nativeModule.calculateMemberTaxLiability,
  generatePerformanceReport: nativeModule.generatePerformanceReport,
  exportSyndicateState: nativeModule.exportSyndicateState,
  importSyndicateState: nativeModule.importSyndicateState,

  // Version info
  version: require('./package.json').version,
  nativeBinding: nativeFile
};
