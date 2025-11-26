/* eslint-disable */
const { existsSync, readFileSync } = require('fs')
const { join } = require('path')

const { platform, arch } = process

let nativeBinding = null
let localFileExisted = false
let loadError = null

function isMusl() {
  // Fix: Use safer alternative without command injection vulnerability
  if (!process.report || typeof process.report.getReport !== 'function') {
    try {
      // Check for musl by examining process features instead of executing shell commands
      const fs = require('fs');
      // Try to read /etc/os-release or check common musl indicators
      if (fs.existsSync('/lib/libc.musl-x86_64.so.1') ||
          fs.existsSync('/lib/ld-musl-x86_64.so.1') ||
          fs.existsSync('/lib/libc.musl-aarch64.so.1')) {
        return true;
      }
      return false;
    } catch (e) {
      return false;
    }
  } else {
    const { glibcVersionRuntime } = process.report.getReport().header
    return !glibcVersionRuntime
  }
}

switch (platform) {
  case 'android':
    switch (arch) {
      case 'arm64':
        localFileExisted = existsSync(join(__dirname, 'neuro-divergent.android-arm64.node'))
        try {
          if (localFileExisted) {
            nativeBinding = require('./neuro-divergent.android-arm64.node')
          } else {
            nativeBinding = require('@neural-trader/neuro-divergent-android-arm64')
          }
        } catch (e) {
          loadError = e
        }
        break
      case 'arm':
        localFileExisted = existsSync(join(__dirname, 'neuro-divergent.android-arm-eabi.node'))
        try {
          if (localFileExisted) {
            nativeBinding = require('./neuro-divergent.android-arm-eabi.node')
          } else {
            nativeBinding = require('@neural-trader/neuro-divergent-android-arm-eabi')
          }
        } catch (e) {
          loadError = e
        }
        break
      default:
        throw new Error(`Unsupported architecture on Android ${arch}`)
    }
    break
  case 'win32':
    switch (arch) {
      case 'x64':
        localFileExisted = existsSync(
          join(__dirname, 'neuro-divergent.win32-x64-msvc.node')
        )
        try {
          if (localFileExisted) {
            nativeBinding = require('./neuro-divergent.win32-x64-msvc.node')
          } else {
            nativeBinding = require('@neural-trader/neuro-divergent-win32-x64-msvc')
          }
        } catch (e) {
          loadError = e
        }
        break
      case 'ia32':
        localFileExisted = existsSync(
          join(__dirname, 'neuro-divergent.win32-ia32-msvc.node')
        )
        try {
          if (localFileExisted) {
            nativeBinding = require('./neuro-divergent.win32-ia32-msvc.node')
          } else {
            nativeBinding = require('@neural-trader/neuro-divergent-win32-ia32-msvc')
          }
        } catch (e) {
          loadError = e
        }
        break
      case 'arm64':
        localFileExisted = existsSync(
          join(__dirname, 'neuro-divergent.win32-arm64-msvc.node')
        )
        try {
          if (localFileExisted) {
            nativeBinding = require('./neuro-divergent.win32-arm64-msvc.node')
          } else {
            nativeBinding = require('@neural-trader/neuro-divergent-win32-arm64-msvc')
          }
        } catch (e) {
          loadError = e
        }
        break
      default:
        throw new Error(`Unsupported architecture on Windows: ${arch}`)
    }
    break
  case 'darwin':
    localFileExisted = existsSync(join(__dirname, 'neuro-divergent.darwin-universal.node'))
    try {
      if (localFileExisted) {
        nativeBinding = require('./neuro-divergent.darwin-universal.node')
      } else {
        nativeBinding = require('@neural-trader/neuro-divergent-darwin-universal')
      }
      break
    } catch {}
    switch (arch) {
      case 'x64':
        localFileExisted = existsSync(join(__dirname, 'neuro-divergent.darwin-x64.node'))
        try {
          if (localFileExisted) {
            nativeBinding = require('./neuro-divergent.darwin-x64.node')
          } else {
            nativeBinding = require('@neural-trader/neuro-divergent-darwin-x64')
          }
        } catch (e) {
          loadError = e
        }
        break
      case 'arm64':
        localFileExisted = existsSync(
          join(__dirname, 'neuro-divergent.darwin-arm64.node')
        )
        try {
          if (localFileExisted) {
            nativeBinding = require('./neuro-divergent.darwin-arm64.node')
          } else {
            nativeBinding = require('@neural-trader/neuro-divergent-darwin-arm64')
          }
        } catch (e) {
          loadError = e
        }
        break
      default:
        throw new Error(`Unsupported architecture on macOS: ${arch}`)
    }
    break
  case 'freebsd':
    if (arch !== 'x64') {
      throw new Error(`Unsupported architecture on FreeBSD: ${arch}`)
    }
    localFileExisted = existsSync(join(__dirname, 'neuro-divergent.freebsd-x64.node'))
    try {
      if (localFileExisted) {
        nativeBinding = require('./neuro-divergent.freebsd-x64.node')
      } else {
        nativeBinding = require('@neural-trader/neuro-divergent-freebsd-x64')
      }
    } catch (e) {
      loadError = e
    }
    break
  case 'linux':
    switch (arch) {
      case 'x64':
        if (isMusl()) {
          localFileExisted = existsSync(
            join(__dirname, 'neuro-divergent.linux-x64-musl.node')
          )
          try {
            if (localFileExisted) {
              nativeBinding = require('./neuro-divergent.linux-x64-musl.node')
            } else {
              nativeBinding = require('@neural-trader/neuro-divergent-linux-x64-musl')
            }
          } catch (e) {
            loadError = e
          }
        } else {
          localFileExisted = existsSync(
            join(__dirname, 'neuro-divergent.linux-x64-gnu.node')
          )
          try {
            if (localFileExisted) {
              nativeBinding = require('./neuro-divergent.linux-x64-gnu.node')
            } else {
              nativeBinding = require('@neural-trader/neuro-divergent-linux-x64-gnu')
            }
          } catch (e) {
            loadError = e
          }
        }
        break
      case 'arm64':
        if (isMusl()) {
          localFileExisted = existsSync(
            join(__dirname, 'neuro-divergent.linux-arm64-musl.node')
          )
          try {
            if (localFileExisted) {
              nativeBinding = require('./neuro-divergent.linux-arm64-musl.node')
            } else {
              nativeBinding = require('@neural-trader/neuro-divergent-linux-arm64-musl')
            }
          } catch (e) {
            loadError = e
          }
        } else {
          localFileExisted = existsSync(
            join(__dirname, 'neuro-divergent.linux-arm64-gnu.node')
          )
          try {
            if (localFileExisted) {
              nativeBinding = require('./neuro-divergent.linux-arm64-gnu.node')
            } else {
              nativeBinding = require('@neural-trader/neuro-divergent-linux-arm64-gnu')
            }
          } catch (e) {
            loadError = e
          }
        }
        break
      case 'arm':
        localFileExisted = existsSync(
          join(__dirname, 'neuro-divergent.linux-arm-gnueabihf.node')
        )
        try {
          if (localFileExisted) {
            nativeBinding = require('./neuro-divergent.linux-arm-gnueabihf.node')
          } else {
            nativeBinding = require('@neural-trader/neuro-divergent-linux-arm-gnueabihf')
          }
        } catch (e) {
          loadError = e
        }
        break
      default:
        throw new Error(`Unsupported architecture on Linux: ${arch}`)
    }
    break
  default:
    throw new Error(`Unsupported OS: ${platform}, architecture: ${arch}`)
}

if (!nativeBinding) {
  if (loadError) {
    throw loadError
  }
  throw new Error(`Failed to load native binding`)
}

const { NeuralForecast, ModelType, listAvailableModels, version, isGpuAvailable } = nativeBinding

module.exports.NeuralForecast = NeuralForecast
module.exports.ModelType = ModelType
module.exports.listAvailableModels = listAvailableModels
module.exports.version = version
module.exports.isGpuAvailable = isGpuAvailable
