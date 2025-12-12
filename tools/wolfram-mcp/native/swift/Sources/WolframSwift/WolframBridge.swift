import Foundation

/// Result from Wolfram computation
public struct WolframResult: Codable, Sendable {
    public let success: Bool
    public let result: String
    public let error: String?
    public let computationTimeMs: Double
    
    public init(success: Bool, result: String, error: String? = nil, computationTimeMs: Double) {
        self.success = success
        self.result = result
        self.error = error
        self.computationTimeMs = computationTimeMs
    }
}

/// Point in Poincaré disk model
public struct PoincarePoint: Codable, Sendable {
    public let x: Double
    public let y: Double
    
    public init(x: Double, y: Double) {
        self.x = x
        self.y = y
    }
    
    /// Complex representation
    public var asComplex: (re: Double, im: Double) {
        (x, y)
    }
    
    /// Squared norm
    public var normSquared: Double {
        x * x + y * y
    }
    
    /// Check if point is inside unit disk
    public var isValid: Bool {
        normSquared < 1.0
    }
}

/// Native Swift bridge for Wolfram computations
public actor WolframBridge {
    
    private let wolframScriptPath: String
    private let defaultTimeout: TimeInterval
    
    public init(wolframScriptPath: String = "/usr/local/bin/wolframscript", timeout: TimeInterval = 30) {
        self.wolframScriptPath = wolframScriptPath
        self.defaultTimeout = timeout
    }
    
    /// Execute WolframScript code locally
    public func execute(code: String, timeout: TimeInterval? = nil) async throws -> WolframResult {
        let start = CFAbsoluteTimeGetCurrent()
        
        let process = Process()
        process.executableURL = URL(fileURLWithPath: wolframScriptPath)
        process.arguments = ["-code", code]
        
        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        
        do {
            try process.run()
            
            // Wait with timeout
            let effectiveTimeout = timeout ?? defaultTimeout
            let deadline = Date().addingTimeInterval(effectiveTimeout)
            
            while process.isRunning && Date() < deadline {
                try await Task.sleep(nanoseconds: 100_000_000) // 100ms
            }
            
            if process.isRunning {
                process.terminate()
                throw WolframError.timeout(effectiveTimeout)
            }
            
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
            
            if process.terminationStatus == 0 {
                let outputData = stdout.fileHandleForReading.readDataToEndOfFile()
                let output = String(data: outputData, encoding: .utf8) ?? ""
                
                // Filter progress messages
                let filteredOutput = output
                    .components(separatedBy: .newlines)
                    .filter { line in
                        !line.contains("Loading from Wolfram") &&
                        !line.contains("Prefetching") &&
                        !line.contains("Connecting")
                    }
                    .joined(separator: "\n")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                
                return WolframResult(
                    success: true,
                    result: filteredOutput,
                    computationTimeMs: elapsed
                )
            } else {
                let errorData = stderr.fileHandleForReading.readDataToEndOfFile()
                let errorOutput = String(data: errorData, encoding: .utf8) ?? "Unknown error"
                
                return WolframResult(
                    success: false,
                    result: "",
                    error: errorOutput,
                    computationTimeMs: elapsed
                )
            }
        } catch let error as WolframError {
            throw error
        } catch {
            throw WolframError.executionFailed(error.localizedDescription)
        }
    }
    
    /// Compute hyperbolic distance in Poincaré disk
    public nonisolated func hyperbolicDistance(_ p1: PoincarePoint, _ p2: PoincarePoint) -> Double {
        guard p1.isValid && p2.isValid else { return .infinity }
        
        let dx = p1.x - p2.x
        let dy = p1.y - p2.y
        let diffNormSq = dx * dx + dy * dy
        
        let denom = sqrt((1 - p1.normSquared) * (1 - p2.normSquared) + diffNormSq)
        let ratio = sqrt(diffNormSq) / denom
        
        return 2.0 * atanh(ratio)
    }
    
    /// Möbius addition in Poincaré disk
    public nonisolated func mobiusAdd(_ a: PoincarePoint, _ b: PoincarePoint) -> PoincarePoint {
        // (a + b) / (1 + conj(a) * b)
        let numRe = a.x + b.x
        let numIm = a.y + b.y
        
        let denomRe = 1.0 + a.x * b.x + a.y * b.y
        let denomIm = a.x * b.y - a.y * b.x
        
        let denomNormSq = denomRe * denomRe + denomIm * denomIm
        
        return PoincarePoint(
            x: (numRe * denomRe + numIm * denomIm) / denomNormSq,
            y: (numIm * denomRe - numRe * denomIm) / denomNormSq
        )
    }
    
    /// Compute STDP weight update
    public nonisolated func stdpWeightUpdate(
        deltaT: Double,
        tauPlus: Double = 20.0,
        tauMinus: Double = 20.0,
        aPlus: Double = 0.1,
        aMinus: Double = 0.12
    ) -> Double {
        if deltaT > 0 {
            return aPlus * exp(-deltaT / tauPlus)
        } else if deltaT < 0 {
            return -aMinus * exp(deltaT / tauMinus)
        }
        return 0
    }
    
    /// Compute Shannon entropy
    public nonisolated func shannonEntropy(_ probabilities: [Double]) -> Double {
        probabilities
            .filter { $0 > 1e-15 }
            .reduce(0) { $0 - $1 * log($1) }
    }
    
    /// Compute softmax
    public nonisolated func softmax(_ values: [Double]) -> [Double] {
        guard let maxVal = values.max() else { return [] }
        let expValues = values.map { exp($0 - maxVal) }
        let sum = expValues.reduce(0, +)
        return expValues.map { $0 / sum }
    }
    
    /// Compute LMSR cost function
    public nonisolated func lmsrCost(_ quantities: [Double], b: Double) -> Double {
        guard let maxQ = quantities.max() else { return 0 }
        let sumExp = quantities.map { exp(($0 - maxQ) / b) }.reduce(0, +)
        return b * (maxQ / b + log(sumExp))
    }
    
    /// Landauer bound for bit erasure
    public nonisolated func landauerBound(temperatureKelvin: Double) -> Double {
        let kB = 1.380649e-23 // Boltzmann constant in J/K
        return kB * temperatureKelvin * log(2.0)
    }
    
    /// Check if WolframScript is available
    public nonisolated func isAvailable() -> Bool {
        FileManager.default.isExecutableFile(atPath: wolframScriptPath)
    }
}

/// Wolfram-specific errors
public enum WolframError: Error, LocalizedError {
    case timeout(TimeInterval)
    case executionFailed(String)
    case notAvailable
    
    public var errorDescription: String? {
        switch self {
        case .timeout(let seconds):
            return "WolframScript timed out after \(seconds) seconds"
        case .executionFailed(let message):
            return "WolframScript execution failed: \(message)"
        case .notAvailable:
            return "WolframScript is not available at the configured path"
        }
    }
}
