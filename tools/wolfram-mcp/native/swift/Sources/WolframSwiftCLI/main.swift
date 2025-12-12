import Foundation
import WolframSwift

/// CLI for Wolfram Swift bridge - can be called from Bun/Node via subprocess
@main
struct WolframSwiftCLI {
    static func main() async {
        let args = CommandLine.arguments
        
        guard args.count >= 2 else {
            printUsage()
            exit(1)
        }
        
        let bridge = WolframBridge()
        
        let command = args[1]
        
        do {
            switch command {
            case "eval", "execute":
                guard args.count >= 3 else {
                    print("Error: Missing code argument")
                    exit(1)
                }
                let code = args[2]
                let timeout = args.count > 3 ? Double(args[3]) ?? 30 : 30
                let result = try await bridge.execute(code: code, timeout: timeout)
                printJSON(result)
                
            case "hyperbolic-distance":
                guard args.count >= 6 else {
                    print("Error: Need x1 y1 x2 y2 arguments")
                    exit(1)
                }
                let p1 = PoincarePoint(x: Double(args[2])!, y: Double(args[3])!)
                let p2 = PoincarePoint(x: Double(args[4])!, y: Double(args[5])!)
                let distance = bridge.hyperbolicDistance(p1, p2)
                print(distance)
                
            case "mobius-add":
                guard args.count >= 6 else {
                    print("Error: Need x1 y1 x2 y2 arguments")
                    exit(1)
                }
                let a = PoincarePoint(x: Double(args[2])!, y: Double(args[3])!)
                let b = PoincarePoint(x: Double(args[4])!, y: Double(args[5])!)
                let result = bridge.mobiusAdd(a, b)
                printJSON(result)
                
            case "stdp":
                guard args.count >= 3 else {
                    print("Error: Need deltaT argument")
                    exit(1)
                }
                let deltaT = Double(args[2])!
                let update = bridge.stdpWeightUpdate(deltaT: deltaT)
                print(update)
                
            case "entropy":
                guard args.count >= 3 else {
                    print("Error: Need probabilities as space-separated values")
                    exit(1)
                }
                let probs = args[2...].compactMap { Double($0) }
                let entropy = bridge.shannonEntropy(probs)
                print(entropy)
                
            case "softmax":
                guard args.count >= 3 else {
                    print("Error: Need values as space-separated numbers")
                    exit(1)
                }
                let values = args[2...].compactMap { Double($0) }
                let result = bridge.softmax(values)
                printJSON(result)
                
            case "lmsr":
                guard args.count >= 4 else {
                    print("Error: Need b and quantities")
                    exit(1)
                }
                let b = Double(args[2])!
                let quantities = args[3...].compactMap { Double($0) }
                let cost = bridge.lmsrCost(quantities, b: b)
                print(cost)
                
            case "landauer":
                guard args.count >= 3 else {
                    print("Error: Need temperature in Kelvin")
                    exit(1)
                }
                let temp = Double(args[2])!
                let bound = bridge.landauerBound(temperatureKelvin: temp)
                print(bound)
                
            case "check":
                let available = bridge.isAvailable()
                print(available ? "available" : "unavailable")
                exit(available ? 0 : 1)
                
            case "version":
                print("wolfram-swift-cli v1.0.0")
                
            default:
                print("Unknown command: \(command)")
                printUsage()
                exit(1)
            }
        } catch {
            print("Error: \(error.localizedDescription)")
            exit(1)
        }
    }
    
    static func printUsage() {
        print("""
        Usage: wolfram-swift-cli <command> [args...]
        
        Commands:
          eval <code> [timeout]     Execute WolframScript code
          hyperbolic-distance x1 y1 x2 y2   Compute hyperbolic distance
          mobius-add x1 y1 x2 y2    Compute Möbius addition
          stdp <deltaT>             Compute STDP weight update
          entropy <p1> <p2> ...     Compute Shannon entropy
          softmax <v1> <v2> ...     Compute softmax
          lmsr <b> <q1> <q2> ...    Compute LMSR cost
          landauer <temp_K>         Compute Landauer bound
          check                     Check WolframScript availability
          version                   Print version
        """)
    }
    
    static func printJSON<T: Encodable>(_ value: T) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(value),
           let json = String(data: data, encoding: .utf8) {
            print(json)
        }
    }
}
