// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "WolframSwift",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "WolframSwift",
            type: .dynamic,
            targets: ["WolframSwift"]
        ),
        .executable(
            name: "wolfram-swift-cli",
            targets: ["WolframSwiftCLI"]
        )
    ],
    dependencies: [],
    targets: [
        .target(
            name: "WolframSwift",
            dependencies: [],
            path: "Sources/WolframSwift"
        ),
        .executableTarget(
            name: "WolframSwiftCLI",
            dependencies: ["WolframSwift"],
            path: "Sources/WolframSwiftCLI"
        )
    ]
)
