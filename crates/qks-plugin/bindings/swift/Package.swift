// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "QKS",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
        .tvOS(.v16),
        .watchOS(.v9)
    ],
    products: [
        .library(
            name: "QKS",
            targets: ["QKS"]
        ),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "QKS",
            dependencies: [],
            path: "Sources/QKS",
            linkerSettings: [
                .linkedLibrary("qks_plugin"),
                .unsafeFlags(["-L../../../../target/release"], .when(configuration: .release)),
                .unsafeFlags(["-L../../../../target/debug"], .when(configuration: .debug)),
            ]
        ),
        .testTarget(
            name: "QKSTests",
            dependencies: ["QKS"],
            path: "Tests/QKSTests"
        ),
    ],
    swiftLanguageVersions: [.v5]
)
