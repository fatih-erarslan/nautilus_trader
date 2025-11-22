#!/usr/bin/env python3
"""
Fix workspace dependencies for autopoiesis project
Ensures all workspace packages have correct dependencies
"""

import os
import toml
from pathlib import Path

def fix_workspace_dependencies():
    """Fix dependencies for all workspace packages"""
    
    workspace_packages = {
        "autopoiesis-core": {
            "description": "Core mathematical and system libraries",
            "dependencies": [
                "anyhow", "async-trait", "serde", "serde_json", "thiserror", 
                "tokio", "tracing", "ndarray", "nalgebra", "num-traits", 
                "num-complex", "statrs", "rand", "rand_distr", "chrono", 
                "uuid", "rust_decimal", "once_cell", "petgraph", 
                "crossbeam-channel", "dashmap"
            ]
        },
        "autopoiesis-ml": {
            "description": "Machine learning components",
            "dependencies": [
                "anyhow", "async-trait", "serde", "serde_json", "thiserror",
                "tokio", "tracing", "autopoiesis-core", "ndarray", "nalgebra",
                "num-traits", "num-complex", "statrs", "rand", "rand_distr",
                "smartcore", "linfa", "linfa-clustering", "linfa-reduction",
                "argmin", "argmin-math", "chrono", "uuid", "rust_decimal",
                "once_cell", "axum", "reqwest", "tower", "tower-http",
                "jsonwebtoken", "url", "crossbeam-channel", "crossbeam-deque",
                "dashmap", "rayon", "bincode", "urlencoding", "sha2"
            ]
        },
        "autopoiesis-consciousness": {
            "description": "Consciousness and awareness systems",
            "dependencies": [
                "anyhow", "async-trait", "serde", "thiserror", "tokio",
                "tracing", "autopoiesis-core", "ndarray", "nalgebra",
                "num-traits", "rand", "chrono", "uuid", "once_cell"
            ]
        },
        "autopoiesis-finance": {
            "description": "Financial domain components",
            "dependencies": [
                "anyhow", "async-trait", "serde", "thiserror", "tokio",
                "tracing", "autopoiesis-core", "rust_decimal", "chrono",
                "uuid", "ta", "statrs", "ndarray"
            ]
        },
        "autopoiesis-engines": {
            "description": "Trading and execution engines",
            "dependencies": [
                "anyhow", "async-trait", "serde", "thiserror", "tokio",
                "tracing", "autopoiesis-core", "autopoiesis-finance", 
                "rust_decimal", "chrono", "uuid", "crossbeam-channel",
                "dashmap", "rayon"
            ]
        },
        "autopoiesis-analysis": {
            "description": "Analysis and observation components",
            "dependencies": [
                "anyhow", "async-trait", "serde", "thiserror", "tokio",
                "tracing", "autopoiesis-core", "ndarray", "nalgebra",
                "statrs", "ta", "chrono", "uuid", "smartcore"
            ]
        },
        "autopoiesis-api": {
            "description": "API and networking components",
            "dependencies": [
                "anyhow", "async-trait", "serde", "serde_json", "thiserror",
                "tokio", "tracing", "autopoiesis-core", "autopoiesis-ml",
                "axum", "reqwest", "tower", "tower-http", "tokio-tungstenite",
                "jsonwebtoken", "url", "chrono", "uuid"
            ]
        },
        "autopoiesis": {
            "description": "Main autopoiesis application",
            "dependencies": [
                "anyhow", "tokio", "tracing", "autopoiesis-core", 
                "autopoiesis-ml", "autopoiesis-consciousness", "autopoiesis-finance",
                "autopoiesis-engines", "autopoiesis-analysis", "autopoiesis-api"
            ]
        }
    }
    
    dev_dependencies = [
        ("tokio-test", "0.4"),
        ("approx", True),
        ("tempfile", True), 
        ("proptest", True),
        ("criterion", True),
        ("pretty_assertions", True),
        ("test-case", True),
        ("quickcheck", True),
        ("quickcheck_macros", True),
        ("serial_test", True)
    ]
    
    for package_name, config in workspace_packages.items():
        package_path = Path(package_name)
        cargo_toml_path = package_path / "Cargo.toml"
        
        if not cargo_toml_path.exists():
            print(f"Creating {cargo_toml_path}")
            package_path.mkdir(exist_ok=True)
            (package_path / "src").mkdir(exist_ok=True)
            (package_path / "src" / "lib.rs").touch()
        
        # Create Cargo.toml content
        cargo_content = {
            "package": {
                "name": package_name,
                "version": {"workspace": True},
                "edition": {"workspace": True},
                "authors": {"workspace": True},
                "license": {"workspace": True},
                "repository": {"workspace": True},
                "description": config["description"]
            },
            "dependencies": {}
        }
        
        # Add dependencies
        for dep in config["dependencies"]:
            if dep.startswith("autopoiesis-"):
                cargo_content["dependencies"][dep] = {"workspace": True}
            else:
                cargo_content["dependencies"][dep] = {"workspace": True}
        
        # Add features for ML package
        if package_name == "autopoiesis-ml":
            cargo_content["features"] = {
                "default": [],
                "benchmarks": ["criterion"],
                "property-tests": ["proptest"]
            }
        else:
            cargo_content["features"] = {"default": []}
        
        # Add dev-dependencies
        dev_deps = {}
        for dep_name, dep_value in dev_dependencies:
            if isinstance(dep_value, bool) and dep_value:
                dev_deps[dep_name] = {"workspace": True}
            else:
                dev_deps[dep_name] = dep_value
        
        cargo_content["dev-dependencies"] = dev_deps
        
        # Write Cargo.toml
        with open(cargo_toml_path, "w") as f:
            toml.dump(cargo_content, f)
        
        print(f"Fixed {cargo_toml_path}")

if __name__ == "__main__":
    fix_workspace_dependencies()
    print("All workspace dependencies fixed!")