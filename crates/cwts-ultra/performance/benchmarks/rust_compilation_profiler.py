#!/usr/bin/env python3
"""
Rust Compilation and Linking Performance Profiler
Specialized analysis for CWTS Rust codebase optimization
"""

import subprocess
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import tempfile
import shutil

class RustCompilationProfiler:
    """Profile Rust compilation performance and optimization opportunities"""
    
    def __init__(self, project_root="/home/kutlu/CWTS/cwts-ultra"):
        self.project_root = Path(project_root)
        self.cargo_manifest = self.project_root / "Cargo.toml"
        self.core_manifest = self.project_root / "core" / "Cargo.toml"
        
    def profile_compilation_configurations(self) -> Dict[str, Dict]:
        """Profile different Rust compilation configurations"""
        configs = {
            "debug": {
                "profile": "dev",
                "lto": "off",
                "codegen-units": 16,
                "opt-level": 0,
                "debug": True,
                "overflow-checks": True
            },
            "release_default": {
                "profile": "release", 
                "lto": "false",
                "codegen-units": 16,
                "opt-level": 3,
                "debug": False,
                "overflow-checks": False
            },
            "release_thin_lto": {
                "profile": "release",
                "lto": "thin", 
                "codegen-units": 16,
                "opt-level": 3,
                "debug": False,
                "overflow-checks": False
            },
            "release_fat_lto": {
                "profile": "release",
                "lto": "fat",
                "codegen-units": 1,
                "opt-level": 3, 
                "debug": False,
                "overflow-checks": False,
                "panic": "abort"
            },
            "release_optimized": {
                "profile": "release",
                "lto": "fat",
                "codegen-units": 1,
                "opt-level": 3,
                "debug": False,
                "overflow-checks": False,
                "panic": "abort",
                "strip": True
            }
        }
        
        results = {}
        
        for config_name, config in configs.items():
            print(f"Profiling compilation configuration: {config_name}")
            
            # Create temporary Cargo.toml with specific configuration
            result = self._measure_compilation_config(config_name, config)
            results[config_name] = result
            
        return results
    
    def _measure_compilation_config(self, config_name: str, config: Dict) -> Dict:
        """Measure compilation performance for specific configuration"""
        
        # Clean previous builds
        self._run_cargo_clean()
        
        # Create custom profile configuration
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(self._generate_cargo_toml(config))
            temp_cargo_file = f.name
        
        try:
            # Copy original and replace with test configuration
            original_cargo = self.cargo_manifest.read_text()
            shutil.copy(temp_cargo_file, str(self.cargo_manifest))
            
            # Measure compilation time
            compile_start = time.perf_counter()
            compile_result = self._run_cargo_build(config["profile"])
            compile_time = time.perf_counter() - compile_start
            
            # Measure binary size
            binary_size = self._get_binary_size(config["profile"])
            
            # Analyze compilation output
            compilation_analysis = self._analyze_compilation_output(compile_result)
            
            result = {
                "compile_time_seconds": compile_time,
                "binary_size_bytes": binary_size,
                "success": compile_result["success"],
                "warnings_count": compilation_analysis["warnings"],
                "llvm_ir_size": compilation_analysis.get("llvm_ir_size", 0),
                "crate_count": compilation_analysis.get("crates_compiled", 0),
                "incremental_compile_enabled": config.get("incremental", True),
                "config": config
            }
            
        finally:
            # Restore original Cargo.toml
            self.cargo_manifest.write_text(original_cargo)
            os.unlink(temp_cargo_file)
        
        return result
    
    def analyze_dependency_compilation_impact(self) -> Dict[str, Dict]:
        """Analyze compilation impact of different dependencies"""
        
        # Read current dependencies
        cargo_content = self.cargo_manifest.read_text()
        
        # Key dependencies to analyze
        dependencies_to_test = [
            {"name": "tokio", "optional": True},
            {"name": "rayon", "optional": False}, 
            {"name": "candle-core", "optional": True},
            {"name": "prometheus", "optional": False},
            {"name": "wide", "optional": True}
        ]
        
        results = {}
        
        # Baseline compilation (all dependencies)
        baseline_time = self._measure_baseline_compilation()
        results["baseline"] = {"compile_time": baseline_time}
        
        for dep in dependencies_to_test:
            if dep["optional"]:
                # Test without optional dependency
                compile_time = self._measure_without_dependency(dep["name"])
                results[f"without_{dep['name']}"] = {
                    "compile_time": compile_time,
                    "time_saved": baseline_time - compile_time,
                    "percentage_improvement": ((baseline_time - compile_time) / baseline_time) * 100
                }
        
        return results
    
    def profile_incremental_compilation(self) -> Dict[str, float]:
        """Profile incremental compilation performance"""
        
        results = {}
        
        # Full clean build
        self._run_cargo_clean()
        full_build_time = self._measure_full_build()
        results["full_build"] = full_build_time
        
        # No-op rebuild (no changes)
        noop_rebuild_time = self._measure_noop_rebuild()
        results["noop_rebuild"] = noop_rebuild_time
        
        # Small change rebuild
        small_change_time = self._measure_small_change_rebuild()
        results["small_change_rebuild"] = small_change_time
        
        # Large change rebuild  
        large_change_time = self._measure_large_change_rebuild()
        results["large_change_rebuild"] = large_change_time
        
        return results
    
    def analyze_llvm_optimization_impact(self) -> Dict[str, Dict]:
        """Analyze LLVM optimization passes impact"""
        
        opt_levels = ["0", "1", "2", "3", "s", "z"]
        results = {}
        
        for opt_level in opt_levels:
            print(f"Testing LLVM optimization level: O{opt_level}")
            
            config = {
                "profile": "release",
                "opt-level": opt_level,
                "lto": "off",
                "codegen-units": 1
            }
            
            result = self._measure_compilation_config(f"opt_{opt_level}", config)
            
            # Add optimization-specific analysis
            result["optimization_level"] = opt_level
            result["size_vs_speed_tradeoff"] = self._analyze_size_speed_tradeoff(result)
            
            results[f"opt_{opt_level}"] = result
            
        return results
    
    def profile_target_cpu_optimizations(self) -> Dict[str, Dict]:
        """Profile target-cpu specific optimizations"""
        
        target_cpus = [
            "generic",
            "native", 
            "x86-64",
            "x86-64-v2",
            "x86-64-v3", 
            "haswell",
            "skylake",
            "znver2",
            "znver3"
        ]
        
        results = {}
        
        for target_cpu in target_cpus:
            print(f"Testing target-cpu: {target_cpu}")
            
            try:
                result = self._measure_target_cpu_compilation(target_cpu)
                results[target_cpu] = result
            except Exception as e:
                results[target_cpu] = {"error": str(e)}
                
        return results
    
    def _run_cargo_clean(self):
        """Clean cargo build artifacts"""
        subprocess.run(["cargo", "clean"], cwd=self.project_root, capture_output=True)
    
    def _run_cargo_build(self, profile="release") -> Dict:
        """Run cargo build and capture output"""
        
        cmd = ["cargo", "build"]
        if profile == "release":
            cmd.append("--release")
        
        start_time = time.perf_counter()
        result = subprocess.run(
            cmd, 
            cwd=self.project_root, 
            capture_output=True, 
            text=True
        )
        build_time = time.perf_counter() - start_time
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr, 
            "build_time": build_time,
            "return_code": result.returncode
        }
    
    def _get_binary_size(self, profile="release") -> int:
        """Get size of compiled binary"""
        
        if profile == "release":
            binary_path = self.project_root / "target" / "release" 
        else:
            binary_path = self.project_root / "target" / "debug"
        
        total_size = 0
        
        # Find all binaries and libraries
        for ext in ['', '.exe', '.so', '.dylib', '.dll']:
            for binary_file in binary_path.glob(f"*{ext}"):
                if binary_file.is_file() and not binary_file.is_symlink():
                    total_size += binary_file.stat().st_size
        
        return total_size
    
    def _analyze_compilation_output(self, compile_result: Dict) -> Dict:
        """Analyze compilation output for insights"""
        
        stdout = compile_result.get("stdout", "")
        stderr = compile_result.get("stderr", "")
        
        # Count warnings
        warnings = stdout.count("warning:") + stderr.count("warning:")
        
        # Count compiled crates
        crates_compiled = stdout.count("Compiling")
        
        # Extract timing information if available
        timing_info = {}
        if "--timings" in stdout:
            # Parse cargo timings output
            timing_info = self._parse_cargo_timings(stdout)
        
        return {
            "warnings": warnings,
            "crates_compiled": crates_compiled,
            "timing_info": timing_info
        }
    
    def _generate_cargo_toml(self, config: Dict) -> str:
        """Generate Cargo.toml content for specific configuration"""
        
        # Read original content
        original_content = self.cargo_manifest.read_text()
        
        # Modify profile section
        profile_section = f"""
[profile.{config['profile']}]
opt-level = {config.get('opt-level', 3)}
lto = "{config.get('lto', 'false')}"
codegen-units = {config.get('codegen-units', 1)}
panic = "{config.get('panic', 'unwind')}"
overflow-checks = {str(config.get('overflow-checks', False)).lower()}
debug = {str(config.get('debug', False)).lower()}
"""
        
        if config.get('strip'):
            profile_section += 'strip = true\n'
        
        # Replace or append profile section
        if f"[profile.{config['profile']}]" in original_content:
            # Replace existing profile section
            lines = original_content.split('\n')
            new_lines = []
            in_profile = False
            
            for line in lines:
                if line.startswith(f"[profile.{config['profile']}]"):
                    in_profile = True
                    new_lines.extend(profile_section.strip().split('\n'))
                    continue
                elif line.startswith('[') and in_profile:
                    in_profile = False
                
                if not in_profile:
                    new_lines.append(line)
            
            return '\n'.join(new_lines)
        else:
            # Append profile section
            return original_content + profile_section
    
    def _measure_baseline_compilation(self) -> float:
        """Measure baseline compilation time with all dependencies"""
        self._run_cargo_clean()
        result = self._run_cargo_build("release")
        return result["build_time"]
    
    def _measure_without_dependency(self, dep_name: str) -> float:
        """Measure compilation time without specific dependency"""
        
        # Create temporary Cargo.toml without the dependency
        original_content = self.cargo_manifest.read_text()
        
        # Simple removal (in production, would be more sophisticated)
        modified_content = original_content
        lines = modified_content.split('\n')
        filtered_lines = [line for line in lines if dep_name not in line]
        modified_content = '\n'.join(filtered_lines)
        
        try:
            self.cargo_manifest.write_text(modified_content)
            self._run_cargo_clean()
            result = self._run_cargo_build("release")
            return result["build_time"]
        finally:
            self.cargo_manifest.write_text(original_content)
    
    def _measure_full_build(self) -> float:
        """Measure full clean build time"""
        result = self._run_cargo_build("release")
        return result["build_time"]
    
    def _measure_noop_rebuild(self) -> float:
        """Measure no-op rebuild time"""
        result = self._run_cargo_build("release")
        return result["build_time"]
    
    def _measure_small_change_rebuild(self) -> float:
        """Measure rebuild time after small change"""
        
        # Make small change to a file
        test_file = self.project_root / "core" / "src" / "lib.rs"
        original_content = test_file.read_text()
        
        try:
            # Add a comment
            modified_content = original_content + "\n// Small change for rebuild test\n"
            test_file.write_text(modified_content)
            
            result = self._run_cargo_build("release")
            return result["build_time"]
        finally:
            test_file.write_text(original_content)
    
    def _measure_large_change_rebuild(self) -> float:
        """Measure rebuild time after large change"""
        
        # Make change to core algorithm file
        test_file = self.project_root / "core" / "src" / "algorithms" / "hft_algorithms.rs"
        original_content = test_file.read_text()
        
        try:
            # Add significant code change
            addition = """
// Large change for rebuild test
pub fn benchmark_test_function() -> f64 {
    let mut result = 0.0;
    for i in 0..1000 {
        result += (i as f64).sqrt();
    }
    result
}
"""
            modified_content = original_content + addition
            test_file.write_text(modified_content)
            
            result = self._run_cargo_build("release") 
            return result["build_time"]
        finally:
            test_file.write_text(original_content)
    
    def _measure_target_cpu_compilation(self, target_cpu: str) -> Dict:
        """Measure compilation with specific target CPU"""
        
        # Set RUSTFLAGS environment variable
        env = os.environ.copy()
        env["RUSTFLAGS"] = f"-C target-cpu={target_cpu}"
        
        self._run_cargo_clean()
        
        start_time = time.perf_counter()
        result = subprocess.run(
            ["cargo", "build", "--release"],
            cwd=self.project_root,
            capture_output=True,
            text=True,
            env=env
        )
        compile_time = time.perf_counter() - start_time
        
        binary_size = self._get_binary_size("release")
        
        return {
            "compile_time": compile_time,
            "binary_size": binary_size,
            "success": result.returncode == 0,
            "target_cpu": target_cpu,
            "stderr": result.stderr if result.returncode != 0 else ""
        }
    
    def _analyze_size_speed_tradeoff(self, result: Dict) -> Dict:
        """Analyze size vs speed tradeoff for optimization"""
        
        compile_time = result.get("compile_time_seconds", 0)
        binary_size = result.get("binary_size_bytes", 0)
        
        # Simple heuristics for tradeoff analysis
        size_efficiency = 1.0 / (binary_size / 1024 / 1024 + 1)  # Smaller is better
        speed_efficiency = 1.0 / (compile_time + 1)  # Faster compilation is better
        
        return {
            "size_efficiency_score": size_efficiency,
            "speed_efficiency_score": speed_efficiency,
            "combined_score": (size_efficiency + speed_efficiency) / 2,
            "size_mb": binary_size / 1024 / 1024,
            "compile_time_minutes": compile_time / 60
        }
    
    def _parse_cargo_timings(self, output: str) -> Dict:
        """Parse cargo timing information"""
        # Placeholder for cargo timing parsing
        return {"timing_data": "parsed"}

def main():
    """Run Rust compilation profiling"""
    print("🦀 Rust Compilation Performance Profiler")
    print("=" * 50)
    
    profiler = RustCompilationProfiler()
    results = {}
    
    print("\n1. Profiling compilation configurations...")
    results["compilation_configs"] = profiler.profile_compilation_configurations()
    
    print("\n2. Analyzing dependency impact...")
    results["dependency_impact"] = profiler.analyze_dependency_compilation_impact()
    
    print("\n3. Profiling incremental compilation...")
    results["incremental_compilation"] = profiler.profile_incremental_compilation()
    
    print("\n4. Analyzing LLVM optimization levels...")
    results["llvm_optimizations"] = profiler.analyze_llvm_optimization_impact()
    
    print("\n5. Profiling target CPU optimizations...")
    results["target_cpu_optimizations"] = profiler.profile_target_cpu_optimizations()
    
    # Save results
    output_file = "/home/kutlu/CWTS/cwts-ultra/performance/benchmarks/rust_compilation_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Rust compilation analysis complete. Results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    main()