#!/usr/bin/env python3
"""
Build script for the Rust Market Analysis System
"""

import subprocess
import sys
import os
from pathlib import Path
import shutil

def check_rust_installed():
    """Check if Rust is installed"""
    try:
        result = subprocess.run(['rustc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Rust detected: {result.stdout.strip()}")
            return True
        else:
            print("❌ Rust not found")
            return False
    except FileNotFoundError:
        print("❌ Rust not found")
        return False

def install_rust():
    """Install Rust using rustup"""
    print("🔧 Installing Rust...")
    try:
        # Download and run rustup
        subprocess.check_call([
            'curl', '--proto', '=https', '--tlsv1.2', '-sSf', 
            'https://sh.rustup.rs', '-o', '/tmp/rustup.sh'
        ])
        subprocess.check_call(['sh', '/tmp/rustup.sh', '-y'])
        
        # Add cargo to PATH
        cargo_path = Path.home() / '.cargo' / 'bin'
        os.environ['PATH'] = f"{cargo_path}:{os.environ['PATH']}"
        
        print("✅ Rust installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Rust: {e}")
        return False

def install_maturin():
    """Install maturin for Python bindings"""
    print("🔧 Installing maturin...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'maturin'])
        print("✅ Maturin installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install maturin: {e}")
        return False

def build_rust_module():
    """Build the Rust module"""
    print("🏗️  Building Rust market analysis module...")
    
    rust_dir = Path(__file__).parent
    
    try:
        # Build in release mode for maximum performance
        result = subprocess.run(
            ['maturin', 'build', '--release', '--out', 'dist'],
            cwd=rust_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Rust module built successfully!")
            print(result.stdout)
            return True
        else:
            print(f"❌ Build failed: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ maturin not found. Installing...")
        if install_maturin():
            return build_rust_module()
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Build error: {e}")
        return False

def install_wheel():
    """Install the built wheel"""
    print("📦 Installing wheel...")
    
    dist_dir = Path(__file__).parent / 'dist'
    wheels = list(dist_dir.glob('*.whl'))
    
    if not wheels:
        print("❌ No wheel files found in dist/")
        return False
    
    # Install the most recent wheel
    latest_wheel = max(wheels, key=lambda p: p.stat().st_mtime)
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '--force-reinstall', str(latest_wheel)
        ])
        print(f"✅ Installed {latest_wheel.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False

def develop_mode():
    """Install in development mode"""
    print("🔧 Installing in development mode...")
    
    rust_dir = Path(__file__).parent
    
    try:
        subprocess.check_call(['maturin', 'develop', '--release'], cwd=rust_dir)
        print("✅ Development installation successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Development installation failed: {e}")
        return False

def run_tests():
    """Run Rust tests"""
    print("🧪 Running Rust tests...")
    
    rust_dir = Path(__file__).parent
    
    try:
        result = subprocess.run(['cargo', 'test'], cwd=rust_dir, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All tests passed!")
            print(result.stdout)
            return True
        else:
            print(f"❌ Tests failed: {result.stderr}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Test error: {e}")
        return False

def benchmark():
    """Run performance benchmarks"""
    print("⚡ Running benchmarks...")
    
    rust_dir = Path(__file__).parent
    
    try:
        subprocess.check_call(['cargo', 'run', '--release', '--bin', 'market_analyzer', 'benchmark'], cwd=rust_dir)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Benchmark error: {e}")
        return False

def clean():
    """Clean build artifacts"""
    print("🧹 Cleaning build artifacts...")
    
    rust_dir = Path(__file__).parent
    
    # Remove Rust build artifacts
    target_dir = rust_dir / 'target'
    if target_dir.exists():
        shutil.rmtree(target_dir)
        print("   Removed target/")
    
    # Remove Python build artifacts
    dist_dir = rust_dir / 'dist'
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
        print("   Removed dist/")
    
    build_dir = rust_dir / 'build'
    if build_dir.exists():
        shutil.rmtree(build_dir)
        print("   Removed build/")
    
    print("✅ Clean completed")

def main():
    """Main build script"""
    print("🚀 Rust Market Analysis System Build Script")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("Usage: python build.py <command>")
        print()
        print("Commands:")
        print("  setup     - Setup Rust and dependencies")
        print("  build     - Build the Rust module")
        print("  install   - Build and install wheel")
        print("  develop   - Install in development mode")
        print("  test      - Run tests")
        print("  benchmark - Run performance benchmarks") 
        print("  clean     - Clean build artifacts")
        print("  all       - Run setup, build, and install")
        return
    
    command = sys.argv[1]
    
    if command == "setup":
        if not check_rust_installed():
            if not install_rust():
                return
        if not install_maturin():
            return
        print("✅ Setup completed!")
        
    elif command == "build":
        if not build_rust_module():
            return
            
    elif command == "install":
        if not build_rust_module():
            return
        if not install_wheel():
            return
            
    elif command == "develop":
        if not develop_mode():
            return
            
    elif command == "test":
        if not run_tests():
            return
            
    elif command == "benchmark":
        if not benchmark():
            return
            
    elif command == "clean":
        clean()
        
    elif command == "all":
        # Full setup, build, and install
        if not check_rust_installed():
            if not install_rust():
                return
        if not install_maturin():
            return
        if not build_rust_module():
            return
        if not develop_mode():
            return
        if not run_tests():
            return
        print("🎉 Full build completed successfully!")
        
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()