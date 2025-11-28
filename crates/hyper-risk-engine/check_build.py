#!/usr/bin/env python3
import subprocess
import sys
import os

os.chdir('/Volumes/Kingston/Developer/Ashina/HyperPhysics')

try:
    result = subprocess.run(
        ['cargo', 'check', '-p', 'hyper-risk-engine', '--lib', '--no-default-features'],
        capture_output=True,
        text=True,
        timeout=120
    )

    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    print(f"\nExit code: {result.returncode}")

    sys.exit(result.returncode)

except subprocess.TimeoutExpired:
    print("Build timed out after 120 seconds")
    sys.exit(1)
except Exception as e:
    print(f"Error running build: {e}")
    sys.exit(1)
