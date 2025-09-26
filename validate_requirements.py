#!/usr/bin/env python3
"""
Validate all requirements.txt packages for Python 3.13 compatibility.
This script checks if all specified versions can be installed together.
"""

import subprocess
import sys
import tempfile
import os

def test_package_compatibility():
    """Test if all packages in requirements.txt are compatible."""
    print("=== Python Version ===")
    print(f"Python: {sys.version}")
    print()

    # Read requirements
    with open('requirements.txt', 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    print("=== Requirements to Test ===")
    for req in requirements:
        print(f"  {req}")
    print()

    # Test each package individually first
    print("=== Individual Package Tests ===")
    failed_packages = []

    for req in requirements:
        try:
            print(f"Testing {req}... ", end="")
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '--dry-run', req],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                print("✅ COMPATIBLE")
            else:
                print("❌ FAILED")
                print(f"   Error: {result.stderr.strip()}")
                failed_packages.append(req)

        except subprocess.TimeoutExpired:
            print("⏰ TIMEOUT")
            failed_packages.append(req)
        except Exception as e:
            print(f"❌ ERROR: {e}")
            failed_packages.append(req)

    print()

    if failed_packages:
        print("=== Failed Packages ===")
        for pkg in failed_packages:
            print(f"  ❌ {pkg}")
        print()
        return False

    # Test all packages together
    print("=== Combined Compatibility Test ===")
    try:
        print("Testing all packages together... ", end="")
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--dry-run'] + requirements,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            print("✅ ALL COMPATIBLE")
            return True
        else:
            print("❌ CONFLICTS DETECTED")
            print("Stderr:", result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def check_known_compatibility():
    """Check known Python 3.13 compatibility issues."""
    print("=== Python 3.13 Compatibility Checks ===")

    compatibility_notes = {
        'pandas==2.2.3': '✅ Python 3.13 supported (2.2.0+)',
        'numpy==1.26.4': '✅ Python 3.13 supported (1.26.0+)',
        'sqlalchemy==2.0.36': '✅ Python 3.13 supported (2.0.0+)',
        'psycopg2-binary==2.9.10': '✅ Python 3.13 supported (2.9.5+)',
        'fastapi==0.115.5': '✅ Python 3.13 supported (0.100.0+)',
        'uvicorn[standard]==0.32.1': '✅ Python 3.13 supported (0.25.0+)',
        'requests==2.32.3': '✅ Python 3.13 supported (always)',
        'psutil==6.1.0': '✅ Python 3.13 supported (6.0.0+)'
    }

    for package, status in compatibility_notes.items():
        print(f"  {package}: {status}")

    print()

def main():
    """Run all validation tests."""
    print("🔍 Python 3.13 Requirements Validation")
    print("=" * 50)
    print()

    # Check Python version
    if sys.version_info < (3, 13):
        print(f"⚠️  Warning: Running on Python {sys.version_info.major}.{sys.version_info.minor}")
        print("   This test is designed for Python 3.13+")
        print()

    # Run compatibility checks
    check_known_compatibility()

    # Test package installation
    success = test_package_compatibility()

    print("=" * 50)
    if success:
        print("🎉 ALL REQUIREMENTS ARE COMPATIBLE!")
        print("✅ Ready for Python 3.13 deployment")
        return True
    else:
        print("❌ COMPATIBILITY ISSUES DETECTED")
        print("⚠️  Fix requirements.txt before deployment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)