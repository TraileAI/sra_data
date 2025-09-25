#!/usr/bin/env python3
"""
Manual API testing script for FastAPI skeleton.

This script provides basic testing functionality when pytest is not available.
"""

import sys
import importlib.util
import traceback
from pathlib import Path

def test_api_imports():
    """Test that API modules can be imported successfully."""
    print("Testing API module imports...")

    try:
        # Test FastAPI skeleton import
        spec = importlib.util.spec_from_file_location(
            "skeleton",
            Path("packages/sra_data/api/skeleton.py")
        )
        if spec and spec.loader:
            skeleton_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(skeleton_module)
            print("✓ FastAPI skeleton module imports successfully")

            # Test app creation function
            if hasattr(skeleton_module, 'create_fastapi_app'):
                print("✓ create_fastapi_app function available")
            else:
                print("✗ create_fastapi_app function not found")

            # Test metadata
            if hasattr(skeleton_module, 'APP_NAME'):
                print(f"✓ APP_NAME defined: {skeleton_module.APP_NAME}")
            if hasattr(skeleton_module, 'APP_VERSION'):
                print(f"✓ APP_VERSION defined: {skeleton_module.APP_VERSION}")

        else:
            print("✗ Failed to load skeleton module spec")
            return False

    except Exception as e:
        print(f"✗ Error importing skeleton module: {e}")
        traceback.print_exc()
        return False

    return True

def test_server_config():
    """Test server configuration."""
    print("\nTesting server configuration...")

    try:
        spec = importlib.util.spec_from_file_location(
            "server",
            Path("server.py")
        )
        if spec and spec.loader:
            server_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(server_module)
            print("✓ Server module imports successfully")

            # Test config function
            if hasattr(server_module, 'get_server_config'):
                config = server_module.get_server_config()
                print(f"✓ Server config generated: {config}")

                # Validate required config keys
                required_keys = ['host', 'port', 'log_level']
                for key in required_keys:
                    if key in config:
                        print(f"✓ Config has {key}: {config[key]}")
                    else:
                        print(f"✗ Config missing {key}")

            else:
                print("✗ get_server_config function not found")

        else:
            print("✗ Failed to load server module spec")
            return False

    except Exception as e:
        print(f"✗ Error importing server module: {e}")
        traceback.print_exc()
        return False

    return True

def test_file_structure():
    """Test that required files exist."""
    print("\nTesting file structure...")

    required_files = [
        "packages/sra_data/api/__init__.py",
        "packages/sra_data/api/skeleton.py",
        "server.py",
        "render.yaml",
        "Dockerfile",
        "tests/features/api_skeleton.feature",
        "tests/fixtures/api_fixtures.py"
    ]

    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")
            all_exist = False

    return all_exist

def test_deployment_configs():
    """Test deployment configuration files."""
    print("\nTesting deployment configurations...")

    # Test render.yaml
    render_config = Path("render.yaml")
    if render_config.exists():
        print("✓ render.yaml exists")
        content = render_config.read_text()
        if "sra-data-processing" in content:
            print("✓ render.yaml contains service name")
        if "python server.py" in content:
            print("✓ render.yaml has correct start command")
    else:
        print("✗ render.yaml missing")

    # Test Dockerfile
    dockerfile = Path("Dockerfile")
    if dockerfile.exists():
        print("✓ Dockerfile exists")
        content = dockerfile.read_text()
        if "python:3.11" in content:
            print("✓ Dockerfile uses Python 3.11")
        if "server.py" in content:
            print("✓ Dockerfile runs server.py")
    else:
        print("✗ Dockerfile missing")

def run_all_tests():
    """Run all manual tests."""
    print("=" * 50)
    print("SRA Data Processing API - Manual Testing")
    print("=" * 50)

    tests = [
        ("File Structure", test_file_structure),
        ("API Imports", test_api_imports),
        ("Server Config", test_server_config),
        ("Deployment Configs", test_deployment_configs)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'-' * 30}")
        print(f"Running: {test_name}")
        print(f"{'-' * 30}")

        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("TESTING SUMMARY")
    print("=" * 50)

    passed = 0
    failed = 0

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n🎉 All tests passed! API skeleton is ready for deployment.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the issues above.")

    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)