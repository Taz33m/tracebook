#!/usr/bin/env python3
"""
Install core dependencies for the order book simulator.
"""

import subprocess
import sys

# Core dependencies needed for basic functionality
CORE_DEPS = [
    "numpy>=1.21.0",
    "numba>=0.56.0", 
    "pandas>=1.3.0",
    "psutil>=5.8.0",
]

# Optional dependencies for full functionality
OPTIONAL_DEPS = [
    "plotly>=5.0.0",
    "dash>=2.0.0",
    "pytest>=6.0.0",
]

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Install dependencies."""
    print("Installing core dependencies for order book simulator...")
    
    # Install core dependencies
    failed_core = []
    for package in CORE_DEPS:
        print(f"Installing {package}...")
        if install_package(package):
            print(f"✓ {package} installed successfully")
        else:
            print(f"✗ Failed to install {package}")
            failed_core.append(package)
    
    # Install optional dependencies
    print("\nInstalling optional dependencies...")
    failed_optional = []
    for package in OPTIONAL_DEPS:
        print(f"Installing {package}...")
        if install_package(package):
            print(f"✓ {package} installed successfully")
        else:
            print(f"✗ Failed to install {package}")
            failed_optional.append(package)
    
    # Summary
    print("\n" + "="*50)
    print("Installation Summary:")
    print(f"Core dependencies: {len(CORE_DEPS) - len(failed_core)}/{len(CORE_DEPS)} installed")
    print(f"Optional dependencies: {len(OPTIONAL_DEPS) - len(failed_optional)}/{len(OPTIONAL_DEPS)} installed")
    
    if failed_core:
        print(f"\nFailed core dependencies: {', '.join(failed_core)}")
        print("⚠️  Some core functionality may not work without these packages.")
    
    if failed_optional:
        print(f"\nFailed optional dependencies: {', '.join(failed_optional)}")
        print("ℹ️  Advanced features like dashboards may not work without these packages.")
    
    if not failed_core:
        print("\n🎉 Core installation completed successfully!")
        print("You can now run: python3 test_system.py")
    else:
        print("\n❌ Core installation had issues. Please install missing packages manually.")

if __name__ == "__main__":
    main()
