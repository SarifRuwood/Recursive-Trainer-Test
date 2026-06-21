"""
Setup script for Windows .exe distribution.

This installs all dependencies needed to build executables.
Run this ONCE: python setup_windows.py
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str = "") -> bool:
    """
    Run a shell command and report status.
    
    Args:
        cmd: Command to run
        description: Description of what's happening
    
    Returns:
        True if successful
    """
    if description:
        print(f"\n{description}...")
    
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✓ {description}" if description else "✓ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}" if description else f"✗ Failed: {e}")
        return False


def main():
    """
    Setup Windows environment for building executables.
    """
    print("\n" + "="*60)
    print("RECURSIVE TRAINER TEST - WINDOWS SETUP")
    print("="*60)
    
    # Step 1: Upgrade pip
    print("\nStep 1: Upgrading pip, setuptools, and wheel...")
    run_command(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        "Upgrading build tools"
    )
    
    # Step 2: Install requirements
    print("\nStep 2: Installing project dependencies...")
    if Path("requirements.txt").exists():
        run_command(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            "Installing requirements from requirements.txt"
        )
    else:
        print("✗ requirements.txt not found!")
        return 1
    
    # Step 3: Install PyInstaller
    print("\nStep 3: Installing PyInstaller...")
    run_command(
        [sys.executable, "-m", "pip", "install", "pyinstaller"],
        "Installing PyInstaller"
    )
    
    # Step 4: Create directories
    print("\nStep 4: Creating necessary directories...")
    Path("dist").mkdir(exist_ok=True)
    Path("build").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    print("✓ Directories created")
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run: python build_executables.py")
    print("  2. Find .exe files in: dist/")
    print("  3. Run the .exe files directly!")
    print("\nOr use: build_all.bat (if available)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
