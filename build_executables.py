"""
Setup script to build standalone .exe files for Windows.

This creates self-contained executables that don't require Python or pip.
Use: python build_executables.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Make sure PyInstaller is installed
try:
    import PyInstaller
except ImportError:
    print("Installing PyInstaller...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    import PyInstaller


def build_exe(script_name: str, exe_name: str, icon_path: str = None) -> bool:
    """
    Build a single .exe using PyInstaller.
    
    Args:
        script_name: Python script to compile
        exe_name: Name of output .exe
        icon_path: Optional path to .ico file
    
    Returns:
        True if successful
    """
    print(f"\n{'='*60}")
    print(f"Building {exe_name}...")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",  # Single executable
        "--console",  # Show console window
        "--distpath", "dist",  # Output directory
        "--buildpath", "build",  # Build directory
        "--specpath", "build",  # Spec file directory
        f"--name={exe_name}",
        script_name,
    ]
    
    # Add icon if provided
    if icon_path and os.path.exists(icon_path):
        cmd.extend(["--icon", icon_path])
    
    try:
        subprocess.check_call(cmd)
        print(f"✓ {exe_name}.exe created successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error building {exe_name}: {e}")
        return False


def main():
    """
    Build all executable files.
    """
    print("\n" + "="*60)
    print("RECURSIVE TRAINER TEST - BUILD EXECUTABLES")
    print("="*60)
    
    # Create output directory
    Path("dist").mkdir(exist_ok=True)
    
    # Build each executable
    executables = [
        ("main.py", "RecursiveTrainer_Main"),
        ("scripts/meta_trainer_demo.py", "RecursiveTrainer_MetaTrainer"),
    ]
    
    results = {}
    for script, exe_name in executables:
        if os.path.exists(script):
            results[exe_name] = build_exe(script, exe_name)
        else:
            print(f"✗ Script not found: {script}")
            results[exe_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("BUILD SUMMARY")
    print("="*60)
    
    successful = sum(1 for v in results.values() if v)
    total = len(results)
    
    for exe_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {exe_name}")
    
    print(f"\nBuilt: {successful}/{total}")
    
    if successful == total:
        print("\n✓ All executables built successfully!")
        print(f"\nLocation: {os.path.abspath('dist')}")
        print("\nYou can now run the .exe files directly!")
        return 0
    else:
        print("\n✗ Some builds failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
