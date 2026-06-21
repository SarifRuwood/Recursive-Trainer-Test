@echo off
REM Batch file to setup and build executables on Windows
REM Just double-click this file to run everything!

echo.
echo ============================================================
echo RECURSIVE TRAINER TEST - WINDOWS SETUP AND BUILD
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

echo Python found!
echo.

REM Step 1: Setup
echo Step 1: Running setup...
python setup_windows.py
if errorlevel 1 (
    echo Setup failed!
    pause
    exit /b 1
)

echo.
echo Step 2: Building executables...
python build_executables.py
if errorlevel 1 (
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo SUCCESS!
echo ============================================================
echo.
echo Executable files are ready in the 'dist' folder:
echo   - RecursiveTrainer_Main.exe
echo   - RecursiveTrainer_MetaTrainer.exe
echo.
echo You can now run these .exe files directly!
echo.
pause
