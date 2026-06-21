@echo off
REM Quick test runner - verifies everything works before full execution
REM Use this to troubleshoot issues

echo.
echo ============================================================
echo RECURSIVE TRAINER TEST - QUICK TEST
echo ============================================================
echo.
echo This will run a quick test to verify everything works.
echo.

REM Check Python
echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)

REM Check if dist folder exists
if not exist "dist" (
    echo ERROR: dist folder not found!
    echo Please run build_all.bat first.
    pause
    exit /b 1
)

REM Check executables
echo.
echo Checking for executables...
if exist "dist\RecursiveTrainer_Main.exe" (
    echo   ✓ RecursiveTrainer_Main.exe found
) else (
    echo   ✗ RecursiveTrainer_Main.exe not found
)

if exist "dist\RecursiveTrainer_MetaTrainer.exe" (
    echo   ✓ RecursiveTrainer_MetaTrainer.exe found
) else (
    echo   ✗ RecursiveTrainer_MetaTrainer.exe not found
)

echo.
echo ============================================================
echo TEST COMPLETE
echo ============================================================
echo.
echo If all items show ✓, you're ready to run the trainers!
echo.
pause
