@echo off
REM Launch script - Run the meta-trainer demo
REM Double-click this to start the meta-trainer

cd /d "%~dp0"

if exist "dist\RecursiveTrainer_MetaTrainer.exe" (
    echo Launching Meta-Trainer...
    start "Meta-Trainer" dist\RecursiveTrainer_MetaTrainer.exe
) else (
    echo ERROR: RecursiveTrainer_MetaTrainer.exe not found!
    echo Please run build_all.bat first to build the executables.
    pause
)
