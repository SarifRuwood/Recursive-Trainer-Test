@echo off
REM Launch script - Run the main training application
REM Double-click this to start the trainer

cd /d "%~dp0"

if exist "dist\RecursiveTrainer_Main.exe" (
    echo Launching Recursive Trainer...
    start "Recursive Trainer" dist\RecursiveTrainer_Main.exe
) else (
    echo ERROR: RecursiveTrainer_Main.exe not found!
    echo Please run build_all.bat first to build the executable.
    pause
)
