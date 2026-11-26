@echo off
cd /d "%~dp0"
python -m pip install -r requirements_frozen.txt
if errorlevel 1 (
 echo FAILED installing dependencies.
 pause
 exit /b 1
)
python -c "import sys; print('Python:',sys.version)"
echo Dependencies installed. Preferred Python version: 3.13.5.
pause
