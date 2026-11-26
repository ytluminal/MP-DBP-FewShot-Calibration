@echo off
cd /d "%~dp0"
python reproduce_rf_equal4.py --stages 1 || exit /b 1
python reproduce_rf_equal4.py --stages 2 || exit /b 1
python reproduce_rf_equal4.py --stages 3 || exit /b 1
python reproduce_rf_equal4.py --stages 4 || exit /b 1
python verify_all.py || exit /b 1
pause
