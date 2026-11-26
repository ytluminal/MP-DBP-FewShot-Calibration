@echo off
cd /d "%~dp0"
python reproduce_rf_equal4.py --stages 1
if errorlevel 1 goto failed
python reproduce_rf_equal4.py --stages 2
if errorlevel 1 goto failed
python reproduce_rf_equal4.py --stages 3
if errorlevel 1 goto failed
python reproduce_rf_equal4.py --stages 4
if errorlevel 1 goto failed
python verify_all.py
if errorlevel 1 goto failed
echo SUCCESS: RF refitted and Equal4 recalculated using verified XGB, LightGBM and CatBoost snapshots.
pause
exit /b 0
:failed
echo FAILED: see console details.
pause
exit /b 1
