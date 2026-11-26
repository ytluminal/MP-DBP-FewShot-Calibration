@echo off
cd /d "%~dp0"
python reproduce_all_five_models.py
if errorlevel 1 (
 echo.
 echo FAILED: see logs folder for the failing stage/model.
 pause
 exit /b 1
)
echo.
echo SUCCESS: all Figure 2c models refitted and audited.
pause
