@echo off
cd /d "%~dp0"
python verify_package.py
if errorlevel 1 (
 echo.
 echo VERIFICATION FAILED. Check your installed Python and package integrity.
 pause
 exit /b 1
)
echo.
echo SUCCESS: archive SHA-256 and Figure 2c 80-model verification completed.
pause
