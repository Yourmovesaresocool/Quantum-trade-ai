@echo off
REM run_update_daily_prices.bat - simplified: no multi-line if/else block,
REM everything prints to the console directly so any failure is visible
REM immediately instead of silently going nowhere.

cd /d "%~dp0"
echo Working directory: %cd%
echo.

echo Activating venv...
call "C:\Users\OM\Desktop\ai-trading-bot\venv\Scripts\activate.bat"
echo.

echo Running update_daily_prices.py...
echo.
python update_daily_prices.py

echo.
echo Done. (update_daily_prices.py's own log is in logs\update_daily_prices.log)
pause