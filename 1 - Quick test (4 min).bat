@echo off
cd /d "%~dp0"
echo ============================================================
echo   Tamale Microgrid EMS - QUICK TEST (about 4 minutes)
echo   Runs the dispatch on a 45-day slice of the test year.
echo ============================================================
echo.
".venv\Scripts\python.exe" run_experiment.py --skip-table1 --test-days 45
echo.
echo ============================================================
echo   Done. Open the "results" folder to see the figures (.png)
echo   and tables (.csv). This window can be closed.
echo ============================================================
pause
