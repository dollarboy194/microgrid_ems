@echo off
cd /d "%~dp0"
echo ============================================================
echo   Tamale Microgrid EMS - FULL REPRODUCTION
echo.
echo   Trains all three forecasters (Random Forest, LSTM,
echo   XGBoost) and runs the whole test year. This takes
echo   roughly 50 minutes - the Random Forest is the slow part.
echo   You can leave it running.
echo ============================================================
echo.
".venv\Scripts\python.exe" run_experiment.py
echo.
echo ============================================================
echo   Done. Tables I and II are printed above and saved to the
echo   "results" folder as table1_forecasters.csv and
echo   table2_ems.csv, along with the five figures.
echo ============================================================
pause
