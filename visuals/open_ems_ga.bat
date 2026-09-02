@echo off
title EMS + GA Live Decisions
cd /d "%~dp0"

echo.
echo  Starting EMS + Genetic Algorithm viewer...
echo.

set "PYEXE="
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
  set "PYEXE=%LocalAppData%\Programs\Python\Python312\python.exe"
)
if not defined PYEXE if exist "%LocalAppData%\Programs\Python\Python311\python.exe" (
  set "PYEXE=%LocalAppData%\Programs\Python\Python311\python.exe"
)
if not defined PYEXE if exist "%ProgramFiles%\Python312\python.exe" (
  set "PYEXE=%ProgramFiles%\Python312\python.exe"
)

if defined PYEXE (
  "%PYEXE%" "%~dp0start_ems_ga.py"
) else (
  where py >nul 2>&1 && (
    py -3 "%~dp0start_ems_ga.py"
  ) || (
    python "%~dp0start_ems_ga.py"
  )
)

if errorlevel 1 (
  echo.
  echo  Something went wrong. See messages above.
  pause
)
