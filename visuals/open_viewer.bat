@echo off
title Tamale Microgrid 3D Viewer
cd /d "%~dp0"

echo.
echo  Starting the 3D microgrid viewer...
echo.

REM Prefer full Python path if present, else py launcher, else python
set "PYEXE="
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "PYEXE=%LocalAppData%\Programs\Python\Python312\python.exe"
if not defined PYEXE if exist "%LocalAppData%\Programs\Python\Python311\python.exe" set "PYEXE=%LocalAppData%\Programs\Python\Python311\python.exe"
if not defined PYEXE if exist "%ProgramFiles%\Python312\python.exe" set "PYEXE=%ProgramFiles%\Python312\python.exe"

if defined PYEXE (
  "%PYEXE%" "%~dp0start_viewer.py"
) else (
  where py >nul 2>&1 && (
    py -3 "%~dp0start_viewer.py"
  ) || (
    python "%~dp0start_viewer.py"
  )
)

if errorlevel 1 (
  echo.
  echo  Something went wrong. See messages above.
  pause
)
