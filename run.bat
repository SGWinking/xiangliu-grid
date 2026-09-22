@echo off
rem ===========================================================================
rem  Dayun Mural Toolkit - Xiangliu Grid - double-click entry
rem
rem  IMPORTANT: this file MUST stay pure ASCII. Not one Chinese character.
rem  cmd.exe reads .bat files using the OEM code page (cp936 on Chinese
rem  Windows), so any non-ASCII byte gets mis-decoded and may be executed as
rem  a command, producing errors like:
rem      'xxx' is not recognized as an internal or external command
rem  All Chinese messages live in run.ps1, which is saved as UTF-8 *with BOM*
rem  so that Windows PowerShell 5.1 reads it correctly.
rem
rem  Verify with:
rem    ..\_series\tools\fix-ps1-encoding.ps1 -Path .\run.ps1,.\run.bat -Check
rem ===========================================================================

chcp 65001 >nul
title Dayun Mural Toolkit - Xiangliu Grid

rem %* forwards any arguments to run.ps1, e.g.  run.bat -NoBrowser
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*

echo.
pause
