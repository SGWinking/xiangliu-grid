@echo off
chcp 65001 >nul 2>&1
title Xiangliu Grid v0.4.1
echo.
echo  Xiangliu Grid / 相柳网格 v0.4.1
echo  ========================================
echo.
echo  Starting server...
echo  Browser will open automatically.
echo  Keep this window open while using the tool.
echo.
start /b "" http://127.0.0.1:8765
"%~dp0xiangliu-grid.exe" 8765
pause
