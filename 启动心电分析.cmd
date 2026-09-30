@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title PulseLab - 心电智能分析
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" start.py
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    echo 未找到 Python。请安装 Python 3.10 或更新版本，并勾选 Add Python to PATH。
    echo 下载地址：https://www.python.org/downloads/
    pause
    exit /b 1
  )
  python start.py
)
if errorlevel 1 (
  echo.
  echo 启动失败，请查看上方提示。请勿关闭窗口前忽略错误。
  pause
)
endlocal
