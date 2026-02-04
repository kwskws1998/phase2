@echo off
REM Open PowerShell and activate the virtual environment automatically
powershell -NoExit -ExecutionPolicy Bypass -Command "& { . .\.venv\Scripts\Activate.ps1; Write-Host '--- Virtual Environment Activated ---' -ForegroundColor Cyan }"
