@echo off
cd /d "%~dp0"

call npm install
if errorlevel 1 goto :end

call npm run build

:end
@pause
