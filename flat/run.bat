@echo off
cd /d "%~dp0"

if not exist "index.html" (
  echo index.html not found next to run.bat.
  goto :end
)

set PORT=8080

:findport
netstat -ano -p tcp | findstr /c:":%PORT% " | findstr /c:"LISTENING" >nul 2>&1
if %errorlevel%==0 (
  set /a PORT+=1
  goto findport
)

where http-server >nul 2>&1
if %errorlevel%==0 (
  call http-server . -a 127.0.0.1 -p %PORT% -c-1 -o
) else (
  call npx --yes http-server . -a 127.0.0.1 -p %PORT% -c-1 -o
)

:end
@pause
