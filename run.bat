@echo off
cd /d "%~dp0"

if not exist "build\index.html" (
  echo build\index.html not found. Run setup.bat first.
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
  call http-server build -a 127.0.0.1 -p %PORT% -c-1 -o
) else (
  call npx --yes http-server build -a 127.0.0.1 -p %PORT% -c-1 -o
)

:end
@pause
