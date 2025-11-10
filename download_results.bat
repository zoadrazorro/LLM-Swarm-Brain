@echo off
REM Download inference results from remote server

set SSH_KEY=C:\Users\jelly\SSH_01
set SERVER=ubuntu@147.185.41.15

echo ==========================================
echo Downloading Inference Results
echo ==========================================
echo.

echo Checking for results files on server...
ssh -i %SSH_KEY% %SERVER% "cd LLM-Swarm-Brain && ls -1t inference_results_*.json 2>/dev/null | head -5"

echo.
echo Downloading latest result file...
for /f "delims=" %%i in ('ssh -i %SSH_KEY% %SERVER% "cd LLM-Swarm-Brain && ls -1t inference_results_*.json 2>/dev/null | head -1"') do set LATEST_FILE=%%i

if "%LATEST_FILE%"=="" (
    echo No results files found on server.
    exit /b 1
)

echo Latest file: %LATEST_FILE%
scp -i %SSH_KEY% %SERVER%:~/LLM-Swarm-Brain/%LATEST_FILE% .

if errorlevel 1 (
    echo ERROR: Download failed
    exit /b 1
)

echo.
echo ==========================================
echo Download Complete!
echo ==========================================
echo File saved to: %CD%\%LATEST_FILE%
echo.
echo To view:
echo   type %LATEST_FILE%
echo   or
echo   python -m json.tool %LATEST_FILE%
