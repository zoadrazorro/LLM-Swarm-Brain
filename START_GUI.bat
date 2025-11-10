@echo off
echo ==========================================
echo Starting LLM-Swarm-Brain GUI
echo ==========================================
echo.

echo [1/2] Starting API Server...
start "API Server" cmd /k "cd api-server && npm install && npm start"

timeout /t 3 /nobreak >nul

echo [2/2] Starting React GUI...
start "React GUI" cmd /k "cd web-gui && npm install && npm start"

echo.
echo ==========================================
echo GUI Starting...
echo ==========================================
echo.
echo API Server: http://localhost:3001
echo React GUI: http://localhost:3000
echo.
echo Press any key to exit...
pause >nul
