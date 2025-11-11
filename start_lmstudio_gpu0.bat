@echo off
echo ========================================
echo Starting LM Studio on GPU 0 (Port 1234)
echo ========================================
echo.

REM Set environment to use only GPU 0
set HIP_VISIBLE_DEVICES=0
set CUDA_VISIBLE_DEVICES=0

echo GPU 0 selected
echo Port: 1234
echo.
echo Starting LM Studio...
echo Please load Phi-4 Q4_K_M model in the LM Studio GUI
echo.

REM Start LM Studio - try common paths
if exist "C:\Users\%USERNAME%\AppData\Local\Programs\LMStudio\LM Studio.exe" (
    start "" "C:\Users\%USERNAME%\AppData\Local\Programs\LMStudio\LM Studio.exe"
) else if exist "C:\Program Files\LMStudio\LM Studio.exe" (
    start "" "C:\Program Files\LMStudio\LM Studio.exe"
) else if exist "C:\Program Files (x86)\LMStudio\LM Studio.exe" (
    start "" "C:\Program Files (x86)\LMStudio\LM Studio.exe"
) else (
    echo ERROR: LM Studio not found!
    echo Please start LM Studio manually and configure it for GPU 0, port 1234
    pause
    exit /b 1
)

echo.
echo LM Studio launched on GPU 0
echo Configure in GUI:
echo   - Load Phi-4 Q4_K_M model
echo   - Go to Local Server tab
echo   - Set port to 1234
echo   - Click Start Server
echo.
pause
