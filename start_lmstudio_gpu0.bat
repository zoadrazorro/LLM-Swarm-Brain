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

REM Start LM Studio (adjust path if needed)
start "" "C:\Users\%USERNAME%\AppData\Local\Programs\LMStudio\LM Studio.exe"

echo.
echo LM Studio launched on GPU 0
echo Configure in GUI:
echo   - Load Phi-4 Q4_K_M model
echo   - Go to Local Server tab
echo   - Set port to 1234
echo   - Click Start Server
echo.
pause
