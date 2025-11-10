@echo off
echo ========================================
echo Starting LM Studio on GPU 1 (Port 1235)
echo ========================================
echo.

REM Set environment to use only GPU 1
set HIP_VISIBLE_DEVICES=1
set CUDA_VISIBLE_DEVICES=1

echo GPU 1 selected
echo Port: 1235
echo.
echo Starting LM Studio...
echo Please load Phi-4 Q4_K_M model in the LM Studio GUI
echo.

REM Start LM Studio (adjust path if needed)
start "" "C:\Users\%USERNAME%\AppData\Local\Programs\LMStudio\LM Studio.exe"

echo.
echo LM Studio launched on GPU 1
echo Configure in GUI:
echo   - Load Phi-4 Q4_K_M model
echo   - Go to Local Server tab
echo   - Set port to 1235
echo   - Click Start Server
echo.
pause
