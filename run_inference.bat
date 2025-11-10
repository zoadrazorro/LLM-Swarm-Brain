@echo off
REM Run inference on remote server
REM Usage: run_inference.bat [mode]
REM   mode: quick, full, or models (default: quick)

set SSH_KEY=C:\Users\jelly\SSH_01
set SERVER=ubuntu@147.185.41.15

set MODE=%1
if "%MODE%"=="" set MODE=quick

echo ==========================================
echo Running LLM-Swarm-Brain Inference
echo ==========================================
echo Mode: %MODE%
echo.

if "%MODE%"=="quick" (
    echo Running quick test (no models, 1 question per level)...
    ssh -i %SSH_KEY% %SERVER% "cd LLM-Swarm-Brain && source venv/bin/activate && python inference_test.py --quick"
) else if "%MODE%"=="full" (
    echo Running full test (no models, all questions)...
    ssh -i %SSH_KEY% %SERVER% "cd LLM-Swarm-Brain && source venv/bin/activate && python inference_test.py"
) else if "%MODE%"=="models" (
    echo Running full test with models (requires GPUs)...
    ssh -i %SSH_KEY% %SERVER% "cd LLM-Swarm-Brain && source venv/bin/activate && python inference_test.py --load-models"
) else (
    echo Invalid mode: %MODE%
    echo Valid modes: quick, full, models
    exit /b 1
)

echo.
echo ==========================================
echo Inference Complete!
echo ==========================================
echo.
echo Fetching latest results file...
ssh -i %SSH_KEY% %SERVER% "cd LLM-Swarm-Brain && ls -lh inference_results_*.json 2>/dev/null | tail -1"

echo.
echo To download results:
echo   scp -i %SSH_KEY% %SERVER%:~/LLM-Swarm-Brain/inference_results_*.json .
echo.
echo To view on server:
echo   ssh -i %SSH_KEY% %SERVER%
echo   cd LLM-Swarm-Brain
echo   cat inference_results_*.json
