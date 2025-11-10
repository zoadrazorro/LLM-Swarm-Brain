@echo off
REM Check GPU status on remote server

set SSH_KEY=C:\Users\jelly\SSH_01
set SERVER=ubuntu@147.185.41.15

echo ==========================================
echo Checking GPU Status on Remote Server
echo ==========================================
echo.

echo [1] GPU Information:
echo.
ssh -i %SSH_KEY% %SERVER% "nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu --format=csv"

echo.
echo [2] Detailed GPU Status:
echo.
ssh -i %SSH_KEY% %SERVER% "nvidia-smi"

echo.
echo [3] CUDA Version:
echo.
ssh -i %SSH_KEY% %SERVER% "nvcc --version 2>/dev/null || echo 'CUDA toolkit not installed'"

echo.
echo [4] PyTorch GPU Check:
echo.
ssh -i %SSH_KEY% %SERVER% "cd LLM-Swarm-Brain && source venv/bin/activate && python -c 'import torch; print(f\"PyTorch version: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"GPU count: {torch.cuda.device_count()}\"); [print(f\"GPU {i}: {torch.cuda.get_device_name(i)}\") for i in range(torch.cuda.device_count())]' 2>/dev/null || echo 'PyTorch not installed or venv not activated'"

echo.
echo ==========================================
