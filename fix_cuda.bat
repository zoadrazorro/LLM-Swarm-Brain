@echo off
echo ==========================================
echo Fixing PyTorch CUDA Support
echo ==========================================
echo.
echo You will be prompted for SSH key passphrase (5047)
echo.

set SSH_KEY=C:\Users\jelly\SSH_01
set SERVER=ubuntu@147.185.41.15

echo [1/2] Reinstalling PyTorch with CUDA 12.1 support...
ssh -i "%SSH_KEY%" %SERVER% "cd LLM-Swarm-Brain && source venv/bin/activate && pip uninstall -y torch torchvision torchaudio && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"

echo.
echo [2/2] Testing CUDA availability...
ssh -i "%SSH_KEY%" %SERVER% "cd LLM-Swarm-Brain && source venv/bin/activate && python -c \"import torch; print('CUDA Available:', torch.cuda.is_available()); print('CUDA Version:', torch.version.cuda); print('GPU Count:', torch.cuda.device_count())\""

echo.
echo ==========================================
echo Done!
echo ==========================================
pause
