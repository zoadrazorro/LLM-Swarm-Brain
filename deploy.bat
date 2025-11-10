@echo off
REM Automated deployment script for LLM-Swarm-Brain

set SSH_KEY=C:\Users\jelly\SSH_01
set SERVER=ubuntu@147.185.41.15

echo ==========================================
echo Auto-Deploying LLM-Swarm-Brain
echo ==========================================

echo.
echo [1/4] Testing connection...
ssh -i %SSH_KEY% -o ConnectTimeout=10 %SERVER% "echo Connected successfully"
if errorlevel 1 (
    echo ERROR: Connection failed
    exit /b 1
)

echo.
echo [2/4] Setting up repository...
ssh -i %SSH_KEY% %SERVER% "if [ -d 'LLM-Swarm-Brain' ]; then echo 'Updating...'; cd LLM-Swarm-Brain; git pull; else echo 'Cloning...'; git clone https://github.com/zoadrazorro/LLM-Swarm-Brain.git; fi"

echo.
echo [3/4] Running setup script...
ssh -i %SSH_KEY% %SERVER% "cd LLM-Swarm-Brain; chmod +x remote_setup.sh; bash remote_setup.sh"

echo.
echo [4/4] Running quick inference test...
ssh -i %SSH_KEY% %SERVER% "cd LLM-Swarm-Brain; source venv/bin/activate; python inference_test.py --quick"

echo.
echo ==========================================
echo Deployment Complete!
echo ==========================================
echo.
echo To view results, SSH into the server:
echo   ssh -i %SSH_KEY% %SERVER%
