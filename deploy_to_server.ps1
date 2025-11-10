# PowerShell script to deploy and run LLM-Swarm-Brain on remote server
# Usage: .\deploy_to_server.ps1

$SSH_KEY = "C:\Users\jelly\SSH_01"
$SERVER = "ubuntu@147.185.41.15"

Write-Host "=========================================="
Write-Host "Deploying LLM-Swarm-Brain to Remote Server"
Write-Host "=========================================="

# Test SSH connection
Write-Host "`n[1/5] Testing SSH connection..."
ssh -i $SSH_KEY -o ConnectTimeout=10 $SERVER "echo 'Connection successful!'"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Cannot connect to server. Check your SSH key and network connection."
    exit 1
}

# Clone/update repository
Write-Host "`n[2/5] Cloning/updating repository on server..."
ssh -i $SSH_KEY $SERVER @"
if [ -d "LLM-Swarm-Brain" ]; then
    echo "Repository exists, pulling latest changes..."
    cd LLM-Swarm-Brain
    git pull
else
    echo "Cloning repository..."
    git clone https://github.com/zoadrazorro/LLM-Swarm-Brain.git
    cd LLM-Swarm-Brain
fi
"@

# Run setup script
Write-Host "`n[3/5] Running setup script..."
ssh -i $SSH_KEY $SERVER @"
cd LLM-Swarm-Brain
chmod +x remote_setup.sh
bash remote_setup.sh
"@

# Check GPU availability
Write-Host "`n[4/5] Checking GPU availability..."
ssh -i $SSH_KEY $SERVER "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'No GPU detected'"

# Run inference test
Write-Host "`n[5/5] Running inference test..."
Write-Host "Choose test mode:"
Write-Host "  1) Quick test (no models, fast)"
Write-Host "  2) Full test without models (architecture test)"
Write-Host "  3) Full test with models (requires GPUs)"

$choice = Read-Host "Enter choice (1-3)"

switch ($choice) {
    "1" {
        Write-Host "`nRunning quick test..."
        ssh -i $SSH_KEY $SERVER @"
cd LLM-Swarm-Brain
source venv/bin/activate
python inference_test.py --quick
"@
    }
    "2" {
        Write-Host "`nRunning full test without models..."
        ssh -i $SSH_KEY $SERVER @"
cd LLM-Swarm-Brain
source venv/bin/activate
python inference_test.py
"@
    }
    "3" {
        Write-Host "`nRunning full test with models (this may take a while)..."
        ssh -i $SSH_KEY $SERVER @"
cd LLM-Swarm-Brain
source venv/bin/activate
python inference_test.py --load-models
"@
    }
    default {
        Write-Host "Invalid choice. Running quick test..."
        ssh -i $SSH_KEY $SERVER @"
cd LLM-Swarm-Brain
source venv/bin/activate
python inference_test.py --quick
"@
    }
}

Write-Host "`n=========================================="
Write-Host "Deployment and Testing Complete!"
Write-Host "=========================================="
Write-Host "`nTo view results, run:"
Write-Host "  ssh -i $SSH_KEY $SERVER"
Write-Host "  cd LLM-Swarm-Brain"
Write-Host "  ls -lh inference_results_*.json"
