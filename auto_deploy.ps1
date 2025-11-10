# Automated deployment script (non-interactive)
# Usage: .\auto_deploy.ps1

$SSH_KEY = "C:\Users\jelly\SSH_01"
$SERVER = "ubuntu@147.185.41.15"

Write-Host "=========================================="
Write-Host "Auto-Deploying LLM-Swarm-Brain"
Write-Host "=========================================="

# Test connection
Write-Host "`n[1/4] Testing connection..."
$result = ssh -i $SSH_KEY -o ConnectTimeout=10 $SERVER "echo 'OK'" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Connected successfully"
} else {
    Write-Host "✗ Connection failed: $result"
    exit 1
}

# Setup repository
Write-Host "`n[2/4] Setting up repository..."
ssh -i $SSH_KEY $SERVER "bash -s" @'
set -e
if [ -d "LLM-Swarm-Brain" ]; then
    echo "Updating existing repository..."
    cd LLM-Swarm-Brain
    git pull
else
    echo "Cloning repository..."
    git clone https://github.com/zoadrazorro/LLM-Swarm-Brain.git
    cd LLM-Swarm-Brain
fi
echo "✓ Repository ready"
'@

# Run setup
Write-Host "`n[3/4] Running setup script..."
ssh -i $SSH_KEY $SERVER "bash -s" @'
set -e
cd LLM-Swarm-Brain
chmod +x remote_setup.sh
bash remote_setup.sh
echo "✓ Setup complete"
'@

# Run quick inference test
Write-Host "`n[4/4] Running quick inference test..."
ssh -i $SSH_KEY $SERVER "bash -s" @'
set -e
cd LLM-Swarm-Brain
source venv/bin/activate
python inference_test.py --quick
echo ""
echo "✓ Inference test complete"
echo ""
echo "Results saved to:"
ls -1t inference_results_*.json 2>/dev/null | head -1 | cat
'@

Write-Host "`n=========================================="
Write-Host "✓ Deployment Complete!"
Write-Host "=========================================="
