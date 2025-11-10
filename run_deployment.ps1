# Automated deployment script for LLM-Swarm-Brain
$SSH_KEY = "C:\Users\jelly\SSH_01"
$SERVER = "ubuntu@147.185.41.15"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Auto-Deploying LLM-Swarm-Brain" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Test connection
Write-Host "`n[1/4] Testing connection..." -ForegroundColor Yellow
try {
    $test = ssh -i $SSH_KEY -o ConnectTimeout=10 $SERVER "echo OK" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Connected successfully" -ForegroundColor Green
    } else {
        throw "Connection failed"
    }
} catch {
    Write-Host "✗ Connection failed" -ForegroundColor Red
    exit 1
}

# Clone/update repository
Write-Host "`n[2/4] Setting up repository..." -ForegroundColor Yellow
ssh -i $SSH_KEY $SERVER @"
if [ -d 'LLM-Swarm-Brain' ]; then
    echo 'Updating repository...'
    cd LLM-Swarm-Brain && git pull
else
    echo 'Cloning repository...'
    git clone https://github.com/zoadrazorro/LLM-Swarm-Brain.git
fi
"@

# Run setup script
Write-Host "`n[3/4] Running setup script (this may take a few minutes)..." -ForegroundColor Yellow
ssh -i $SSH_KEY $SERVER @"
cd LLM-Swarm-Brain
chmod +x remote_setup.sh
bash remote_setup.sh
"@

# Run inference test
Write-Host "`n[4/4] Running quick inference test..." -ForegroundColor Yellow
ssh -i $SSH_KEY $SERVER @"
cd LLM-Swarm-Brain
source venv/bin/activate
python inference_test.py --quick
echo ''
echo 'Results file:'
ls -1t inference_results_*.json 2>/dev/null | head -1
"@

Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "✓ Deployment Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "`nTo view full results, SSH into the server:" -ForegroundColor Yellow
Write-Host "  ssh -i $SSH_KEY $SERVER" -ForegroundColor White
