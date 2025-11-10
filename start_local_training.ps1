# Quick Start Script for Dual-GPU Local Training
# Run this after LM Studio is set up with Phi-4

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "DUAL-GPU LOCAL TRAINING - Phi-4 on 2x AMD Radeon RX 7900 XT" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

# Check if LM Studio is running
Write-Host "Checking LM Studio server..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:1234/v1/models" -TimeoutSec 2 -ErrorAction Stop
    Write-Host "✓ LM Studio server is running on port 1234" -ForegroundColor Green
} catch {
    Write-Host "✗ LM Studio server not detected on port 1234" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please start LM Studio server with Phi-4 model:" -ForegroundColor Yellow
    Write-Host "  1. Open LM Studio" -ForegroundColor White
    Write-Host "  2. Go to 'Local Server' tab" -ForegroundColor White
    Write-Host "  3. Load Phi-4 (Q4_K_M) model" -ForegroundColor White
    Write-Host "  4. Start server on port 1234" -ForegroundColor White
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Hardware: 2x AMD Radeon RX 7900 XT (40GB total)" -ForegroundColor White
Write-Host "  Model: Phi-4 (14B parameters, Q4_K_M)" -ForegroundColor White
Write-Host "  Neurons: 4 (2 per GPU)" -ForegroundColor White
Write-Host "  RAG Sharing: ENABLED (syncs with cloud)" -ForegroundColor White
Write-Host ""

# Ask for number of questions
$questions = Read-Host "How many questions to train on? (default: 100)"
if ([string]::IsNullOrWhiteSpace($questions)) {
    $questions = 100
}

Write-Host ""
Write-Host "Starting local training with $questions questions..." -ForegroundColor Green
Write-Host ""

# Run training
python train_local_7900xt.py --questions $questions --max-steps 2 --batch-size 5 --save-interval 20

Write-Host ""
Write-Host "Training complete!" -ForegroundColor Green
Write-Host "Check training_checkpoints/local/ for results" -ForegroundColor Yellow
