# Verify Dual-GPU LM Studio Setup

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "VERIFYING DUAL-GPU SETUP" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

# Check GPU 0 (Port 1234)
Write-Host "Checking GPU 0 (Port 1234)..." -ForegroundColor Yellow
try {
    $response1 = Invoke-RestMethod -Uri "http://localhost:1234/v1/models" -TimeoutSec 3
    Write-Host "✓ GPU 0 server is running" -ForegroundColor Green
    Write-Host "  Model: $($response1.data[0].id)" -ForegroundColor White
} catch {
    Write-Host "✗ GPU 0 server not responding" -ForegroundColor Red
    Write-Host "  Run: .\start_lmstudio_gpu0.bat" -ForegroundColor Yellow
}

Write-Host ""

# Check GPU 1 (Port 1235)
Write-Host "Checking GPU 1 (Port 1235)..." -ForegroundColor Yellow
try {
    $response2 = Invoke-RestMethod -Uri "http://localhost:1235/v1/models" -TimeoutSec 3
    Write-Host "✓ GPU 1 server is running" -ForegroundColor Green
    Write-Host "  Model: $($response2.data[0].id)" -ForegroundColor White
} catch {
    Write-Host "✗ GPU 1 server not responding" -ForegroundColor Red
    Write-Host "  Run: .\start_lmstudio_gpu1.bat" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=" * 80 -ForegroundColor Cyan

# Summary
if ($response1 -and $response2) {
    Write-Host "✓ BOTH GPUS READY!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Neuron Distribution:" -ForegroundColor Yellow
    Write-Host "  GPU 0 (Port 1234): Neurons 0-1 (Perception, Attention, Memory, Reasoning)" -ForegroundColor White
    Write-Host "  GPU 1 (Port 1235): Neurons 2-3 (Creative, Analytical, Synthesis, Meta)" -ForegroundColor White
    Write-Host ""
    Write-Host "Start training with:" -ForegroundColor Green
    Write-Host "  python train_local_7900xt.py --questions 100" -ForegroundColor White
} elseif ($response1) {
    Write-Host "⚠ Only GPU 0 ready. Start GPU 1 server." -ForegroundColor Yellow
} elseif ($response2) {
    Write-Host "⚠ Only GPU 1 ready. Start GPU 0 server." -ForegroundColor Yellow
} else {
    Write-Host "✗ No servers running. Run setup_dual_gpu.ps1" -ForegroundColor Red
}

Write-Host ""
