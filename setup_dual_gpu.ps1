# Dual-GPU LM Studio Setup Script
# Configures two LM Studio servers for dual 7900XT training

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "DUAL-GPU SETUP - 2x AMD Radeon RX 7900 XT" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

Write-Host "This script will help you set up two LM Studio instances:" -ForegroundColor Yellow
Write-Host "  GPU 0 -> Port 1234" -ForegroundColor White
Write-Host "  GPU 1 -> Port 1235" -ForegroundColor White
Write-Host ""

# Check if LM Studio is installed
$lmStudioPath = "$env:LOCALAPPDATA\Programs\LMStudio\LM Studio.exe"
if (-not (Test-Path $lmStudioPath)) {
    Write-Host "LM Studio not found at default location." -ForegroundColor Red
    Write-Host "Please install LM Studio from https://lmstudio.ai/" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "✓ LM Studio found" -ForegroundColor Green
Write-Host ""

# Instructions
Write-Host "SETUP INSTRUCTIONS:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. First Instance (GPU 0):" -ForegroundColor Cyan
Write-Host "   - Run: start_lmstudio_gpu0.bat" -ForegroundColor White
Write-Host "   - Load Phi-4 Q4_K_M model" -ForegroundColor White
Write-Host "   - Go to 'Local Server' tab" -ForegroundColor White
Write-Host "   - Set port to 1234" -ForegroundColor White
Write-Host "   - Click 'Start Server'" -ForegroundColor White
Write-Host ""

Write-Host "2. Second Instance (GPU 1):" -ForegroundColor Cyan
Write-Host "   - Run: start_lmstudio_gpu1.bat" -ForegroundColor White
Write-Host "   - Load Phi-4 Q4_K_M model" -ForegroundColor White
Write-Host "   - Go to 'Local Server' tab" -ForegroundColor White
Write-Host "   - Set port to 1235" -ForegroundColor White
Write-Host "   - Click 'Start Server'" -ForegroundColor White
Write-Host ""

Write-Host "3. Verify Both Servers:" -ForegroundColor Cyan
Write-Host "   curl http://localhost:1234/v1/models" -ForegroundColor White
Write-Host "   curl http://localhost:1235/v1/models" -ForegroundColor White
Write-Host ""

Write-Host "4. Start Training:" -ForegroundColor Cyan
Write-Host "   python train_local_7900xt.py --questions 100" -ForegroundColor White
Write-Host ""

$choice = Read-Host "Start first instance now? (y/n)"
if ($choice -eq "y" -or $choice -eq "Y") {
    Write-Host ""
    Write-Host "Launching LM Studio for GPU 0..." -ForegroundColor Green
    
    # Set environment for GPU 0
    $env:HIP_VISIBLE_DEVICES = "0"
    $env:CUDA_VISIBLE_DEVICES = "0"
    
    Start-Process $lmStudioPath
    
    Write-Host "✓ LM Studio launched" -ForegroundColor Green
    Write-Host ""
    Write-Host "Configure it for GPU 0 (port 1234), then run:" -ForegroundColor Yellow
    Write-Host "  .\start_lmstudio_gpu1.bat" -ForegroundColor White
    Write-Host "to start the second instance." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Setup script complete!" -ForegroundColor Green
