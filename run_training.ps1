param([string]$Run = 'run_01', [int]$Epochs = 35, [int]$BatchSize = 8)
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Create .venv and install requirements.txt first; see README.md.' }
Push-Location -LiteralPath $taskRoot
try {
    if (-not (Test-Path -LiteralPath 'data\manifest.json')) {
        & $taskPython download_subset.py
        if ($LASTEXITCODE -ne 0) { throw 'Dataset preparation failed.' }
    }
    if (-not (Test-Path -LiteralPath 'pretrained\hub\checkpoints\resnet50-11ad3fa6.pth')) {
        & $taskPython download_weights.py
        if ($LASTEXITCODE -ne 0) { throw 'Pretrained weight download failed.' }
    }
    & $taskPython train.py --smoke --run smoke
    if ($LASTEXITCODE -ne 0) { throw 'GPU smoke check failed.' }
    & $taskPython train.py --run $Run --epochs $Epochs --batch-size $BatchSize
    if ($LASTEXITCODE -ne 0) { throw 'Training failed.' }
} finally {
    Pop-Location
}
