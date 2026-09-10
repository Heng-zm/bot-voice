param (
    [string]$User = "chuokimheng.2852ceb0",
    [string]$Server = "my.anajak.cloud",
    [int]$Port = 2022,
    [string]$KeyFile = ""
)

$scriptDir = $PSScriptRoot
if (-not $scriptDir -and $MyInvocation.MyCommand.Path) {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
}
if (-not $scriptDir) {
    $scriptDir = (Get-Location).Path
}

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " Bot Voice SFTP Uploader - Anajak Cloud (my.anajak.cloud:2022)" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

if (-not $User) {
    Write-Host ""
    Write-Host "Note: On Anajak Cloud panel, find your SFTP username in Settings -> SFTP Details." -ForegroundColor Yellow
    $User = (Read-Host "Enter your SFTP Username").Trim()
}

if (-not $User) {
    Write-Host "Error: SFTP Username is required." -ForegroundColor Red
    exit 1
}

$batchFile = Join-Path $scriptDir "upload.sftp"
if (-not (Test-Path $batchFile)) {
    Write-Host "Error: upload.sftp not found in $scriptDir" -ForegroundColor Red
    exit 1
}

$sftpArgs = @("-P", "$Port", "-o", "StrictHostKeyChecking=no")
if ($KeyFile) {
    $sftpArgs += @("-i", "$KeyFile")
}
$sftpArgs += @("-b", "$batchFile", "${User}@${Server}")

Write-Host ""
Write-Host "Connecting to ${User}@${Server}:${Port} via SFTP..." -ForegroundColor Yellow
Write-Host "Enter your Anajak Cloud account password when prompted." -ForegroundColor Magenta
Write-Host ""

Set-Location $scriptDir
& sftp @sftpArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "==================================================================" -ForegroundColor Green
    Write-Host " ✅ UPLOAD TO ANAJAK CLOUD COMPLETED SUCCESSFULLY!" -ForegroundColor Green
    Write-Host "==================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "👉 Next step: Go to your Anajak Cloud Panel (https://my.anajak.cloud)" -ForegroundColor Cyan
    Write-Host "   and click 'Start' / 'Restart' on your server console!" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "❌ SFTP upload failed (Exit code: $LASTEXITCODE)." -ForegroundColor Red
    Write-Host "💡 Troubleshooting:" -ForegroundColor Yellow
    Write-Host "   - Verify your SFTP Username (from Panel -> Settings -> SFTP Details)" -ForegroundColor Gray
    Write-Host "   - Verify your Anajak Cloud account password" -ForegroundColor Gray
    Write-Host "   - Ensure Host is my.anajak.cloud and Port is 2022" -ForegroundColor Gray
    Write-Host ""
}
