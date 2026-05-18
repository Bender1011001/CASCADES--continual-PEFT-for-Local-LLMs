# PowerShell script to package ONLY what is needed for Colab
# Run this from the root of the project

$ZipFile = "CASCADES_Colab_Minimal.zip"

If (Test-Path $ZipFile) {
    Remove-Item $ZipFile
}

Write-Host "Creating minimal CASCADES Colab package..." -ForegroundColor Cyan

# Clean out pycache and artifacts before zipping
Get-ChildItem -Path "cascades", "app" -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "app/data/conversations.db" -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path "app/data/checkpoints" -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue

# Package only source files
Compress-Archive -Path "cascades", "app", "local_extract_cascades.py", "requirements.txt" -DestinationPath $ZipFile

$sizeMB = [math]::Round((Get-Item $ZipFile).Length / 1KB)
Write-Host "Done! $ZipFile ($sizeMB KB)" -ForegroundColor Green
Write-Host "Upload this to Google Drive along with abliteratedqwen3.zip and takeout_chunks_32k/"
