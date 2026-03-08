# PowerShell script to package ONLY what is needed for Colab
# Run this from the root of the project

$ZipFile = "CASCADES_Colab_Minimal.zip"

If (Test-Path $ZipFile) {
    Remove-Item $ZipFile
}

Write-Host "Creating minimal CASCADES Colab package..." -ForegroundColor Cyan

# We ONLY take the required source files, skipping models, data, and cache
Compress-Archive -Path "cascades", "app", "local_extract_cascades.py", "requirements.txt" -DestinationPath $ZipFile

Write-Host "Done! The file $ZipFile is ready." -ForegroundColor Green
Write-Host "Upload this small zip file to Google Drive along with the takeout_chunks_32k folder."
