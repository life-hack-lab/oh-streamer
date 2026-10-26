# Optional portable VSR dependency. Run from any working directory.
$ErrorActionPreference = 'Stop'
$destination = Join-Path $PSScriptRoot 'nvencc'
if (Test-Path -LiteralPath $destination) {
    throw "Already exists: $destination. Keep the installed version or choose a separate folder and set OH_STREAMER_NVENC_C."
}
$archive = Join-Path $PSScriptRoot 'NVEncC_9.35_x64.7z'
Invoke-WebRequest 'https://github.com/rigaya/NVEnc/releases/download/9.35/NVEncC_9.35_x64.7z' -OutFile $archive
$expected = '8D80C91B4D38F2065B760A9B13D7835080D9971569B172FBFF6C376FD429FEBD'
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $expected) {
    throw "NVEncC archive checksum mismatch: $archive"
}
New-Item -ItemType Directory -Path $destination | Out-Null
& tar -xf $archive -C $destination
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath (Join-Path $destination 'NVEncC64.exe'))) {
    throw "NVEncC extraction failed. Archive retained at $archive"
}
Remove-Item -LiteralPath $archive
Write-Output "Installed NVEncC 9.35 to $destination. Enable 投放 RTX 超解析 in the player cast menu."
