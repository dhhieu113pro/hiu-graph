param(
    [switch]$ForceIndex
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

function Read-DotEnv([string]$Name) {
    $line = Get-Content .env -ErrorAction Stop | Where-Object { $_ -match "^$Name=" } | Select-Object -First 1
    if (-not $line) { return $null }
    return ($line -replace "^$Name=", "").Trim().Trim('"')
}

function Test-Http([string]$Url) {
    try {
        Invoke-RestMethod -Uri $Url -TimeoutSec 2 | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Test-Tcp([string]$HostName, [int]$Port) {
    return (Test-NetConnection -ComputerName $HostName -Port $Port -WarningAction SilentlyContinue -InformationLevel Quiet)
}

if (-not (Test-Path .env)) {
    throw "Missing .env. Copy .env.example to .env and set LLAMA_CPP_MODEL first."
}

$llamaBaseUrl = Read-DotEnv "LLAMA_CPP_BASE_URL"
$llamaModel = Read-DotEnv "LLAMA_CPP_MODEL"
$llamaModelName = Read-DotEnv "LLAMA_CPP_MODEL_NAME"
$mcpPort = Read-DotEnv "MCP_PORT"
if (-not $mcpPort) { $mcpPort = "8011" }
$mcpBaseUrl = "http://127.0.0.1:$mcpPort"

if (-not $llamaBaseUrl) { $llamaBaseUrl = "http://127.0.0.1:8080" }
if (-not $llamaModelName) { $llamaModelName = "local-model" }
if (-not $llamaModel) { throw "LLAMA_CPP_MODEL is not set in .env." }

if (-not (Test-Http "$llamaBaseUrl/health")) {
    if (-not (Get-Command llama-server.exe -ErrorAction SilentlyContinue)) {
        throw "llama-server.exe was not found in PATH. Install llama.cpp first."
    }
    if (-not (Test-Path $llamaModel)) { throw "GGUF model not found: $llamaModel" }

    Write-Host "Starting llama.cpp ($llamaModelName)..."
    Start-Process -FilePath "llama-server.exe" `
        -ArgumentList @("-m", $llamaModel, "--alias", $llamaModelName, "--host", "127.0.0.1", "--port", "8080", "-c", "8192", "-np", "2", "--reasoning", "off", "--temp", "0.2", "--n-predict", "1200") `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden

    for ($attempt = 0; $attempt -lt 60 -and -not (Test-Http "$llamaBaseUrl/health"); $attempt++) {
        Start-Sleep -Seconds 1
    }
    if (-not (Test-Http "$llamaBaseUrl/health")) { throw "llama.cpp did not become healthy." }
}

$requiredOutput = @(
    "output/entities.parquet",
    "output/relationships.parquet",
    "output/communities.parquet",
    "output/community_reports.parquet",
    "output/text_units.parquet"
)
$needsIndex = $ForceIndex -or (($requiredOutput | Where-Object { -not (Test-Path $_) }).Count -gt 0)
if ($needsIndex) {
    if ($ForceIndex -and (Test-Path "cache")) {
        Remove-Item -LiteralPath (Resolve-Path "cache").Path -Recurse -Force
        New-Item -ItemType Directory -Path "cache" -Force | Out-Null
    }
    Write-Host "Running GraphRAG indexing..."
    $env:PYTHONPATH = Join-Path $projectRoot "src"
    uv --cache-dir .uv-cache run python -m maf_graphrag.core.index
    if ($LASTEXITCODE -ne 0) { throw "GraphRAG indexing failed with exit code $LASTEXITCODE." }
} else {
    Write-Host "GraphRAG index already exists; skipping indexing."
}

if (-not (Test-Tcp "127.0.0.1" ([int]$mcpPort))) {
    Write-Host "Starting MCP server..."
    $logPath = Join-Path $projectRoot "logs\mcp_server_bootstrap.log"
    $errorPath = Join-Path $projectRoot "logs\mcp_server_bootstrap.err.log"
    Start-Process -FilePath "uv" `
        -ArgumentList @("--cache-dir", ".uv-cache", "run", "python", "run_mcp_server.py") `
        -WorkingDirectory $projectRoot `
        -RedirectStandardOutput $logPath `
        -RedirectStandardError $errorPath `
        -WindowStyle Hidden
    Start-Sleep -Seconds 2
}

Write-Host "Local stack is ready: llama.cpp=$llamaBaseUrl, MCP=$mcpBaseUrl/mcp"
