param(
    [string]$Registry = "ghcr.io",
    [string]$Image = "dhhieu113pro/hiu-graph",
    [string]$Tag = "latest"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker Desktop is required to build and publish the image."
}

$requiredOutput = @(
    "output/entities.parquet",
    "output/relationships.parquet",
    "output/communities.parquet",
    "output/community_reports.parquet",
    "output/text_units.parquet",
    "output/lancedb"
)
$missing = $requiredOutput | Where-Object { -not (Test-Path $_) }
if ($missing) {
    throw "GraphRAG index is incomplete. Run .\bootstrap_local.ps1 first. Missing: $($missing -join ', ')"
}

$imageRef = "$Registry/$Image`:$Tag"
Write-Host "Building $imageRef with the generated GraphRAG index..."
docker build --tag $imageRef .
if ($LASTEXITCODE -ne 0) { throw "Docker build failed." }

Write-Host "Publishing $imageRef..."
docker push $imageRef
if ($LASTEXITCODE -ne 0) { throw "Docker push failed. Run 'docker login $Registry' first." }

Write-Host "Published $imageRef"
