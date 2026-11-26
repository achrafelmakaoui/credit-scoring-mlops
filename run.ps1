# Relance toute la chaîne : build, entraînement, enregistrement du champion, API.
# Usage : .\run.ps1        démarre la pile
#         .\run.ps1 -Down  arrête la pile et supprime le registre MLflow (volume)
param([switch]$Down)

Set-Location $PSScriptRoot
if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env" }

if ($Down) {
    docker compose down -v
} else {
    docker compose up --build
}