# PowerShell deployment script for the GAN web app

Write-Host "Building Docker image..." -ForegroundColor Green
docker-compose build

Write-Host "Starting services..." -ForegroundColor Green
docker-compose up -d

Write-Host "Web app is running at http://localhost:8501" -ForegroundColor Cyan
Write-Host "To view logs: docker-compose logs -f" -ForegroundColor Yellow
Write-Host "To stop: docker-compose down" -ForegroundColor Yellow

