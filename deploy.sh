#!/bin/bash
# Deployment script for the GAN web app

echo "Building Docker image..."
docker-compose build

echo "Starting services..."
docker-compose up -d

echo "Web app is running at http://localhost:8501"
echo "To view logs: docker-compose logs -f"
echo "To stop: docker-compose down"

