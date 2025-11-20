#!/bin/bash
# Complete deployment script

echo "=========================================="
echo "GAN Image Enhancer - Complete Deployment"
echo "=========================================="
echo ""

# Check if training is complete
echo "Checking training status..."
python scripts/check_training_status.py

echo ""
echo "Building Docker image..."
docker-compose build

echo ""
echo "Starting services..."
docker-compose up -d

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "Web app is running at: http://localhost:8501"
echo ""
echo "Useful commands:"
echo "  - View logs: docker-compose logs -f"
echo "  - Stop: docker-compose down"
echo "  - Restart: docker-compose restart"
echo ""

