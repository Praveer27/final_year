#!/bin/bash

# Gesture2Speech Quick Start Script

echo "=================================="
echo "Gesture2Speech - Quick Start"
echo "=================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/installed" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    touch venv/installed
fi

# Setup directories
echo "Setting up project directories..."
python main.py setup

echo ""
echo "=================================="
echo "Setup complete!"
echo "=================================="
echo ""
echo "Available commands:"
echo "  python main.py process-videos    # Process videos to frames"
echo "  python main.py extract-keypoints # Extract hand keypoints"
echo "  python main.py augment-data      # Augment dataset"
echo "  python main.py train             # Train model"
echo "  python main.py inference         # Run inference"
echo "  python main.py pipeline          # Run complete pipeline"
echo ""
echo "For more information, run: python main.py --help"
echo ""

# Made with Bob
