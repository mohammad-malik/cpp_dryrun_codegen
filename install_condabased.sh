#!/bin/bash

# Exit on any error
set -e

# Check if conda is installed and in PATH
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed or not in PATH."
    echo "Please install Conda first or use install.sh for non-Conda setup."
    exit 1
fi

# Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    echo "Visit https://docs.docker.com/get-docker/ for installation instructions."
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo "Docker daemon is not running. Please start Docker:"
    echo "For macOS: Open Docker Desktop application"
    echo "For Linux: Run 'sudo systemctl start docker'"
    echo "After starting Docker, run this script again."
    exit 1
fi

# Clone the repository
echo "Cloning repository..."
git clone https://github.com/mohammad-malik/cpp_dryrun_codegen.git
cd cpp_dryrun_codegen

# Create conda environment from environment.yml
echo "Creating conda environment..."
conda env create -f environment.yml

# Create required directories
mkdir -p cache logs output

# Create .env file template
if [ ! -f .env ]; then
    echo "Creating .env file template..."
    echo "GOOGLE_API_KEY=your_api_key_here" > .env
    echo "Please edit .env and add your Google API key from https://makersuite.google.com/app/apikey"
fi

# Build Docker image
echo "Building Docker image..."
cd run_cpp
if ! docker build -t cpp-runner .; then
    echo "Docker build failed. Please ensure:"
    echo "1. Docker daemon is running"
    echo "2. You have sufficient permissions"
    echo "3. You have internet connection"
    exit 1
fi

echo "Installation complete!"
echo "To run the application:"
echo "1. Edit .env and add your Google API key"
echo "2. Ensure Docker daemon is running"
echo "3. Run: conda activate cpp_quiz"
echo "4. Run: run.sh"