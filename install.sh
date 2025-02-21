#!/bin/bash

# Exit on any error
set -e

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Git is not installed. Please install git first."
    exit 1
fi

# Check if python3 is installed and version is 3.10 or higher
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.10 or higher."
    exit 1

elif ! python3 -c "import sys; assert sys.version_info >= (3, 10)" 2>/dev/null; then
    echo "Python 3.10 or higher is required. Your version is too old."
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

# Create and activate virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing Python requirements..."
pip install -r requirements.txt

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
echo "3. Run: run.sh"
