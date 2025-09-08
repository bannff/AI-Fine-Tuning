#!/bin/bash

# CyberGym Setup Script

set -e

echo "🏋️ Setting up CyberGym - Large-Scale Cybersecurity Evaluation Framework..."

# Parse command line arguments
DATASET_MODE="subset"
STORAGE_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            DATASET_MODE="full"
            shift
            ;;
        --subset)
            DATASET_MODE="subset"
            shift
            ;;
        --storage-path)
            STORAGE_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--full|--subset] [--storage-path PATH]"
            exit 1
            ;;
    esac
done

# Check if we're in the cybergym directory
if [ ! -f "README.md" ]; then
    echo "❌ Please run this script from the cybergym directory"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️ Virtual environment not detected. Activating parent environment..."
    if [ -f "../venv/bin/activate" ]; then
        source ../venv/bin/activate
    else
        echo "❌ No virtual environment found. Please run ../setup.sh first"
        exit 1
    fi
fi

# Check Docker installation
echo "🐳 Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed or not in PATH"
    echo "💡 Please install Docker from https://docs.docker.com/engine/install/"
    exit 1
fi

# Check Docker daemon
if ! docker info &> /dev/null; then
    echo "❌ Docker daemon is not running"
    echo "💡 Please start Docker daemon"
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not available"
    echo "💡 Please install Docker Compose"
    exit 1
fi

echo "✅ Docker and Docker Compose are available"

# Check storage requirements
if [ "$DATASET_MODE" = "full" ]; then
    echo "⚠️ Full dataset mode selected (~10TB required)"
    
    # Check available disk space
    if [ -n "$STORAGE_PATH" ]; then
        AVAILABLE_SPACE=$(df "$STORAGE_PATH" | awk 'NR==2 {print $4}')
    else
        AVAILABLE_SPACE=$(df . | awk 'NR==2 {print $4}')
    fi
    
    # Convert to GB (assuming df output is in KB)
    AVAILABLE_GB=$((AVAILABLE_SPACE / 1024 / 1024))
    
    if [ $AVAILABLE_GB -lt 10240 ]; then
        echo "❌ Insufficient storage space. Available: ${AVAILABLE_GB}GB, Required: 10TB"
        echo "💡 Use --subset mode or specify --storage-path with sufficient space"
        exit 1
    fi
    
    echo "✅ Sufficient storage available: ${AVAILABLE_GB}GB"
else
    echo "📦 Subset mode selected (~100GB required)"
fi

# Install CyberGym specific dependencies
echo "📦 Installing CyberGym specific dependencies..."
pip install docker docker-compose requests beautifulsoup4 pyyaml

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p results
mkdir -p logs
mkdir -p data
mkdir -p docker

# Set up storage path
if [ -n "$STORAGE_PATH" ]; then
    echo "💾 Setting up storage path: $STORAGE_PATH"
    mkdir -p "$STORAGE_PATH"
    ln -sf "$STORAGE_PATH" data/cybergym-data
    export CYBERGYM_DATA_PATH="$STORAGE_PATH"
else
    export CYBERGYM_DATA_PATH="$(pwd)/data"
fi

# Create Docker Compose configuration
echo "🐳 Creating Docker configuration..."
cat > docker/docker-compose.yml << 'EOF'
version: '3.8'

services:
  cybergym-sandbox:
    image: ubuntu:22.04
    container_name: cybergym-sandbox
    volumes:
      - ../data:/cybergym-data:ro
      - ../results:/results:rw
    environment:
      - CYBERGYM_MODE=sandbox
    networks:
      - cybergym-net
    command: tail -f /dev/null
    
  cybergym-poc-server:
    image: python:3.9
    container_name: cybergym-poc-server
    volumes:
      - ../data:/cybergym-data:ro
      - ../results:/results:rw
    ports:
      - "8080:8080"
    environment:
      - CYBERGYM_MODE=poc_server
    networks:
      - cybergym-net
    command: python -m http.server 8080

networks:
  cybergym-net:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
EOF

# Create configuration file
echo "⚙️ Creating configuration file..."
cat > cybergym_config.py << EOF
"""
CyberGym Configuration

Configuration settings for CyberGym cybersecurity evaluation framework.
"""

import os

# Dataset Configuration
DATASET_CONFIG = {
    "mode": "$DATASET_MODE",
    "data_path": "$CYBERGYM_DATA_PATH",
    "subset_size": 10 if "$DATASET_MODE" == "subset" else None,
    "categories": [
        "SQL Injection",
        "Cross-Site Scripting (XSS)", 
        "File Upload Vulnerabilities",
        "Command Injection",
        "Authentication Bypass",
        "Directory Traversal",
        "Buffer Overflow",
        "Privilege Escalation",
        "Remote Code Execution",
        "Information Disclosure"
    ]
}

# Docker Configuration
DOCKER_CONFIG = {
    "network_name": "cybergym-net",
    "container_timeout": 300,
    "sandbox_image": "ubuntu:22.04",
    "poc_server_port": 8080,
    "compose_file": "docker/docker-compose.yml"
}

# Model Configuration
MODEL_CONFIG = {
    "max_tokens": 1024,
    "temperature": 0.1,
    "top_p": 0.9,
    "generation_timeout": 120,
    "max_poc_attempts": 3
}

# Evaluation Configuration
EVAL_CONFIG = {
    "max_scenarios": None,  # None for all available
    "timeout_per_scenario": 300,  # 5 minutes per scenario
    "verify_pocs": True,
    "save_intermediate": True,
    "verbose": True
}

# Output Configuration
OUTPUT_CONFIG = {
    "results_dir": "results",
    "logs_dir": "logs", 
    "save_raw_responses": True,
    "generate_reports": True,
    "poc_storage": "results/pocs"
}
EOF

# Create evaluation wrapper script
echo "🔗 Creating evaluation wrapper script..."
cat > run_evaluation.py << 'EOF'
#!/usr/bin/env python3
"""
CyberGym Evaluation Runner

Wrapper script for running CyberGym evaluations with the PHI-3 cybersecurity model.
This is a scaffold - full implementation requires CyberGym dataset and Docker setup.
"""

import argparse
import sys
import os
from pathlib import Path

def main():
    """Main function for CyberGym evaluation."""
    parser = argparse.ArgumentParser(
        description="CyberGym Cybersecurity Evaluation Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--sample", action="store_true", help="Run sample evaluation")
    parser.add_argument("--full", action="store_true", help="Run full evaluation")
    parser.add_argument("--categories", type=str, help="Comma-separated vulnerability categories")
    parser.add_argument("--generate_poc", action="store_true", help="Generate PoC for vulnerabilities")
    parser.add_argument("--vuln_id", type=str, help="Specific vulnerability ID to evaluate")
    parser.add_argument("--temperature", type=float, default=0.1, help="Model temperature")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max tokens to generate")
    
    args = parser.parse_args()
    
    print("🏋️ CyberGym - Large-Scale Cybersecurity Evaluation")
    print("=" * 50)
    
    # Check configuration
    config_path = Path("cybergym_config.py")
    if not config_path.exists():
        print("❌ Configuration file not found")
        print("💡 Run ./setup.sh to create configuration")
        sys.exit(1)
    
    # Check Docker setup
    compose_path = Path("docker/docker-compose.yml")
    if not compose_path.exists():
        print("❌ Docker configuration not found")
        print("💡 Run ./setup.sh to create Docker setup")
        sys.exit(1)
    
    print("🚧 CyberGym integration is not yet implemented.")
    print("")
    print("To complete the integration:")
    print("1. Implement dataset loading and parsing")
    print("2. Create PHI-3 model wrapper for PoC generation")
    print("3. Set up Docker containers for vulnerability testing")
    print("4. Implement PoC verification system")
    print("5. Configure results analysis and reporting")
    print("")
    print("Current setup status:")
    print(f"  ✅ Configuration file: {config_path}")
    print(f"  ✅ Docker setup: {compose_path}")
    print(f"  📁 Data directory: {Path('data').absolute()}")
    print(f"  📁 Results directory: {Path('results').absolute()}")
    print("")
    print("📚 See README.md for detailed implementation guide")

if __name__ == "__main__":
    main()
EOF

chmod +x run_evaluation.py

# Create dataset verification script
cat > verify_dataset.py << 'EOF'
#!/usr/bin/env python3
"""
CyberGym Dataset Verification

Verifies the integrity and structure of the CyberGym dataset.
"""

import os
import sys
from pathlib import Path

def main():
    """Verify dataset structure and integrity."""
    print("🔍 CyberGym Dataset Verification")
    print("=" * 40)
    
    # Load configuration
    try:
        from cybergym_config import DATASET_CONFIG
        data_path = Path(DATASET_CONFIG["data_path"])
        mode = DATASET_CONFIG["mode"]
    except ImportError:
        print("❌ Configuration not found. Run ./setup.sh first.")
        sys.exit(1)
    
    print(f"📊 Dataset mode: {mode}")
    print(f"📁 Data path: {data_path}")
    
    # Check if data directory exists
    if not data_path.exists():
        print("⚠️ Data directory does not exist")
        print("💡 Dataset download not yet implemented")
        return
    
    # Check for expected structure
    expected_dirs = ["scenarios", "pocs", "metadata"]
    found_dirs = []
    
    for expected_dir in expected_dirs:
        dir_path = data_path / expected_dir
        if dir_path.exists():
            found_dirs.append(expected_dir)
            print(f"✅ Found: {expected_dir}/")
        else:
            print(f"❌ Missing: {expected_dir}/")
    
    if len(found_dirs) == len(expected_dirs):
        print("✅ Dataset structure verified")
    else:
        print("⚠️ Dataset structure incomplete")
        print("💡 Full dataset integration pending")

if __name__ == "__main__":
    main()
EOF

chmod +x verify_dataset.py

echo ""
echo "✅ CyberGym setup complete!"
echo ""
echo "🚧 Current Status: Scaffold Ready (Dataset Mode: $DATASET_MODE)"
echo ""
echo "Configuration:"
echo "  📊 Dataset mode: $DATASET_MODE"
echo "  💾 Data path: $CYBERGYM_DATA_PATH"
echo "  🐳 Docker config: docker/docker-compose.yml"
echo ""
echo "Next steps to complete integration:"
echo "1. Implement dataset download and parsing"
echo "2. Create PHI-3 model wrapper for PoC generation"
echo "3. Set up vulnerability testing containers"
echo "4. Implement PoC verification system"
echo "5. Configure results analysis"
echo ""
echo "Test current setup:"
echo "  python verify_dataset.py"
echo "  python run_evaluation.py --sample"
echo ""
echo "📚 See README.md for detailed implementation guide"

if [ "$DATASET_MODE" = "full" ]; then
    echo ""
    echo "⚠️ Full dataset mode selected:"
    echo "  • Requires ~10TB storage space"
    echo "  • Dataset download not yet implemented"
    echo "  • Consider subset mode for development"
fi
