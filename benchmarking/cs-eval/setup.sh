#!/bin/bash

# CS-Eval Benchmark Setup Script

set -e

echo "📊 Setting up CS-Eval Cybersecurity Evaluation Benchmark..."

# Check if we're in the cs-eval directory
if [ ! -f "README.md" ]; then
    echo "❌ Please run this script from the cs-eval directory"
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

# Install CS-Eval specific dependencies
echo "📦 Installing CS-Eval specific dependencies..."
pip install datasets huggingface_hub

# Create results directory
echo "📁 Creating results directory..."
mkdir -p results

# Check Hugging Face login status
echo "🔐 Checking Hugging Face authentication..."
python -c "
try:
    from huggingface_hub import whoami
    user = whoami()
    print(f'✅ Logged in to Hugging Face as: {user[\"name\"]}')
except Exception as e:
    print('⚠️ Not logged in to Hugging Face. Some datasets may not be accessible.')
    print('💡 Run: huggingface-cli login')
"

# Test dataset access
echo "🧪 Testing CS-Eval dataset access..."
python -c "
try:
    from datasets import load_dataset
    print('🔄 Loading CS-Eval dataset (this may take a moment)...')
    dataset = load_dataset('cseval/cs-eval', split='test[:5]')  # Load just 5 samples for testing
    print(f'✅ Dataset loaded successfully! Sample size: {len(dataset)}')
    print(f'📋 Available columns: {list(dataset.features.keys())}')
except Exception as e:
    print(f'❌ Error loading dataset: {e}')
    print('💡 Make sure you have internet access and Hugging Face authentication')
    exit(1)
" || exit 1

# Create config file if it doesn't exist
if [ ! -f "config.py" ]; then
    echo "⚙️ Creating default configuration file..."
    cat > config.py << 'EOF'
"""
CS-Eval Configuration

Default configuration for CS-Eval benchmark evaluation.
"""

# Model Configuration
MODEL_CONFIG = {
    "max_tokens": 512,
    "temperature": 0.1,
    "top_p": 0.9,
    "batch_size": 10
}

# Evaluation Configuration  
EVAL_CONFIG = {
    "max_questions_per_category": None,  # None for all questions
    "categories_to_evaluate": None,  # None for all categories
    "save_intermediate_results": True,
    "verbose": True
}

# Output Configuration
OUTPUT_CONFIG = {
    "results_dir": "results",
    "save_raw_responses": True,
    "generate_report": True
}

# Categories mapping (for filtering if needed)
CATEGORIES = [
    "Network Security",
    "Cryptography", 
    "Web Security",
    "System Security",
    "Software Security",
    "Mobile Security",
    "Database Security",
    "Cloud Security", 
    "Risk Management",
    "Digital Forensics",
    "Security Management"
]
EOF
fi

echo ""
echo "✅ CS-Eval setup complete!"
echo ""
echo "Next steps:"
echo "1. Run evaluation: python evaluate_phi3.py"
echo "2. View results: cat results/latest_evaluation.json"
echo "3. Check report: ls results/*.md"
echo ""
echo "💡 Tips:"
echo "  - Use --help flag to see all available options"
echo "  - Results are saved with timestamps for comparison"
echo "  - Edit config.py to customize evaluation parameters"
