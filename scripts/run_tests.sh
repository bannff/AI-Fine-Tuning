#!/bin/bash
# Temporary test runner
set -e

echo "🧪 Setting up QA environment..."
python3 -m venv .venv_qa
source .venv_qa/bin/activate

echo "📦 Installing testing framework..."
pip install --quiet pytest hypothesis
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "🏃 Running Tests..."
pytest tests/ -v

echo "🗑️  Cleaning up QA environment..."
deactivate
rm -rf .venv_qa

echo "✅ QA Validation completed successfully!"
