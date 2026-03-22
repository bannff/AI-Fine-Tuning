#!/bin/bash
# Temporary formatting script to ensure perfect code quality
set -e

echo "🌟 Setting up temporary aesthetic environment..."
python3 -m venv .venv_format
source .venv_format/bin/activate

echo "📦 Installing formatters..."
pip install --quiet ruff black isort

echo "🧹 Running Ruff (Lint/Fix)..."
ruff check --fix .

echo "📝 Running isort..."
isort .

echo "✒️  Running Black..."
black .

echo "🗑️  Cleaning up..."
deactivate
rm -rf .venv_format

echo "✅ All code is now perfectly formatted and clean!"
