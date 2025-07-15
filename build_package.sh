#!/bin/bash
# Build script for PyPI readiness testing
# This script builds the package but does NOT publish it

set -e

echo "🔧 Building S3 Vectors CLI package for PyPI readiness..."

# Check if we're in the right directory
if [ ! -f "setup.py" ] || [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: setup.py or pyproject.toml not found. Run from project root."
    exit 1
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/

# Install build dependencies if not present
echo "📦 Installing build dependencies..."
python -m pip install --upgrade build twine

# Build source distribution and wheel
echo "🏗️  Building package..."
python -m build

# Verify package contents
echo "📋 Package contents:"
echo "Source distribution:"
tar -tzf dist/*.tar.gz | head -20

echo -e "\nWheel contents:"
python -m zipfile -l dist/*.whl | head -20

# Check package metadata and structure
echo "🔍 Validating package metadata..."
twine check dist/*

# Display package info
echo "📊 Package information:"
ls -lh dist/

echo ""
echo "✅ Build complete! Package is PyPI-ready."
echo "📦 Distribution files created in dist/ directory"
echo "🚫 Package NOT published (as intended)"
echo ""
echo "When ready to publish to PyPI:"
echo "  twine upload dist/*"
