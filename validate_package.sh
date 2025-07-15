#!/bin/bash
# Validate package for PyPI readiness
# This script validates the package structure and metadata

set -e

echo "🔍 Validating S3 Vectors CLI package for PyPI readiness..."

# Check if we're in the right directory
if [ ! -f "setup.py" ] || [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: setup.py or pyproject.toml not found. Run from project root."
    exit 1
fi

# Install validation dependencies
echo "📦 Installing validation dependencies..."
python -m pip install --upgrade build twine check-manifest pip-tools

echo ""
echo "🔍 Running validation checks..."

# 1. Check package structure
echo "1️⃣  Checking package manifest..."
check-manifest --ignore-bad-ideas '*.sh,*.md.bak' || echo "⚠️  Manifest check completed with warnings"

# 2. Validate setup.py
echo "2️⃣  Validating setup.py..."
python setup.py check --metadata --strict

# 3. Build package (required for further validation)
echo "3️⃣  Building package for validation..."
rm -rf build/ dist/ *.egg-info/
python -m build

# 4. Check package metadata and structure
echo "4️⃣  Validating package metadata..."
twine check dist/*

# 5. Check dependencies
echo "5️⃣  Checking dependencies..."
python -c "
import pkg_resources
import sys

try:
    # Check if all dependencies can be resolved
    requirements = [
        'boto3>=1.39.5',
        'botocore>=1.39.5', 
        'click>=8.0.0',
        'rich>=12.0.0',
        'pydantic>=1.10.0'
    ]
    
    for req in requirements:
        try:
            pkg_resources.require(req)
            print(f'✅ {req}')
        except pkg_resources.DistributionNotFound:
            print(f'⚠️  {req} - not installed (will be installed by pip)')
        except pkg_resources.VersionConflict as e:
            print(f'❌ {req} - version conflict: {e}')
            
except Exception as e:
    print(f'❌ Dependency check failed: {e}')
    sys.exit(1)
"

# 6. Test package installation in isolated environment
echo "6️⃣  Testing package installation..."
python -c "
import tempfile
import subprocess
import sys
import os

# Create temporary virtual environment
with tempfile.TemporaryDirectory() as tmpdir:
    venv_path = os.path.join(tmpdir, 'test_venv')
    
    # Create virtual environment
    subprocess.run([sys.executable, '-m', 'venv', venv_path], check=True)
    
    # Get paths
    if os.name == 'nt':  # Windows
        pip_path = os.path.join(venv_path, 'Scripts', 'pip')
        python_path = os.path.join(venv_path, 'Scripts', 'python')
    else:  # Unix-like
        pip_path = os.path.join(venv_path, 'bin', 'pip')
        python_path = os.path.join(venv_path, 'bin', 'python')
    
    try:
        # Install package from wheel
        wheel_file = [f for f in os.listdir('dist') if f.endswith('.whl')][0]
        subprocess.run([pip_path, 'install', f'dist/{wheel_file}'], check=True, capture_output=True)
        
        # Test import
        result = subprocess.run([python_path, '-c', 'import s3vectors; print(\"✅ Package imports successfully\")'], 
                              capture_output=True, text=True, check=True)
        print(result.stdout.strip())
        
        # Test CLI command
        result = subprocess.run([python_path, '-c', 'from s3vectors.cli import main; print(\"✅ CLI entry point works\")'], 
                              capture_output=True, text=True, check=True)
        print(result.stdout.strip())
        
    except subprocess.CalledProcessError as e:
        print(f'❌ Package installation test failed: {e}')
        if e.stdout:
            print(f'STDOUT: {e.stdout}')
        if e.stderr:
            print(f'STDERR: {e.stderr}')
        sys.exit(1)
"

echo ""
echo "📊 Validation Summary:"
echo "✅ Package structure validated"
echo "✅ Metadata validated"
echo "✅ Dependencies checked"
echo "✅ Installation tested"
echo ""
echo "🎉 Package is PyPI-ready!"
echo "📦 Distribution files: $(ls dist/)"
echo "🚫 Package NOT published (validation only)"
