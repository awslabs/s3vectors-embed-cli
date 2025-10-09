#!/usr/bin/env python3
"""Setup script for S3 Vectors CLI."""

from setuptools import setup, find_packages

# Import version from the package
exec(open("s3vectors/__version__.py").read())

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="s3vectors-embed-cli",
    version=__version__,
    author="Vaibhav Sabharwal",
    author_email="vsabhar@amazon.com",
    maintainer="Vaibhav Sabharwal",
    maintainer_email="vsabhar@amazon.com",
    description="Standalone CLI for S3 Vector operations with Bedrock embeddings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/awslabs/s3vectors-embed-cli",
    project_urls={
        "Bug Reports": "https://github.com/awslabs/s3vectors-embed-cli/issues",
        "Source": "https://github.com/awslabs/s3vectors-embed-cli",
        "Documentation": "https://github.com/awslabs/s3vectors-embed-cli#readme",
        "Homepage": "https://github.com/awslabs/s3vectors-embed-cli",
    },
    packages=find_packages(),
    keywords="aws, s3, vectors, embeddings, bedrock, cli, machine-learning, ai",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Systems Administration",
        "Topic :: Utilities",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: Console",
    ],
    python_requires=">=3.9",
    install_requires=[
        "boto3>=1.39.5",
        "botocore>=1.39.5",
        "click>=8.0.0",
        "rich>=12.0.0",
        "pydantic>=1.10.0",
    ],
    entry_points={
        "console_scripts": [
            "s3vectors-embed=s3vectors.cli:main",
        ],
    },
)
