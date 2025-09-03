#!/bin/bash

# Script to update llama.cpp to the latest version
# This script fetches the latest release tag and updates the Dockerfile

set -e

echo "🔍 Checking for latest llama.cpp version..."

# Get the latest release tag
LATEST_TAG=$(curl -s https://api.github.com/repos/ggml-org/llama.cpp/releases/latest | jq -r '.tag_name')

if [ "$LATEST_TAG" = "null" ] || [ -z "$LATEST_TAG" ]; then
    echo "❌ Failed to fetch latest version from GitHub API"
    exit 1
fi

echo "📋 Latest llama.cpp version: $LATEST_TAG"

# Check current version in Dockerfile
CURRENT_TAG=$(grep "git checkout" Dockerfile | awk '{print $3}' || echo "not found")
echo "📋 Current version in Dockerfile: $CURRENT_TAG"

if [ "$CURRENT_TAG" = "$LATEST_TAG" ]; then
    echo "✅ Already using the latest version ($LATEST_TAG)"
    exit 0
fi

echo "🔄 Updating Dockerfile to use version $LATEST_TAG..."

# Update the Dockerfile
sed -i "s/git checkout .*/git checkout $LATEST_TAG \&\&/" Dockerfile

echo "✅ Updated Dockerfile to use llama.cpp version $LATEST_TAG"
echo ""
echo "🚀 To apply the changes, run:"
echo "   docker compose up -d --build"
echo ""
echo "📝 Note: This will rebuild the containers with the latest llama.cpp version"
