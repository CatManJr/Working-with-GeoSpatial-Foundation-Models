#!/bin/bash
# Fix permissions and clean up before running dev mode

echo "🔧 Pre-flight checks..."
echo ""

# 1. Fix npm permissions
echo "Fixing npm permissions..."
sudo chown -R $(whoami) "$HOME/.npm"

# 2. Clean npm cache
echo "Cleaning npm cache..."
npm cache clean --force

# 3. Set UV link mode for ExFAT compatibility
echo "Setting UV_LINK_MODE=copy for ExFAT compatibility..."
export UV_LINK_MODE=copy

echo ""
echo "✅ Pre-building checks complete!"
echo ""
echo "Now you can run: zsh run_dev.sh"