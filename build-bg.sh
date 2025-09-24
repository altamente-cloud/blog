#!/bin/bash

# GitHub repository configuration
GITHUB_REPO="git@github.com:altamente-cloud/blog-bg.git"  # Update this with your actual GitHub repo
GITHUB_PAGES_BRANCH="bg"              # Branch for GitHub Pages
CNAME_DOMAIN="bg.yurigolub.me"       # Custom domain for GitHub Pages
BUILD_DIR="dist/bg"
CONTENT_SOURCE="../bg/content"
CONFIG_SOURCE="../bg/config.toml"
HUGO_DIR="hugo-server"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to clean up on exit
cleanup() {
    print_status "Cleaning up..."
    cd "$SCRIPT_DIR"
    if [ -L "$HUGO_DIR/content" ]; then
        rm "$HUGO_DIR/content"
        print_status "Removed symlink"
    fi
    # Restore original config if backup exists
    if ls "$HUGO_DIR"/config.toml.backup.* 1> /dev/null 2>&1; then
        LATEST_BACKUP=$(ls -t "$HUGO_DIR"/config.toml.backup.* 2>/dev/null | head -1)
        if [ -f "$LATEST_BACKUP" ]; then
            cp "$LATEST_BACKUP" "$HUGO_DIR/config.toml"
            print_status "Restored original config.toml"
        fi
    fi
}

# Set up trap to ensure cleanup happens
trap cleanup EXIT

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

print_status "Starting Bulgarian site build and deployment..."

# Step 0: Clean build directory
print_status "Step 0: Cleaning build directory..."
if [ -d "$BUILD_DIR" ]; then
    rm -rf "$BUILD_DIR"
    print_status "Removed existing build directory"
fi
mkdir -p "$BUILD_DIR"

# Step 1: Create symlink from bg folder to hugo-server/content and copy config
print_status "Step 1: Creating symlink and copying config..."
cd "$HUGO_DIR"

# Remove existing content directory or symlink if it exists
if [ -e "content" ]; then
    if [ -L "content" ]; then
        rm "content"
        print_status "Removed existing symlink"
    elif [ -d "content" ]; then
        print_warning "Found existing content directory, backing it up..."
        mv "content" "content.backup.$(date +%Y%m%d_%H%M%S)"
    fi
fi

# Backup existing config.toml if it exists
if [ -f "config.toml" ]; then
    cp "config.toml" "config.toml.backup.$(date +%Y%m%d_%H%M%S)"
    print_status "Backed up existing config.toml"
fi

# Create the symlink
ln -s "$CONTENT_SOURCE" "content"
if [ $? -eq 0 ]; then
    print_status "Created symlink: content -> $CONTENT_SOURCE"
else
    print_error "Failed to create symlink"
    exit 1
fi

# Copy the config file
cp "$CONFIG_SOURCE" "config.toml"
if [ $? -eq 0 ]; then
    print_status "Copied config: $CONFIG_SOURCE -> config.toml"
else
    print_error "Failed to copy config file"
    exit 1
fi

# Step 2: Build Hugo site
print_status "Step 2: Building Hugo site..."

# Clear Hugo's generated resources to ensure fresh CSS compilation
if [ -d "resources/_gen" ]; then
    rm -rf "resources/_gen"
    print_status "Cleared Hugo generated resources cache"
fi

hugo --destination "../$BUILD_DIR" --minify --cleanDestinationDir
if [ $? -eq 0 ]; then
    print_status "Hugo build completed successfully"
else
    print_error "Hugo build failed"
    exit 1
fi

# Step 2.5: Create CNAME file for custom domain
if [ ! -z "$CNAME_DOMAIN" ]; then
    print_status "Step 2.5: Creating CNAME file for custom domain..."
    cd "$SCRIPT_DIR/$BUILD_DIR"
    echo "$CNAME_DOMAIN" > CNAME
    print_status "Created CNAME file with domain: $CNAME_DOMAIN"
fi

# Step 3: Deploy to GitHub Pages
print_status "Step 3: Deploying to GitHub Pages..."

cd "$SCRIPT_DIR/$BUILD_DIR"

# Initialize git if not already a repo
if [ ! -d ".git" ]; then
    git init
    print_status "Initialized git repository"
fi

# Configure git for GitHub Pages
git remote remove origin 2>/dev/null || true
git remote add origin "$GITHUB_REPO"

# Pull the latest changes from GitHub Pages branch
print_status "Pulling latest changes from $GITHUB_PAGES_BRANCH branch..."
git fetch origin
git checkout -B "$GITHUB_PAGES_BRANCH"

# Check if we need to pull existing content and clean for fresh deployment
if git ls-remote --heads origin | grep -q "$GITHUB_PAGES_BRANCH"; then
    # Reset to remote branch to get proper git history but keep working directory
    git reset --soft "origin/$GITHUB_PAGES_BRANCH"
    print_status "Reset git history to latest $GITHUB_PAGES_BRANCH branch"
else
    print_status "Creating new $GITHUB_PAGES_BRANCH branch"
fi

# Add all files
git add .
git add -A  # Make sure to include deletions

# Check if there are any changes to commit
if git diff --cached --quiet; then
    print_warning "No changes to deploy"
else
    # Commit and push
    git commit -m "Deploy Bulgarian site - $(date '+%Y-%m-%d %H:%M:%S')"
    git push -u origin "$GITHUB_PAGES_BRANCH" --force
    if [ $? -eq 0 ]; then
        print_status "Successfully deployed to GitHub Pages!"
        print_status "Your site should be available at: https://$CNAME_DOMAIN"
    else
        print_error "Failed to push to GitHub"
        exit 1
    fi
fi

# Step 4: Remove symlink and restore config (handled by trap cleanup function)
print_status "Build and deployment completed successfully!"