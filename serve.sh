#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_usage() {
    echo -e "${BLUE}Usage:${NC} $0 <language>"
    echo -e "${BLUE}Languages:${NC}"
    echo -e "  ${GREEN}bg${NC} - Bulgarian version"
    echo -e "  ${GREEN}ru${NC} - Russian version"
    echo ""
    echo -e "${BLUE}Examples:${NC}"
    echo -e "  $0 bg    # Serve Bulgarian site"
    echo -e "  $0 ru    # Serve Russian site"
}

# Function to clean up on exit
cleanup() {
    print_status "Cleaning up..."
    cd "$SCRIPT_DIR"
    if [ -L "$HUGO_DIR/content" ]; then
        rm "$HUGO_DIR/content"
        print_status "Removed content symlink"
    fi
    # Restore original config if backup exists
    if [ -f "$HUGO_DIR/config.toml.serve.backup" ]; then
        mv "$HUGO_DIR/config.toml.serve.backup" "$HUGO_DIR/config.toml"
        print_status "Restored original config.toml"
    fi
}

# Set up trap to ensure cleanup happens
trap cleanup EXIT INT TERM

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
HUGO_DIR="hugo-server"

# Check if language argument is provided
if [ $# -eq 0 ]; then
    print_error "No language specified"
    print_usage
    exit 1
fi

LANGUAGE="$1"

# Validate language argument and set paths
case "$LANGUAGE" in
    "bg")
        CONTENT_SOURCE="../bg/content"
        CONFIG_SOURCE="../bg/config.toml"
        LANGUAGE_NAME="Bulgarian"
        ;;
    "ru")
        CONTENT_SOURCE="../ru/content"
        CONFIG_SOURCE="../ru/config.toml"
        LANGUAGE_NAME="Russian"
        ;;
    *)
        print_error "Invalid language: $LANGUAGE"
        print_usage
        exit 1
        ;;
esac

print_status "Starting Hugo development server for $LANGUAGE_NAME ($LANGUAGE)..."

# Check if source directories exist
if [ ! -d "$SCRIPT_DIR/${CONTENT_SOURCE#../}" ]; then
    print_error "Content directory not found: $SCRIPT_DIR/${CONTENT_SOURCE#../}"
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/${CONFIG_SOURCE#../}" ]; then
    print_error "Config file not found: $SCRIPT_DIR/${CONFIG_SOURCE#../}"
    exit 1
fi

cd "$SCRIPT_DIR/$HUGO_DIR"

# Step 1: Setup symlinks and config
print_status "Step 1: Setting up symlinks and configuration..."

# Remove existing content directory or symlink if it exists
if [ -e "content" ]; then
    if [ -L "content" ]; then
        rm "content"
        print_status "Removed existing content symlink"
    elif [ -d "content" ]; then
        print_warning "Found existing content directory, backing it up..."
        mv "content" "content.backup.$(date +%Y%m%d_%H%M%S)"
    fi
fi

# Backup existing config.toml if it exists
if [ -f "config.toml" ]; then
    cp "config.toml" "config.toml.serve.backup"
    print_status "Backed up existing config.toml"
fi

# Create the content symlink
ln -s "$CONTENT_SOURCE" "content"
if [ $? -eq 0 ]; then
    print_status "Created content symlink: content -> $CONTENT_SOURCE"
else
    print_error "Failed to create content symlink"
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

# Step 2: Start Hugo development server
print_status "Step 2: Starting Hugo development server..."
print_status "Server will be available at: http://localhost:1313"
print_status "Press Ctrl+C to stop the server and cleanup"
echo ""

# Start Hugo server with development settings
hugo serve --bind="0.0.0.0" --buildDrafts --buildFuture --disableFastRender

# Cleanup will be handled by the trap function when the script exits