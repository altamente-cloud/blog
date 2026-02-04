# Hugo Build and Deploy Scripts

This repository contains build scripts for deploying a Hugo site to GitHub Pages, plus a development server script for local testing.

**Note:** Bulgarian version is currently disabled. Only Russian version is active.

## Setup

### 1. Configure GitHub Repository
Before running the build scripts, update the GitHub repository configuration at the top of each script:

```bash
# In build-bg.sh and build-ru.sh
GITHUB_REPO="git@github.com:altamente-cloud/blog.git"  # Your actual GitHub repo
```

### 2. Prerequisites
- Hugo installed and available in PATH
- Git configured with your GitHub credentials
- GitHub repository set up for GitHub Pages

## Scripts

### `serve.sh` - Development Server
Starts a local Hugo development server for the specified language.

**Usage:**
```bash
./serve.sh <language>
```

**Languages:**
- `ru` - Russian version (active)
- `bg` - Bulgarian version (disabled)

**What it does:**
1. Creates symlink from language content folder to `hugo-server/content`
2. Copies language-specific config to `hugo-server/config.toml`
3. Starts Hugo development server with live reload
4. Automatically cleans up symlinks and restores config on exit

**Examples:**
```bash
./serve.sh ru    # Serve Russian site at http://localhost:1313
```

**Features:**
- **Live reload**: Changes to content files are automatically reflected
- **Draft support**: Shows draft content with `--buildDrafts`
- **Future posts**: Shows future-dated posts with `--buildFuture`
- **Network access**: Binds to `0.0.0.0` for access from other devices
- **Automatic cleanup**: Restores original config and removes symlinks on exit

### `build-ru.sh` - Russian Site Build and Deploy
Builds and deploys the Russian version of the site.

**What it does:**
0. Cleans the build directory (`dist/ru`)
1. Creates a symlink from `ru/content` folder to `hugo-server/content`
2. Copies `ru/config.toml` to `hugo-server/config.toml`
3. Builds the Hugo site to `dist/ru`
4. Deploys the site to GitHub Pages (`ru` branch)
5. Removes symlinks and restores original config

**Usage:**
```bash
./build-ru.sh
```

## Directory Structure
```
├── bg/                           # Bulgarian version (disabled)
│   ├── content/                  # Bulgarian content files
│   │   ├── _index.md
│   │   ├── pages/
│   │   └── posts/
│   └── config.toml              # Bulgarian-specific config
├── ru/                          # Russian version (active)
│   ├── content/                 # Russian content files
│   └── config.toml             # Russian-specific config
├── hugo-server/                # Hugo site configuration
│   ├── config.toml            # Base config (backed up/restored)
│   ├── themes/
│   │   └── hugo-winston-theme/ # Winston theme files
│   └── assets/
│       └── css/extended/       # Custom CSS overrides
│           └── compact.css     # Compact theme overrides
├── dist/                       # Build output (created by scripts)
│   └── ru/                    # Russian build
├── serve.sh                   # Development server script
└── build-ru.sh               # Russian build script
```

## Features

### Safety Features
- **Automatic cleanup**: Symlinks and configs are automatically restored even if scripts fail
- **Backup protection**: Existing content directories and configs are backed up
- **Error handling**: Scripts exit gracefully on errors with clear error messages
- **Change detection**: Only commits and pushes if there are actual changes

### Output
- **Colored status messages**: Clear visual feedback with green info, yellow warnings, and red errors
- **Progress tracking**: Step-by-step progress indication
- **URL notification**: Shows the GitHub Pages URL after successful deployment

### GitHub Pages Configuration
- **Russian site**: Deployed to `ru` branch of `git@github.com:altamente-cloud/blog-ru.git`
- **Force push**: Ensures clean deployment state
- **Automatic branch creation**: Creates branches if they don't exist

## Language-Specific Configurations

### Russian (`ru/config.toml`):
```toml
languageCode = "ru" 
[menu]
  [[menu.main]]
    name = 'Главная'     # Home
  [[menu.main]]
    name = "Статьи"      # Articles  
  [[menu.main]]
    name = "О сайте"     # About the site
  [[menu.main]]
    name = "По темам"    # By topics
```

## Development Workflow

### 1. Local Development
```bash
# Start development server for Russian version
./serve.sh ru
```

### 2. Build and Deploy
```bash
# Build and deploy Russian version
./build-ru.sh
```

## Troubleshooting

### Permission Issues
If you get permission errors, make sure the scripts are executable:
```bash
chmod +x serve.sh build-ru.sh
```

### Git Authentication
Make sure your GitHub credentials are configured:
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

For private repositories, you may need to use a personal access token instead of password authentication.

### Hugo Not Found
Ensure Hugo is installed and available in your PATH:
```bash
hugo version
```

### Port Already in Use
If port 1313 is already in use, Hugo will automatically try the next available port. Check the console output for the actual URL.

### Symlink Issues
The scripts handle symlink creation and cleanup automatically. If you encounter issues, you can manually remove symlinks:
```bash
rm hugo-server/content  # Remove content symlink if it exists
```

### Config Restoration
If config restoration fails, you can manually restore from backups:
```bash
ls hugo-server/config.toml.*  # List backup files
cp hugo-server/config.toml.serve.backup hugo-server/config.toml  # Restore from backup
```