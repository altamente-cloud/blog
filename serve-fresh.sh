#!/bin/bash

# Fresh server restart script - clears all caches and restarts on fixed port

echo "Killing existing Hugo servers..."
pkill -f "hugo server" 2>/dev/null || true
sleep 1

echo "Clearing Hugo cache..."
rm -rf hugo-server/resources/_gen
rm -rf hugo-server/public

echo "Starting fresh Hugo server for Russian on port 1313..."
cd hugo-server
rm -f content config.toml
ln -s ../ru/content content
cp ../ru/config.toml config.toml

hugo server --bind 0.0.0.0 --port 1313 --buildDrafts --buildFuture
