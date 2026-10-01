#!/bin/bash
# Extract all .7z archives in a directory (default: ~/Desktop/P76154862/DiffLocks_Dataset)

set -u

TARGET_PATH="${1:-$HOME/Desktop/P76154862/DiffLocks_Dataset}"

if [ ! -d "$TARGET_PATH" ]; then
    echo "❌ Directory not found: $TARGET_PATH"
    exit 1
fi

if ! command -v 7z >/dev/null 2>&1; then
    echo "❌ 7z command not found. Install p7zip (e.g. 'sudo apt install p7zip-full')."
    exit 1
fi

shopt -s nullglob
archives=("$TARGET_PATH"/*.7z)
shopt -u nullglob

if [ ${#archives[@]} -eq 0 ]; then
    echo "❌ No .7z files found in $TARGET_PATH"
    exit 1
fi

echo -e "\nFound ${#archives[@]} archive(s) in $TARGET_PATH\n"

for archive in "${archives[@]}"; do
    filename=$(basename "$archive")
    echo "📦 Extracting $filename ..."
    7z x -y -o"$TARGET_PATH" "$archive"
    if [ $? -ne 0 ]; then
        echo "❌ Error extracting $filename. Exiting script."
        exit 1
    fi
    echo "✅ Done: $filename"
done

echo -e "\n🎉 All archives extracted to $TARGET_PATH"
