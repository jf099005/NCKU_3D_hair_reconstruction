#!/bin/bash
# Extract all .7z archives in a directory (default: ~/Desktop/P76154862/DiffLocks_Dataset)
#
# History is kept in <TARGET_PATH>/extract_history.log (one tab-separated line per attempt:
# time, status, archive, message). Archives already logged as OK are skipped on later runs.
# Archives that fail (corrupt / cannot be opened) are logged as FAIL and skipped, so the
# script keeps going; they are retried on the next run. Delete a line (or the whole log)
# to force re-extraction.

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

HISTORY_LOG="$TARGET_PATH/extract_history.log"
touch "$HISTORY_LOG"

log_history() {  # status archive_name message
    printf '%s\t%s\t%s\t%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" "$2" "$3" >> "$HISTORY_LOG"
}

already_done() {  # archive_name
    awk -F'\t' -v f="$1" '$2 == "OK" && $3 == f { found = 1 } END { exit !found }' "$HISTORY_LOG"
}

shopt -s nullglob
archives=("$TARGET_PATH"/*.7z)
shopt -u nullglob

if [ ${#archives[@]} -eq 0 ]; then
    echo "❌ No .7z files found in $TARGET_PATH"
    exit 1
fi

echo -e "\nFound ${#archives[@]} archive(s) in $TARGET_PATH"
echo -e "History log: $HISTORY_LOG\n"

n_ok=0
n_skip=0
failed=()

for archive in "${archives[@]}"; do
    filename=$(basename "$archive")

    if already_done "$filename"; then
        echo "⏭️  Skipping $filename (already extracted)"
        n_skip=$((n_skip + 1))
        continue
    fi

    echo "📦 Extracting $filename ..."
    err_file=$(mktemp)
    if 7z x -y -o"$TARGET_PATH" "$archive" 2> "$err_file"; then
        log_history "OK" "$filename" ""
        echo "✅ Done: $filename"
        n_ok=$((n_ok + 1))
    else
        rc=$?
        msg=$(grep -v '^\s*$' "$err_file" | tr '\n\t' '  ' | cut -c1-300)
        log_history "FAIL" "$filename" "exit $rc: $msg"
        echo "❌ Cannot extract $filename (exit $rc), skipping. ${msg}"
        failed+=("$filename")
    fi
    rm -f "$err_file"
done

echo -e "\n🎉 Finished: ${n_ok} extracted, ${n_skip} skipped, ${#failed[@]} failed"
if [ ${#failed[@]} -gt 0 ]; then
    echo "Failed archives (will be retried on next run):"
    printf '   - %s\n' "${failed[@]}"
    exit 2
fi
