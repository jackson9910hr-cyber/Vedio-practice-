#!/bin/sh
# Deletes every personal photo/video/audio file and render this workspace created.
# Run after the finished video has been delivered to the user.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
find "$ROOT" -path "$ROOT/.git" -prune -o -type d \( -name assets -o -name stills -o -name frames -o -name output \) -print -exec rm -rf {} + 2>/dev/null
find "$ROOT" -path "$ROOT/.git" -prune -o -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.gif' -o -iname '*.webp' \
  -o -iname '*.heic' -o -iname '*.heif' -o -iname '*.mp4' -o -iname '*.mov' -o -iname '*.m4v' -o -iname '*.webm' \
  -o -iname '*.wav' -o -iname '*.mp3' -o -iname '*.m4a' -o -iname '*.log' \) -print -delete
# Chat uploads and scratch copies inside the session container
rm -rf "$HOME"/.claude/uploads/* /tmp/claude-*/*/*/scratchpad/* 2>/dev/null
# Verify
LEFT=$(find "$ROOT" "$HOME/.claude/uploads" /tmp -xdev -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.heic' -o -iname '*.mp4' -o -iname '*.mov' -o -iname '*.wav' \) 2>/dev/null | grep -v '/.git/' | wc -l)
echo "purge done — personal media files remaining: $LEFT"
