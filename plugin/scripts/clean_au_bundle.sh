#!/bin/bash
# Strip resource forks and xattrs from a JUCE-built AU bundle so codesign
# accepts it. Without this, codesign fails with:
#   "resource fork, Finder information, or similar detritus not allowed".
#
# Idempotent — safe to run multiple times. Invoked by release.yml before
# signing, and may be called by hand for local notarization builds.
#
# Usage:
#   ./clean_au_bundle.sh "/path/to/Holy Shifter.component"
set -euo pipefail

AU_PATH="${1:?usage: $0 <Path/To/Bundle.component>}"
if [[ ! -d "$AU_PATH" ]]; then
    echo "Bundle not found: $AU_PATH" >&2
    exit 1
fi

# Derive the inner Mach-O binary name from the bundle.
AU_NAME="$(basename "$AU_PATH" .component)"
AU_BINARY="$AU_PATH/Contents/MacOS/$AU_NAME"

# 1. Empty the resource-fork via named-fork path (silently OK if absent).
[[ -e "$AU_BINARY" ]] && cat /dev/null > "${AU_BINARY}/..namedfork/rsrc" 2>/dev/null || true

# 2. Cat-then-replace to rebuild the binary with only its data fork.
if [[ -e "$AU_BINARY" ]]; then
    cat "$AU_BINARY" > "${AU_BINARY}.clean"
    rm -f "$AU_BINARY"
    mv "${AU_BINARY}.clean" "$AU_BINARY"
    chmod +x "$AU_BINARY"
fi

# 3. Strip xattrs from every file in the bundle.
find "$AU_PATH" -exec xattr -c {} \; 2>/dev/null || true

# 4. Delete AppleDouble (._*) sidecars and .DS_Store files.
find "$AU_PATH" -name "._*" -delete 2>/dev/null || true
find "$AU_PATH" -name ".DS_Store" -delete 2>/dev/null || true
command -v dot_clean >/dev/null && dot_clean "$AU_PATH" 2>/dev/null || true

# 5. Remove any pre-existing signature so we start fresh.
find "$AU_PATH" -name "_CodeSignature" -type d -exec rm -rf {} + 2>/dev/null || true

echo "Cleaned: $AU_PATH"
