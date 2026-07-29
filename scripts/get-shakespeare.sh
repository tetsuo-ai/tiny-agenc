#!/usr/bin/env bash
#
# get-shakespeare.sh -- fetch the pinned classic tiny-shakespeare corpus
# (~1.1MB) as a comparison dataset.  The default progress sampler needs
# a sufficiently long plain-text corpus containing a newline.

set -euo pipefail

URL="https://raw.githubusercontent.com/karpathy/char-rnn/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt"
EXPECTED_SHA256="86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
OUT="${1:-data/shakespeare.txt}"

out_dir="$(dirname "$OUT")"
mkdir -p "$out_dir"
temp_dir="$(mktemp -d "$out_dir/.tiny-agenc-data.XXXXXX")"
temp_file="$temp_dir/shakespeare.txt"

cleanup() {
    rm -f -- "$temp_file"
    rmdir -- "$temp_dir" 2>/dev/null || true
}
trap cleanup EXIT

curl -sSfL "$URL" -o "$temp_file"
actual_sha256="$(sha256sum "$temp_file" | awk '{print $1}')"
if [ "$actual_sha256" != "$EXPECTED_SHA256" ]; then
    echo "get-shakespeare: checksum mismatch" >&2
    exit 1
fi

if [ -e "$OUT" ]; then
    chmod --reference="$OUT" "$temp_file"
fi
mv -- "$temp_file" "$OUT"
rmdir -- "$temp_dir"
trap - EXIT
echo "get-shakespeare: $(stat -c%s "$OUT") bytes -> $OUT"
