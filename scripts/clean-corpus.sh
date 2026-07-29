#!/usr/bin/env bash
#
# clean-corpus.sh -- distill the raw model output into a training corpus.
#
# The NIGHT GRID format is a tiny grammar, so cleaning is a whitelist:
# accepted scene needs a well-formed transmission header followed by at
# least one dialogue line from a known handle.  The cleaner inserts a
# blank separator before each accepted header.  Everything the generator
# got wrong (chatter, markdown, malformed lines) fails the grammar and
# disappears.  Unicode is transliterated to plain ASCII first so the
# character vocabulary stays small.
#
# Usage: clean-corpus.sh [raw-file] [clean-file]

set -euo pipefail

RAW_FILE="${1:-data/cyberpunk.raw.txt}"
CLEAN_FILE="${2:-data/cyberpunk.txt}"

MAX_REPEATS_PER_LINE=3   # guard against excessive exact-line repetition

HEADER_PATTERN="^=== TRANSMISSION [0-9]{4} // SECTOR ([1-9]|1[0-2]): [A-Z0-9][A-Z0-9 '-]* ===$"
DIALOGUE_PATTERN="^(RAZR|GHOST|VYPR|NYX|JINX|DOC|WIRES|MOTH): [^ ].*[.!?][\"']?$"

to_ascii() {
    iconv -f UTF-8 -t ASCII//TRANSLIT -c | LC_ALL=C tr -cd '\n -~'
}

apply_grammar() {
    awk -v header="$HEADER_PATTERN" \
        -v dialogue="$DIALOGUE_PATTERN" \
        -v max_repeats="$MAX_REPEATS_PER_LINE" '
        $0 ~ header { pending = $0; scene_open = 0; next }
        /^===/      { pending = ""; scene_open = 0; next }
        $0 ~ dialogue && (pending != "" || scene_open) {
            if (seen[$0] >= max_repeats)
                next
            seen[$0]++
            if (pending != "") {
                print ""
                print pending
                pending = ""
                scene_open = 1
            }
            print
            next
        }
    '
}

squeeze_blank_runs() {
    cat -s
}

report() {
    local file="$1"
    echo "clean-corpus: $(stat -c%s "$file") bytes -> $file"
    echo "clean-corpus: vocabulary of $(fold -w1 "$file" | sort -u | wc -l) distinct characters"
    echo "clean-corpus: $(grep -cE '^=== TRANSMISSION' "$file") transmission headers"
    for handle in RAZR GHOST VYPR NYX JINX DOC WIRES MOTH; do
        echo "clean-corpus:   $handle: $(grep -c "^$handle:" "$file") lines"
    done
}

if [ ! -r "$RAW_FILE" ]; then
    echo "clean-corpus: cannot read $RAW_FILE" >&2
    exit 1
fi

raw_path="$(realpath -m -- "$RAW_FILE")"
clean_path="$(realpath -m -- "$CLEAN_FILE")"
if [ "$raw_path" = "$clean_path" ]; then
    echo "clean-corpus: raw and clean paths must be different" >&2
    exit 1
fi

clean_dir="$(dirname "$CLEAN_FILE")"
mkdir -p "$clean_dir"
temp_dir="$(mktemp -d "$clean_dir/.tiny-agenc-clean.XXXXXX")"
temp_file="$temp_dir/corpus"

cleanup() {
    rm -f -- "$temp_file"
    rmdir -- "$temp_dir" 2>/dev/null || true
}
trap cleanup EXIT

to_ascii < "$RAW_FILE" | apply_grammar | squeeze_blank_runs > "$temp_file"

if ! grep -qE "$HEADER_PATTERN" "$temp_file" \
    || ! grep -qE "$DIALOGUE_PATTERN" "$temp_file"; then
    echo "clean-corpus: no complete valid scene found in $RAW_FILE" >&2
    exit 1
fi

if [ -e "$CLEAN_FILE" ]; then
    chmod --reference="$CLEAN_FILE" "$temp_file"
fi
mv -- "$temp_file" "$CLEAN_FILE"
rmdir -- "$temp_dir"
trap - EXIT
report "$CLEAN_FILE"
