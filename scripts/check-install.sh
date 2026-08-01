#!/usr/bin/env bash
#
# Exercise the staged install instead of assuming that a successful copy
# operation produced a usable package.
set -euo pipefail

PREFIX_ROOT=${1:?usage: check-install.sh PREFIX_ROOT}
EXECUTABLE=$PREFIX_ROOT/bin/tiny-agenc
MODEL=$PREFIX_ROOT/share/tiny-agenc/tiny-agenc.bin
DOC_DIR=$PREFIX_ROOT/share/doc/tiny-agenc

fail() {
    printf 'check-install: %s\n' "$*" >&2
    exit 1
}

cleanup() {
    rm -rf -- "$temporary_dir"
}

[[ -x $EXECUTABLE ]] || fail "missing executable: $EXECUTABLE"
[[ -r $MODEL ]] || fail "missing bundled checkpoint: $MODEL"

for document in \
    CHANGELOG.md CITATION.cff EVIDENCE.md EVIDENCE.sha256 LICENSE \
    MODEL_CARD.md NOTICE README.md VERSION
do
    [[ -r $DOC_DIR/$document ]] ||
        fail "missing installed documentation: $DOC_DIR/$document"
done

version=$(<"$DOC_DIR/VERSION")
[[ $("$EXECUTABLE" --version) == "tiny-agenc $version" ]] ||
    fail "installed executable and VERSION disagree"

help=$("$EXECUTABLE" --help)
grep -qF 'tiny-agenc train --data FILE' <<< "$help" ||
    fail "installed help does not describe training"
grep -qF 'tiny-agenc sample --model FILE' <<< "$help" ||
    fail "installed help does not describe sampling"

checkpoint_hash=$(sha256sum "$MODEL" | awk '{print $1}')
[[ $checkpoint_hash == \
   b425ee5d0a2b09168114fa4fd7e755725fa63f4809b1cdb3034f9603b131863d ]] ||
    fail "installed checkpoint hash is $checkpoint_hash"

temporary_dir=$(mktemp -d)
trap cleanup EXIT HUP INT TERM

"$EXECUTABLE" sample --model "$MODEL" --prompt 'RAZR:' \
    --length 1 --temperature 0.8 --seed 1337 \
    > "$temporary_dir/sample.txt" 2> "$temporary_dir/sample.err"
grep -q '^RAZR:' "$temporary_dir/sample.txt" ||
    fail "installed executable did not sample from the installed checkpoint"
grep -q '815360 parameters' "$temporary_dir/sample.err" ||
    fail "installed checkpoint did not report the expected architecture"

printf 'check-install: staged executable, model, metadata, and sampling agree\n'
