#!/usr/bin/env bash
#
# Verify the central evidence manifest, reproducible derivations, executable
# artifacts, and the exact metrics quoted from the measured runs.
set -euo pipefail

EXECUTABLE=${1:-./tiny-agenc}
BIGRAM=${2:-./build/openmp-1/bigram}
REPOSITORY_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPOSITORY_ROOT"
export LC_ALL=C

fail() {
    printf 'check-evidence: %s\n' "$*" >&2
    exit 1
}

expect_value() {
    local label=$1
    local actual=$2
    local expected=$3

    if [[ $actual != "$expected" ]]; then
        fail "$label is $actual, expected $expected"
    fi
}

expect_hash() {
    local label=$1
    local path=$2
    local expected=$3
    local actual

    actual=$(sha256sum "$path" | awk '{print $1}')
    expect_value "$label SHA-256" "$actual" "$expected"
}

manifest_contains() {
    local path=$1

    awk -v path="$path" '$2 == path { found = 1 }
        END { exit found ? 0 : 1 }' EVIDENCE.sha256
}

if ! sha256sum --check --strict --quiet EVIDENCE.sha256; then
    fail "EVIDENCE.sha256 does not match the current evidence inputs"
fi

for source in src/*.c src/*.h tests/*.c; do
    manifest_contains "$source" ||
        fail "EVIDENCE.sha256 does not cover current source or test $source"
done
for log in book/logs/*; do
    [[ -f $log ]] || continue
    manifest_contains "$log" ||
        fail "EVIDENCE.sha256 does not cover evidence log $log"
done

expect_hash "canonical raw corpus" data/cyberpunk.raw.txt \
    38151a5b4b6a813d66488540b3b44cc678f9b28df6c85f4ab5f52408b63e811e
expect_hash "clean corpus" data/cyberpunk.txt \
    cbf7f8326ddba8b345414c84daaa07c2df7b2f74ae5fcd33ff30eba08bdb658c
expect_hash "training split" data/cyberpunk.train.txt \
    81e4d8744add73032f5df569582272a4a28fc89a032a18d3a0c52125de6fbae7
expect_hash "validation split" data/cyberpunk.val.txt \
    ec24c14fca09facba0bb3283b17a8c06c58eff45df91e362af5419ac80d8cf31
expect_hash "bundled checkpoint" tiny-agenc.bin \
    b425ee5d0a2b09168114fa4fd7e755725fa63f4809b1cdb3034f9603b131863d

expect_value "raw corpus size" \
    "$(stat -c%s data/cyberpunk.raw.txt)" 1200000
expect_value "clean corpus size" \
    "$(stat -c%s data/cyberpunk.txt)" 1089394
expect_value "training split size" \
    "$(stat -c%s data/cyberpunk.train.txt)" 982693
expect_value "validation split size" \
    "$(stat -c%s data/cyberpunk.val.txt)" 106701
expect_value "clean corpus transmission count" \
    "$(grep -c '^=== TRANSMISSION' data/cyberpunk.txt)" 2011
expect_value "clean corpus vocabulary size" \
    "$(fold -w1 data/cyberpunk.txt | sort -u | wc -l)" 80
expect_value "training transmission count" \
    "$(grep -c '^=== TRANSMISSION' data/cyberpunk.train.txt)" 1809
expect_value "validation transmission count" \
    "$(grep -c '^=== TRANSMISSION' data/cyberpunk.val.txt)" 202

historical_source_hash=46c88bbcb83af0b52c103d7a5e70540b546dffff81a752139e25b8b1ce0cca5e
historical_source_commit=989ed6422b1a6f98874867ce016eb4b74532c614
current_source_hash=$(
    find src -maxdepth 1 -type f -print0 \
        | sort -z \
        | xargs -0 sha256sum \
        | sha256sum \
        | awk '{print $1}'
)
grep -qF "$historical_source_hash" EVIDENCE.md ||
    fail "EVIDENCE.md lost the historical measured source hash"
grep -qF "$historical_source_commit" EVIDENCE.md ||
    fail "EVIDENCE.md lost the historical measured source commit"
grep -qF "$current_source_hash" EVIDENCE.md ||
    fail "EVIDENCE.md does not identify the current hardened source aggregate"
grep -qF "$historical_source_hash" book/logs/train-cyberpunk-5000.log ||
    fail "full-run log lost the historical measured source hash"
grep -qF "$historical_source_hash" \
    book/logs/validation-cyberpunk-5000-summary.log ||
    fail "validation log lost the historical measured source hash"

bigram_output=$(
    "$BIGRAM" data/cyberpunk.train.txt data/cyberpunk.val.txt
)
grep -qF \
    'full-file add-one loss 2.326050 | perplexity 10.2374' \
    <<< "$bigram_output"
grep -qF \
    'fixed-window add-one loss 2.320951 | perplexity 10.1854' \
    <<< "$bigram_output"

magic=$(od -An -tx1 -N4 tiny-agenc.bin | tr -d '[:space:]')
expect_value "checkpoint magic" "$magic" 54414743

temporary_dir=$(mktemp -d)
cleanup() {
    rm -rf -- "$temporary_dir"
}
trap cleanup EXIT

"$EXECUTABLE" sample --model tiny-agenc.bin \
    --prompt 'RAZR:' --length 1 --temperature 0.8 --seed 1337 \
    > "$temporary_dir/sample.txt" 2> "$temporary_dir/sample.err"
grep -q '^RAZR:' "$temporary_dir/sample.txt"
grep -q '815360 parameters' "$temporary_dir/sample.err"

"$EXECUTABLE" sample --model tiny-agenc.bin \
    --prompt '=== TRANSMISSION 0999 // SECTOR 9: ' \
    --length 400 --temperature 0.8 --seed 1337 \
    > "$temporary_dir/showcase.txt" 2> "$temporary_dir/showcase.err"
if ! cmp -s "$temporary_dir/showcase.txt" book/logs/showcase-sample.txt; then
    fail "bundled checkpoint no longer replays the recorded showcase sample"
fi

grep -qF \
    'b425ee5d0a2b09168114fa4fd7e755725fa63f4809b1cdb3034f9603b131863d' \
    book/logs/train-cyberpunk-5000.log
grep -qF 'step     1/5000 | loss 4.4395' \
    book/logs/train-cyberpunk-5000.log
grep -qF 'step  5000/5000 | loss 0.7668' \
    book/logs/train-cyberpunk-5000.log
grep -qF \
    '0753089369482f5ac9e4cb80c940c90b58f38493019ddcc4478ced61242350b8' \
    book/logs/validation-cyberpunk-5000-summary.log
for report in \
    'step     1/5000 | loss 4.4463 | val 4.0430' \
    'step    50/5000 | loss 2.4230 | val 2.4210' \
    'step   250/5000 | loss 1.8567 | val 1.8301' \
    'step  1000/5000 | loss 1.1218 | val 1.1182' \
    'step  2000/5000 | loss 0.9470 | val 0.9790' \
    'step  3000/5000 | loss 0.8414 | val 0.9320' \
    'step  4000/5000 | loss 0.7987 | val 0.9105' \
    'step  5000/5000 | loss 0.7638 | val 0.9023'
do
    grep -qF "$report" book/logs/validation-cyberpunk-5000-summary.log
done
grep -qF \
    '8d8e1d20f37e9a54ba908277251a8a9b61f4cac6920aa86abd2c9f33c0cd144e' \
    book/logs/validation-cyberpunk-5000-summary.log

printf 'check-evidence: manifest, derivations, baseline, checkpoint, samples, and metrics agree\n'
