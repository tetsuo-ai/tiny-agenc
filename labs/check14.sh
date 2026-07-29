#!/usr/bin/env bash
set -euo pipefail

executable=$1
labs_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ $executable != /* ]]; then
    executable=$(
        cd -- "$(dirname -- "$executable")"
        printf '%s/%s\n' "$PWD" "$(basename -- "$executable")"
    )
fi

temporary_dir=$(mktemp -d)
trap 'rm -rf -- "$temporary_dir"' EXIT

run_captured() {
    if "$@" > "$temporary_dir/out" 2> "$temporary_dir/err"; then
        return 0
    else
        return $?
    fi
}

run_from_temporary_dir() {
    (
        cd -- "$temporary_dir"
        "$@"
    )
}

expect_success() {
    local label=$1
    shift

    if ! run_captured "$@"; then
        printf 'check-14: expected success: %s\n' "$label" >&2
        sed 's/^/check-14: stderr: /' "$temporary_dir/err" >&2
        exit 1
    fi
    if [[ -s $temporary_dir/err ]]; then
        printf 'check-14: unexpected diagnostic on success: %s\n' "$label" >&2
        sed 's/^/check-14: stderr: /' "$temporary_dir/err" >&2
        exit 1
    fi
}

expect_failure() {
    local label=$1
    local required_text=$2
    shift 2

    if run_captured "$@"; then
        printf 'check-14: expected failure: %s\n' "$label" >&2
        exit 1
    fi
    if [[ -s $temporary_dir/out ]]; then
        printf 'check-14: failure wrote to stdout: %s\n' "$label" >&2
        exit 1
    fi
    if [[ ! -s $temporary_dir/err ]] \
       || ! head -n 1 "$temporary_dir/err" | grep -q '^tiny-agenc:'; then
        printf 'check-14: diagnostic lacks the tiny-agenc prefix: %s\n' \
            "$label" >&2
        sed 's/^/check-14: stderr: /' "$temporary_dir/err" >&2
        exit 1
    fi
    if [[ -n $required_text ]] \
       && ! grep -qF -- "$required_text" "$temporary_dir/err"; then
        printf 'check-14: diagnostic omits %q: %s\n' \
            "$required_text" "$label" >&2
        sed 's/^/check-14: stderr: /' "$temporary_dir/err" >&2
        exit 1
    fi
}

expect_usage_failure() {
    local label=$1
    shift

    expect_failure "$label" "usage:" "$@"
    if ! grep -qF 'tiny-agenc train' "$temporary_dir/err" \
       || ! grep -qF 'tiny-agenc sample' "$temporary_dir/err"; then
        printf 'check-14: usage omits a subcommand: %s\n' "$label" >&2
        exit 1
    fi
}

expect_success "top-level help" "$executable" --help
if ! grep -qF 'tiny-agenc train' "$temporary_dir/out" \
   || ! grep -qF 'tiny-agenc sample' "$temporary_dir/out" \
   || ! grep -qF 'tiny-agenc --help' "$temporary_dir/out" \
   || ! grep -qF 'tiny-agenc --version' "$temporary_dir/out"; then
    printf 'check-14: --help omits a public command path\n' >&2
    exit 1
fi

expect_success "top-level version" "$executable" --version
release_version=$(tr -d '\r\n' < "$labs_dir/../VERSION")
printf 'tiny-agenc %s\n' "$release_version" > "$temporary_dir/expected-version"
if ! cmp -s "$temporary_dir/expected-version" "$temporary_dir/out"; then
    printf 'check-14: --version output is not exact\n' >&2
    exit 1
fi

printf 'abcabcabc\n' > "$temporary_dir/train.txt"
printf 'abcX\n' > "$temporary_dir/unknown-validation.txt"
ln "$temporary_dir/train.txt" "$temporary_dir/train-alias.txt"
truncate -s 268435457 "$temporary_dir/oversized-corpus.txt"

expect_usage_failure "missing subcommand" "$executable"
expect_failure "unknown subcommand" "unknown command" "$executable" unknown
expect_failure "subcommand help is not a second help surface" \
    "invalid option" "$executable" train --help
expect_failure "missing training data flag" "requires --data" \
    "$executable" train
expect_failure "missing sample model flag" "--model" "$executable" sample
expect_failure "extra sample argument" "unexpected argument" \
    "$executable" sample --model "$temporary_dir/missing.bin" extra
expect_failure "unknown training option" "invalid option" \
    "$executable" train --unknown
expect_failure "missing option argument" "invalid option" \
    "$executable" train --data
expect_failure "malformed integer" "--steps" \
    "$executable" train --data "$temporary_dir/train.txt" --steps 12oops
expect_failure "malformed learning rate" "--lr" \
    "$executable" train --data "$temporary_dir/train.txt" --lr NaN
expect_failure "negative seed" "--seed" \
    "$executable" train --data "$temporary_dir/train.txt" --seed -1
expect_failure "oversized corpus" "268435456" \
    "$executable" train --data "$temporary_dir/oversized-corpus.txt" \
    --steps 1 --layers 1 --heads 1 --width 8 --block 4 --batch 1
expect_failure "impossible geometry" "impossible model configuration" \
    "$executable" train --data "$temporary_dir/train.txt" \
    --steps 1 --layers 1 --heads 2 --width 7 --block 4 --batch 1
expect_failure "memory preflight" "model needs" \
    "$executable" train --data "$temporary_dir/train.txt" \
    --steps 1 --layers 1 --heads 1 --width 16384 --block 4 --batch 1
expect_failure "training output must not overwrite its input" "--out" \
    "$executable" train --data "$temporary_dir/train.txt" \
    --out "$temporary_dir/train.txt" --steps 1 \
    --layers 1 --heads 1 --width 8 --block 4 --batch 1
expect_failure "validation must be a distinct file" "distinct files" \
    "$executable" train --data "$temporary_dir/train.txt" \
    --val-data "$temporary_dir/train-alias.txt" --steps 1 \
    --layers 1 --heads 1 --width 8 --block 4 --batch 1
expect_failure "validation byte absent from training" "absent" \
    "$executable" train --data "$temporary_dir/train.txt" \
    --val-data "$temporary_dir/unknown-validation.txt" --steps 1 \
    --layers 1 --heads 1 --width 8 --block 4 --batch 1
expect_failure "missing checkpoint" "$temporary_dir/missing.bin" \
    "$executable" sample --model "$temporary_dir/missing.bin"
expect_failure "malformed temperature" "--temperature" \
    "$executable" sample --model "$temporary_dir/missing.bin" \
    --temperature infinity

model_path="$temporary_dir/trained.bin"
expect_success "one valid training step with the safe default output" \
    run_from_temporary_dir "$executable" train \
    --data "$temporary_dir/train.txt" \
    --steps 1 --layers 1 --heads 1 --width 8 \
    --block 4 --batch 1 --lr 0.001 --seed 7
if [[ ! -s $model_path ]] \
   || ! grep -qF 'tiny-agenc:' "$temporary_dir/out" \
   || ! grep -qF 'checkpoint saved' "$temporary_dir/out"; then
    printf 'check-14: valid training did not produce its checkpoint contract\n' >&2
    exit 1
fi

if ! run_captured "$executable" sample --model "$model_path" \
        --prompt ab --length 4 --temperature 0.8 --seed 7; then
    printf 'check-14: valid sampling failed\n' >&2
    sed 's/^/check-14: stderr: /' "$temporary_dir/err" >&2
    exit 1
fi
if [[ ! -s $temporary_dir/out ]] \
   || grep -qF 'parameters |' "$temporary_dir/out"; then
    printf 'check-14: sampled text stdout is empty or contains the banner\n' >&2
    exit 1
fi
if [[ ! -s $temporary_dir/err ]] \
   || ! head -n 1 "$temporary_dir/err" | grep -q '^tiny-agenc: .*parameters'; then
    printf 'check-14: sampling architecture banner is absent from stderr\n' >&2
    exit 1
fi

printf 'check-14: information, dispatch, validation, and stream contracts passed\n'
