#!/usr/bin/env bash
#
# Exercise the public command line without depending on generated data.
set -euo pipefail

EXECUTABLE=${1:-./tiny-agenc}
REPOSITORY_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

case "$EXECUTABLE" in
    /*) ;;
    *) EXECUTABLE="$REPOSITORY_ROOT/${EXECUTABLE#./}" ;;
esac

if [[ ! -x $EXECUTABLE ]]; then
    printf 'smoke-cli: executable not found: %s\n' "$EXECUTABLE" >&2
    exit 1
fi

temporary_dir=$(mktemp -d)
cleanup() {
    rm -rf -- "$temporary_dir"
}
trap cleanup EXIT

training_data="$temporary_dir/train.txt"
validation_data="$temporary_dir/validation.txt"
split_source="$temporary_dir/records.txt"
split_train="$temporary_dir/records.train.txt"
split_validation="$temporary_dir/records.val.txt"
split_train_replay="$temporary_dir/records.replay.train.txt"
split_validation_replay="$temporary_dir/records.replay.val.txt"
split_train_other="$temporary_dir/records.other.train.txt"
split_validation_other="$temporary_dir/records.other.val.txt"
checkpoint="$temporary_dir/model.bin"
checkpoint_without_validation="$temporary_dir/model-without-validation.bin"
training_log="$temporary_dir/train.log"

expect_failure() {
    local label=$1
    shift

    if "$@" > "$temporary_dir/failure.out" 2> "$temporary_dir/failure.err"; then
        printf 'smoke-cli: expected failure: %s\n' "$label" >&2
        exit 1
    fi
    if [[ ! -s $temporary_dir/failure.err ]]; then
        printf 'smoke-cli: failure had no diagnostic: %s\n' "$label" >&2
        exit 1
    fi
    local first_line

    IFS= read -r first_line < "$temporary_dir/failure.err"
    if [[ $first_line != tiny-agenc:* ]]; then
        printf 'smoke-cli: unprefixed diagnostic: %s: %s\n' \
            "$label" "$first_line" >&2
        exit 1
    fi
}

expect_failure_contains() {
    local label=$1
    local expected=$2
    shift 2

    expect_failure "$label" "$@"
    if ! grep -qF -- "$expected" "$temporary_dir/failure.err"; then
        printf 'smoke-cli: diagnostic for %s did not contain %s\n' \
            "$label" "$expected" >&2
        exit 1
    fi
}

"$EXECUTABLE" --help \
    > "$temporary_dir/help.out" 2> "$temporary_dir/help.err"
test ! -s "$temporary_dir/help.err"
grep -qF 'tiny-agenc train --data FILE' "$temporary_dir/help.out"
grep -qF 'tiny-agenc sample --model FILE' "$temporary_dir/help.out"
grep -qF 'tiny-agenc --help' "$temporary_dir/help.out"
grep -qF 'tiny-agenc --version' "$temporary_dir/help.out"

printf 'tiny-agenc 1.0.0\n' > "$temporary_dir/version.expected"
"$EXECUTABLE" --version \
    > "$temporary_dir/version.out" 2> "$temporary_dir/version.err"
test ! -s "$temporary_dir/version.err"
cmp "$temporary_dir/version.expected" "$temporary_dir/version.out"

expect_failure_contains "missing subcommand" "usage:" "$EXECUTABLE"
expect_failure_contains "unknown subcommand" "usage:" \
    "$EXECUTABLE" unknown
expect_failure_contains "subcommand help is not ambiguous" "usage:" \
    "$EXECUTABLE" train --help
expect_failure_contains "train requires data" "--data" \
    "$EXECUTABLE" train
expect_failure_contains "sample requires an explicit checkpoint" "--model" \
    "$EXECUTABLE" sample
expect_failure_contains "unknown train option" "usage:" \
    "$EXECUTABLE" train --wat
expect_failure_contains "missing train option argument" "usage:" \
    "$EXECUTABLE" train --data

for transmission in 0001 0002 0003 0004 0005 0006 0007 0008 0009 0010; do
    {
        printf '=== TRANSMISSION %s // SECTOR 1: GRID ===\n' "$transmission"
        printf 'RAZR: GRID GRID.\n'
        printf 'DOC: GRID RAZR 0123456789.\n\n'
    } >> "$split_source"
done

VALIDATION_PERCENT=20 \
    bash "$REPOSITORY_ROOT/scripts/split-corpus.sh" \
    "$split_source" "$split_train" "$split_validation" \
    > "$temporary_dir/split.log"

VALIDATION_PERCENT=20 \
    bash "$REPOSITORY_ROOT/scripts/split-corpus.sh" \
    "$split_source" "$split_train_replay" "$split_validation_replay" \
    > "$temporary_dir/split-replay.log"
cmp "$split_train" "$split_train_replay"
cmp "$split_validation" "$split_validation_replay"

VALIDATION_PERCENT=20 SPLIT_SEED=1338 \
    bash "$REPOSITORY_ROOT/scripts/split-corpus.sh" \
    "$split_source" "$split_train_other" "$split_validation_other" \
    > "$temporary_dir/split-other.log"
if cmp -s "$split_train" "$split_train_other" \
   && cmp -s "$split_validation" "$split_validation_other"; then
    printf 'smoke-cli: changing the split seed did not change the split\n' >&2
    exit 1
fi

if [[ $(grep -c '^=== TRANSMISSION' "$split_train") -ne 8 ]] \
    || [[ $(grep -c '^=== TRANSMISSION' "$split_validation") -ne 2 ]]; then
    printf 'smoke-cli: scene-level split produced the wrong record counts\n' >&2
    exit 1
fi
sort "$split_source" > "$temporary_dir/source-lines.txt"
sort "$split_train" "$split_validation" > "$temporary_dir/split-lines.txt"
cmp "$temporary_dir/source-lines.txt" "$temporary_dir/split-lines.txt"

cp -- "$split_train" "$training_data"
cp -- "$split_validation" "$validation_data"

collision_source="$temporary_dir/collision-source.txt"
collision_backup="$temporary_dir/collision-source.backup"
collision_hardlink="$temporary_dir/collision-hardlink.txt"
collision_symlink="$temporary_dir/collision-symlink.txt"
cp -- "$training_data" "$collision_source"
cp -- "$collision_source" "$collision_backup"
ln "$collision_source" "$collision_hardlink"
ln -s "$collision_source" "$collision_symlink"

collision_train=(
    train --steps 1 --layers 1 --heads 1 --width 8
    --block 8 --batch 1 --seed 1337
)

expect_failure_contains "output cannot overwrite training data" "--data" \
    "$EXECUTABLE" "${collision_train[@]}" \
    --data "$collision_source" --out "$collision_source"
cmp "$collision_backup" "$collision_source"
expect_failure_contains "hardlink output alias cannot overwrite training data" \
    "--data" "$EXECUTABLE" "${collision_train[@]}" \
    --data "$collision_source" --out "$collision_hardlink"
cmp "$collision_backup" "$collision_source"
expect_failure_contains "symlink output alias cannot overwrite training data" \
    "--data" "$EXECUTABLE" "${collision_train[@]}" \
    --data "$collision_source" --out "$collision_symlink"
cmp "$collision_backup" "$collision_source"
expect_failure_contains "output cannot overwrite validation data" "--val-data" \
    "$EXECUTABLE" "${collision_train[@]}" \
    --data "$training_data" --val-data "$collision_source" \
    --out "$collision_hardlink"
cmp "$collision_backup" "$collision_source"
expect_failure_contains "training and validation data must be distinct" \
    "distinct files" "$EXECUTABLE" "${collision_train[@]}" \
    --data "$collision_source" --val-data "$collision_source" \
    --out "$temporary_dir/distinct-output.bin"
expect_failure_contains "data aliases must also be distinct" \
    "distinct files" "$EXECUTABLE" "${collision_train[@]}" \
    --data "$collision_source" --val-data "$collision_hardlink" \
    --out "$temporary_dir/distinct-alias-output.bin"
cmp "$collision_backup" "$collision_source"

"$EXECUTABLE" train \
    --data "$training_data" \
    --val-data "$validation_data" \
    --out "$checkpoint" \
    --steps 2 --layers 1 --heads 1 --width 8 \
    --block 8 --batch 2 --seed 1337 \
    > "$training_log"

test -s "$checkpoint"
grep -qE 'step[[:space:]]+1/2 .* val ' "$training_log"
grep -qF 'checkpoint saved to' "$training_log"

"$EXECUTABLE" train \
    --data "$training_data" \
    --out "$checkpoint_without_validation" \
    --steps 2 --layers 1 --heads 1 --width 8 \
    --block 8 --batch 2 --seed 1337 \
    > "$temporary_dir/train-without-validation.log"

cmp "$checkpoint" "$checkpoint_without_validation"

"$EXECUTABLE" sample --model "$checkpoint" \
    --prompt 'RAZR:' --length 24 --seed 1337 \
    > "$temporary_dir/sample-a.txt" 2> "$temporary_dir/sample-a.err"
"$EXECUTABLE" sample --model "$checkpoint" \
    --prompt 'RAZR:' --length 24 --seed 1337 \
    > "$temporary_dir/sample-b.txt" 2> "$temporary_dir/sample-b.err"
"$EXECUTABLE" sample --model "$checkpoint" \
    --prompt 'RAZR:@' --length 24 --seed 1337 \
    > "$temporary_dir/sample-unknown.txt" 2> "$temporary_dir/sample-unknown.err"

cmp "$temporary_dir/sample-a.txt" "$temporary_dir/sample-b.txt"
cmp "$temporary_dir/sample-a.err" "$temporary_dir/sample-b.err"
cmp "$temporary_dir/sample-a.txt" "$temporary_dir/sample-unknown.txt"
cmp "$temporary_dir/sample-a.err" "$temporary_dir/sample-unknown.err"

printf 'short\n' > "$temporary_dir/short.txt"
printf '@@@@@@@@@@@@@@@@@\n' > "$temporary_dir/unseen.txt"
printf 'not a checkpoint\n' > "$temporary_dir/malformed.bin"
truncate -s $((256 * 1024 * 1024 + 1)) "$temporary_dir/oversized.txt"
awk 'BEGIN { for (i = 0; i < 70000; i++) printf "%s", i % 2 ? "a" : "\n" }' \
    > "$temporary_dir/resource-corpus.txt"

common_train=(
    train --data "$training_data" --out "$temporary_dir/failed.bin"
    --steps 1 --layers 1 --heads 1 --width 8
    --block 8 --batch 1 --seed 1337
)

expect_failure "missing validation file" \
    "$EXECUTABLE" "${common_train[@]}" \
    --val-data "$temporary_dir/absent.txt"
expect_failure "undersized validation corpus" \
    "$EXECUTABLE" "${common_train[@]}" \
    --val-data "$temporary_dir/short.txt"
expect_failure "validation bytes absent from training vocabulary" \
    "$EXECUTABLE" "${common_train[@]}" \
    --val-data "$temporary_dir/unseen.txt"
expect_failure_contains "oversized corpus is rejected before allocation" \
    "exceeds the platform limit" \
    "$EXECUTABLE" train --data "$temporary_dir/oversized.txt" \
    --out "$temporary_dir/oversized.bin" --steps 1 \
    --layers 1 --heads 1 --width 8 --block 8 --batch 1
expect_failure "malformed numeric flag" \
    "$EXECUTABLE" train --data "$training_data" --steps nope
expect_failure "non-finite learning rate" \
    "$EXECUTABLE" train --data "$training_data" --lr NaN
expect_failure "negative seed" \
    "$EXECUTABLE" train --data "$training_data" --seed -1
expect_failure "invalid model geometry" \
    "$EXECUTABLE" train --data "$training_data" --steps 1 \
    --layers 1 --heads 2 --width 7 --block 8 --batch 1
expect_failure "model exceeds the CLI memory ceiling" \
    "$EXECUTABLE" train --data "$temporary_dir/resource-corpus.txt" \
    --steps 1 --layers 1 --heads 256 --width 256 \
    --block 65536 --batch 16
expect_failure "missing sample checkpoint" \
    "$EXECUTABLE" sample --model "$temporary_dir/absent.bin"
expect_failure "malformed sample checkpoint" \
    "$EXECUTABLE" sample --model "$temporary_dir/malformed.bin"
expect_failure_contains "extra command argument" "usage:" \
    "$EXECUTABLE" sample --model "$checkpoint" extra

printf 'smoke-cli: help, path safety, size policy, train, save/load, sampling, and diagnostics passed\n'
