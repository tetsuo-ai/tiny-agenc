#!/usr/bin/env bash
#
# Split a cleaned corpus between complete blank-line-delimited records.
#
# Usage: split-corpus.sh [source [train-output [validation-output]]]
# Set VALIDATION_PERCENT to an integer from 1 through 50 (default 10).
# Set SPLIT_SEED to an unsigned integer (default 1337).
set -euo pipefail

REPOSITORY_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE=${1:-data/cyberpunk.txt}
TRAIN_OUTPUT=${2:-data/cyberpunk.train.txt}
VALIDATION_OUTPUT=${3:-data/cyberpunk.val.txt}
VALIDATION_PERCENT=${VALIDATION_PERCENT:-10}
SPLIT_SEED=${SPLIT_SEED:-1337}
ORDER_HELPER=${SPLIT_ORDER_HELPER:-"$REPOSITORY_ROOT/build/split-order"}

cleanup() {
    rm -f -- "$temp_train" "$temp_validation" "$assignments" \
        "$temp_dir/train.backup" "$temp_dir/validation.backup"
    rmdir -- "$temp_dir" 2>/dev/null || true
}

restore_train_output() {
    if [[ -e $temp_dir/train.backup ]]; then
        mv -- "$temp_dir/train.backup" "$TRAIN_OUTPUT"
        return
    fi
    rm -f -- "$TRAIN_OUTPUT"
}

if [[ ! $VALIDATION_PERCENT =~ ^[0-9]{1,2}$ ]]; then
    printf 'split-corpus: VALIDATION_PERCENT must be an integer in [1, 50]\n' >&2
    exit 1
fi
validation_percent_decimal=$((10#$VALIDATION_PERCENT))
if ((validation_percent_decimal < 1 || validation_percent_decimal > 50)); then
    printf 'split-corpus: VALIDATION_PERCENT must be an integer in [1, 50]\n' >&2
    exit 1
fi
if [[ ! $SPLIT_SEED =~ ^[0-9]+$ ]]; then
    printf 'split-corpus: SPLIT_SEED must be an unsigned integer\n' >&2
    exit 1
fi
if [[ ! -x $ORDER_HELPER ]]; then
    printf 'split-corpus: missing %s; run `make build/split-order`\n' \
        "$ORDER_HELPER" >&2
    exit 1
fi
if [[ ! -r $SOURCE ]]; then
    printf 'split-corpus: cannot read %s\n' "$SOURCE" >&2
    exit 1
fi

source_path=$(realpath -m -- "$SOURCE")
train_path=$(realpath -m -- "$TRAIN_OUTPUT")
validation_path=$(realpath -m -- "$VALIDATION_OUTPUT")

if [[ $source_path == "$train_path" || $source_path == "$validation_path" \
      || $train_path == "$validation_path" ]]; then
    printf 'split-corpus: source, train, and validation paths must differ\n' >&2
    exit 1
fi

train_dir=$(dirname -- "$TRAIN_OUTPUT")
validation_dir=$(dirname -- "$VALIDATION_OUTPUT")
mkdir -p -- "$train_dir" "$validation_dir"

train_dir_path=$(realpath -- "$train_dir")
validation_dir_path=$(realpath -- "$validation_dir")
if [[ $train_dir_path != "$validation_dir_path" ]]; then
    printf 'split-corpus: train and validation outputs must share a directory\n' >&2
    exit 1
fi

temp_dir=$(mktemp -d "$train_dir/.tiny-agenc-split.XXXXXX")
temp_train="$temp_dir/train"
temp_validation="$temp_dir/validation"
assignments="$temp_dir/assignments"

trap cleanup EXIT

record_count=$(awk 'BEGIN { RS = "" } END { print NR }' "$SOURCE")
if ((record_count < 2)); then
    printf 'split-corpus: need at least two blank-line-delimited records\n' >&2
    exit 1
fi

train_count=$((record_count * (100 - validation_percent_decimal) / 100))
if ((train_count < 1)); then
    train_count=1
fi
if ((train_count >= record_count)); then
    train_count=$((record_count - 1))
fi
validation_count=$((record_count - train_count))

"$ORDER_HELPER" "$record_count" "$validation_count" "$SPLIT_SEED" \
    > "$assignments"

awk -v train="$temp_train" \
    -v validation="$temp_validation" \
    -v assignments="$assignments" '
    BEGIN {
        RS = "\n"
        while ((getline destination < assignments) > 0) {
            count++
            split_to[count] = destination
        }
        close(assignments)
        RS = ""
        ORS = "\n\n"
    }
    {
        destination = split_to[NR] == "validation" ? validation : train
        print $0 > destination
    }
' "$SOURCE"

if [[ -e $TRAIN_OUTPUT ]]; then
    chmod --reference="$TRAIN_OUTPUT" "$temp_train"
fi
if [[ -e $VALIDATION_OUTPUT ]]; then
    chmod --reference="$VALIDATION_OUTPUT" "$temp_validation"
fi

if [[ -e $TRAIN_OUTPUT ]]; then
    cp -p -- "$TRAIN_OUTPUT" "$temp_dir/train.backup"
fi
if [[ -e $VALIDATION_OUTPUT ]]; then
    cp -p -- "$VALIDATION_OUTPUT" "$temp_dir/validation.backup"
fi

mv -- "$temp_train" "$TRAIN_OUTPUT"
if ! mv -- "$temp_validation" "$VALIDATION_OUTPUT"; then
    restore_train_output
    printf 'split-corpus: could not install both outputs; restored train output\n' >&2
    exit 1
fi

rm -f -- "$temp_dir/train.backup" "$temp_dir/validation.backup" "$assignments"
rmdir -- "$temp_dir"
trap - EXIT

train_records=$(grep -cE '^=== TRANSMISSION' "$TRAIN_OUTPUT" || true)
validation_records=$(grep -cE '^=== TRANSMISSION' "$VALIDATION_OUTPUT" || true)

printf 'split-corpus: train %s bytes, %s transmissions -> %s\n' \
    "$(stat -c%s "$TRAIN_OUTPUT")" "$train_records" "$TRAIN_OUTPUT"
printf 'split-corpus: validation %s bytes, %s transmissions -> %s\n' \
    "$(stat -c%s "$VALIDATION_OUTPUT")" "$validation_records" \
    "$VALIDATION_OUTPUT"
printf 'split-corpus: deterministic scene shuffle seed %s\n' "$SPLIT_SEED"
