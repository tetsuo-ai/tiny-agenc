#!/usr/bin/env bash
#
# Prove that every staged linker boundary still accepts the answer key.
set -euo pipefail

REPOSITORY_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
temporary_dir=$(mktemp -d)

cleanup() {
    make -C "$REPOSITORY_ROOT/labs" clean >/dev/null 2>&1 || true
    rm -rf -- "$temporary_dir"
}
trap cleanup EXIT

work="$temporary_dir/work"
bash "$REPOSITORY_ROOT/labs/start.sh" "$work" >/dev/null
cp -- "$REPOSITORY_ROOT"/src/*.c "$work"/
cp -- "$REPOSITORY_ROOT/src/mat.h" "$work/mat.h"
cp -- "$REPOSITORY_ROOT/src/model_internal.h" "$work/model_internal.h"
cp -- "$REPOSITORY_ROOT/labs/answer-spec.c" "$work/spec.c"

make -C "$REPOSITORY_ROOT/labs" clean >/dev/null
for stage in 01 02 03 04 05 06 08 09 10 11 12 13 14 15 16; do
    make -C "$REPOSITORY_ROOT/labs" WORK="$work" "check-$stage"
done

if [[ ${OPENMP:-0} == 1 ]]; then
    make -C "$REPOSITORY_ROOT/labs" WORK="$work" check-parallel
fi

printf 'check-labs: every staged answer-key boundary passed\n'
