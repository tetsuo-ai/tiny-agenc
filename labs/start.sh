#!/usr/bin/env bash
#
# Create a separate build-track workspace without overwriting existing work.
set -euo pipefail

LABS_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY_ROOT=$(cd -- "$LABS_DIR/.." && pwd)
DESTINATION=${1:-"$LABS_DIR/work"}

if [[ -e $DESTINATION ]]; then
    printf 'start-lab: refusing to overwrite %s\n' "$DESTINATION" >&2
    exit 1
fi

mkdir -p -- "$DESTINATION"
cp -- "$REPOSITORY_ROOT"/src/*.h "$DESTINATION"/
cp -- "$LABS_DIR/mat.h.in" "$DESTINATION/mat.h"
cp -- "$LABS_DIR/model_internal.h.in" "$DESTINATION/model_internal.h"

for module in util rng tokenizer dataset ops param; do
    {
        printf '#include "%s.h"\n\n' "$module"
        printf '/* TODO: implement %s after reading its chapter. */\n' "$module"
    } > "$DESTINATION/$module.c"
done

for module in model model_parameters model_memory model_forward \
              model_backward checkpoint model_sampling; do
    {
        printf '#include "model_internal.h"\n\n'
        printf '/* TODO: implement %s after reading its chapter. */\n' "$module"
    } > "$DESTINATION/$module.c"
done

{
    printf '#include "model.h"\n'
    printf '#include "dataset.h"\n'
    printf '#include "tokenizer.h"\n\n'
    printf '/* TODO: implement the command line after Chapter 14. */\n'
} > "$DESTINATION/main.c"

{
    printf '#ifndef TINY_AGENC_LAB_SPEC_H\n'
    printf '#define TINY_AGENC_LAB_SPEC_H\n\n'
    printf '#include <stddef.h>\n\n'
    printf '#include "model.h"\n\n'
    printf 'int lab_config_valid(int vocab, int block, int width,\n'
    printf '                     int heads, int layers, int batch);\n'
    printf 'size_t lab_parameter_count(int vocab, int block, int width, int layers);\n\n'
    printf '#endif\n'
} > "$DESTINATION/spec.h"

{
    printf '#include "spec.h"\n\n'
    printf '/* TODO: make the Chapter 1 geometry executable. */\n'
} > "$DESTINATION/spec.c"

printf 'start-lab: created %s\n' "$DESTINATION"
printf 'start-lab: begin with `make -C %q WORK=%q check-01`\n' \
    "$LABS_DIR" "$DESTINATION"
