# Contributing to Tiny AgenC

Tiny AgenC is an executable explanation. A change is ready when the code
is correct, the relevant lesson remains clear, and the evidence says what
it actually proves.

## Before changing code

Build a serial baseline:

```sh
make OPENMP=0 check-all
```

Use a focused target while working. `make check-forward`,
`make check-backward`, `make check-model`, and `make check-cli` provide
smaller witnesses. If a public behavior changes, add or strengthen a
test that fails for the old behavior.

## Book and lab changes

- Keep the direct, conversational voice already present.
- Do not use Unicode em dashes.
- Keep code claims synchronized with the reference implementation.
- Prefer links labeled with source symbols over brittle line anchors.
- Preserve the two tracks: reading the answer key and building every
  runtime module in `labs/work`.
- Run `make check-book` and `make check-labs`.

## Final verification

```sh
make OPENMP=0 check-all
make OPENMP=1 check-parallel
make check-sanitizers
```

If you change training math, data ordering, compiler flags, defaults, or
the checkpoint, regenerate the affected evidence and record the full
command, hashes, seed, and environment. Do not rewrite a measured claim
from memory. `make check-evidence` ties the current source, deterministic
split, bundled checkpoint, baseline, and recorded metrics together.

By submitting a contribution, you agree that software contributions are
licensed under Apache License 2.0. Book, corpus, and model-artifact
contributions are licensed under Creative Commons Attribution 4.0
International, as described in their local license notices.
