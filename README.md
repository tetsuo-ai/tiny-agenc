# Tiny AgenC

A character-level autoregressive transformer with multi-head causal
self-attention, layernorm, GELU, residual streams, AdamW, and
hand-written backpropagation. It trains from scratch on a CPU in about
2,700 lines of C11 implementation code, excluding headers and tests,
with no third-party C library dependencies.
Run it and watch noise become language in your terminal.

It ships with its own dataset: **NIGHT GRID**, ~1MB of cyberpunk
transmission logs synthesized by a local LLM. It is structured so
word-like text, recurring slang, speaker tags, and the transmission
format become increasingly recognizable during training.

```text
=== TRANSMISSION 1017 // SECTOR 3: BLACKOUT DISTRICT ===
RAZR: Found a trace, but it's gone cold like this district in blackout.
GHOST: The Grid's like a maze. No easy exits when you're marked.
DOC: We need to stay one step ahead of the chrome.
```

## Requirements

The reference environment is GNU/Linux. Building the program requires a
C11 compiler, GNU Make, the platform C library, and `libm`. The default
build also requires compiler support for OpenMP:

```sh
make                 # OpenMP build
make OPENMP=0        # serial build when OpenMP is unavailable
```

The default optimization flags produce a binary for compatible
GNU/Linux CPUs rather than tuning it to the build machine. Use
`make NATIVE=1` to add `-march=native`. The recorded 5,000-step evidence
used `NATIVE=1`; its exact compiler and executable hashes are recorded
with the logs.

## Quickstart

```sh
make                 # builds ./tiny-agenc with OpenMP by default
make corpus          # cleans the committed raw corpus without network access

./tiny-agenc sample --model tiny-agenc.bin --prompt "RAZR:" --temperature 0.8
./tiny-agenc train --data data/cyberpunk.txt --out trained.bin
./tiny-agenc sample --model trained.bin --prompt "RAZR:" --temperature 0.8

make check           # gradient checks plus integration contracts
make overfit         # prove the model can memorize one fixed batch
```

The default training run takes about four minutes on the recorded
32-core machine. `trained.bin` is ignored by Git, so training does not
replace the bundled, versioned checkpoint.

The corpus scripts target GNU/Linux and use Bash, `awk`, `grep`,
`iconv`, and GNU coreutils. `make corpus` only cleans the committed raw
artifact. `make generate-corpus` explicitly extends it and additionally
requires `curl`, `jq`, and a reachable Ollama server with the selected
model. `make data` fetches the pinned Shakespeare comparison corpus and
requires `curl` and `sha256sum`.

Training narrates itself: a loss line every 50 steps and a fresh sample of
the model's writing every 250, so the learning is visible. The core
architecture and training controls have flags
(`--layers --heads --width --block --batch --steps --lr --seed`).

For a held-out measurement, split only between complete transmissions
and pass the validation half explicitly:

```sh
make validation-data
./tiny-agenc train \
    --data data/cyberpunk.train.txt \
    --val-data data/cyberpunk.val.txt \
    --out trained-with-validation.bin
make bigram-baseline
```

Validation uses four fixed batches from its own random stream. It is a
stable comparison during one run, and it does not silently alter which
training windows or progress samples the model sees.

## The map

The stateful components use opaque interfaces in the style of Hanson's
*C Interfaces and Implementations*. `mat`, `ops`, `util`, and `main`
remain deliberately small value-type or function modules. Reading order,
if you read it like a book:

| Chapter | Files | What it teaches |
|---|---|---|
| Foundations | `util.[hc]`, `rng.[hc]` | error-checked wrappers; reproducible randomness (PCG32, Box-Muller) |
| Data | `tokenizer.[hc]`, `dataset.[hc]` | byte-level vocabularies; next-token training pairs |
| Shape | `mat.h` | a matrix is a pointer and a shape; views, not owners |
| Math | `ops.[hc]` | every transformer op as a `_forward`/`_backward` pair; attention is three readable loops; causality is a loop bound |
| Learning | `param.[hc]` | a learnable tensor with its AdamW state; bias correction, decoupled weight decay |
| The blueprint | `model.[hc]`, `model_internal.h`, `model_parameters.c` | public contracts, private shapes, parameter order, and tied embeddings |
| The arenas | `model_memory.c` | checked memory estimates, ownership, and two-pass placement |
| The graph | `model_forward.c`, `model_backward.c` | compact pre-norm decoder wiring in both directions |
| Persistence | `checkpoint.c`, `model_sampling.c` | defensive save/load and autoregressive generation |
| The referee | `tests/gradcheck.c`, `tests/integration.c` | central differences, independent forward examples, contracts, save/load, and a tiny smoke run |
| The terminal | `main.c` | command-line parsing and user-facing output |

The full walk-through lives in [`book/`](book/README.md). It can be
read as a guided source tour or followed as a build-along. The staged
[`labs/`](labs/README.md) workspace lets you implement one subsystem at
a time. Focused witnesses isolate early modules with known-good support;
later checks integrate more and more of your own workspace. The
completed `src/` tree remains the answer key.

Useful verification targets:

```sh
make check-foundations  # allocation and deterministic RNG contracts
make check-data         # tokenizer and next-token batches
make check-forward      # independent forward witnesses
make check-backward     # finite differences for every operation backward
make check-optimizer    # AdamW, clipping, and update policy
make check-model        # exact architecture and whole-model gradients
make check-sampling     # temperature, categorical draws, and fresh context
make check-labs         # rebuild every runtime module through 16 stages
make check-evidence     # verify the evidence manifest and recorded metrics
make check-metadata     # keep release and compiled versions synchronized
make check-install      # stage and exercise the installed program and model
make check-sanitizers   # core suite under AddressSanitizer and UBSan
make check-all          # full math, CLI, labs, overfit, and book checks
```

## Provenance

The NIGHT GRID corpus can be extended with `make generate-corpus` (any
Ollama model; `dolphin3` by default) and is distilled by
`scripts/clean-corpus.sh`, which enforces a line grammar as a whitelist:
lines that aren't a well-formed transmission header or dialogue from a
known handle simply don't survive. Training accepts any sufficiently
long plain-text file; the default progress sampler also requires the
corpus to contain a newline.

The bundled checkpoint is documented in [`MODEL_CARD.md`](MODEL_CARD.md).
Measured training and validation evidence lives in Chapter 17 rather
than being inferred from a pleasing sample. The central
[`EVIDENCE.md`](EVIDENCE.md) explains what is reproducible, and
[`EVIDENCE.sha256`](EVIDENCE.sha256) pins every local input and recorded
artifact checked by `make check-evidence`.

## Install

The serial executable is self-contained apart from the platform C
library and `libm`. Installation also includes the bundled checkpoint,
licenses, version metadata, model card, and evidence record:

```sh
make OPENMP=0
make OPENMP=0 DESTDIR=/tmp/tiny-agenc-package PREFIX=/usr install

/tmp/tiny-agenc-package/usr/bin/tiny-agenc sample \
    --model /tmp/tiny-agenc-package/usr/share/tiny-agenc/tiny-agenc.bin \
    --prompt "RAZR:"
```

Omit `DESTDIR` for a normal installation. `PREFIX` defaults to
`/usr/local`; with that prefix, the model is installed at
`/usr/local/share/tiny-agenc/tiny-agenc.bin` and the reading material at
`/usr/local/share/doc/tiny-agenc/`. The complete book and build labs
remain part of the source distribution rather than the runtime install.

## Licensing and citation

The C source, tests, labs, scripts, build files, and CI configuration
are licensed under [Apache License 2.0](LICENSE). The
[book](book/LICENSE.md), [NIGHT GRID corpus](data/LICENSE.md), and
bundled checkpoint are licensed under Creative Commons Attribution 4.0
International. Attribution and artifact details are in [`NOTICE`](NOTICE)
and [`MODEL_CARD.md`](MODEL_CARD.md). Citation metadata is available in
[`CITATION.cff`](CITATION.cff), and release history is in
[`CHANGELOG.md`](CHANGELOG.md).
