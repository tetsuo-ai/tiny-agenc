# Tiny AgenC Evidence Record

This file separates three different promises:

1. the committed inputs and recorded artifacts have exact identities;
2. cleaning, splitting, baselines, loading, and sampling can be replayed
   quickly;
3. the two measured 5,000-step training runs can be replayed when the
   recorded compiler, flags, source, data, and seeds are available.

`make check-evidence` verifies the first two promises. It does not
pretend to rerun eight minutes of training in every ordinary test pass.
The full training commands and environment remain explicit below.

## Canonical data boundary

Raw generation was stochastic and its generator-model digest was not
recorded. The historical generator output reported 1,200,020 bytes, but
that exact byte stream was not retained. The committed
`data/cyberpunk.raw.txt` is the canonical starting point:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| canonical raw corpus | 1,200,000 | `38151a5b4b6a813d66488540b3b44cc678f9b28df6c85f4ab5f52408b63e811e` |
| cleaned corpus | 1,089,394 | `cbf7f8326ddba8b345414c84daaa07c2df7b2f74ae5fcd33ff30eba08bdb658c` |
| training split | 982,693 | `81e4d8744add73032f5df569582272a4a28fc89a032a18d3a0c52125de6fbae7` |
| validation split | 106,701 | `ec24c14fca09facba0bb3283b17a8c06c58eff45df91e362af5419ac80d8cf31` |

Cleaning the canonical raw file produces 2,011 complete transmissions
and an 80-byte vocabulary. The seeded 90/10 record split produces 1,809
training transmissions and 202 validation transmissions.

The missing historical 20 bytes are not recoverable from this
repository, and no result depends on pretending otherwise. Every
training claim begins at the canonical raw artifact and the cleaned
corpus hash above.

## Recorded build

The measured runs used:

- AMD Ryzen Threadripper PRO 9975WX, 32 physical cores;
- GCC 13.3.0 from Ubuntu 24.04;
- OpenMP enabled with `OMP_NUM_THREADS` unset;
- `-O3 -ffast-math -march=native`;
- seed 1337;
- repository commit
  `989ed6422b1a6f98874867ce016eb4b74532c614`;
- source-tree aggregate SHA-256
  `46c88bbcb83af0b52c103d7a5e70540b546dffff81a752139e25b8b1ce0cca5e`;
- executable SHA-256
  `b86086c42993fb6f64ffdcad6c0315555984b619855f12251b54368719967557`.

The repository now defaults to a CPU-portable build. Use `NATIVE=1` to
request the recorded `-march=native` setting. Exact floating-point
replay remains a property of a compatible toolchain and machine;
behavioral checks remain the cross-platform contract.

The historical build identity above and the training results in the
following sections belong to that historical source snapshot. Commit
`5dab9eb` changed book diagrams only and has the same source aggregate.
Current hardening is pinned and verified separately below. It is not
relabeled as the implementation that produced the historical
measurements.

## Hardened-source regression replay

The 1.0.0 source aggregate, computed with the same sorted-source-hash
procedure, is:

```text
76a076b3e0f78a87bcbfa0756ba1efcd4770143a31b679a8d8f5b9300e01be8c
```

The defensive hardening changed source code around input limits,
checkpoint preflight, CLI diagnostics, and optimizer rejection. To
detect an accidental change to ordinary finite training arithmetic, an
independent audit built both commit
`989ed6422b1a6f98874867ce016eb4b74532c614` and the hardened tree with
GCC 13.3.0, OpenMP disabled, and the exact historical optimization flags
`-O3 -march=native -ffast-math`. The historical tree used
`make OPENMP=0`; the hardened tree used
`make OPENMP=0 NATIVE=0 OPTFLAGS="-O3 -march=native -ffast-math"`.
Both executables trained on
`data/cyberpunk.txt`, whose SHA-256 is
`cbf7f8326ddba8b345414c84daaa07c2df7b2f74ae5fcd33ff30eba08bdb658c`.
The witness used a one-layer, one-head, width-16 model with block 16,
batch 2, learning rate 0.001, and seed 1337:

```sh
for steps in 1 25 50; do
    build/compatibility/historical/tiny-agenc train \
        --data data/cyberpunk.txt \
        --out "build/compatibility/historical-${steps}.bin" \
        --steps "$steps" --layers 1 --heads 1 --width 16 \
        --block 16 --batch 2 --lr 0.001 --seed 1337
    build/compatibility/hardened/tiny-agenc train \
        --data data/cyberpunk.txt \
        --out "build/compatibility/hardened-${steps}.bin" \
        --steps "$steps" --layers 1 --heads 1 --width 16 \
        --block 16 --batch 2 --lr 0.001 --seed 1337
    cmp "build/compatibility/historical-${steps}.bin" \
        "build/compatibility/hardened-${steps}.bin"
    sha256sum "build/compatibility/historical-${steps}.bin" \
        "build/compatibility/hardened-${steps}.bin"
done
```

| Steps | Shared SHA-256 | Byte comparison |
|---:|---|---|
| 1 | `4256961c5ccb28c17d04b1a336a25f2e47f4f188cbbca21f02f4854a7b26826d` | exact |
| 25 | `b17ee0fa148082d744b853d2c42a16741334460f96ce7e215cacf2ed7b2551c2` | exact |
| 50 | `9afa53a84f930c6903db963263aaad206052a47a3c3ea6821ee602baf038736b` | exact |

The final hardened source was then built with `CC=cc OPENMP=1 NATIVE=1`.
That audit executable has SHA-256
`af4d022732742e4f17f1b32c43e47bff27c2744ac1097b77cb731282d3100942`.
It replayed both 5,000-step commands below. The full-corpus checkpoint
matched `b425ee5d0a2b09168114fa4fd7e755725fa63f4809b1cdb3034f9603b131863d`
exactly. The held-out checkpoint matched
`0753089369482f5ac9e4cb80c940c90b58f38493019ddcc4478ced61242350b8`
exactly, and all eight selected loss pairs matched the recorded summary.

These are post-hardening regression results. The historical executable
hash, timestamps, and performance measurements remain attributed only
to the historical measured source snapshot.

A later construction-boundary fix made `model_new` retain its checked
configuration and size preflight when assertions are disabled. Later
source cleanup divided the training command, bounded file reader,
tokenizer loader, and checkpoint path into small named stages while
retaining their operation order and TAGC version 1 payload. The source
aggregate after that cleanup was:

```text
16f4b7f49a92edb59d9640a9480ad4ddd9da0f283d968da2b15913e8ebb01b8b
```

`make check-ndebug` builds that source with `-DNDEBUG`, passes the
6,640 model gradient checks and 43 model integration checks, and
verifies that invalid geometry exits with failure. The valid-model
arithmetic is unchanged. The compatibility training replays above
remain attributed to the earlier `76a076...` hardened aggregate rather
than being relabeled as measurements of this later boundary fix.

The checkpoint refactor was also compared directly with commit
`af73d2d2561c03f55ce021d1b7c6d9bbb1c87179`. Both trees were built on
x86-64 Ubuntu 24.04 with GCC 13.3.0, OpenMP disabled, and the default
`-O3 -ffast-math` flags. They ran this one-step witness against
`labs/tiny-corpus.txt`, whose SHA-256 is
`ff1124610dc483a1b8ccafdb9650438ad98436bb143a130569d104da0161f980`:

```sh
./tiny-agenc train \
    --data labs/tiny-corpus.txt --out /tmp/tiny-agenc-witness.bin \
    --steps 1 --layers 1 --heads 1 --width 8 \
    --block 8 --batch 1 --seed 1337
```

The old and new 4,286-byte checkpoints compared equal and shared
SHA-256
`dd647b63ce090c2bb28e42f52a6431b07e295a7bbcb86d61c805f993c39d101d`.
The checkpoint failure harness separately pins a 121-byte deterministic
TAGC writer fixture to the pre-refactor bytes on its recorded
little-endian IEEE binary32 platform.

The numerical-stage refactor was compared directly with merged commit
`a4dfaae`. Both trees were built on x86-64 Ubuntu 24.04 with GCC
13.3.0, OpenMP disabled, and the default `-O3 -ffast-math` flags. Each
trained a one-layer, one-head, width-8 model for 50 steps against
`labs/tiny-corpus.txt`, using block 8, batch 1, and seed 1337. The
checkpoints compared equal and shared SHA-256
`4adcaab12765e7d2b0802851a2bbb64c723c90263dea725c2f5cbe7d0f1627e7`.
The merged baseline used:

```sh
make OPENMP=0
./tiny-agenc train \
    --data labs/tiny-corpus.txt \
    --out /tmp/tiny-agenc-pr2-baseline-a4dfaae.bin \
    --steps 50 --layers 1 --heads 1 --width 8 \
    --block 8 --batch 1 --seed 1337 \
    > /tmp/tiny-agenc-pr2-baseline-a4dfaae.log 2>&1
./tiny-agenc sample \
    --model /tmp/tiny-agenc-pr2-baseline-a4dfaae.bin \
    --prompt 'RAZR:' --length 24 --temperature 0.8 --seed 1337 \
    > /tmp/tiny-agenc-pr2-baseline-a4dfaae.sample \
    2> /tmp/tiny-agenc-pr2-baseline-a4dfaae.sample.err
```

The refactored checkout used the same arguments with current output
paths:

```sh
make OPENMP=0
./tiny-agenc train \
    --data labs/tiny-corpus.txt --out /tmp/tiny-agenc-pr2-current.bin \
    --steps 50 --layers 1 --heads 1 --width 8 \
    --block 8 --batch 1 --seed 1337 \
    > /tmp/tiny-agenc-pr2-current.log 2>&1
./tiny-agenc sample \
    --model /tmp/tiny-agenc-pr2-current.bin \
    --prompt 'RAZR:' --length 24 --temperature 0.8 --seed 1337 \
    > /tmp/tiny-agenc-pr2-current.sample \
    2> /tmp/tiny-agenc-pr2-current.sample.err
cmp /tmp/tiny-agenc-pr2-baseline-a4dfaae.bin \
    /tmp/tiny-agenc-pr2-current.bin
cmp /tmp/tiny-agenc-pr2-baseline-a4dfaae.sample \
    /tmp/tiny-agenc-pr2-current.sample
cmp /tmp/tiny-agenc-pr2-baseline-a4dfaae.sample.err \
    /tmp/tiny-agenc-pr2-current.sample.err
sha256sum /tmp/tiny-agenc-pr2-baseline-a4dfaae.bin \
    /tmp/tiny-agenc-pr2-current.bin \
    /tmp/tiny-agenc-pr2-current.sample \
    /tmp/tiny-agenc-pr2-current.sample.err
```

Sampling produced byte-identical standard output and diagnostics.
Their SHA-256 values were
`9e77d01c4baa58e2480be73defca3f0bd94f6c66892c266c009aa5ce621629d3`
and
`0fc63e99d7eb837b2f3010e1942f43fc6bee33cd1496ed82f9220e639dec81ba`.

The current source aggregate after that refactor is:

```text
b7620842985e2b00756ab5948f65592e088243080a255b63f5fb6e6ea368ff2b
```

## Full-corpus showcase

```sh
make NATIVE=1 corpus
make NATIVE=1
mkdir -p build/evidence
env -u OMP_NUM_THREADS ./tiny-agenc train \
    --data data/cyberpunk.txt \
    --out build/evidence/full-run.bin \
    --steps 5000 --layers 4 --heads 4 --width 128 \
    --block 128 --batch 32 --lr 0.001 --seed 1337
sha256sum build/evidence/full-run.bin
```

The recorded run began at loss `4.4395`, ended at loss `0.7668`, and
produced:

```text
b425ee5d0a2b09168114fa4fd7e755725fa63f4809b1cdb3034f9603b131863d
```

That checkpoint is the bundled `tiny-agenc.bin`. The complete stdout is
`book/logs/train-cyberpunk-5000.log`.

## Held-out validation

```sh
make NATIVE=1 validation-data
env -u OMP_NUM_THREADS ./tiny-agenc train \
    --data data/cyberpunk.train.txt \
    --val-data data/cyberpunk.val.txt \
    --out build/evidence/validation-run.bin \
    --steps 5000 --layers 4 --heads 4 --width 128 \
    --block 128 --batch 32 --lr 0.001 --seed 1337
sha256sum build/evidence/validation-run.bin
```

| Step | Training loss | Fixed validation loss |
|---:|---:|---:|
| 1 | 4.4463 | 4.0430 |
| 50 | 2.4230 | 2.4210 |
| 250 | 1.8567 | 1.8301 |
| 1,000 | 1.1218 | 1.1182 |
| 2,000 | 0.9470 | 0.9790 |
| 3,000 | 0.8414 | 0.9320 |
| 4,000 | 0.7987 | 0.9105 |
| 5,000 | 0.7638 | 0.9023 |

The validation checkpoint SHA-256 is
`0753089369482f5ac9e4cb80c940c90b58f38493019ddcc4478ced61242350b8`.
It is a recorded result, not a bundled artifact. Replaying its recorded
sample produces SHA-256
`8d8e1d20f37e9a54ba908277251a8a9b61f4cac6920aa86abd2c9f33c0cd144e`.

The add-one bigram baselines are:

- full validation file: loss `2.326050`, perplexity `10.2374`;
- matched fixed windows: loss `2.320951`, perplexity `10.1854`.

## Manifest coverage

`EVIDENCE.sha256` pins:

- canonical, cleaned, and split corpora;
- the bundled checkpoint;
- the build recipe and corpus-generation, cleaning, and splitting code;
- every reference implementation source and header;
- every C test, including the baseline implementation;
- the evidence checker and this record;
- the CLI, lab, installation, and release-metadata check scripts;
- all recorded training, validation, measurement, and showcase logs.

Changing a pinned file makes `make check-evidence` fail until the
evidence is deliberately reviewed. Updating a checksum is not a
substitute for rerunning any experiment affected by that change.
