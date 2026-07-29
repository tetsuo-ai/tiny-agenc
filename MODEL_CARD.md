# Tiny AgenC Model Card

## Model

The bundled `tiny-agenc.bin` is a compact character-level
autoregressive transformer trained from scratch on NIGHT GRID.

| Property | Value |
|---|---:|
| Artifact version | 1.0.0 |
| Metadata date | 2026-07-26 |
| Parameters | 815,360 |
| Transformer blocks | 4 |
| Attention heads | 4 |
| Residual width | 128 |
| Maximum context | 128 byte tokens |
| Maximum training batch | 32 sequences |
| Vocabulary | 80 sorted ASCII bytes |
| Training steps | 5,000 |
| Seed | 1,337 |

Its checkpoint uses Tiny AgenC's same-platform TAGC version 1 format.
It contains the configuration, canonical byte vocabulary, parameter
values, and a CRC32 corruption check. It does not contain optimizer
moments or enough state to resume the exact training trajectory.

Checkpoint SHA-256:
`b425ee5d0a2b09168114fa4fd7e755725fa63f4809b1cdb3034f9603b131863d`.
The bundled artifact was migrated once from the retired framing by
changing only its magic and appending CRC32. Configuration, vocabulary,
and every learned float bit were compared byte for byte.
The same seeded prompt produced the same token sequence through both
loaders.

## Intended use

This is an educational artifact. It supports the book's sampling
examples, checkpoint exercises, and source-level experiments. It is
useful for learning how a small autoregressive transformer works end to
end.

It is not an assistant, a factual reference, a safety classifier, or a
model for decisions about people. Its short context, synthetic training
set, and tiny capacity make it unsuitable for production language tasks.

## Training data

NIGHT GRID is roughly one MiB of synthetic cyberpunk transmission logs.
It was generated with a locally operated language model and filtered
through a strict whitelist of transmission headers, known speaker tags,
ASCII text, and record boundaries. The generator and cleaner are in
`scripts/gen-corpus.sh` and `scripts/clean-corpus.sh`.

The exact generator-model digest was not recorded, and generation is
stochastic. `data/cyberpunk.raw.txt` is therefore the canonical source
artifact. Cleaning and seeded splitting are reproducible from that file;
raw generation is not claimed to be bit-for-bit reproducible.

The canonical raw artifact is 1,200,000 bytes with SHA-256
`38151a5b4b6a813d66488540b3b44cc678f9b28df6c85f4ab5f52408b63e811e`.
The historical generation log ended at 1,200,020 bytes. Those historical
extra bytes were not retained, and this repository does not claim they
can be reconstructed. [`EVIDENCE.md`](EVIDENCE.md) separates that
historical observation from the reproducible path beginning at the
committed canonical file.

The bundled checkpoint was trained on the complete cleaned corpus. It is
therefore a fitting and sampling artifact, not held-out evidence. The
book records a separate deterministic 90/10 scene-level run for
validation claims.

## Evaluation

The repository includes:

- exact and independent forward witnesses;
- operation-level and whole-model finite-difference checks;
- a one-batch overfit test;
- deterministic save, load, and sampling checks;
- a matched add-one bigram validation baseline;
- serial and OpenMP arithmetic-invariance checks;
- malformed-input and hostile-checkpoint tests.

The full commands and measured training evidence live in Chapter 17,
`book/logs/`, and [`EVIDENCE.md`](EVIDENCE.md). Results are narrow
measurements of this implementation and corpus. They are not claims
about general language understanding.

## Limitations

The model often learns speaker tags, local spelling, and transmission
format while producing fragile or incoherent content. It can reproduce
or remix training phrases. Synthetic data can still contain stereotypes,
unsafe language, or artifacts inherited from its generator. No safety
fine-tuning or output filter is present.

The checkpoint stores native 32-bit integers and floats. Use it on a
compatible platform and let the loader verify its magic, dimensions,
vocabulary, payload length, finite values, resource ceiling, and CRC32.

## Artifact license

Copyright 2026 Tiny AgenC contributors.

The bundled checkpoint is licensed under the
[Creative Commons Attribution 4.0 International
License](https://creativecommons.org/licenses/by/4.0/).

Attribution: "Tiny AgenC model by Tiny AgenC contributors."
