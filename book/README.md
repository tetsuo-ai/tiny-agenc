# Tiny AgenC: The Book

*Build an autoregressive transformer from first principles, in C, one
tested piece at a time.*

Tiny AgenC is a working character-level language model with causal
multi-head attention, layer normalization, GELU feed-forward blocks,
residual streams, tied embeddings, AdamW, and backpropagation written
by hand. This book explains the machine and gives you a path to rebuild
it.

Throughout the book, **autoregressive transformer** means a causal
decoder trained to predict the next token. The name describes what the
machine does instead of borrowing a product identity. `GPT-2` appears
only when we refer to the historical model or source that influenced a
specific design choice.

NIGHT GRID is normalized to ASCII, so one byte is one character token
for this project. Newlines and spaces are tokens too, even though they
are not visible glyphs. This convenient byte-to-token equality does not
hold for general UTF-8 text.

The prose and the source have different jobs. The prose supplies the
map, the derivations, the engineering decisions, and the checkpoints
that tell you whether you are still on the road. The source in
[`src/`](../src) is the complete reference implementation. Code excerpts
come from that implementation and are labeled when shortened.

## Who this is for

You need to be able to program, in any language you like. Nothing
else is assumed: the C idioms, the math, and machine learning itself
are each taught the first time they appear. The full contract is in
[Chapter 0](00-introduction.md#who-this-is-for).

The reference environment is GNU/Linux with a C11 compiler, `libm`, and
the libacl development headers and library. OpenMP is optional. The
default model uses about 380 MiB while training. The first-light model
later in the book is deliberately much smaller.

## Two ways through

**The reading track.** Keep the completed source open, follow the
chapters in order, and run each verification command. This is the
fastest route from "transformers feel magical" to "I can point at every
loop."

**The building track.** Work in your own implementation directory and
use [`labs/`](../labs/README.md) as the checklist. Read a chapter, write
the named functions without copying their bodies, then run the matching
check. Open the reference implementation only after your version passes
or when you have a specific question. The book tells you what each
stage must do, not merely what the finished program happens to contain.

Start that workspace from the repository root:

```sh
bash labs/start.sh
make -C labs check-01
```

The first check is expected to fail until you implement its TODO. The
building track is a multi-session project; each passing chapter leaves
you a small, working boundary to return to.

Commands of the form `make -C labs check-NN` test your building
workspace. Root targets such as `make check-forward` test the completed
answer key. The chapter checkpoints show both when the distinction is
useful.

## Quickstart

From the repository root:

```sh
make                         # build ./tiny-agenc
make corpus                  # derive data/cyberpunk.txt from the committed raw corpus
make check                   # mathematical and integration checks

./tiny-agenc train --data data/cyberpunk.txt --out trained.bin
./tiny-agenc sample --model trained.bin --prompt "RAZR:"
```

`trained.bin` is your new local checkpoint. The committed
`tiny-agenc.bin` showcase remains untouched.

OpenMP is enabled by default. If the compiler does not provide it:

```sh
make OPENMP=0
make OPENMP=0 check
```

The default executable is not tuned to the build host. The recorded
performance run used `make NATIVE=1`, which adds `-march=native`; use
that only for a local build you do not intend to move to another CPU.

The corpus scripts target GNU/Linux and use Bash, `awk`, `grep`,
`iconv`, and GNU coreutils. The committed raw corpus is already large
enough, so `make corpus` cleans it locally without contacting Ollama.
`make generate-corpus` is the separate, explicit mutation path.
Extending the raw corpus requires `curl`, `jq`, and a reachable Ollama
server.

## The rhythm

Most chapters finish with the same four signposts:

- **Build:** the smallest useful piece to implement.
- **Verify:** the command or observation that tests it.
- **Expected:** what success looks like.
- **Common failures:** what the usual mistakes look like.

Those signposts are not homework pasted onto the story. They are the
story. A model becomes understandable when each claim can be made to
answer for itself.

## Chapters

| # | Chapter | You will learn |
|---|---------|----------------|
| 0 | [Introduction](00-introduction.md) | what we are building, why C, and the payoff |
| 1 | [The Map](01-the-map.md) | the complete architecture, notation, tensor shapes, and parameter count |
| 2 | [Foundations](02-foundations.md) | utilities, exact binary I/O, PCG32, Box-Muller, and reproducibility |
| 3 | [Data](03-data.md) | corpus design, durable tokenization, and next-token pairs |
| 4 | [Poor Man's Tensors](04-poor-mans-tensors.md) | `Mat` views, row-major storage, ownership, and activation lifetimes |
| 5 | [The Forward Pass](05-forward-pass.md) | every transformer operation from intuition to loops |
| 6 | [Backprop by Hand](06-backprop-by-hand.md) | the chain rule made mechanical and every backward operation |
| 7 | [Trust, but Verify](07-trust-but-verify.md) | finite differences, random projections, and mixed tolerances |
| 8 | [AdamW](08-adamw.md) | `Param` storage, SGD and moment histories, AdamW bias correction and decay, global clipping, and value-only I/O |
| 9 | [Parameters and the Private Blueprint](09-parameters-and-the-blueprint.md) | the canonical parameter registry, supplied private layout, tied head, and model-wide chores |
| 10 | [Memory Planning and Arenas](10-memory-planning.md) | checked byte estimates, two-pass placement, lifetimes, and short views |
| 11 | [Wiring the Model Forward](11-wiring-the-model-forward.md) | exact block and stack wiring, latest-call state, inference, and the tied head |
| 12 | [Wiring the Model Backward](12-wiring-the-model-backward.md) | reverse stack order, residual and tied-gradient meetings, and bounded evidence |
| 13 | [Durable Checkpoints](13-durable-checkpoints.md) | TAGC layout, immutable loads, resource bounds, metadata-preserving replacement, and durability |
| 14 | [The Command Line](14-the-command-line.md) | help, version, parsing, diagnostics, streams, and setup boundaries |
| 15 | [The Training Loop](15-the-training-loop.md) | batch, zero, forward, backward, update, evaluate, save, repeat |
| 16 | [Sampling](16-sampling.md) | newest-logit selection, temperature, finite weighted draws, fresh context, and replay |
| 17 | [The Training Run](17-the-training-run.md) | matched baselines, add-one bigrams, perplexity, and honest reading of recorded losses, samples, and timings |
| 18 | [Epilogue](18-epilogue.md) | the complete lifecycle, transferable roles, Tiny-specific boundaries, and the next controlled experiment |

## Appendices

- [Appendix A: Derivations](appendix-a-derivations.md) expands the
  equations behind every local backward loop, plus Adam's zero-start
  correction.
- [Appendix B: Sources and Further Reading](appendix-b-sources.md)
  maps questions from the chapters that first teach each idea and the
  current C to original papers, standards, and larger implementations.
- [Appendix C: Capstone Labs](appendix-c-capstone-labs.md) turns eight
  exposed boundaries into controlled experiments with prerequisite
  links, contract maps, targeted witnesses, and honest stopping points.

Chapters 5 through 17 are the load-bearing span. Read slowly there.
Chapters 0 through 4 prepare the pieces. Chapter 18 separates durable
roles from Tiny-specific choices; the appendices slow the derivations,
trace the sources, and turn open boundaries into controlled
experiments.

The book is licensed under Creative Commons Attribution 4.0
International. See [`LICENSE.md`](LICENSE.md) for attribution terms.
