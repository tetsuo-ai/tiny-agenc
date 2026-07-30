# Chapter 0: Introduction

This book builds, from an empty file, a program that learns to write
by studying a text file. Here is what that looks like. One command
starts a training run. On the recorded 32-core machine, 4 minutes
16.77 seconds later, it is done. This is the shortened opening:

```
$ ./tiny-agenc train --data data/cyberpunk.txt --out trained.bin
tiny-agenc: 815360 parameters | 376.5 MiB buffers | vocab 80 | 4 layers x 4 heads x 128 wide
tiny-agenc: 1089394 tokens of training data from data/cyberpunk.txt
step     1/5000 | loss 4.4395 |  105.9 ms/step
```

The short command uses the same model and training defaults as the
recorded command, but names a convenient local output file. The
quoted time belongs to a GCC 13.3 build that spread supported loops
across CPU cores with OpenMP; [Chapter 5](05-forward-pass.md) shows
where. That build also used `NATIVE=1` to tune its instructions for
an AMD Threadripper PRO 9975WX. The
[complete log](logs/train-cyberpunk-5000.log) records the exact
command, machine, build hashes, and every later line. The
[evidence record](../EVIDENCE.md#recorded-build) pins the build flags.
Together they document that run; they do not promise the same speed
on another machine.

Reading the banner: the model has 815,360 adjustable numbers. Those
numbers are its **parameters**. The `376.5 MiB buffers` part is its
reserved working memory; one MiB is 1,048,576 bytes.

NIGHT GRID is the text the model studies.
[Chapter 3](03-data.md) builds and names a collection like this a
**corpus**. This file is ASCII, so one byte holds one character. The
program treats each byte as one unit to read or predict. It calls that
unit a **token**
([Chapter 3](03-data.md#give-each-byte-a-compact-number)). A displayed
character can occupy several bytes in other text encodings. The
banner's `vocab 80` says the model's whole alphabet is the 80 distinct
characters in its training text. `step 1/5000` marks the first of
5,000 training rounds; [Chapter 1](01-the-map.md) maps what one round
must do. Each round grades the model's next-character guesses on one
group of snippets. The banner calls that grade the **loss**. It is not
a running average. Lower is better.
[Chapter 5](05-forward-pass.md) makes the score exact.

Before the first step, 813,056 parameters hold small random values.
The other 2,304 either multiply an incoming value by 1 or add 0, so
they initially leave that value unchanged. All 815,360 can change
during training.
[Chapter 1](01-the-map.md) catalogs the starting recipe,
[Chapter 2](02-foundations.md) builds the replayable random draws,
and [Chapter 9](09-parameters-and-the-blueprint.md) constructs the
initializer in C.

Predict which will appear first: the repeated shapes of speaker
labels and transmission headers, or coherent dialogue. The first
captured sample comes after step 250:

```
KRAZR: Weett too ad tad neet shar fre-ckeort. Stam lanndourene gon blo womertt ll theee te't shad beoout icth ror on and iner.
WIRAZR: Yeear s ount as ond the landenans thaser teach eme.
JINX: Sthefor
```

The rigid shapes arrive before coherent prose: two labels are
malformed, one is valid, and the dialogue is still mostly
word-shaped noise. The voices come from NIGHT GRID, the book's
synthetic corpus of cyberpunk radio chatter.
[Chapter 3](03-data.md) explains its design, the history of its
generated raw text, and the reproducible cleaning step.

By step 500 the model is attempting transmission headers, and by
step 1000 the speaker-and-dialogue layout is recognizable. At step
5000 the recorded loss is 0.7668. Hand the saved model half a header
and it completes a valid one, continues the scene, and starts the
next. This shortened transcript omits the architecture banner that
the sample command also prints:

```
$ ./tiny-agenc sample --model trained.bin --prompt "=== TRANSMISSION 0999 // SECTOR 9: " --length 400 --temperature 0.8 --seed 1337
=== TRANSMISSION 0999 // SECTOR 9: ICE MARKET ===
DOC: What's the pad heat? Fast that corpo labyrief if we're late.
RAZR: GHOST, you sure the is sector at the Neon Docks' woun'rate out out there?
GHOST: WIRES, stay frosty; the corpo langes in the Grid.

=== TRANSMISSION 0420 // SECTOR 10: BLACKOUT DISTRICT ===
```

The command writes by choosing one character, appending it, and
running the model again. [Chapter 1](01-the-map.md) builds that loop
and names it **autoregressive**. The machine inside the loop is a
compact, decoder-only transformer with multi-head causal
self-attention, layer normalization, GELU feed-forward layers,
residual streams, tied embeddings, and an AdamW optimizer.
Backpropagation and that optimizer train those parts.

If none of those names mean anything to you, you are holding the right
book. They are labels here, not explanations.
[Chapter 1](01-the-map.md) puts every part on the map.
[Chapters 5](05-forward-pass.md), [6](06-backprop-by-hand.md),
[8](08-adamw.md), and [9](09-parameters-and-the-blueprint.md) then
construct the operations, backpropagation, optimizer, and tied tables
from arithmetic and code you already understand. By the last page
that sentence reads as a parts list, the way "wheels, axle, frame"
reads to someone who has assembled a cart. These are core transformer
components;
[Chapter 18](18-epilogue.md#carry-roles-not-filenames) shows which
roles carry into another implementation and which choices belong to
this one. Ours is small: 815,360 parameters instead of billions or
more.

And it is written in about 3,600 lines of implementation C, plus
small headers. It calls no PyTorch, NumPy, or BLAS, and no library
computes the parameter changes for it. Underneath are explicit memory
requests, ordinary loops, and a small set of math functions. Each C
form is introduced when the book first uses it.

## Run it before you read it

You do not have to take either shortened transcript on faith. The
training log linked above is the complete captured stdout. The full
generated stdout is
[`logs/showcase-sample.txt`](logs/showcase-sample.txt), and
`make check-evidence` replays it from the committed artifacts.

Better: the finished machine is in this repository, next to
`tiny-agenc.bin`, a saved-model file.
[Chapter 13](13-durable-checkpoints.md) builds and names that kind of
file a **checkpoint**. The reference environment is GNU/Linux with GNU
Make, a C11 compiler, `libm` (the C math library), the `libacl`
development package, and the corpus tools listed in the
[book quickstart](README.md#quickstart). OpenMP is optional; the serial
fallback is under Common failures below. From the repository root,
build the reference executable and make it talk before reading a single
equation:

```sh
make
./tiny-agenc sample --model tiny-agenc.bin \
    --prompt "=== TRANSMISSION 0999 // SECTOR 9: " \
    --length 400 --temperature 0.8 --seed 1337
```

The first command builds the complete reference executable. The
second loads the committed showcase checkpoint and asks it to
continue a transmission header. It also repeats the model banner in
the terminal. `--length` requests 400 new byte tokens, which are
characters for this ASCII model. `--temperature` controls how varied
the choices are; [Chapter 16](16-sampling.md) builds that control.
`--seed` replays the same random draws;
[Chapter 2](02-foundations.md) builds that guarantee.

Matching generated text also requires the same source, checkpoint,
compiler behavior, flags, and platform. Under those conditions the
fixed seed prints the ICE MARKET scene quoted above. If yours
diverges, `make check-evidence` checks the committed hashes before it
compares generated text. A hash failure identifies a changed
artifact; matching hashes with a different sample can point to
platform-dependent floating-point behavior.
[Chapters 2](02-foundations.md) and [16](16-sampling.md) explain where
that sensitivity lives.

Then start training your own model in a second terminal, and leave it
running while you read the rest of this chapter:

```sh
make corpus
./tiny-agenc train --data data/cyberpunk.txt --out trained.bin
```

`make corpus` derives the training text from the committed raw
corpus without contacting the network. Training uses about 380 MiB
of buffers. The recorded timing belongs to the `NATIVE=1` build
identified above; plain `make` favors a binary that can move between
compatible CPUs. Your time will depend on your compiler, CPU, and how
many CPU cores the program uses at once. The `ms/step` report gives
you an estimate: at `600.0`
milliseconds per training step, 5,000 steps take about 50 minutes,
plus the time for periodic samples and checkpoint saves along the
way. The explicit `--out trained.bin` keeps your new checkpoint
separate from the committed showcase.

Watch the first `step` line, after the two banner lines. A model with
no learned preference makes nearly equal guesses among the 80
characters. [Chapter 5](05-forward-pass.md) derives why those guesses
give a loss near 4.38. The random initialization, seed, first group
of snippets, and floating-point environment move the exact value;
the recorded first loss is 4.4395.

Should every later loss report be smaller than the one before it?
Watch before answering. Each report grades a newly selected group of
text snippets, so individual readings bounce. The trend falling over
many reports is what learning looks like from the outside. The rest
of this book is the inside.

## Who this is for

You need to have written simple programs with functions, loops, and
arrays, in any language you like. Nothing else is assumed.

Not C: every C feature this project needs is explained at first
contact. When [Chapter 1](01-the-map.md) first reaches a C pointer, it
explains that the value stores an address.
[Chapter 2](02-foundations.md) then shows how C requests, counts, and
releases bytes of memory before the implementation relies on those
operations.

Not math: every mathematical tool is built on the page the first time
it is needed, from arithmetic up. Every new operation starts with
two to four numbers you can check by hand before the code uses it.
[Chapter 2](02-foundations.md) constructs `exp` and `log` for its
random-number machinery. [Chapters 5](05-forward-pass.md) and
[6](06-backprop-by-hand.md) build the model's probability and slope
calculations, and [Chapter 7](07-trust-but-verify.md) checks those
slopes by nudging numbers and measuring what moves. If you once knew
calculus and lost it, nothing here assumes you kept it.
[Appendix A](appendix-a-derivations.md) is a slower second pass
through the algebra for the curious, never a prerequisite.

Not machine learning: the book's working premise is that you have
never trained a model before.

## Why C, of all things

Because this C implementation puts the stored numbers, loops, and
learning calculations in view.

Many framework tutorials eventually hit the same wall:
`loss.backward()`. That method asks the framework to work out how the
loss responds to every parameter. An optimizer uses those results to
change the parameters, but most of the arithmetic that links a wrong
guess back to the responsible numbers disappears into the call. You
are told to use a rule that follows one change through the next, but
you do not see it happen. [Chapter 6](06-backprop-by-hand.md)
constructs that mechanism and names it the chain rule.

In this codebase, even the backward part of attention is a short loop
nest you can put your finger on.
[Chapter 6](06-backprop-by-hand.md) first works the arithmetic, then
walks the matching
[`attention_head_backward`](../src/ops.c) loop line by line.

There is a second, sneakier reason. Frameworks can make a shaped
collection of numbers feel like a magic object with behaviors.
[Chapter 1](01-the-map.md) shows the concrete representation used
here: the address of the first number, plus row and column counts.
The calculation that turns input text into next-character guesses is
ordinary loops. Training repeatedly adjusts the parameters so those
guesses fit the corpus more closely. Once you have seen those steps
directly, larger models become less mysterious, even though scale
brings architectural and systems changes of its own.

The payoff is concrete. After this book you can open a production
model's source and recognize the role of its core transformer and
training pieces, then locate the places where it differs from this
one. You can debug a Tiny AgenC run because every printed
measurement has a code path and an expected scale. And when someone
tells you a model "understands", you can name what this model
computes and decide for yourself what that word is worth.

## What you are holding

The code lives in [`src/`](../src), and the book and code keep each
other honest. The book gives you the architecture, derivations,
implementation order, and checkpoints. The source gives you the full
reference answer. There are
[two ways through](README.md#two-ways-through). On the reading
track you follow the chapters with the finished source open and run
each chapter's verification against it. On the building track you
write every runtime module yourself in a separate workspace, using
[`labs/`](../labs/README.md) as the checklist, and consult the
reference only when a test or a question earns the peek. To take
the building track, stage that workspace now, from the repository
root:

```sh
bash labs/start.sh
```

The script creates `labs/work/` and refuses to overwrite it if it
already exists. That refusal protects work from an earlier session.
The building-track checks that begin in
[Chapter 1](01-the-map.md)
(`make -C labs check-01`) assume this workspace exists.

Code excerpts are taken from the working source, built with
`-Wall -Wextra -Werror`. Those flags ask the compiler to report a
broad set of suspicious C and make any warning stop the build. The
mathematical checker compares the handwritten loss-slope
calculations, called gradients in
[Chapter 6](06-backprop-by-hand.md), and the
[AdamW](08-adamw.md) arithmetic against independent references. It
prints the exact number of comparisons when `make check` runs; the
integration checks cover the surrounding data, checkpoint, and
sampling paths. Captured losses and timings in `book/logs/` carry
their measured command and environment;
[`EVIDENCE.md`](../EVIDENCE.md) identifies the committed evidence
artifacts and states what `make check-evidence` does and does not
prove. Exact floating-point results depend on the matching source,
checkpoint, compiler, flags, and platform.

The model even has its own dataset. Rather than use the Shakespeare
corpus common in small language-model tutorials, we synthesized
**NIGHT GRID**: a megabyte of cyberpunk transmission logs generated
by a language model running locally. The cleaned corpus admits only
eight speaker handles trading hard-boiled dialogue across numbered
sectors. It was distilled through a grammar whitelist
([Chapter 3](03-data.md)). The corpus was designed to make learning
visible:
word-shaped text, recurring slang, speaker tags, and the full
transmission format become increasingly recognizable as training
progresses.

## The route

Chapters 1 to 4 draw the architecture, build checked memory requests
and reproducible randomness, prepare the corpus, and store each grid
of numbers in three fields. Chapters 5 to 7 implement the forward
pass, derive the backward pass by hand, and build the checker that
tests both. Chapter 8 builds the optimizer.
Chapters 9 to 12 assemble parameters, memory, forward wiring, and
backward wiring in four separate stages. Chapter 13 makes saved
models defensive, Chapter 14 builds the command-line interface, and
Chapter 15 closes the training loop. Chapter 16 turns
next-character probabilities back into text.
[Chapter 17](17-the-training-run.md#two-runs-two-claims) replays the
recorded run and compares progress on training text with performance
on text held aside from training.
[Chapter 18](18-epilogue.md#the-whole-machine-on-one-page) closes the
three routes, then identifies what carries into another implementation
and what changes.

One warning before you start: after this, `loss.backward()` will never
impress you again.

## Build checkpoint: prove the artifact is alive

Everything in this checkpoint already appeared above; close the first
sitting by finishing it. You are not learning by typing commands yet.
You are establishing that the destination exists, on your machine.

**Build.** Build the executable and derive the corpus if you have not
already done so. First inspect the run started earlier. If it is still
going, wait for it. If it printed the final save confirmation shown
below, reuse its `trained.bin`. Run the 5,000-step command below only
when neither condition applies; an interrupted run may have left an
intermediate file with that name.

```sh
make
make corpus
./tiny-agenc train --data data/cyberpunk.txt --out trained.bin
```

When run, the training command keeps the terminal occupied until the
final checkpoint is ready. Wait until it prints:

```text
tiny-agenc: checkpoint saved to trained.bin
```

This ordering also avoids loading a second full-capacity model while
the trainer still holds roughly 380 MiB of buffers.

**Verify.** Run the quick checks, then sample the committed showcase
and your finished model:

```sh
make check
make check-evidence
./tiny-agenc sample --model tiny-agenc.bin \
    --prompt "=== TRANSMISSION 0999 // SECTOR 9: " \
    --length 400 --temperature 0.8 --seed 1337
./tiny-agenc sample --model trained.bin \
    --prompt "=== TRANSMISSION 0999 // SECTOR 9: " \
    --length 400 --temperature 0.8 --seed 1337
```

**Expected.** The build finishes cleanly. The corpus cleaner reports
1089394 bytes and a vocabulary of 80 distinct characters. The
gradient checker, the referee that compares the hand-derived math
against numerical estimates
([Chapter 7](07-trust-but-verify.md)), reports that all checks passed,
and the integration checks finish without an error. On a compatible
replay environment, the evidence check also finishes without an error
and the committed checkpoint reproduces the recorded ICE MARKET text.

On the recorded environment, the training run begins at loss 4.4395
and ends at 0.7668. These are evidence values, not cross-platform
acceptance thresholds: another toolchain or machine may produce
different final digits. A loss report can rise even in a healthy run.
Look for a downward trend across many reports. Treat your freshly
trained sample as a qualitative check: compare whether it has picked
up the recurring header and dialogue form of NIGHT GRID. Exact wording
is not promised.

**Common failures.**

- If the compiler rejects `-fopenmp`, add `OPENMP=0` to every `make`
  command in this chapter, including `make OPENMP=0 check-evidence`.
- If `iconv` or another corpus tool is missing, install the GNU/Linux
  dependencies listed in the [book quickstart](README.md#quickstart).
- If your machine cannot spare roughly 380 MiB for the default model,
  read on without the full training run. If the showcase checkpoint
  cannot allocate its full-capacity buffers either, skip both
  `make check-evidence` and the showcase sample, but still run
  `make check`. [Chapter 15](15-the-training-loop.md) begins with a
  first-light configuration that is intentionally small, and nothing
  before that chapter requires a finished run of your own.
- If the evidence hashes pass but the showcase replay differs, the
  compiler, flags, math library, or platform may have changed the
  floating-point choices. [Chapters 2](02-foundations.md) and
  [16](16-sampling.md) locate that sensitivity.
- If `trained.bin` is missing or produces an early-training sample
  while the trainer is still running, wait for the final
  `checkpoint saved` line. If the run stopped, restart it. The trainer
  writes intermediate checkpoints every 1,000 steps.

If you ran the full training command, your model has now finished.
Judge its progress by the longer loss trend, not one pair of reports.
[Chapter 1](01-the-map.md) draws the complete machine it has become,
before you build any part of it.

---

[Contents](README.md) | [Next: The Map](01-the-map.md)
