# Tiny AgenC Build Track

The completed implementation in `src/` is the answer key. This
directory lets you build the same program in a separate workspace, one
observable boundary at a time.

Early checks link one learner-owned module with known-good support.
Later checks steadily remove that support until every implementation
line is yours. The staged linker is part of the teaching method: a
tokenizer should not require a finished transformer before it can be
tested.

## Start a workspace

From the repository root:

```sh
bash labs/start.sh
```

This creates `labs/work/`, copies stable interfaces, and creates source
modules with TODO markers. `mat.h` begins as a Chapter 4 template. The
Chapter 9 private model blueprint is supplied in full because later
stages link separately compiled learner and support files that must
agree on one exact structure layout. Existing work is never
overwritten.

To use another directory:

```sh
bash labs/start.sh /absolute/path/to/my-tiny-agenc
make -C labs WORK="/absolute/path/to/my-tiny-agenc" check-02
```

## The rule

Read the chapter, implement the named surface, and run its target.
Consult the matching file in `src/` after your version passes or when
you can state the exact question that has you stuck.

There is no honor system hiding in the build. You own the learning
strategy. The separation exists so the book can function as a workshop
without pretending the solution repository vanished.

## Checkpoints

| Chapter | Build | Command | Independent success condition |
|---|---|---|---|
| 1 | executable model geometry in `spec.c` | `make -C labs check-01` | default count is 815,360 |
| 2 | `util.c`, `rng.c` | `make -C labs check-02` | utility contracts and published deterministic draws pass |
| 3 | `tokenizer.c`, `dataset.c` | `make -C labs check-03` | durable sorted vocabulary and shifted batches pass |
| 4 | `mat.h` views | `make -C labs check-04` | sharing, bounds, and assertion contracts pass |
| 5 | forward operations in `ops.c` | `make -C labs check-05` | exact nonuniform math witnesses pass |
| 6 | backward operations in `ops.c` | `make -C labs check-06` | operation finite differences pass |
| 7 | inspect, mutate, and repair the referee | `make -C labs check-06` | a controlled layernorm error is caught |
| 8 | `param.c` | `make -C labs check-08` | AdamW, scaling, zeroing, and parameter I/O match references |
| 9 | supplied private blueprint and `model_parameters.c` | `make -C labs check-09` | exact parameter order, shapes, initialization, count, and clipping pass |
| 10 | `model_memory.c` | `make -C labs check-10` | full and shortened arena views pass |
| 11 | `model_forward.c` | `make -C labs check-11` | whole-model known losses and target sensitivity pass |
| 12 | `model_backward.c` | `make -C labs check-12` | every parameter connects and finite differences pass |
| 13 | `checkpoint.c` | `make -C labs check-13` | TAGC round-trip, CRC, and resource limits pass |
| 14 | command parsing and setup in `main.c` | `make -C labs check-14` | information paths, valid work, and prefixed failures use the right streams |
| 15 | the learner’s training loop | `make -C labs check-15` | CLI loss falls and validation does not steer |
| 16 | `model_sampling.c` and sample command | `make -C labs check-16` | temperature, fresh context, and replay pass |

Chapter 7 builds a test technique rather than a runtime module. Read the
random-projection driver, predict which comparisons a layernorm mutation
should break, and watch the controlled mistake produce structured
evidence.

## What the staged linker is doing

- Chapters 2 through 5 compile only the learner modules named by that
  chapter.
- Chapter 6 combines learner operations with the answer-key model so
  operation derivatives can be tested before assembly.
- Chapter 9 constructs learner parameters without requiring activation
  memory. The learner and support files share the supplied private
  layout exactly. Its checkpoint verifies both copies before compiling,
  so layout drift fails with a direct diagnostic instead of reaching a
  mixed-file call.
- Chapter 10 drives learner arenas with known-good model wiring.
- Chapters 11 and 12 replace that wiring with the learner’s forward and
  backward files.
- Chapter 14 temporarily supplies answer-key sampling so CLI work does
  not wait for Chapter 16.
- Chapter 16 is the first stage where every runtime module comes from
  the learner workspace.

The support code never decides whether the learner implementation is
right by comparing two copies of it. Exact examples, known answers,
connectivity witnesses, finite differences, learning behavior, and
round-trip invariants each answer a different question.

## Useful habits

- Keep assertions enabled.
- Compile serially first. Parallelism is not a debugging aid.
- Do not change an interface merely to make one implementation easier
  unless you can explain the new contract.
- Reduce a failure to the smallest named operation before editing.
- Commit after every passing checkpoint. A known-good chapter is a much
  friendlier place to return than a memory of one.

## Verify the parallel path

The chapter checkpoints compile serially by default so a race is not
confused with a mathematical mistake. Once the Chapter 6 operation
checks pass, compile the learner's `ops.c` with OpenMP and compare one
thread with four:

```sh
make -C labs WORK="$PWD/labs/work" check-parallel
```

That target performs an OpenMP sub-build and crosses the threshold for
both matmul and attention, forward and backward. The results must match
bit for bit across thread counts. After the complete build passes,
compile every learner module under OpenMP too:

```sh
make -C labs WORK="$PWD/labs/work" OPENMP=1 check-16
```

If the compiler does not provide OpenMP, keep using the serial
checkpoints. The parallel target is an additional portability-specific
witness, not a prerequisite for the mathematics.

## Finish

When `check-16` passes, build your executable:

```sh
make -C labs WORK="$PWD/labs/work" tiny-agenc
```

Then run the same first-light and full training commands used in
Chapters 15 and 17. Matching every captured floating-point digit across
compilers is not the final test. Matching the architecture, invariants,
loss scale, learning behavior, and deterministic replay within one
build is.
