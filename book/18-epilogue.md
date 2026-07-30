# Chapter 18: Epilogue

You have now followed a working compact autoregressive transformer from
raw text through training, held-out grading, checkpoints, and sampling.
Chapter 0 asked you to run that machine while most of its parts were
still labels. Chapter 17 ended with an honest report card for the
assembled result.

The closing test is concrete.

**Predict:** draw the training, held-out, and generation routes from
memory. Mark where a known answer is available and where learned
parameter values can change after the model has been initialized or
loaded.

## The whole machine on one page

All three routes reuse the forward calculation. What surrounds that
calculation determines what the result means.

```text
TRAINING
training text
      |
      +-> visible ids -> forward with parameters -> bets
      |                                               |
      +-> known next ids -----------------------------+-> loss
                                                            |
                                                 backward -> gradients
                                                            |
                                                   AdamW adjustment
                                                            |
                                                   parameters change

HELD-OUT GRADING
held-aside text
      |
      +-> visible ids -> forward with parameters -> bets
      |                                               |
      +-> known next ids -----------------------------+-> loss report
                                                             |
                                                  no backward, no update

GENERATION
checkpoint file/current training state -> fixed parameters
                                             |
prompt ids ------------------------------> forward -> newest bet sheet
    ^                                                     |
    +----------------- append one drawn id <---------------+
                          no known answer, no update
```

The visible ids enter the predictor. The known next ids bypass it and
meet the resulting bets only at the loss. That loss starts the backward
route, and [`model_step`](../src/model.c) uses the resulting gradients
to adjust the parameters.

Held-out grading preserves the same separation, but its permission ends
at the loss report. Chapter 15's
[no-correction boundary](15-the-training-loop.md#grade-without-sending-corrections-back)
calls no backward operation and no update.

Generation has no known next id. Chapter 16's
[generation loop](16-sampling.md#the-complete-generation-loop) reads
the newest bet sheet, draws one id, appends it, and runs forward again.
It has no loss from which to construct a correction. Sampling text
during training does not make the sampled text part of training. The
fixed values for one generation route may come from a checkpoint or
from the current training state after an update; that route does not
adjust them.

Initialization and checkpoint loading first fill parameter storage.
After that boundary, ordinary forward calls can replace working
activations and the latest-forward record, but they do not adjust the
learned values. A checkpoint copies those values into a file. It does
not teach them.

This is the distinction Chapter 1 first mapped. The intervening
chapters supplied the arithmetic, code, and failure boundaries behind
every arrow.

## What you can now rebuild

The important verb is rebuild. Familiar vocabulary is cheap. A working
model whose pointers, loops, gradients, tests, and failure boundaries
you can account for is a different kind of understanding.

The lifecycle above is not an architecture poster. Every arrow has a
source boundary you have already walked:

| Question | Tiny AgenC boundary | Where it was built |
|---|---|---|
| What pieces enter? | [`tokenizer_new`](../src/tokenizer.c) and [`dataset_batch`](../src/dataset.c) | [Chapter 3](03-data.md#the-complete-data-boundary) |
| Where are the bets made? | [`model_forward`](../src/model_forward.c) and the loops in [`ops.c`](../src/ops.c) | [Chapters 5](05-forward-pass.md) and [11](11-wiring-the-model-forward.md) |
| Where do corrections come from? | [`model_backward`](../src/model_backward.c) and the reverse loops in [`ops.c`](../src/ops.c) | [Chapters 6](06-backprop-by-hand.md) and [12](12-wiring-the-model-backward.md) |
| Where do learned values move? | [`model_step`](../src/model.c) and [`param.c`](../src/param.c) | [Chapters 8](08-adamw.md), [9](09-parameters-and-the-blueprint.md), and [15](15-the-training-loop.md) |
| Where is storage planned and persisted? | [`model_memory.c`](../src/model_memory.c) and [`checkpoint.c`](../src/checkpoint.c) | [Chapters 10](10-memory-planning.md) and [13](13-durable-checkpoints.md) |
| How does text leave? | [`model_sample`](../src/model_sampling.c) fills generated ids; [`tokenizer_decode`](../src/tokenizer.c) returns one byte | [Chapter 16](16-sampling.md) |
| What licenses a claim? | focused checks, held-out grades, matched baselines, logs, and hashes | [Chapters 7](07-trust-but-verify.md) and [17](17-the-training-run.md) |

You can start at any row and follow the dependency in either direction.
From `model_step`, trace backward to the gradient that moved one
parameter and forward to the next batch that reads its new value. From
one generated byte, trace backward to the draw, its probability, its
logit, the final token-table row, and the visible context that produced
it.

That ability is more portable than this repository's filenames.

## Carry roles, not filenames

Another implementation may divide its files differently or replace a
mechanism. Search for the role and its contract before searching for a
familiar symbol.

The conceptual inventory you built here carries forward. A query
remains a recognizable idea even when its implementation changes. The
same is true of a token id, a residual addition, a next-token grade, and
a gradient. Their storage or surrounding block can differ without
making the question they answer disappear.

Use these questions when opening another decoder-only transformer:

| Question to settle | Tiny AgenC's answer | What may differ elsewhere |
|---|---|---|
| What is one text unit? | one vocabulary byte maps to one id and back | one id may cover several bytes |
| Where does position enter? | one learned position row is added to each token row | another position mechanism may fill the role |
| Where is no-peeking enforced? | attention visits keys only through the query position | the storage and loop schedule may differ |
| How is context mixed? | scaled query-key bets mix value rows | head layout or the mixing operation may differ |
| How is each row edited? | a four-times-wider GELU feed-forward path | width and row-editing mechanism may differ |
| What supplies the grade? | mean next-token cross-entropy on known ids | a different grade changes what training rewards |
| How are learned values adjusted? | clipped AdamW over one parameter registry | update state and rules may differ |
| What survives a process exit? | tokenizer, geometry, and learned values in TAGC version 1 | a format may save more or less state |
| What supports the claim? | targeted checks plus matched held-out evidence | every changed contract needs a matching witness |

This table is a reading method, not a claim that every language model
contains Tiny AgenC with larger dimensions. If another source has no
causal attention, locate its no-peeking mechanism before assuming the
same role exists. If it saves optimizer and random-stream state, its
checkpoint contract is stronger than
[TAGC version 1](13-durable-checkpoints.md#state-the-version-1-promise-exactly).
If it changes the text unit, loss per token no longer gives a direct
comparison with this byte model.

## Trace one change across the contracts

Scale can force an implementation change, but "larger" is not one
number. Work one source-grounded change before looking outward.

Use Chapter 1's
[dimension letters](01-the-map.md#the-shapes-now-have-somewhere-to-live):
hold batch size `B`, head count `H`, channel width `C`, and layer count
`L` fixed. Change only the context capacity from `T = 128` to
`T = 256`.

**Predict:** how many score slots does one sequence and one attention
head need before and after the change?

```text
old score grid     128 rows * 128 columns = 16,384 slots
new score grid     256 rows * 256 columns = 65,536 slots

row growth           2 times
column growth        2 times
slot growth          2 * 2 = 4 times
```

Chapter 10 counted one complete score grid as
[`B*H*T*T` floats per layer](10-memory-planning.md#attention-owns-the-expensive-square).
Backward storage has a same-sized gradient grid. With `B` and `H`
fixed, each score-shaped term becomes four times as large. Total model
memory does not become exactly four times as large. Token tables,
parameter gradients, and width-sized rows follow other counts.

The learned position table changes from `128*C` to `256*C` entries.
The byte vocabulary does not change. The attention role and its
query-key-value arithmetic do not change, but the loops and storage
visit more positions. The configuration checks, memory report,
parameter count, checkpoint geometry, and affected witnesses must all
accept the new shape.

One changed field therefore reaches several contracts:

```text
context capacity
    -> position parameters
    -> attention work and score storage
    -> arena requirements
    -> checkpoint geometry
    -> tests and measured resource claims
```

Predicting that reach before editing is how a scale change stays
visible.

## Know what belongs to this box

The roles above transfer. The following choices and observations belong
to Tiny AgenC unless a new experiment establishes something else.

### Text units can change

Tiny AgenC's tokenizer has 80 byte tokens for the recorded corpus, and
`tokenizer_decode` returns one `char`. Chapter 3 constructed a
[subword token](03-data.md#give-each-byte-a-compact-number) that can
cover a recurring multi-byte piece. A tokenizer built from such pieces
still emits ids, but decoding, vocabulary-sized parameter tables,
checkpoint data, context units, and fair loss comparisons all change.

Appendix C turns that boundary into a
[variable-length-token capstone](appendix-c-capstone-labs.md#7-learn-a-bpe-vocabulary).
The chapter does not assume that replacing bytes automatically improves
the model. It asks for round trips and matched units of comparison.

### Attention storage can change

Tiny AgenC materializes a `T*T` forward score grid for every sequence
and head, or `B*H*T*T` per layer. It retains a same-sized gradient grid
for backward. That choice keeps both loops readable at `T = 128`.

Another implementation can evaluate pieces of the same attention
calculation while retaining less of the full score grid in main memory.
That changes the execution schedule and floating-point grouping even
when the mathematical attention result is the same. Appendix B points
to [an exact tiled attention
algorithm](appendix-b-sources.md#flashattention) for the longer story.

This distinction matters when reading fast code: an equation and its
storage plan are separate contracts.

### Hardware changes ownership problems

Tiny AgenC is one process using CPU loops. OpenMP lets threads share its
arrays, and the arenas give every working region one explicit lifetime.

A larger implementation may divide calculations and stored arrays
across other processors or several machines. It must then account for
communication, copies, partial results, and failures outside one
process. Those are real engineering problems, but this book did not
build them. [Appendix B's larger C
implementation](appendix-b-sources.md#llmc) is a source-level next
step.

### A text continuation model is not an assistant

Everything here produces a model that continues text. Giving it a
question-shaped prompt does not add a known-answer training route for
following instructions, a check that its statements are true, or a
safety boundary.
Systems intended for assistant behavior use additional data, grades,
and evaluations. Their exact recipes vary.

Chapter 17 stated the measured boundary:
[NIGHT GRID loss and samples](17-the-training-run.md#state-what-the-experiment-leaves-unanswered)
do not evaluate factual accuracy, instruction following, long-document
recall, or another corpus. More parameters do not convert those missing
measurements into evidence.

### Scale produces several measures

The recorded corpus contains 1,089,394 byte tokens. The 5,000-step
showcase presented 20,480,000 target slots because windows were drawn
with replacement. The second number is not a larger corpus. It counts
repeated training questions.

Parameter dimensions, context capacity, distinct data, batch size, and
step count are experiment settings. Parameter count, presented target
slots, working memory, arithmetic, and wall time are consequences that
answer different questions. They are not independent knobs: the
context change above affected several at once. Changing a setting
creates a new experiment. Chapter 17's timings and samples remain
attached to the recorded configuration and machine.

## Choose your next mutation

[Appendix C](appendix-c-capstone-labs.md) turns the exposed boundaries
into a graduated set of capstone labs. Each names a goal, likely files,
acceptance criteria, hints, and the observation the work is meant to
expose.

Begin with the observation-focused
[sampling](appendix-c-capstone-labs.md#1-temperature-safari) and
[validation](appendix-c-capstone-labs.md#2-validation-experiments)
experiments if the difference between a draw and a grade is still
uncertain. Break and repair a
[gradient](appendix-c-capstone-labs.md#3-break-a-gradient-watch-the-referee)
or strengthen the
[matched baseline](appendix-c-capstone-labs.md#4-beat-the-bigram-honestly)
if the evidence boundary is the weak point.

The later labs change execution state, position handling, text units,
or scale. Before changing any of them, write four lines:

```text
contract that will change:
mistake the current witnesses could miss:
new witness that will object:
claim the result still will not earn:
```

Choose the mutation that attacks the part you understand least. A
familiar feature name is not an implementation plan. The four lines
force the proposed change back into shapes, ownership, arithmetic, and
evidence.

## Build checkpoint: explain the artifact

**Build.** On a blank page, recreate the three-route diagram at the
start of this chapter. Mark the known answers, the loss, the backward
boundary, the parameter adjustment, the checkpoint, and the sampling
loop. Put one source symbol and one witness beside each route.

Then choose one Appendix C experiment without changing code yet. Fill
in the four-line contract above. The new witness must be able to fail
for the mistake you are worried about.

**Verify.** From the unchanged completed repository, run the portable
serial gate:

```sh
make OPENMP=0 check-all
```

**Expected.** The completed reference suite exits successfully.
`overfit`, the book and lab checks, the evidence agreement, and the
staged install all report success.

Your diagram puts known answers and loss in training and held-out
grading. Only training continues through backward and `model_step`.
Generation has no known answer, loss, backward pass, or update. A
checkpoint carries learned values and the tokenizer across a process
boundary, but TAGC version 1 does not carry the optimizer, random
streams, or exact training position.

One valid source-and-witness answer is:

| Route | Source symbol | Witness |
|---|---|---|
| training | [`model_step`](../src/model.c) | `make overfit` |
| held-out grading | [`measure_validation`](../src/main.c) | `make check-cli` compares checkpoints made with and without validation |
| generation | [`model_sample`](../src/model_sampling.c) | `make check-sampling` |

This checkpoint proves the completed repository still satisfies its
declared contracts. It does not prove that a later mutation works.

**Common failures.**

- Treating one generated sample as a held-out grade gives a random
  trace an answer sheet it never had.
- Putting the draw inside the predictor confuses choosing from a bet
  sheet with producing the sheet.
- Assuming a shared role requires identical storage hides valid
  implementation differences.
- Calling one score-storage term four times larger does not make total
  memory four times larger.
- Comparing corpus tokens with presented target slots mixes distinct
  data with repeated questions.
- Running the clean-tree evidence gate after intentionally changing a
  pinned source or data file should expose the changed identity. It is
  not, by itself, a math failure.
- Changing several dimensions at once leaves no controlled answer to
  which change caused an observation.

## The last word

The next time someone calls this language model an inscrutable
artifact, remember what is actually in this box: a tokenizer you could
write from memory, embeddings, a stack of blocks made from a handful of
operations, a softmax, and an optimizer. The chain rule connects the
grade back to every learned value. Finite differences compare the
implemented return with measured slopes; the source walk establishes
the intended route.

The complete implementation fits in about 3,600 lines of C, excluding
headers and tests. Smallness is not the claim that larger systems are
the same. It gave you one chance to see every shape, pointer, loop,
gradient, ownership boundary, and witness together.

The next source may replace half of Tiny AgenC's choices. Start at its
text ids, find the calculation that makes one next-token bet, and trace
how a known answer can change stored parameters. You now know what a
complete answer looks like.

---

[Previous: The Training Run](17-the-training-run.md) | [Contents](README.md) | [Next: Appendix A](appendix-a-derivations.md)
