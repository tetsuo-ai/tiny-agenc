# Chapter 1: The Map

Chapter 0 showed three different moments. Before the first training
step, the model's adjustable numbers held starting values. Training
changed those numbers over 5,000 steps and saved them. A later
`sample` command loaded the saved numbers and generated text without
changing them.

This chapter connects those moments. First, a model with no learned
preference makes wrong predictions against known text. Then repeated
training steps make those predictions less wrong. After the
saved-model boundary, generation uses what training learned. Only
then will we open the prediction calculation and map its parts.

```text
TRAINING

repeat 5,000 times:

corpus -> draw known snippets -> model inputs and known answers
current parameters and model inputs -> predict -> score rows
score rows and known answers -> grade -> adjust parameters

after the final adjustment -> save trained.bin

GENERATION

trained.bin -> load fixed parameters ----+
                                         +-> predict -> choose -> append
prompt -> visible text ------------------+                   |
               ^                                             |
               +---------------- repeat ---------------------+

visible text begins as the prompt
no grade and no adjustment
```

The two loops happen at different times. Training reads answers from
the corpus and changes parameters. Generation appends the model's own
choices while leaving the parameters alone.

## The model begins broken

Chapter 0 introduced a parameter as one adjustable number. A new
model needs a valid value in every parameter before it can calculate
anything. Tiny AgenC uses three starting patterns:

```text
numbers inside adjustable grids small repeatable random values near zero
learned multipliers             exactly 1
learned additions               exactly 0
```

Multiplying by one and adding zero initially leave a value unchanged.
The random values break ties between otherwise identical paths. None
of these starting numbers came from the corpus, so none represents
learned knowledge.
[Chapter 2](02-foundations.md#rng-randomness-you-can-subpoena)
constructs how the random draws can be repeated exactly. Here they
are starting values, not facts learned from text.

Choosing these starting values is **parameter initialization**. The
exact recipe depends on each parameter's job; we will catalog it
after all the parameter shapes are visible.

Use a paper-sized corpus with no word for the reader to complete:

```text
ACDB
ACDB
...
```

Its five possible characters are newline, `A`, `B`, `C`, and `D`.
Suppose an untrained paper model produces these constructed scores
after reading `A`:

```text
possible next character     raw score
newline                     -0.3
A                            0.4
B                           -0.1
C                            0.2
D                            0.1
```

Before reading on, find the largest score.

The largest is `0.4`, beside `A`. The corpus answer is `C`, so this
prediction is wrong. The scores are teaching values, not output
captured from Tiny AgenC. Their job is to show the starting
condition: an untrained model can rank characters, but the ranking
has no learned reason to favor the answer.

## Known text supplies the answers

The corpus already contains what follows every position. Hide its
last character and predict the three missing entries:

```text
known text       A  C  D  B
model inputs     A  C  D
known answers    ?  ?  ?
```

Shift the same text left by one place to reveal them:

```text
known text       A  C  D  B
model inputs     A  C  D
known answers    C  D  B
```

The source calls the known answers `targets`.
[Chapter 3](03-data.md#one-extra-token-supplies-every-answer)
constructs these adjacent input-answer pairs. Training does not ask
the model to invent a character and then treat that invention as an
answer. The corpus supplies every answer.

### No peeking at the answer

Imagine grading the first prediction while letting it read all three
model-input characters:

```text
prediction after A   may read A C D   answer: C
```

The answer `C` is already in view. A machine could copy the next
character, receive a perfect grade, and learn nothing about
prediction. The honest version restricts what each position may
consult:

```text
position making a bet     positions it may consult
0                         0
1                         0 1
2                         0 1 2
```

Written as a square, the same rule forms a triangle:

```text
                         position consulted
                         0    1    2
position making bet  0  yes   .    .
                     1  yes  yes    .
                     2  yes  yes  yes
```

A dot marks a future position, so no calculation is performed there.
Position `t` may use positions `0` through `t`, never `t + 1` or
beyond. This no-peeking rule is **causality**.

## From known snippets to one adjustment

One short snippet would make every adjustment depend on one small
place in the corpus. Training instead grades a group of snippets
together.

The default group contains 32 snippets, each 128 positions long.
Before reading on, multiply those two numbers to find how many known
input-answer pairs the group supplies.

```text
32 * 128 = 4,096 input-answer pairs
```

A group of sequences graded together is a **batch**. The default
batch therefore supplies 4,096 pairs.

The model produces 80 character scores at every one of those
positions:

```text
4,096 positions * 80 scores = 327,680 raw scores
```

Those scores are compared with the 4,096 known answers and reduced to
one grade, the loss introduced in Chapter 0. The recorded first grade
was `4.4395`.

The complete round follows this order:

```text
draw known snippets from the corpus
                 |
erase correction signals left by the previous batch
                 |
make score rows and grade them against known answers
                 |
trace correction signals back through the calculations
                 |
offer every parameter its next adjustment
```

Make the last line concrete with one paper parameter. Suppose the
updater has already worked out an adjustment of `+0.01`:

```text
parameter before adjustment      0.20
offered adjustment              +0.01
                                -----
parameter after adjustment       ?
```

Predict the last number before reading on. It is `0.21`. This
constructed adjustment is not output captured from Tiny AgenC.
Chapter 6 will determine the returning signal and Chapter 8 will show
how AdamW turns that signal into the actual adjustment.

One batch, one grade, one backward calculation, and one visit to the
parameter updater form a **training step**, or **step** for short.
This is the unit counted by `step 1/5000`.

[Chapter 5](05-forward-pass.md#grade-one-next-token-bet) constructs
the grading calculation.
[Chapter 6](06-backprop-by-hand.md#build-a-local-rate-from-a-nudge)
constructs the returning correction signals, called gradients.
[Chapter 8](08-adamw.md#one-scalar-update-under-glass) constructs the
updater, AdamW.
[Chapter 15](15-the-training-loop.md#the-complete-learning-step)
joins them in C. Chapter 1 needs their order, not their arithmetic.

The next step draws another batch from the corpus. It does not append
a model choice to make new training text.

## Learning takes many steps

One adjustment should not turn the broken scorecard into a competent
model. Extend the paper example to all three input positions. These
rows are schematic snapshots separated by many batches:

```text
model state       after A   after AC   after ACD   top choices right
random start         A          A           A             0 of 3
later                C          A           A             1 of 3
later still          C          D           A             2 of 3
trained paper model  C          D           B             3 of 3
```

The table does not claim that a particular step produces a particular
row. It shows the direction of learning: known answers that received
too little support should receive more on later encounters. The real
model grades every score, not only the top choice.

The recorded Chapter 0 run gives the larger view. Its loss fell from
`4.4395` at step 1 to `0.7668` at step 5,000, but not on every report:

![The 101 recorded training-loss measurements from step 1 through
step 5,000](figures/01-training-loss.svg)

The [figure script](../scripts/figures/01_training_loss.py) reads all
101 reports directly from the
[committed training log](logs/train-cyberpunk-5000.log). The bumps
matter. Every report grades a newly drawn batch, so a healthy run can
score worse on one batch and still improve over many steps.

### Progress samples are inspections

The training command also printed generated text every 250 steps.
That can look like generation is part of the update unless the source
order is made explicit:

```text
training step 250 finishes its adjustment
                 |
                 +--> generate an inspection sample
                 |    no answers, grade, backward pass, or adjustment
                 |    discard the generated characters
                 |
training step 251 draws another batch from the corpus
```

The step-250 sample was still broken: it began
`KRAZR: Weett too ad tad`. The sample let the human see what the
current parameters could produce. It did not become training data.
The source even gives batch selection and progress sampling separate
random generators, so inspecting the model does not change which
corpus snippets train next.

## The saved-model boundary

After the final adjustment, the training command saves the model
description, vocabulary, and learned parameters to `trained.bin`:

```text
random starting values
          |
          v
5,000 training steps change the parameters
          |
          v
save trained.bin
======================= end of training =======================
load trained.bin in a later sample command
          |
          v
generate with no parameter adjustments
```

[Chapter 13](13-durable-checkpoints.md#the-tagc-version-1-layout)
constructs this saved-model file and names it a **checkpoint**. The
file is the handoff between the two commands. During generation the
loaded parameters stay fixed because that call path never runs the
backward calculation or updater.

## Generation uses what training learned

Return to the final row of the paper progression. If that model now
chooses its top-scoring character, one possible continuation is:

```text
prompt A     -> choose C -> append -> text AC
text AC      -> choose D -> append -> text ACD
text ACD     -> choose B -> append -> text ACDB
```

The model receives no answer telling it what should follow. It
chooses from its current scores, appends that character, and runs
again on the longer text. Writing one character at a time, each choice
using the preceding text available to the model, is what
**autoregressive** means here.

The paper trace always takes the largest score so its path is easy to
check. Tiny AgenC's real generation path makes a controlled random
choice from all the scores.
[Chapter 16](16-sampling.md#a-largest-only-rule-throws-information-away)
constructs that choice.

The two modes now stay separate:

| Property | Training | Generation |
|---|---|---|
| Text comes from | corpus batches | prompt plus chosen characters |
| Known next-character answers | yes | no |
| Grade and backward calculation | yes | no |
| Parameter adjustment | yes | no |
| Append a model choice | no | yes |

## One predictor, called in two modes

Training and generation separately reuse one score-producing
calculation:

```text
TRAINING CALL

corpus ids ----> shared predictor ----> score rows --+
known answers ---------------------------------------+-> grade

GENERATION CALL

visible ids -> shared predictor -> score rows
                                    |
                                    +-> newest row -> choose -> append
```

Carrying ids one way through the model to raw character scores is a
**forward pass**. When known answers are supplied, `model_forward`
continues through grading and returns a loss. With no answers, it
stops after the scores. Generation calls that no-answer form and
performs no update.

We will open the shared score calculation first. Its route is:

```text
character ids
     |
opening rows of notes
     |
repeated editing blocks
     |
final steadying
     |
raw character scores
```

## From a character to a row of notes

The rest of the chapter will use one toy machine. Its sizes are small
enough to draw and count:

```text
2 sequences per batch
3 positions per sequence
4 note slots per position
2 side-by-side consultation groups
1 repeated editing stage
5 possible characters
```

These are teaching sizes, not the defaults. Each size receives its
letter after the corresponding part of the machine is visible. The
same geometry then scales to the real model.

### Four different things

Suppose the toy alphabet assigns these ids:

```text
character       newline   A   B   C   D
id                    0   1   2   3   4
```

For the first `A` in `"ACD"`, four objects must not be confused:

```text
character     "A"                         printed text
id             1                          address in the alphabet
position       0                          place in this sequence
state row     [ 0.3  0.1  0.3  0.0 ]     changing notes about this place
```

[Chapter 3](03-data.md#give-each-byte-a-compact-number) will give the
name *token* to one piece of text that receives one id. In this ASCII
model, that later term refers to one character. The id tells the
model which character occurred. The position tells it where that
occurrence sits. The state row is working information that later
calculations revise.

An identity lookup alone cannot distinguish two occurrences of the
same character. For `"ACA"`, it would do this:

```text
A at position 0 -> identity notes for A
A at position 2 -> identity notes for A
```

The two rows would begin identically even though one `A` is first and
the other is third. The model therefore owns a second learned row for
each possible position. For a hand-sized example, let the identity row
for `A` and the location row for position 0 be:

```text
identity notes for A       [ 0.2  -0.1   0.4   0.0 ]
location notes for pos 0   [ 0.1   0.2  -0.1   0.0 ]
                            --------------------------
opening notes              [ 0.3   0.1   0.3   0.0 ]
```

Every column is added separately. `0.2 + 0.1 = 0.3` in the first
column; `-0.1 + 0.2 = 0.1` in the second.
[Chapter 5](05-forward-pass.md#give-each-id-a-row-of-opening-notes)
builds these two table lookups and their addition, then gives the
mechanism its formal name. This chapter needs the result: one opening
row of notes for each text position.

### Rows, columns, and channels

Place the three state rows for `"ACD"` one under another:

```text
                         note slot
                     0     1     2     3
position 0, A     [ 0.3   0.1   0.3   0.0 ]
position 1, C     [ 0.0   0.4  -0.2   0.2 ]
position 2, D     [ 0.1  -0.1   0.5   0.3 ]
                           |
                           +-- slot 1 across every row
```

This grid has three rows and four columns. A rectangular grid of
numbers is a **matrix**. Its **shape** is written rows by columns,
always in that order, so this matrix has shape `3 x 4`.

In this book, a row is one thing and the columns hold that thing's
measurements. Here each row is the model's running notes about one
position. The same slot across every row forms a column. A column
index in the model's state is called a **channel**. In the toy grid,
channel 1 contains `0.1`, `0.4`, and `-0.1`.

No one channel has a fixed English meaning. The pattern across all
four toy channels, or all 128 real channels, carries the useful
information. When this book says "the state at a text position," it
means that position's row of notes.

## One trip through the machine

Printed characters need conversion before the model can receive them.
The complete input route now has enough pieces to draw end to end:

```text
characters
    |
    v
character-to-id conversion          outside the model
    |
    v
integer ids                         shared predictor begins
    |
    v
opening rows of notes
    |
    v
revise the rows in each block
    |
    v
steady the final rows
    |
    v
one score per row and vocabulary entry
                                    shared predictor ends
```

Zone 1 is the input conversion. Zones 2 through 4 are the shared
predictor. Grading and choosing sit after it. Training supplies known
answers after the score rows exist. Generation passes the newest
score row to its chooser.

### Zone 1: characters become ids

Zone 1 is the character-to-id boundary built at the start of this
chapter. Printed `"A"` becomes its integer alphabet address. The
converter hands that address to the model, and `model_forward` begins
with ids.
[Chapter 3](03-data.md#tokenizer-the-alphabet-sorted) will build the
complete two-way conversion.

### Zone 2: opening notes

The id chooses an identity row. The position chooses a location row.
Their sum makes the opening notes. The two occurrences of `A` now
start differently:

```text
A at position 0 = identity(A) + location(0)
A at position 2 = identity(A) + location(2)
```

The identity part records what character sits there. The location
part distinguishes first from third.

### Zone 3: repeated editing blocks

Opening notes contain identity and location, but no information from
other positions. The row for `D` cannot yet record that `A` and `C`
came before it. Each repeated stage gives the notes two chances to
change:

1. each row consults itself and earlier rows, then adds an edit;
2. each row works within its own channels, then adds another edit.

Suppose one channel starts at `3.0`. The first edit contributes `0.1`
and the second contributes `-0.2`. Predict the value leaving the
stage.

```text
start               3.0
add first edit      0.1
                    ---
after first edit    3.1
add second edit    -0.2
                    ---
leaving value       2.9
```

This repeated two-edit stage is a **transformer block**, or **block**
for short. Tiny AgenC uses four blocks, each with its own learned
numbers.

The notes entering a branch remain available while that branch makes
an edit. Adding the first edit creates a new state, and that new state
feeds the second branch. The changing state is previewed here as the
residual stream.
[Chapter 5](05-forward-pass.md#add-an-edit-without-erasing-the-notes)
constructs its additions in full. A block amends the state instead of
discarding it and starting over.

### The first edit: consult earlier rows

Only one kind of operation lets positions exchange information. For
the row at position 1, it can draw from rows 0 and 1:

```text
row 0: notes about A ----+
row 1: notes about C ----+--> one edit for row 1
row 2: notes about D      X    not visible
```

The operation needs a way to decide which visible rows matter and how
much information to take from each.
[Chapter 5](05-forward-pass.md#let-one-position-consult-the-visible-past)
constructs that calculation under the name **attention**. It gives
each row three temporary roles, query, key, and value, and turns
comparisons into mixing shares. Chapter 1 does not need that
arithmetic yet. It needs the behavior: a position may consult its own
and earlier notes, never future notes.

Tiny AgenC does not build the no-peeking triangle and then hide its
upper half. Its attention loop visits only the permitted positions.
This is the exact controlling line from
[`attention_head_forward`](../src/ops.c):

```c
for (int source = 0; source <= time_index; source++)
```

Read its three clauses from left to right. `int source = 0` creates an
integer counter at the first position. `source <= time_index` permits
the loop body while the position being consulted is no later than the
position asking. `source++` advances the counter after each visit. Once
`source` becomes `time_index + 1`, the middle test rejects it. In this
source,
causality is a loop bound.

The consultation happens in several channel groups side by side.
That fact will explain the head-size division when we open the shapes.

### The second edit: work within one row

Consultation moves information between positions. A different
operation transforms each row without reading another position:

```text
row 0 -> widen its channels -> bend the numbers -> narrow -> edit row 0
row 1 -> widen its channels -> bend the numbers -> narrow -> edit row 1
row 2 -> widen its channels -> bend the numbers -> narrow -> edit row 2
```

For the real width of 128, the row widens to 512 numbers and narrows
back to 128.
[Chapter 5](05-forward-pass.md#put-a-bend-between-widen-and-narrow)
builds the multiply, the smooth bend called GELU, and the narrowing
operation. It names this branch the feed-forward layer or MLP. Its
local picture is enough here: consultation works across positions;
the MLP works across channels inside one position.

### Prepare a copy, preserve the highway

As blocks add edits, one row can arrive at a branch with much larger
numbers than another. Compare these two rows:

```text
row A    [ 1  2 ]
row B    [ 10 20 ]
```

Their entries have the same one-to-two relationship, but the second
row is ten times larger. Sending raw rows into every editor makes the
scale seen by an editor depend on everything added earlier.

Tiny AgenC therefore prepares a steadied copy for each branch. The
editor reads that copy, but its edit is added to the branch's
unchanged input:

```text
branch input x ----+-------------------------> add ----> output y
                   |                           ^
                   +-> steady copy -> editor --+
```

[Chapter 5](05-forward-pass.md#keep-one-rows-scale-from-controlling-the-next-operation)
constructs the steadying operation, layer normalization. Putting that
preparation before a branch and its addition after the branch is the
**pre-norm** block pattern.

The first addition changes the state from `x` to `y`. The second
branch therefore begins with `y`, not the old `x`:

```text
x -> steady -> consult -> mix -> first edit
x + first edit = y

y -> steady -> widen -> bend -> narrow -> second edit
y + second edit = z
```

The first branch consults positions. The second works inside each row.
Both preserve the row count and note width. Four blocks repeat this
two-edit sequence, and one final steadying operation closes the stack.

### Zone 4: notes become character scores

After the last block, each row still has 128 channels. The machine
needs one score for each of its 80 possible characters. It already
owns an 80 by 128 identity table whose row for `A` was used at the
entrance. At the exit, it reads the same table in the other direction:

```text
at the entrance             at the exit
id A -> row A of table      final notes -> score against every table row
```

The same learned table performs two jobs: character lookup at the
entrance and output scoring at the exit.
[Chapter 9](09-parameters-and-the-blueprint.md#one-block-all-its-learned-tensors)
constructs this one-table, two-jobs arrangement and gives it its
standard name.

The raw output scores are logits, previewed for
[Chapter 5](05-forward-pass.md#grade-one-next-token-bet). They are the
end of the shared predictor. Training grades all its score rows.
[Chapter 16](16-sampling.md#drawing-from-the-distribution) constructs
how generation chooses from the newest row.

The complete machine now has an earned name. A **transformer** here is
the route from opening identity-and-location notes, through repeated
consult-and-edit blocks, to next-character scores.

Tiny AgenC has no second stack reading a separate source sequence. It
has this one preceding-text-to-next-character route. That arrangement
is called **decoder-only**.

## The shapes now have somewhere to live

The plain-language route and the exact route are the same machine.
The letters attach sizes to parts that are already visible:

| Name | What it counts | Toy | Default |
|---|---|---:|---:|
| `B` | sequences in one batch | 2 | 32 |
| `T` | positions in one sequence | 3 | 128 |
| `C` | channels in each notes row | 4 | 128 |
| `H` | attention channel groups | 2 | 4 |
| `L` | repeated blocks | 1 | 4 |
| `V` | vocabulary entries and output scores | 5 | 80 |
| `R` | rows after stacking a batch, `B*T` | 6 | 4,096 |

Six are base dimensions. `R` is calculated from `B` and `T`, so no
configuration field stores it.

### Stack a batch into rows

One sequence supplies a `T x C` notes matrix. A batch starts as `B`
such matrices, which can be described as `B x T x C`. We now need one
word that covers a row, a matrix, and this collection of matrices. A
**tensor** is any shaped collection of numbers. A matrix is one kind
of tensor; `B x T x C` is another.

Tiny AgenC stacks the collection's rows into one tall `R x C` matrix:

```text
B x T x C  ->  (B*T) x C  =  R x C
```

Use the toy `B = 2`, `T = 3`, and `C = 4`:

```text
sequence 0                         sequence 1
position 0 [ 10 11 12 13 ]         position 0 [ 20 21 22 23 ]
position 1 [ 14 15 16 17 ]         position 1 [ 24 25 26 27 ]
position 2 [ 18 19 20 21 ]         position 2 [ 28 29 30 31 ]
```

Sequence 0 is stored first, then sequence 1:

```text
row 0 [ 10 11 12 13 ]     sequence 0, position 0
row 1 [ 14 15 16 17 ]     sequence 0, position 1
row 2 [ 18 19 20 21 ]     sequence 0, position 2
row 3 [ 20 21 22 23 ]     sequence 1, position 0
row 4 [ 24 25 26 27 ]     sequence 1, position 1
row 5 [ 28 29 30 31 ]     sequence 1, position 2
```

Before reading on, recover the sequence and position for row 4.

Each earlier sequence contributes `T = 3` rows. Sequence `s`,
position `p` therefore lands at:

```text
r = s*T + p
4 = 1*3 + 1
```

Undoing the formula gives sequence 1 and position 1. Integer division
finds the number of complete groups of three, and the remainder finds
the place within that group:

```text
4 div 3 = 1
4 mod 3 = 1
```

The stack is bookkeeping, not a merge. Position numbers restart at
zero in every sequence. Attention recovers both sequence and position
so no row consults a different sequence. Other operations can process
the tall matrix one row at a time.

Storing a `B x T x C` collection as `(B*T) x C` is **batch
flattening**. At the defaults:

```text
R = B*T = 32*128 = 4,096 rows
```

### Split channels into equal groups

Consultation happens `H` times side by side. The toy row has four
channels and two groups:

```text
channel index       0   1 | 2   3
attention group     0   0 | 1   1
```

Each group receives `C/H = 4/2 = 2` channels. That per-group width is
the **head size**.
[Chapter 5](05-forward-pass.md#split-the-channels-into-separate-heads)
gives an attention group its standard name, a head, and constructs
what each head computes.

Predict whether `C = 5`, `H = 2` can be split this way. One group
would receive two channels and one would need three. The groups no
longer have one shared shape, so the configuration is rejected.
`C` must divide evenly by `H`.

The default split is:

```text
C = 128 channels
H = 4 heads
head size = C/H = 32 channels
```

Changing `H` divides the same 128 channels differently. It does not
create another copy of the model's learned tables.

### Capacity and the active call

A prediction call cannot keep an unlimited past in working memory.
The default model reserves room for at most 128 positions.

The Chapter 0 sample still produced 400 new characters because
generation made many calls. After the text exceeded 128 characters,
each call used only the newest 128:

```text
all generated text:  [ older text ][ newest 128 characters ]
                                      ^ used for the next bet
```

[Chapter 16](16-sampling.md#one-choice-must-become-the-next-question)
builds this sliding window.

The configured maximum number of positions in one sequence is `T`,
called the **context length**. It is the ceiling on how far one call
can look back.

The configured `B` and `T` are capacity. A training call may use the
full 32 by 128. Sampling can use `batch = 1` and a shorter active
`time`. Temporary rows then number `batch * time`; learned tables keep
their configured sizes.
[Chapter 10](10-memory-planning.md#capacity-is-not-the-current-shape)
builds short matrix views over full-capacity storage.

The C source groups the six configured capacities in a description
named `ModelConfig`. For now, its field names are source labels. The
module map later shows how C builds that description.

The six base dimensions meet C under these names:

| Letter | `ModelConfig` field | Set by |
|---|---|---|
| `V` | `vocab_size` | the corpus |
| `T` | `block_size` | `--block` |
| `C` | `d_model` | `--width` |
| `H` | `head_count` | `--heads` |
| `L` | `layer_count` | `--layers` |
| `B` | `batch_size` | `--batch` |

The same machine sees different active shapes in its two modes. A
full-size training call uses every configured batch and time slot:

```text
TRAINING AT FULL CAPACITY

ids B x T -> opening notes R x C -> L blocks -> scores R x V
                  R = B*T                         |
                                                  v
                                           grade all R rows
```

A generation call carries one sequence and only the visible part of
that sequence:

```text
GENERATION

ids 1 x time -> notes time x C -> L blocks -> scores time x V
                                                        |
                                                        v
                                             use row time - 1

parameters are read, never adjusted
```

The default training call has `R = 32*128 = 4,096`. That number does
not describe every generation call. If generation can currently see
17 characters, it makes 17 temporary rows and uses row 16 for the
next-character scores. The learned tables keep their configured
shapes in both modes.

### Read one table in the other direction

The token table has shape `V x C`: one `C`-wide identity row for each
of `V` vocabulary entries. Final notes have shape `R x C`. Scoring
needs `R x V`: one row of `V` scores for each of the `R` positions.

Start with a smaller `2 x 3` grid:

```text
[ 1.0  0.5  2.0 ]
[ 0.0  3.0  1.5 ]
```

Predict where `2.0`, currently row 0 and column 2, lands when rows and
columns swap:

```text
[ 1.0  0.0 ]
[ 0.5  3.0 ]
[ 2.0  1.5 ]
```

It moves to row 2, column 0. The original rows are now columns. The
same grid read with rows and columns swapped is its **transpose**,
written with a superscript `T`.

The superscript is not the context dimension. In the scoring shape,
it says to read the `V x C` token table as `C x V`:

```text
final notes        token table transposed        scores
   R x C                    C x V                 R x V
```

The matching `C` widths allow every notes row to be compared with
every token row.
[Chapter 5](05-forward-pass.md#make-every-output-from-one-input-row)
constructs the row-by-row multiplication.

## What stays and what belongs to one trip

Run two training batches in sequence. Which values must survive from
the first run so the second can benefit from it?

```text
                     batch 7                    batch 8

learned numbers   tables before  ------->  adjusted tables
                       |                         |
                       v                         v
working values    notes for batch 7        notes for batch 8
                  keep for backward        keep for backward
                  then reuse storage       then reuse storage
```

The 815,360 adjustable numbers survive and are updated. Chapter 0
introduced them as **parameters**. A learned parameter is also often
called a **weight**.

The opening notes, prepared copies, branch results, and scores are
computed anew for each call. A value computed during a forward pass is
an **activation**. Input ids and grading answers are supplied data,
not activations, though Tiny AgenC stores them beside the activations
until backward finishes.
[Chapter 11](11-wiring-the-model-forward.md#the-caller-cannot-own-the-saved-ids)
constructs the exact latest-forward record and shows why another
forward call replaces it.

The word *weight* has a second use in attention. A learned weight
survives between calls. An attention weight is a temporary mixing
share calculated for the current text.
[Chapter 5](05-forward-pass.md#let-one-position-consult-the-visible-past)
makes the difference concrete.

### The activation ledger

Backward needs facts from the forward trip. Recomputing every earlier
operation whenever one fact is needed would repeat much of the work.
Tiny AgenC saves the needed activations until that batch's backward
pass is done.

The storage follows the machine:

```text
supplied data     ids and answers
                       |
activations       opening notes          R x C
                       |
                  prepared rows and edits
                       |
                  raw scores             R x V

parameters        token table, position table, block tables
                  survive after this trip ends
```

Training keeps these working values because backward will need them.
Once backward finishes, their storage can be reused by the next
batch. Generation has no backward pass, so it uses working values only
for the current prediction. Tiny AgenC still reserves the
full-capacity buffers up front; the active call uses views over the
rows it needs.

Some block operations need larger or differently shaped temporary
areas. The attention area grows with the square of `T`, because each
position reserves room to consult visible positions. Listing every
buffer here would hide the route under allocation details.
[Chapter 10](10-memory-planning.md#attention-owns-the-expensive-square)
draws the complete memory plan after the operations themselves are
known.
[Chapter 4](04-poor-mans-tensors.md#put-the-shape-beside-the-address)
constructs who owns that storage and how long a view may use it.

## Every learned number

The learned tables also have shapes. We will count them in two passes:
the 248-number toy first, then the 815,360-number default.

### Count the toy

For `V = 5`, `T = 3`, `C = 4`, and `L = 1`, predict whether `B = 2`
or `H = 2` will appear in the count.

`B` changes how many sequences use the same learned tables. `H`
divides the same channels into groups. Neither creates a new parameter
table, so neither appears.

Count the two opening tables:

```text
token table       V*C = 5*4 = 20
position table    T*C = 3*4 = 12
                  -------------
                  32 parameters
```

One block prepares a copy twice, once before each edit. Each
preparation owns a gain row and a bias row:

```text
first preparation     gain + bias     2 rows of C
second preparation    gain + bias     2 rows of C
                                      -----------
                                      4*C = 16
```

The gain row holds one multiplier per channel. The bias row holds one
addition per channel. Their exact steadying job belongs to Chapter 5;
their `C`-wide shapes are enough for this count.

The consultation branch makes three `C`-wide roles from one `C`-wide
row. The source shortens query, key, and value to `QKV` and packs
their learned connections together as one `3C x C` table. It then
uses one `C x C` table to mix the consulted result. The within-row
branch widens from `C` to `4C`, then narrows from `4C` to `C`.

Count the one block:

```text
QKV weights       3C*C = 3*4*4 = 48
projection        C*C  =   4*4 = 16
MLP up            4C*C = 4*4*4 = 64
MLP down          C*4C = 4*4*4 = 64
four norm rows    4C   =     16
                                  ---
                                  208 parameters
```

The final norm adds `2C = 8`. Total:

```text
32 + 208 + 8 = 248
```

### Generalize the shapes

A weight matrix reading `C` inputs and producing `3C` outputs is stored
as `3C x C`: one row for each output.
[Chapter 5](05-forward-pass.md#make-every-output-from-one-input-row)
builds the multiplication and names a multiply-by-a-weight-matrix step
a linear transform.

Read the two MLP shapes against that rule. The up transform reads `C`
and produces `4C`, so its matrix is `4C x C`. The down transform reads
`4C` and produces `C`, so its matrix is `C x 4C`.

| Parameter | Shape | How many |
|---|---:|---:|
| token table | `V x C` | `V*C` |
| position table | `T x C` | `T*C` |
| norm 1 gain and bias, per block | `1 x C` each | `2*C` |
| packed QKV weights, per block | `3C x C` | `3*C*C` |
| attention projection, per block | `C x C` | `C*C` |
| norm 2 gain and bias, per block | `1 x C` each | `2*C` |
| MLP up weights, per block | `4C x C` | `4*C*C` |
| MLP down weights, per block | `C x 4C` | `4*C*C` |
| final norm gain and bias | `1 x C` each | `2*C` |
| final scoring stage | token table reused | `0` new |

The four block matrices have no added bias rows. The layer
normalizations do have their own gains and biases. Tiny AgenC also
never randomly silences working values during training. Those are
exact choices of this architecture, not rules for every transformer.

### Count the default

`C^2` is a compact spelling of `C*C`. One block contains:

```text
3C^2 + C^2 + 4C^2 + 4C^2 + 4C
= 12C^2 + 4C
```

Call the total number of parameters `P`:

```text
opening tables  (V + T)C
each block      12C^2 + 4C
final norm      2C

P = (V + T)C + L(12C^2 + 4C) + 2C
```

Substitute the defaults:

```text
P = (80 + 128)128 + 4(12*128^2 + 4*128) + 2*128
  = 26,624 + 4(197,120) + 256
  = 26,624 + 788,480 + 256
  = 815,360
```

The formula is a function, not a memorized constant. Change `V`, `T`,
`C`, or `L`, and the answer changes. `B` changes activations. `H`
changes the channel split. Neither changes this count.

The final scoring stage adds zero because it reuses the token table.
If it allocated a second `V x C` table, the count would be larger by
`V*C = 80*128 = 10,240`.

If your program prints another number, stop there. A wrong parameter
count is not a cosmetic disagreement. It means you built a different
model.

### How the learned numbers start

The opening showed the three starting patterns before training:
small repeatable random matrix values, gains of one, and biases of
zero. The shapes now give the exact recipe somewhere to attach.

The gain and bias form a learned volume knob. Its starting action is
neutral: multiply by one, then add zero.

| Parameter group | Initialization |
|---|---|
| token and position tables | Gaussian, spread `0.02` |
| QKV and MLP-up matrices | Gaussian, spread `0.02` |
| projection and MLP-down matrices | Gaussian, spread `0.02/sqrt(2L)` |
| normalization gains and biases | ones and zeroes |

[Chapter 2](02-foundations.md#turn-two-flat-draws-into-a-bell)
constructs repeatable randomness, the bell-shaped Gaussian rule, and
spread.
[Chapter 9](09-parameters-and-the-blueprint.md#initialization-is-architecture)
applies them to these tables. The smaller `0.02/sqrt(2L)` start belongs
to matrices whose results are added back to the residual stream. There
are two such write-backs per block.

The table and formula match
[`model_create_parameters`](../src/model_parameters.c) and
[`model_parameter_float_count`](../src/model.c).

## The map becomes C modules

The machine is conceptually one route, but putting its entire
implementation in one file would let every part reach every detail.
The source instead gives each group of files and functions one
contract. Such a group is a **module**.

The diagram is a teaching map of the main data and compute paths, not
an exhaustive include graph:

```text
main.c ---> model ---> ops ---> mat        compute spine
   |          |
   |          +----> param ---> rng        learnable state
   |          +----> tokenizer             saved vocabulary
   +----> dataset ---> tokenizer
   +----> util                              shared support
```

| Module | One contract |
|---|---|
| `util` | checked memory, errors, file reads, integers, clock |
| `rng` | replayable random fractions and bell-shaped values |
| `tokenizer` | known bytes to ids and valid ids back to bytes |
| `dataset` | corpus ids and shifted input/answer batches |
| `mat` | rows and columns describing separate float storage |
| `ops` | forward and backward numerical operations |
| `param` | learned values, correction signals, update history |
| `model` | transformer wiring, memory, step, sample, save, load |
| `main` | command parsing, train/sample orchestration, output |

This is also the book's itinerary.
[Chapter 2](02-foundations.md) builds `util` and `rng`;
[Chapter 3](03-data.md) builds `tokenizer` and `dataset`;
[Chapter 4](04-poor-mans-tensors.md) builds `mat`;
[Chapters 5](05-forward-pass.md) and [6](06-backprop-by-hand.md) build
the two directions through `ops`; [Chapter 8](08-adamw.md) builds
`param`; Chapters [9](09-parameters-and-the-blueprint.md) through
[13](13-durable-checkpoints.md) and [16](16-sampling.md) build the model
files; and Chapters [14](14-the-command-line.md) through
[16](16-sampling.md) build the command-line program.

### An address instead of exposed fields

Five modules keep state between calls: `rng`, `tokenizer`, `dataset`,
`param`, and `model`. If a caller could change every field directly,
the module's rules would be optional.

Consider a model object somewhere in memory:

```text
caller                              model module

Model *m  ---- stores address ----> [ private Model fields ]
   |
   +------ model_forward(m, ...) --> public operation
```

An address tells C where an object lives. A value that stores an
address is a **pointer**. The declaration `Model *m` means that `m`
is a pointer to a `Model`.

The `ModelConfig` table above grouped six named configuration values.
C calls a group of named fields a **struct**. A `Model` also has named
fields, but its public header does not reveal them.
[Chapter 2](02-foundations.md#read-bytes-without-trusting-the-file)
explains the C spelling that creates such a short type name.

The field definition lives in the private `model_internal.h` header
used by the model implementation. Callers can hold the address and
pass it to public functions, but cannot edit the hidden
representation. A pointer used through such a boundary is an
**opaque handle**.

### When copying the description is useful

`ModelConfig` makes the opposite choice from `Model`. Its six integer
fields are public and small. Assigning one configuration to another
copies the complete description:

```text
ModelConfig a: [ V | T | C | H | L | B ]
                         |
                         | assignment copies all six fields
                         v
ModelConfig b: [ V | T | C | H | L | B ]
```

A small public struct intended to be copied as a complete description
is a **value type**.
[Chapter 4](04-poor-mans-tensors.md#put-the-shape-beside-the-address)
applies that trade to `Mat`, whose description refers to separate
float storage, and constructs who may free those floats and how long
other code may use them.

The remaining modules, `mat`, `ops`, `util`, and `main`, keep no
mutable private object between calls. They are collections of
operations.
A module with functions but no retained private state is a
**function module**.

### Two rules for finding your way

The source map gives the reader two navigation rules.

First, public headers describe allowed interactions. Private headers
hold the fields needed to implement them. When you want to learn what
a caller may do, read the public header. When a later chapter opens a
module, it will show the private representation beside the code that
maintains its rules.

Second, inside `ops.c`, each numerical backward operation sits beside
the forward operation it reverses. In C, `x += y` computes `x + y`
and stores the result back in `x`. When two reverse paths return to
one value, `+=` preserves both contributions.
[Chapter 6](06-backprop-by-hand.md#add-every-returning-path) constructs
that gradient-accumulation rule from a value used twice.
[Chapter 12](12-wiring-the-model-backward.md#one-block-has-two-reverse-meeting-points)
later applies it at both residual meetings, between stacked blocks, and
at the two uses of the tied token table.

Other C rules arrive with the first code that needs them. They are not
extra pieces of the transformer map.

## Build checkpoint: specify the machine

The Chapter 0 workspace already contains `ModelConfig`, `spec.h`, and
the dimension ceilings from [`model.h`](../src/model.h). `spec.h`
declares two functions:

```text
lab_config_valid      reject impossible or unsafe geometry
lab_parameter_count   count every learned number
```

Their job is to turn this chapter's map into executable claims.

### Make the geometry safe

`V`, `T`, `C`, `H`, and `L` must be positive and within their named
ceilings. `B` must be positive. Two rules combine dimensions:

| Check | Reason |
|---|---|
| `B*T <= MODEL_MAX_TOKENS_PER_PASS` | flattened rows must fit |
| `C % H == 0` | each head needs a whole channel count |

In C, `%` produces the remainder. The toy `4 % 2` is zero, so four
channels split evenly between two heads. `5 % 2` is one, so five
channels do not.

Conditions joined by `&&` are tested left to right and stop when one
is false. Test that `H` is positive before evaluating `C % H`.
Remainder by zero is invalid.

The product `B*T` has a separate C edge. Multiplying two `int` values
can overflow before a wider destination receives the result. This
does not help:

```c
long long rows = batch * block;
```

The multiplication on the right still happens as `int`. Convert one
input first:

```c
long long rows = (long long)batch * block;
```

The cast `(long long)` tells C to convert `batch` before multiplying,
so the multiplication itself uses the wider type.

### Count in the type C uses for object sizes

Parameter counts use `size_t`, C's unsigned integer type for object
sizes and element counts. Every input must enter that type before an
addition or multiplication could overflow as `int`. The safe lab
expression is:

```c
size_t c = (size_t)width;

return ((size_t)vocab + (size_t)block) * c
     + (size_t)layers * (12 * c * c + 4 * c)
     + 2 * c;
```

`vocab` and `block` are each converted before their addition.
`layers` is converted before it multiplies the per-block count.
Casting an already-overflowed `int` result to `size_t` would be too
late.

**Build.** Implement `lab_config_valid` and `lab_parameter_count` in
`labs/work/spec.c`. Do not create parameters yet. After both functions
have an answer, compare the production predicate with
[`model_config_valid`](../src/model.c).

**Verify.**

```sh
make -C labs check-01
# answer key: make check-model
```

**Expected.** The default configuration reports `815360`. Width 128
with four heads is accepted; width 127 with four heads is rejected.
Zero and over-ceiling dimensions are rejected. The exact `B*T`
ceiling is accepted and one row beyond it is rejected. The
large-shape count succeeds without `int` overflow.

**Common failures.**

- `819968` is the default count after adding biases to all four linear
  transforms in every block. This model does not have those biases.
- A count larger by `V*C` usually means a separate output table was
  added instead of reusing the token table.
- In C, `!=` means "is not equal to." Accepting a configuration when
  `C % H != 0` postpones failure until attention tries to split the
  channels.
- A crash on `H = 0` means the remainder ran before the positive-head
  check.
- A wrong large-shape count means some multiplication happened as
  `int` before conversion to `size_t`.
- If `labs/work/spec.c` is missing, run `bash labs/start.sh` from the
  repository root, as in Chapter 0.

## Check the map from memory

Cover the chapter. Write `ACDB`, then put its three model inputs and
three known answers beneath it. Circle where every answer came from.
Recreate the untrained score row after `A`, identify its unsupported
top choice, and sketch the four learning snapshots. Note that each
snapshot may stand many batches apart.

Now draw these two timelines from memory:

```text
TRAINING

current parameters ----+
batch input ids -------+-> predict -> score rows --+
batch answers -------------------------------------+-> grade -> adjust

repeat through step 5,000:
adjusted parameters and a newly drawn corpus batch enter the next step
after the final adjustment -> save trained.bin

GENERATION

fixed saved parameters --+
visible text ------------+-> predict -> choose -> append
       ^                                          |
       +----------------- repeat -----------------+
```

Before uncovering the chapter, mark where parameters change. Mark the
point after which they stay fixed. Then add the short progress sample
that may run between training steps. Its output must not point into
the next step's answers.

Below the timelines, draw the one predictor that both modes call:

```text
ids -> opening notes -> repeated edits -> raw scores
```

Training grades its score rows. Generation chooses from its newest
row. Neither action belongs inside the shared predictor.

Finally, recover what `B`, `T`, `C`, `H`, `L`, `V`, and `R = B*T`
count. Label one parameter and one activation. If the generation path
contains a grade or an adjustment, or the training path contains an
appended model choice, return to the two-mode table. If a shape is
missing, return to the matching route section.

Chapter 2 now builds the ground under that map: memory requests that
cannot fail silently, and randomness that can be replayed on demand.

---

[Previous: Introduction](00-introduction.md) | [Contents](README.md) |
[Next: Foundations](02-foundations.md)
