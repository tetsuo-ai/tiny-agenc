# Chapter 11: Wiring the Model Forward

Every operation in this chapter has already passed its own test. Our
new failure mode is order.

Chapter 5 built the forward operations one at a time. Chapter 9 gave
every learned matrix one exact place in the model. Chapter 10 gave
every temporary result a reusable place in memory. None of that says
which result the next operation must read.

A correct layernorm can normalize the wrong stream. A correct residual
addition can preserve the wrong earlier state. A correct block can run
twice while its second call receives the original input again. This
chapter connects the complete score route and gives that wiring its own
executable witness.

First, place this work inside one training step:

```text
clear gradients -> forward -> backward -> adjust parameters
                     ^
                     |
                 Chapter 11
```

Only the final adjustment changes parameter values. Forward reads the
current values and writes temporary results. Backward will calculate
gradients in Chapter 12. The adjustment machinery from Chapters 8 and
9 will use those gradients later.

Suppose one token-table row starts as:

```text
before model_forward     [ 0.10  -0.20 ]
after model_forward      [ 0.10  -0.20 ]
```

**Predict:** which later phase can change the row?

The adjustment can. Forward cannot. This remains true whether or not
the call receives known answers. A call with answers can produce a
grade, but a grade is not an update.

## Targets join after the scores

[Chapter 1 separated the shared score
route](01-the-map.md#one-predictor-called-in-two-modes) from its two
callers. The exact whole-model wiring keeps that separation visible:

```text
WITH TARGETS

tokens -> embedding -> blocks -> final norm -> logits
logits -> probabilities
probabilities + targets -> mean loss

WITHOUT TARGETS

tokens -> embedding -> blocks -> final norm -> logits
                                                    |
                                                    +-> return 0
                                                        no loss computed
```

Tokens enter the calculation at the beginning. Target ids are copied
into model-owned storage near the beginning too, but no score-producing
operation reads them. They affect the calculation only after the logits
exist, when cross-entropy grades the probability assigned to each
target. There is no arrow from targets into embedding, attention,
logits, or probabilities.

Use one row with two possible next tokens. Let its logits be:

```text
[ 0  log(3) ]
```

[Chapter 5's
softmax](05-forward-pass.md#turn-arbitrary-scores-into-usable-shares)
subtracts the largest logit, exponentiates, and divides by the total.
The same calculation can be written without the subtraction because
these values are small:

```text
exp values       [ 1     3 ]
total              1 + 3 = 4
probabilities    [ 1/4   3/4 ]
```

Now change only the answer:

```text
target id 1:  loss = -log(3/4) = 0.28768...
target id 0:  loss = -log(1/4) = 1.38629...
```

**Predict:** did changing the target change either logit?

No. It changed which unchanged probability was graded. The probability
row is also unchanged because softmax reads logits, not targets. If
there is no target, cross-entropy is skipped and this API returns
`0.0f` to mean that it did not compute a loss. That zero is not a
perfect grade.

This distinction prevents two common misreadings:

- Supplying targets does not make `model_forward` learn.
- Omitting targets does not make `model_forward` choose a character.

The first call still needs later backward and adjustment phases. The
second call leaves raw score rows for a later caller. Chapter 16 will
build the choice-and-append loop around those rows.

## One block has two highway states

The [residual
highway](05-forward-pass.md#add-an-edit-without-erasing-the-notes)
passes through two branches in every block. A flat list of operation
names hides the fact that the highway changes between them.

Strip away the already-tested branch arithmetic and keep two
two-channel edits. Begin with:

```text
incoming stream x          [ 10  20 ]
attention edit p           [  1  -2 ]
```

**Predict:** what continues after the first residual addition?

```text
r1 = x + p                 [ 11  18 ]
```

The MLP branch reads that new stream and produces another edit:

```text
MLP edit d, made from r1   [  3   4 ]
```

The second residual must preserve `r1`:

```text
r2 = r1 + d                [ 14  22 ]   correct
r2 = x  + d                [ 13  24 ]   loses the attention edit
```

Both additions are valid elementwise arithmetic. Only one wires the
intended block. The two highway states are visible in the block:

```text
x --------------------------------------------> (+) -> r1
 \-> norm1 -> QKV -> attention -> project p -----^

r1 -------------------------------------------> (+) -> r2
  \-> norm2 -> up -> GELU -> down d -------------^
```

The branch reads a steadied copy, while the unnormalized highway stays
available for the addition. Chapter 1 named that arrangement
[pre-norm](01-the-map.md#prepare-a-copy-preserve-the-highway), and
Chapter 5 built every operation inside it.

One block can now be spoken without losing either left side:

```text
n1  = layernorm(x)
qkv = matmul(n1, Wqkv)
a   = causal_attention(qkv)
p   = matmul(a, Wproj)
r1  = x + p

n2  = layernorm(r1)
u   = matmul(n2, Wup)
g   = gelu(u)
d   = matmul(g, Wdown)
r2  = r1 + d
```

Read the two layernorm inputs and the two residual left sides:

```text
first branch:   normalize x,  then add its edit to x
second branch:  normalize r1, then add its edit to r1
```

That is the wiring invariant the C must preserve.

## Attach the current shapes

The model owns the arena allocation. A block stores capacity-sized
`Mat` descriptions in its own fields; those descriptions point into
the arena. A particular call needs local descriptions for only its
active batch and time.

The Chapter 11 witness configures:

```text
V=5, block capacity=4, C=8, H=2, L=1, batch capacity=2
```

Its full call uses two sequences of four positions, so `R = 2*4 = 8`.
Its short call uses one sequence of three positions, so `R = 1*3 = 3`.
Apply [Chapter 10's current-shape
rule](10-memory-planning.md#capacity-is-not-the-current-shape):

| Field | Full `B=2, T=4` | Short `B=1, T=3` |
|---|---:|---:|
| residual-width values | `8 x 8` | `3 x 8` |
| packed QKV | `8 x 24` | `3 x 24` |
| attention scores | `16 x 4` | `6 x 3` |
| widened MLP values | `8 x 32` | `3 x 32` |
| logits and probabilities | `8 x 5` | `3 x 5` |

The score row count is `B*H*T`. **Predict:** for one sequence, two
heads, and three positions, how many rows and columns are needed?

```text
rows = 1*2*3 = 6
cols = current T = 3
```

A leading `6 x 4` view would retain the capacity stride and be wrong.
Chapter 10 already implemented the required `6 x 3` reshape. Chapter
11 requests that view before it calls attention.

For general active shapes, the complete block ledger is:

| Value | Shape |
|---|---:|
| incoming stream, `normed1` | `R x C` |
| packed QKV | `R x 3C` |
| attention scores | `B*H*T x T` |
| attended, projected, `after_attention` | `R x C` |
| `normed2` | `R x C` |
| up projection, activated | `R x 4C` |
| down projection, `after_mlp` | `R x C` |

The first `R` entries of each mean and reciprocal-standard-deviation
array accompany the two layernorm outputs. They are plain float
arrays, not `Mat` fields. Layernorm writes exactly one of each per
active input row.

## Walk one block in source order

Here is the complete `block_forward` from
[`model_forward.c`](../src/model_forward.c):

```c
static Mat block_forward(const Model *m, Block *b, Mat stream,
                         int batch, int time)
{
    BlockTensors a =
        model_block_views(&b->acts, batch * time, time, m->cfg.head_count);

    layernorm_forward(a.normed1, b->means1, b->rstds1, stream,
                      param_values(b->norm1_gain).vals,
                      param_values(b->norm1_bias).vals);
    matmul_forward(a.qkv, a.normed1, param_values(b->qkv_weights));
    attention_forward(a.attended, a.scores, a.qkv, time, m->cfg.head_count);
    matmul_forward(a.projected, a.attended, param_values(b->proj_weights));
    residual_forward(a.after_attention, stream, a.projected);

    layernorm_forward(a.normed2, b->means2, b->rstds2, a.after_attention,
                      param_values(b->norm2_gain).vals,
                      param_values(b->norm2_bias).vals);
    matmul_forward(a.up, a.normed2, param_values(b->up_weights));
    gelu_forward(a.activated, a.up);
    matmul_forward(a.down, a.activated, param_values(b->down_weights));
    residual_forward(a.after_mlp, a.after_attention, a.down);

    return a.after_mlp;
}
```

The signature gives the helper four kinds of information. `m` supplies
read-only configuration such as the head count. `b` points to this
layer's parameters, views, and saved statistics. `stream` is the
borrowed `Mat` entering the layer. `batch` and `time` describe this
call, not the model's capacity. Chapter 2 introduced file-private
`static` and read-only `const`; neither changes those roles.

The first statement asks `model_block_views` for active descriptions.
It copies one `BlockTensors` collection of `Mat` descriptions into
local variable `a`. It does not copy the floats and does not allocate.
The descriptions disappear on return; the arena-backed values remain
until a later forward call overwrites them.

The first `layernorm_forward` reads `stream`, writes `a.normed1`, and
saves one mean and reciprocal standard deviation per active row.
`param_values` supplies the learned gain and bias values. No gradient
or optimizer field is involved.

The next matrix multiplication turns `R x C` normalized notes into
`R x 3C` packed queries, keys, and values. Attention reads that packed
matrix, writes `R x C` attended notes, and stores its current-stride
weights in `a.scores`. The projection returns the branch to `R x C`.

Now the first residual handoff appears:

```c
residual_forward(a.after_attention, stream, a.projected);
```

It adds the projected edit to the original incoming `stream`, producing
the numeric example's `r1`.

The second layernorm therefore reads `a.after_attention`. The up
projection widens `C` channels to `4C`; GELU bends each value; the down
projection returns to `C`. The second residual handoff is:

```c
residual_forward(a.after_mlp, a.after_attention, a.down);
```

Its left side is the first residual result, not `stream`. Returning
`a.after_mlp` copies a three-field `Mat` description by value. It does
not copy the `R*C` result floats.

Cover the code and answer these four questions:

```text
first layernorm reads       stream
first residual preserves   stream
second layernorm reads      after_attention
second residual preserves  after_attention
```

If those four names are right, the two highway states are right.

## One returned view feeds the next block

The complete model may contain more than one block. Calling each block
with `embedded` would create several independent edits of the opening
notes. The intended stack carries each result into the next layer:

```text
embedded
   |
   v
block 0 after_mlp
   |
   v
block 1 after_mlp
   |
   v
...
   |
   v
block L-1 after_mlp
   |
   v
final layernorm
```

Reuse the earlier numbers. If block 0 returns `[14, 22]`, block 1 must
receive `[14, 22]`, not the original `[10, 20]`.

The local `stream` variable carries that borrowed view. Assigning a
new `Mat` to it changes three description fields: address, rows, and
columns. The result floats stay in block 0's arena region. No
residual-width matrix is copied between blocks.

The Chapter 11 lab configures `L=1`, so its fixed output cannot directly
test this multi-block handoff. The source walk establishes the
assignment. Chapter 12's whole-model derivative witness uses two
layers and will exercise both directions through the stack.

## The caller cannot own the saved ids

Forward and backward are separate function calls. Suppose a dataset
hands the model these arrays:

```text
caller inputs       [ 0  1  2 ]
caller targets      [ 1  2  3 ]
```

After `model_forward` returns, the dataset is free to reuse its batch
buffers:

```text
same caller arrays  [ 4  4  4 ]
```

Chapter 12's backward call must still know that the earlier input ids
were `[0, 1, 2]` and the answers were `[1, 2, 3]`. Saving only the
caller's pointers would give those arrays a lifetime the model cannot
enforce.

Chapter 10 allocated two model-owned integer buffers. The final private
fields introduced structurally in Chapter 9 are these selected exact
lines from [`model_internal.h`](../src/model_internal.h):

```c
    int    *tokens;
    int    *targets;
    int     batch;
    int     time;
    int     has_targets;
```

The model copies the active ids and saves the active shape. It also
records whether the newest call supplied targets. Every call's
score-route values, saved layernorm statistics, logits, copied tokens,
active shape, and target-status flag form its **latest-forward record**.
When targets exist, their copied ids and the fresh probabilities join
that record. This is a lifetime contract spread across model fields,
not a separate C struct.

Only the newest call has a current record. A later call reuses the same
addresses and replaces the earlier record's applicability. The
capacity descriptions remain in the model and block fields.

## Begin the whole-model call

The first half of the complete `model_forward` establishes that record.
This is exact source:

```c
float model_forward(Model *m, const int *tokens, const int *targets,
                    int batch, int time)
{
    assert(batch >= 1 && batch <= m->cfg.batch_size);
    assert(time >= 1 && time <= m->cfg.block_size);

    int rows = batch * time;

    m->batch       = batch;
    m->time        = time;
    m->has_targets = targets != NULL;
    memcpy(m->tokens, tokens, (size_t)rows * sizeof *tokens);
    if (targets != NULL)
        memcpy(m->targets, targets, (size_t)rows * sizeof *targets);
```

The signature accepts borrowed input ids and optional borrowed target
ids. `const` prevents this function from changing either caller array
through those pointers. The model itself is mutable because the call
must write its latest-forward record and activation values.

The two assertions require a positive active batch and time within the
configured capacities. Given a valid model configuration, their
product fits the configured maximum, so `rows = batch*time` is the
active flattened row count.

The next three assignments publish the active shape and target status.
In C, comparing a pointer with `NULL` produces true when an address was
supplied and false when it was not.

Chapter 3 built [`memcpy`](03-data.md#draw-every-legal-start)
as a byte copy between valid, non-overlapping regions. The first copy
moves exactly `rows` integers into model-owned storage. `sizeof
*tokens` means the size of one pointed-to integer; the cast moves the
nonnegative row count into C's object-size type before multiplication.

Targets are different. When their pointer is non-null, the second copy
refreshes the first `rows` target entries. When it is null, the copy is
skipped. Old target bytes may remain in memory, but `has_targets` says
they do not belong to the newest call.

The caller may reuse both arrays after this function returns. Backward
will read the model-owned copies.

## Open the stream and walk upward

The next exact source chunk creates opening notes and threads the local
view through the block stack:

```c
    Mat stream = mat_first_rows(m->embedded, rows);

    embedding_forward(stream, m->tokens, param_values(m->token_table),
                      param_values(m->position_table), time);
    for (int layer = 0; layer < m->cfg.layer_count; layer++)
        stream = block_forward(m, &m->blocks[layer], stream, batch, time);
```

`m->embedded` describes capacity storage. `mat_first_rows` returns a
view of its first active `rows` without moving floats.

Embedding reads the copied token ids, the token table, and the position
table. It receives the current `time`, not the block capacity. For a
two-sequence call with `T=3`, flattened rows map to positions like
this:

```text
flattened row       0  1  2  3  4  5
row % time          0  1  2  0  1  2
```

The second sequence therefore restarts at position zero. Passing a
capacity of four would instead give row 3 position 3 and break the
sequence boundary.

The loop visits block indices from zero upward. The assignment matters:

```c
stream = block_forward(...);
```

Each call receives the view returned by the previous call. After the
last iteration, `stream` describes the last block's `after_mlp`
values. With one layer it describes block 0's output; with four, it
describes block 3's.

## Finish with the tied head

Block output still has `C` channels per row. The final route steadies
those notes, then produces one score per vocabulary id:

```c
    Mat final_normed = mat_first_rows(m->final_normed, rows);
    Mat logits       = mat_first_rows(m->logits, rows);

    layernorm_forward(final_normed, m->final_means, m->final_rstds, stream,
                      param_values(m->final_gain).vals,
                      param_values(m->final_bias).vals);
    matmul_forward(logits, final_normed, param_values(m->token_table));
```

Both destinations narrow capacity storage to the active rows. Final
layernorm reads the last block's returned stream and writes `R x C`
notes plus `R` saved means and reciprocal standard deviations.

Chapter 9 constructed the
[tied head](09-parameters-and-the-blueprint.md#one-table-two-jobs).
The same `m->token_table` used for the opening lookup now supplies
`V x C` weight rows. The ordinary Chapter 5 matrix multiplication has:

```text
input             R x C
weight rows       V x C
output logits     R x V
```

For one two-channel row, use:

```text
final notes             [ 2  -1 ]

token-table row 0       [ 1   0 ]   score =  2
token-table row 1       [ 0   1 ]   score = -1
token-table row 2       [ 1   1 ]   score =  1
```

The three dot products become one `1 x 3` logit row:

```text
[ 2  -1  1 ]
```

No output-weight field exists to choose by accident. This line reads
the token-table values; it neither copies them nor changes them.

## Stop when there is no answer sheet

Generation needs the raw scores for visible text, but it has no known
next character to grade. Passing a made-up target would calculate a
meaningless loss and mark that call as eligible for backward.

The last exact source lines make grading optional:

```c
    if (!m->has_targets)
        return 0.0f;
    return crossentropy_forward(mat_first_rows(m->probs, rows), logits,
                                m->targets);
}
```

Chapter 3 introduced logical NOT: `!m->has_targets` is true when the
newest call supplied no targets. That path returns after logits have
been written and before cross-entropy runs. It does not refresh
`m->probs`; old or uninitialized probability bytes have no meaning for
this call.

When targets exist, the final line narrows probability storage to
`R x V`, converts each logit row into probabilities, grades the copied
target id, and returns the mean loss.

Now the no-answer path has been constructed. A target-free
`model_forward` call is this program's **inference mode**.

| Result of this call | Targets supplied | Targets are `NULL` |
|---|---:|---:|
| shared route through logits runs | yes | yes |
| fresh probabilities and loss | yes | no |
| returned float | mean loss | `0.0f` sentinel |
| newest record may feed backward | yes | no |
| parameter adjustment | no | no |
| character choice or append | no | no |

Inference mode is smaller than generation. It leaves private raw
scores in `m->logits`. Chapter 16's `model_sample` will read the newest
row, apply temperature and softmax, draw one id, append it, and call
forward again. None of those actions happen here.

Likewise, a target-bearing call is smaller than training. A validation
caller can ask for a loss and stop. Learning occurs only if the caller
later runs backward and an accepted parameter adjustment.

## Only the newest forward record survives

The arena exists to be reused. A second forward call replaces the first
record's applicability even when both calls have targets:

```text
graded call A -> graded call B -> backward
                                  uses B, not A
```

**Predict:** after call B, can backward still mean call A?

No. The model has one arena, one pair of id buffers, one saved batch and
time, and one target-status flag. Its stored capacity descriptions stay
unchanged. Call B creates local active views and overwrites the active
contents that its route defines and may later read.

A target-free call writes a new score route but skips two pieces:

```text
graded call A -> target-free call B -> backward
                                      invalid: B has no answers
```

After B:

- B's embedded, ordinary block, final-normalized, saved-statistic, and
  logit prefixes plus its written causal score entries hold its values;
- copied tokens, batch, and time describe B;
- target bytes and probabilities may still contain A's values;
- `has_targets` is false;
- the gradient arena has not been touched by forward.

Unused capacity tails and the unread upper triangle of causal score
rows may also retain older bytes. They are outside the values B defines
or a matching backward call would read. Replacing the record is a
statement about which call the saved state belongs to, not a claim that
every arena byte was erased.

Mixing B's activations with A's probabilities and targets would not
differentiate either call. The API therefore requires backward to
follow the immediately preceding target-bearing forward.

Chapter 12 enforces that programming contract with this selected exact
source:

```c
void model_backward(Model *m)
{
    assert(m->has_targets);
```

In the repository's normal builds, assertions are enabled and this
misuse stops at the assertion. If a caller compiles with `NDEBUG`, the
assertion disappears. The source has no second runtime rejection, so
such a build must uphold the call-order precondition itself. This is a
programmer contract, not a recoverable input error.

A later target-bearing forward creates a new eligible record. Backward
does not clear `has_targets`. [Chapter
12](12-wiring-the-model-backward.md#start-from-clean-destinations)
explains why a second unzeroed backward call compounds stale
intermediate gradients and is outside the supported contract.

## The complete source route

The walked source chunks join into one short file:

```text
model_forward
|
+-- assert active batch and time preconditions
+-- record shape and copy active ids
+-- embed copied tokens
+-- block 0 -> block 1 -> ... -> block L-1
+-- final layernorm
+-- token-table matmul -> logits
|
+-- no targets  -> return the no-loss sentinel
|
+-- targets     -> probabilities and mean loss
```

Search both functions for a parameter-changing operation. Every
learned object enters through `param_values`. There is no
`param_gradient`, `model_backward`, clipping call, or AdamW step in
[`model_forward.c`](../src/model_forward.c).

There is also no allocation. All destinations came from the arenas and
id buffers constructed in Chapter 10. The call changes contents, not
capacity or ownership.

## A whole-model fixed-output witness

[Chapter 7](07-trust-but-verify.md#two-referees-watch-different-calculations)
showed that an independently worked known answer can referee the
intended equation. No such derivation record exists for Chapter 11's
whole-model loss literals. Here the source walk establishes the
intended wiring, and the fixed outputs pin that assembled behavior
against later regressions.

The lab uses this configuration and seed:

```text
V=5, block capacity=4, C=8, H=2, L=1, batch capacity=2
seed=17
```

`model_new` initializes the parameters from that seed, and the witness
runs no backward or adjustment. This model is deliberately untrained.
The fixture checks deterministic wiring, not learned text quality, and
it does not generate text.

Its full input loop creates these two flattened sequences:

```text
sequence 0 inputs     [ 0  1  2  3 ]
sequence 0 targets    [ 1  2  3  4 ]

sequence 1 inputs     [ 4  0  1  2 ]
sequence 1 targets    [ 0  1  2  3 ]
```

The first check compares the returned loss with the fixed expected
literal in [`labs/check11.c`](../labs/check11.c). The literal is a
seeded regression fixture, not a universal transformer value or a
hand-derived mathematical oracle. This chapter does not reprint it as
measured evidence because no `EVIDENCE.md` entry or committed log
records how it was captured.

The next call supplies the same inputs and a null target pointer. Its
check proves that the call returns the `0.0f` no-loss sentinel. It does
not inspect the private logits, so the complete null-target score route
is established by the source walk here and exercised through sampling
in Chapter 16.

The shortened fixture is:

```text
inputs                 [ 0  1  2 ]
targets                [ 1  2  3 ]
changed targets        [ 1  4  3 ]
```

One check compares the `1 x 3` call with its fixed expected loss. The
last changes only the middle target and requires the returned loss to
move. That catches an implementation that ignores the supplied answer
ids.

The four checks share one implementation and one model. They do not
independently prove every private contract. In particular, they do not
inspect the copied ids or status fields, parameter immutability, fresh
null-target logits, or any intermediate activation. With `L=1`, they
also cannot prove the multi-layer stream handoff. Chapter 9's blueprint
proves structural tying; Chapter 12's two-layer connectivity and
whole-model finite-difference checks cover different boundaries.

The witness is still useful. A fixed full output detects later ordering
and shape regressions from the recorded reference, while the short and
changed-target cases separate two likely failure regions. Its
limitations say where source review and later witnesses remain
necessary.

## The call boundary has sharp edges

`model_forward` is a low-level C API for valid model-owned work. Its
caller must provide:

```text
1 <= batch <= configured batch capacity
1 <= time  <= configured block capacity
batch*time readable token ids
batch*time readable target ids when targets is non-null
every token and target id in [0, V)
```

The source assertions check active batch and time. Embedding and
cross-entropy reach lower-level assertions for ids. These are
programmer-error checks, not command-line validation.

Compiling with `NDEBUG` removes them. Invalid dimensions can then form
out-of-capacity views or an oversized byte copy, and an invalid id can
index outside a table. The current public function returns a `float`,
not an error code, so it has no recoverable invalid-call path. Keep
assertions enabled while building the lab and uphold these preconditions
in any assertion-free caller.

Valid short calls remain supported. Short means positive dimensions
within capacity, not zero rows.

## Build checkpoint: connect the forward graph

**Build.** Implement `model_forward.c`. Begin with the file-private
`block_forward`: request Chapter 10's current-shape views, run both
pre-norm branches in order, preserve the correct highway state at each
residual, and return `after_mlp`.

Then implement `model_forward`: assert active dimensions, publish the
latest shape and target status, copy active ids, open the embedded view,
walk blocks upward, apply final normalization and the tied token-table
head, and grade only when targets exist. Do not allocate, clear
gradients, update parameters, select a character, or reimplement
`model_block_views`.

**Verify.**

```sh
make -C labs check-11
# answer key
make -C labs WORK=../src check-11
```

**Expected.**

```text
check-11: all 4 model-forward checks passed
```

**Common failures.**

- **Both fixed-output checks fail:** find the first wrong handoff in
  source order. Confirm the Chapter 5 operations, Chapter 9 parameters,
  and Chapter 10 views still pass before changing the wiring.
- **The full case passes but the short case fails:** attention scores
  probably retained capacity width, or another destination retained
  capacity rows.
- **The batched case asserts while the short case runs:** embedding may
  be using a flattened row or block capacity as its position instead of
  `row % time`.
- **The null-target call crashes or returns nonzero:** target copying or
  grading is unconditional, or the early return occurs in the wrong
  place.
- **Changing one target leaves loss unchanged:** cross-entropy received
  stale or caller-independent targets.
- **Outputs are plausible but the fixtures fail:** check that the second
  layernorm reads `after_attention` and the second residual preserves
  it.
- **A multi-layer mistake still passes:** this checkpoint has `L=1`.
  Verify that the loop assigns each returned `Mat` back to `stream`.

The assembled route now stores raw next-token score rows and, when
answers are supplied, returns a grade. It has not changed one
parameter. Chapter 12 will route that grade backward through the
latest target-bearing record.

---

[Previous: Memory Planning and Arenas](10-memory-planning.md) | [Contents](README.md) | [Next: Wiring the Model Backward](12-wiring-the-model-backward.md)
