# Chapter 12: Wiring the Model Backward

Every backward operation in this chapter has already passed its own
derivative check. Our new failure mode is wiring.

Chapter 11 connected the route that turns token ids into a grade.
Chapter 6 built the local rules that send that grade backward. A correct
local rule can still read the wrong saved value, run before its arriving
gradient exists, or replace a contribution that should meet another
one.

Place the new work inside one training step:

```text
clear gradients -> forward -> backward -> adjust parameters
                                ^
                                |
                            Chapter 12
```

Backward is [Chapter 11's exact forward
list](11-wiring-the-model-forward.md#the-complete-source-route) read
from bottom to top, with one extra responsibility: gradients from every
path must meet instead of replacing one another.

Use a two-entry parameter as a stand-in:

```text
                              values       gradient
before model_backward       [ 0.10 -0.20 ] [ 0.00  0.00 ]
```

**Predict:** after `model_backward`, which column may differ?

```text
                              values       gradient
after model_backward        [ 0.10 -0.20 ] [ 0.70  0.40 ]
```

Only the gradient column. Backward calculates how the grade responds to
each learned value. It does not apply those corrections. Chapter 15
will put the later adjustment inside the repeated training loop.

## One reverse pass has three prerequisites

The calculation that produced the grade left behind values needed by
its reverse. A backward call describes that calculation only while
three facts still hold:

```text
1. the newest forward call supplied targets
2. every gradient destination was zeroed for this cycle
3. parameter values still match the values used by that forward call
```

The first rule comes from Chapter 11's
[latest-forward
record](11-wiring-the-model-forward.md#only-the-newest-forward-record-survives):

```text
target-bearing A -> backward                    valid for A

target-bearing A -> target-free B -> backward   invalid: B has no grade

target-bearing A -> target-bearing B -> backward
                                      uses B, not A
```

The source checks the target flag:

```c
assert(m->has_targets);
```

That assertion checks only the first rule. It cannot tell whether
gradient storage is stale or a parameter changed after forward. It also
disappears in a build compiled with `NDEBUG`, as
[Chapter 11](11-wiring-the-model-forward.md#only-the-newest-forward-record-survives)
explains.

Why must the parameters wait? Consider one dot product:

```text
x       = [ 3  4 ]
w       = [ 2 -1 ]
y       = 3*2 + 4*(-1) = 2
arriving d_y = 2
```

[Chapter 6's matmul
backward](06-backprop-by-hand.md#matmul-sends-contributions-to-inputs-and-weights) uses the
forward weights:

```text
d_x = d_y * w = 2 * [ 2 -1 ] = [ 4 -2 ]
```

Suppose an updater changes `w` to `[1.5, -0.5]` before backward reaches
this operation. It would calculate:

```text
d_x = 2 * [ 1.5 -0.5 ] = [ 3 -1 ]
```

That gradient describes a dot product that did not produce `y = 2`.

**Predict:** can clipping or AdamW run between a matching forward and
backward?

No. The complete reverse pass must keep the saved activations and the
parameter values that produced them.

## Start from clean destinations

[Chapter 6's returning-path
construction](06-backprop-by-hand.md#add-every-returning-path) made
local backward operations add contributions. A shared destination may
receive:

```text
returning path A       [ 2  1 ]
returning path B       [ 3 -2 ]
correct total          [ 5 -1 ]
```

Addition solves the meeting-path problem only if the destination begins
at zero. Begin instead with an old result:

```text
old destination        [ 7 -4 ]
after path A            [ 9 -3 ]
after path B            [12 -5 ]   wrong for this cycle
```

**Predict:** does forward erase the old `[7, -4]`?

No. Forward writes the value arena. It does not touch parameter
gradients or the activation-gradient arena.

Chapter 9 built the exact clearing operation:

```c
void model_zero_gradients(Model *m)
{
    for (int i = 0; i < m->param_count; i++)
        param_zero_gradient(m->params[i]);
    memset(m->gradient_arena, 0,
           m->gradient_floats * sizeof *m->gradient_arena);
}
```

The registry loop clears every learned parameter's gradient. The
`memset` clears every temporary activation-gradient destination in one
write over the arena. See the
[line-by-line walk](09-parameters-and-the-blueprint.md#apply-one-operation-to-the-whole-model)
for the C details.

This call is required before the first reverse pass too. Parameter
gradient allocations happen to begin cleared, but Chapter 10's
activation-gradient arena comes from `emalloc`, whose bytes are
uninitialized.

The supported rhythm is:

```text
zero gradients -> one target-bearing forward -> one backward
```

Calling `model_backward` again without zeroing is not a way to obtain
exactly twice the gradient. The second call first enlarges `d_logits`.
Later operations then read enlarged intermediate gradients while adding
into destinations that already contain the first pass. That stale state
compounds through the graph.

Most externally meaningful input and parameter-gradient destinations
accumulate. There is one local distinction from Chapter 6:
attention's private `d_scores` workspace overwrites each active score
derivative before consuming it. It is scratch inside one attention
backward, not another public path-meeting destination. Therefore
"every backward operation uses `+=`" would be too broad.

## Start at the scalar grade

The model's loss is one number. Backpropagation begins with the slope
of that number with respect to itself:

```text
d_loss = 1
```

That seed is implicit in `crossentropy_backward`; there is no `d_loss`
argument. For one row, reuse Chapter 6's
[cross-entropy result](06-backprop-by-hand.md#send-the-bet-sheet-grade-back-to-logits):

```text
probabilities        [ 0.2  0.5  0.3 ]
target id              1
row count              1
```

Each entry is `(probability - target_indicator) / row_count`. With one
row the division changes nothing.

**Predict:** which logit receives the negative entry?

```text
id 0                  (0.2 - 0) / 1 =  0.2
id 1                  (0.5 - 1) / 1 = -0.5
id 2                  (0.3 - 0) / 1 =  0.3
d_logits                                 [ 0.2 -0.5  0.3 ]
```

The target logit. Raising that logit would lower the loss, so its loss
slope is negative.

From there, the complete route is:

```text
loss
  |
  v
d_logits
  |
  v
tied output matmul
  |
  v
final layernorm
  |
  v
block L-1
  |
  v
...
  |
  v
block 0
  |
  v
embedding
```

The arrows do not introduce new calculus. Each one calls a local
backward operation already derived and checked in Chapters 6 and 7.
Chapter 12 decides when to call it and where each result belongs.

## One block has two reverse meeting points

Chapter 11 followed two edits along the
[residual highway](11-wiring-the-model-forward.md#one-block-has-two-highway-states):

```text
x -> attention edit -> r1 -> MLP edit -> r2
```

The forward values in its small example were:

```text
x                 [10 20]
attention edit    [ 1 -2]
r1                [11 18]
MLP edit          [ 3  4]
r2                [14 22]
```

Backward starts with an arriving gradient for `r2`. The second
residual addition gave `r2` two parents, so the gradient follows a
direct highway path and the MLP branch:

```text
                              arriving d_r2
                              /           \
                    direct path           MLP branch
                              \           /
                               combined d_r1
                              /           \
                    direct path           attention branch
                              \           /
                                combined d_x
```

Use these two-entry contributions:

```text
arriving d_r2                 [ 2.0 -1.0 ]

direct residual path to r1    [ 2.0 -1.0 ]
MLP path returning to r1      [ 0.5  3.0 ]
```

**Predict:** what must `d_r1` contain before the attention residual
unwinds?

```text
total d_r1                    [ 2.5  2.0 ]
```

Now split that total through the first residual. Its direct path puts
`[2.5, 2.0]` into `d_x`. Suppose the attention branch returns:

```text
attention path returning x    [-1.0  0.25]
```

**Predict:** what is the final `d_x`?

```text
total d_x                     [ 1.5  2.25 ]
```

If the MLP branch assigned `[0.5, 3.0]` over the direct path, the first
meeting would lose `[2.0, -1.0]`. If the attention branch assigned over
`d_x`, the second meeting would lose `[2.5, 2.0]`.

Those two totals are the mental key to `block_backward`:

```text
g.after_attention   holds the combined d_r1
d_stream            holds the combined d_x
```

## Match saved values with gradient destinations

Most local derivatives need values from forward. Matmul backward reads
its input and weights. GELU backward reads its input. Layernorm backward
reads its input, gain, saved mean, and saved reciprocal standard
deviation. Attention backward reads packed QKV values and saved
attention weights. [Residual
backward](06-backprop-by-hand.md#residual-the-highway-works-in-reverse)
is the exception: it adds the arriving gradient into both parents
without reading a numeric forward value.

The block therefore opens two current-shape descriptions:

```c
BlockTensors a =
    model_block_views(&b->acts, batch * time, time, m->cfg.head_count);
BlockTensors g =
    model_block_views(&b->grads, batch * time, time, m->cfg.head_count);
```

These are exact lines from
[`block_backward`](../src/model_backward.c). `a` names saved
**activations** from the latest forward call. `g` names their matching
**gradient** destinations, plus the private `scores` scratch. The
helper shortens capacity-sized descriptions to the active `batch` and
`time`; it allocates and copies no floats.

For the Chapter 12 fixture:

```text
B=2, active T=6, C=16, H=2, V=13
R=B*T=12, widened channels=4*C=64
```

The active block views are:

| `BlockTensors` field | Active shape |
|---|---:|
| residual-width fields | `12 x 16` |
| packed QKV | `12 x 48` |
| attention weights and score scratch | `24 x 6` |
| MLP up and activated fields | `12 x 64` |

The latest-forward record also supplies active fields outside those two
`BlockTensors` descriptions:

| Other saved field or gradient | Active shape |
|---|---:|
| final normalized fields | `12 x 16` |
| logits, probabilities, and `d_logits` | `12 x 13` |
| each layernorm mean or reciprocal standard deviation | `12` |
| copied token or target ids | `12` |

The attention row count is `B*H*T = 2*2*6 = 24`. Chapter 10 owns the
[active-view construction](10-memory-planning.md#capacity-is-not-the-current-shape);
this chapter applies the same shapes in reverse.

Probabilities and layernorm summaries are saved coefficients. The
gradient arena has no `d_probs`, `d_mean`, or `d_rstd` field.
Cross-entropy backward reads the probabilities and produces
`d_logits`. Layernorm backward reads the summaries and produces the
layernorm input gradient.

Three block arguments complete the pairing:

```text
stream      saved x entering this block
d_stream    shared destination for d_x
g.after_mlp arriving d_r2 from the route above
```

The caller supplies `stream` and `d_stream`. The block's own active
views supply everything inside it.

## The last block unwinds first

One block's input is the previous block's output. With two layers:

```text
forward:

embedded -> block 0 -> block 1 -> final norm -> loss

backward:

d_embedded <- block 0 <- block 1 <- final norm <- loss
```

Use a smaller arithmetic stand-in:

```text
block 0: b0 = [2*x0,  2*x1]
block 1: b1 = [3*b0[0], -b0[1]]
arriving d_b1 = [1, 1]
```

Block 1 first returns:

```text
d_b0 = [3, -1]
```

**Predict:** what can block 0 return after receiving that gradient?

```text
d_x = [6, -2]
```

Running block 0 first would give it no gradient from block 1.

Chapter 10's selectors make the shared boundaries explicit:

| Selector argument | Selected stream |
|---:|---|
| `0` | embedded values or `d_embedded` |
| `1` | block 0 `after_mlp` values or gradient |
| `2` | block 1 `after_mlp` values or gradient |
| `L` | the final block's output values or gradient |

For `L=2`, `last` is one. Therefore `last + 1` is two, and selecting
stream two reaches block 1's output.

The reverse loop must count downward with a signed integer:

```c
for (int layer = last; layer >= 0; layer--)
```

After layer zero, `layer` becomes `-1` and the condition stops the loop.
An unsigned counter would follow [Chapter 2's wrap
rule](02-foundations.md#build-a-generator-from-state), moving from zero
to a large positive value instead.

## One tied object receives two returns

The [tied
head](09-parameters-and-the-blueprint.md#one-table-two-jobs) uses one
token-table `Param` twice. Forward first reads token rows to open the
stream, then uses the same table as the final scoring weights.

Backward reaches those uses at opposite ends of the route:

```text
token-table gradient
|
+-- output-head contribution: loss -> output matmul
|
+-- embedding contribution:   loss -> blocks -> embedding
```

Use one two-channel row:

```text
output-head contribution      [ 0.4 -0.1 ]
embedding contribution        [ 0.3  0.5 ]
```

**Predict:** what must the shared gradient contain after both paths?

```text
combined token-table gradient [ 0.7  0.4 ]
```

Both calls receive a `Mat` view from
`param_gradient(m->token_table)`. The two copied `Mat` descriptions
point into the same gradient region owned by one `Param`; they are not
two arrays that later happen to be combined. Clearing that region
between calls would cut one use of the tied weight out of learning.
Assigning the embedding result would leave only `[0.3, 0.5]` in the
example and erase the scoring use.

The need, the two residual meetings, the saved views, the descending
stack, and the tied return are now visible. This assembly is the
**whole-model backward wiring**: one source-ordered reverse traversal of
the newest target-bearing forward record, starting from cleared
destinations, walking blocks from `L-1` to zero, and adding wherever
residual or tied paths meet.

Its boundary is narrow. One cleared state supports one matching reverse
pass. The operation writes gradients while parameter values and saved
forward values remain unchanged. It stops before clipping or updating.

## Walk one block in source order

Here is the exact function header and the two view constructions:

```c
static void block_backward(const Model *m, Block *b, Mat stream, Mat d_stream,
                           int batch, int time)
{
    BlockTensors a =
        model_block_views(&b->acts, batch * time, time, m->cfg.head_count);
    BlockTensors g =
        model_block_views(&b->grads, batch * time, time, m->cfg.head_count);
```

The function is file-private. `m` supplies configuration, `b` supplies
one block's parameters and storage, and the two `Mat` arguments name
the saved input and its shared gradient destination.

Forward ended this block at the MLP residual. Backward begins there:

```c
    residual_backward(g.after_attention, g.down, g.after_mlp);
    matmul_backward(g.activated, param_gradient(b->down_weights), g.down,
                    a.activated, param_values(b->down_weights));
    gelu_backward(g.up, g.activated, a.up);
    matmul_backward(g.normed2, param_gradient(b->up_weights), g.up,
                    a.normed2, param_values(b->up_weights));
    layernorm_backward(g.after_attention,
                       param_gradient(b->norm2_gain).vals,
                       param_gradient(b->norm2_bias).vals,
                       g.normed2, a.after_attention,
                       param_values(b->norm2_gain).vals,
                       b->means2, b->rstds2);
```

The first line adds the arriving `g.after_mlp` into both parents:
`g.after_attention` is the direct path to `r1`, while `g.down` enters
the MLP branch.

The first matmul call sends `g.down` into `g.activated`, the gradient of
the activated values, and into the down-weight gradient. Its saved
coefficients are `a.activated` and
`param_values(b->down_weights)`.

GELU backward sends that result into `g.up` while reading the saved
pre-GELU values in `a.up`.

The next matmul sends `g.up` into `g.normed2` and the up-weight
gradient. It reads `a.normed2` and the unchanged up weights.

The layernorm call finishes the branch. It adds into the gain and bias
gradients, and it adds the returned branch contribution into
`g.after_attention`. That destination already holds the direct
`[2.0, -1.0]` contribution in the numeric example. This is the first
load-bearing meeting point.

Now the combined gradient reaches the attention residual:

```c
    residual_backward(d_stream, g.projected, g.after_attention);
    matmul_backward(g.attended, param_gradient(b->proj_weights), g.projected,
                    a.attended, param_values(b->proj_weights));
    attention_backward(g.qkv, g.scores, g.attended, a.qkv, a.scores,
                       time, m->cfg.head_count);
    matmul_backward(g.normed1, param_gradient(b->qkv_weights), g.qkv,
                    a.normed1, param_values(b->qkv_weights));
    layernorm_backward(d_stream,
                       param_gradient(b->norm1_gain).vals,
                       param_gradient(b->norm1_bias).vals,
                       g.normed1, stream,
                       param_values(b->norm1_gain).vals,
                       b->means1, b->rstds1);
}
```

Residual backward adds `g.after_attention` directly into `d_stream` and
into the projected-branch gradient.

Projection matmul backward sends the branch through its learned weights.
Attention backward uses the saved QKV values and attention weights; its
`g.scores` field is the private derivative workspace mentioned earlier.
QKV matmul backward then reaches `g.normed1`, the gradient of the first
normalized values, and the packed QKV-weight gradient.

The final layernorm call adds the attention branch into `d_stream`,
which already holds the direct highway contribution. It also adds into
the first gain and bias gradients. This is the second load-bearing
meeting point.

Every learned matrix has the same pairing:

```text
param_gradient(...)   destination that receives a correction
param_values(...)     unchanged coefficients read by the derivative
```

No parameter value is a destination. The function allocates nothing,
clears nothing, and calls no updater.

**Predict:** if the two residual calls traded places, would either
local derivative become mathematically wrong?

No. The local functions would remain correct, but the attention branch
would start before the MLP branch had finished building its arriving
gradient. Whole-model wiring is a separate correctness problem.

## Walk the whole model in source order

The exact top of `model_backward` establishes its saved shape and final
views:

```c
void model_backward(Model *m)
{
    assert(m->has_targets);

    int rows = m->batch * m->time;
    int last = m->cfg.layer_count - 1;

    Mat final_normed   = mat_first_rows(m->final_normed, rows);
    Mat d_final_normed = mat_first_rows(m->d_final_normed, rows);
    Mat d_logits       = mat_first_rows(m->d_logits, rows);
    Mat last_stream    = model_stream_into(m, last + 1, rows);
    Mat d_last_stream  = model_d_stream_into(m, last + 1, rows);
```

The function accepts no batch, time, token, or target arguments.
`m->batch`, `m->time`, copied ids, saved values, and `has_targets` all
belong to the latest-forward record. This prevents the caller from
supplying a shape or answer list that disagrees with the saved
calculation.

`rows` reconstructs `R=B*T`. `last` names block `L-1`. The three
leading-row calls expose active final fields, while the two selectors
open the final residual stream and its matching gradient destination.

The scalar loss now enters the learned route:

```c
    crossentropy_backward(d_logits, mat_first_rows(m->probs, rows),
                          m->targets);
    matmul_backward(d_final_normed, param_gradient(m->token_table), d_logits,
                    final_normed, param_values(m->token_table));
    layernorm_backward(d_last_stream,
                       param_gradient(m->final_gain).vals,
                       param_gradient(m->final_bias).vals,
                       d_final_normed, last_stream,
                       param_values(m->final_gain).vals,
                       m->final_means, m->final_rstds);
```

Cross-entropy adds the implicit unit-loss result into `d_logits` using
the saved probabilities and copied targets.

Matmul backward follows the tied scoring use. It adds into
`d_final_normed` and makes the first contribution to the token-table
gradient. The saved final-normalized values and token-table parameter
values supply its coefficients.

Final layernorm sends the result into the last residual stream and the
final gain and bias gradients. Its saved input, gain, means, and
reciprocal standard deviations all belong to the same forward call.

Only now does the last block have an arriving gradient:

```c
    for (int layer = last; layer >= 0; layer--)
        block_backward(m, &m->blocks[layer],
                       model_stream_into(m, layer, rows),
                       model_d_stream_into(m, layer, rows),
                       m->batch, m->time);
```

Each iteration selects block `layer`'s saved input and the shared
gradient destination for that input. Unwinding it fills the arriving
gradient needed by the next lower layer. The loop body has no braces
because its single statement is the `block_backward` call.

After block zero, the route has reached `d_embedded`:

```c
    embedding_backward(param_gradient(m->token_table),
                       param_gradient(m->position_table),
                       mat_first_rows(m->d_embedded, rows),
                       m->tokens, m->time);
}
```

Embedding backward reads the copied token ids and saved active time. It
adds into the position-table gradient and makes the second contribution
to the same token-table gradient used by the output-head call.

The function returns `void`. Its result lives in parameter-gradient and
activation-gradient storage. It does not return a newly allocated
gradient object.

## Stop when the gradients are complete

Backward ends before clipping and adjustment:

```text
zero -> forward -> backward -> complete gradients
                                  |
                                  +-- Chapter 12 stops

later caller -> global clip -> AdamW update
```

[Chapter 9's model-wide
step](09-parameters-and-the-blueprint.md#apply-one-operation-to-the-whole-model)
will read the complete parameter registry. Chapter 15 will place the
verified call order inside the training loop.

Do not update block 0 while block 1 is still computing gradients.
Backward equations assume that every parameter value saved or read from
forward remains unchanged until the whole reverse pass is finished.

The complete supported order is:

```text
zero
forward with targets
backward once
clip
update
```

There is no allocation, sampling, text generation, cache refresh,
gradient clear, clipping, or parameter update inside
`model_backward.c`.

## Two witnesses make different claims

The operation-level derivatives already survived Chapter 6.
[Chapter 7's central-difference
referee](07-trust-but-verify.md#measure-a-slope-from-nearby-values)
can now examine the assembled model. A second check is needed because a
derivative test can agree with a route that forward and backward both
omit.

### Find a completely unused parameter object

Consider:

```text
loss = a*a
parameters [a, b] = [2, 3]
```

**Predict:** which coordinate has slope zero at `[2, 3]`?

```text
gradient          = [4, 0]
```

Finite differences agree with both entries. Nudging `a` changes the
loss with slope four. Nudging unused `b` changes nothing, so analytic
and numeric slopes are both zero.

Suppose the intended model says both parameters should participate.
Another check can require the parameter object's gradient to contain at
least one finite nonzero entry.

The Chapter 12 fixture calls this a **parameter-connectivity witness**.
Its promise is precise: for this fixed input, each registered `Param`
must contain at least one finite nonzero gradient.

That is weaker than checking every coordinate. If `[a, b]` were stored
inside one `Param`, its gradient `[4, 0]` would pass because `a` is
nonzero. In Tiny AgenC, queries, keys, and values share one packed QKV
weight `Param`; a connected value region could hide a disconnected
query region from this witness.

### Derive the connectivity count

The fixture in [`labs/check12.c`](../labs/check12.c) uses:

```text
V=13, block capacity=8, C=16, H=2, L=2, B=2
active T=6, seed=4242
```

Its first four generated ids can be checked from the source formulas:

```text
i             0   1   2   3
input         1   4   7  10     (3*i + 1) modulo 13
target        2   7  12   4     (5*i + 2) modulo 13
```

There are four parameter objects outside the blocks and eight per
block:

```text
Param objects = 4 + 8*L
              = 4 + 8*2
              = 20
```

Count their scalar values:

```text
token and position tables    (V + block_capacity)*C
                           = (13 + 8)*16
                           = 336

each block                  = 12*C*C + 4*C
                           = 12*16*16 + 4*16
                           = 3136

two blocks                 = 2*3136
                           = 6272

final gain and bias        = 2*C
                           = 32

total parameter scalars    = 336 + 6272 + 32
                           = 6640
```

The `12*C*C` term collects QKV's `3*C*C`, projection's `C*C`,
and the up and down matrices' `4*C*C` each. The `4*C` term collects
two gains and two biases.

The executable performs:

```text
1 finite positive loss check
+ 6640 finite parameter-gradient checks
+ 20 one-nonzero-entry-per-Param checks
= 6661 checks
```

It does not inspect activation-gradient finiteness, require every
parameter coordinate to be nonzero, or compare an exact gradient value.

### Nudge every learned scalar

The whole-model case in
[`tests/gradcheck.c`](../tests/gradcheck.c) uses fixed id seed 42 and
model seed 7. It:

```text
zeroes all gradients
runs one target-bearing forward
runs one analytic backward
for each of 6640 learned scalars:
    measure loss with the value raised by 0.01
    measure loss with the value lowered by 0.01
    restore the value
    compare the measured slope with its stored gradient
```

The model already returns a scalar mean loss, so this test uses that
loss directly. Chapter 7's random projection was needed to turn a
matrix operation's many outputs into one grade; no projection is
needed here.

Agreement is judged with Chapter 7's
[mixed tolerance](07-trust-but-verify.md#decide-when-two-float-answers-agree):

```text
allowed difference
    = 0.001 + 0.02 * max(abs(analytic), abs(numeric))
```

The 6,640 comparisons prove agreement between the implemented forward
and backward model at this fixed fixture, nudge, and tolerance. They
exercise two layers, the tied token-table contributions, and active
time six inside capacity eight.

Neither witness replaces the other. Together they still do not prove
an independently intended architecture. Forward and backward can omit
the same subregion and agree on a zero slope. The checks also do not
exercise stale gradients, repeated backward calls, invalid call order,
assertion-free misuse, parameter updates, or every possible shape and
seed. The source walk establishes the intended route; the two
executables guard different failures in its implementation.

## The complete reverse route

Read the file once more at model scale:

```text
model_backward
|
+-- require a newest target-bearing record
+-- recover active rows and the final residual stream
+-- loss -> logits -> tied head -> final layernorm
+-- block L-1 -> block L-2 -> ... -> block 0
+-- embedding -> position table and the same tied token table
|
+-- leave complete gradients; do not adjust parameters
```

The route above is the exact reverse assembly previewed in
Chapter 1 and prepared by Chapters 6 through 11.

## Build checkpoint: unwind the graph

**Build.** Implement `model_backward.c`. Begin with file-private
`block_backward`: request current-shape activation and gradient views,
split the MLP residual, unwind its branch, split the attention
residual, and unwind its branch. Preserve the two shared-gradient
meeting points.

Then implement `model_backward`: require the newest target-bearing
record, recover its shape, begin at cross-entropy, unwind the tied head
and final layernorm, walk blocks from last to first, and finish at
embedding. Use the same token-table gradient for both tied paths. Do
not clear gradients, allocate storage, or call `model_step` inside
backward.

**Verify.**

```sh
make -C labs check-12
# answer key
make -C labs WORK=../src check-12
```

**Expected.**

```text
check-12: all 6661 model-backward checks passed
gradcheck[model]: all 6640 checks passed
```

**Common failures.**

- **The entry assertion fails:** the newest forward call did not supply
  targets. An intervening target-free call replaced the eligible
  record.
- **The first result is nonfinite or changes between fresh cycles:**
  both the parameter gradients and activation-gradient arena must be
  cleared before forward and backward.
- **Connectivity passes but finite differences fail:** inspect reverse
  order, both residual meetings, saved activation arguments, and the
  tied-table sum.
- **Finite differences pass but connectivity fails:** forward and
  backward may consistently omit a complete parameter object.
- **Token-table gradients are too small:** one tied contribution was
  overwritten or the shared gradient was cleared between its uses.
- **Early-layer parameters fail:** confirm that the layer loop counts
  downward and that `model_d_stream_into` selects the lower block's
  output-gradient destination.
- **A hang or invalid block access follows layer zero:** the descending
  layer counter may be unsigned and wrapping.
- **Calling backward twice gives surprising values:** an unzeroed
  replay compounds stale intermediate gradients and is outside the
  supported contract.
- **Parameter values change before `model_step`:** a backward call
  wrote through `param_values` instead of `param_gradient`.

The model's forward and backward routes are now connected. Chapter 13
decides which state survives after the process exits.

---

[Previous: Wiring the Model Forward](11-wiring-the-model-forward.md) | [Contents](README.md) | [Next: Durable Checkpoints](13-durable-checkpoints.md)
