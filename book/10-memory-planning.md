# Chapter 10: Memory Planning and Arenas

Parameters survive every batch. Activations do not.

[Chapter 1](01-the-map.md#the-activation-ledger) called every
temporary value computed along the score route an activation. Its
contents belong to one forward/backward cycle, not learned across
cycles.

That sounds like permission to allocate activations inside
`model_forward`, free them, and repeat. It is also a reliable way to
bury the shape of the model under allocator traffic. Every operation
would need allocation and ownership rules beside its arithmetic. The
model would also need to remember which pieces to release.

Chapter 0 printed a more disciplined promise before the first training
step:

```text
815360 parameters | 376.5 MiB buffers
```

Chapter 9 accounted for the parameters and gave each
[`Param`](08-adamw.md#four-aligned-buffers-make-one-learnable-object)
its own four-part allocation. The banner's buffer total combines those
persistent `Param` regions with two temporary float blocks and two id
caches. This chapter accounts for the entire total. It plans the
maximum temporary footprint once and reuses it for every training
step.

First build that arrangement at a size that fits on one line.

## Two views, seven floats, one owner

Suppose a calculation needs a `2 x 2` grid followed by a `1 x 3` grid.
The shapes demand four floats and three floats:

```text
first grid     2*2 = 4 slots
second grid    1*3 = 3 slots
total                7 slots
```

Two separate allocations would work. They would also create two owners
and two releases. Nothing about the calculations requires that split.
One seven-float allocation can hold both:

```text
slot       0  1  2  3 | 4  5  6
use       first grid  | second grid
shape        2 x 2    |    1 x 3
```

The first description starts at slot 0. The second starts at slot 4:

```c
/* A seven-float teaching sketch, not a source excerpt. */
float *storage = emalloc(7 * sizeof *storage);
Mat first  = mat_make(storage,     2, 2);
Mat second = mat_make(storage + 4, 1, 3);
```

**Predict:** under this ownership plan, which variable is responsible
for the eventual `free`: `storage`, `first`, or `second`?

Only `storage` owns an allocation. Both `Mat` values borrow parts of
it, so the owner releases `storage` once. `first.vals` happens to hold
the same address, but copying an owner's address into a view does not
transfer ownership. Freeing `second.vals` would try to release an
address in the middle of an allocation. Freeing both `first.vals` and
`storage` would release the same allocation twice.

A large allocation reserved for many smaller, related uses is an
**arena**. The smaller uses are views into it. Tiny AgenC plans the
maximum training footprint once, allocates two float arenas, and reuses
them for every step. The same construction now spans model-sized
storage.

One owner solves the release problem. It does not yet explain how to
keep dozens of region offsets and the allocation total in agreement.

## Three lifetimes share one allocation

An arena does not make every lifetime equal. It puts three different
lifetimes beside the same memory.

The allocation lifetime is longest. `model_new` obtains an arena, and
`model_free` releases its base pointer. The allocation remains alive
across every batch in between.

The content lifetime is shorter. A forward call overwrites the value
slots that its operations will later read. Unread causal score cells
and unused capacity may hold uninitialized or older bits without
current meaning. A new
[gradient-accumulation cycle](06-backprop-by-hand.md#add-every-returning-path)
clears the gradient arena before backward adds into it. The bytes
persist while their meanings change.

The description lifetime is independent. Capacity-sized `Mat`
descriptions live in `Model` and its blocks. A call with fewer rows
makes smaller descriptions by value. Those local descriptions
disappear when the call returns, but the arena and the stored
capacity descriptions remain.

```text
allocation:  model_new [--------------------------------] model_free
contents:               [batch 1][batch 2][batch 3] ...
call views:              [view 1] [view 2] [view 3] ...
```

[Chapter 4's ownership
rule](04-poor-mans-tensors.md#what-a-mat-never-owns-or-remembers)
still controls the picture. A view neither creates storage nor extends
its lifetime.

## Values and gradients are different lifetimes

Could one slot hold a forward value and its gradient? Reuse [Chapter
6's square
operation](06-backprop-by-hand.md#build-a-local-rate-from-a-nudge),
`y = x*x`. Forward saves `x=2`. Backward will need that saved 2 to
compute `d_x = 2*x*d_y`.

The training order clears gradients before forward. **Predict:** after
forward saves `x=2` into the aliased slot, what starting value would
`d_x` have?

It would start at 2, not zero. An accumulating update would produce
`d_x = 2 + 2*2*1 = 6` instead of 4. Clearing the slot again after
forward would not repair the plan: that would erase saved `x`, so
backward would read zero and produce zero. No clearing order preserves
both meanings in one slot.

Values and gradients therefore need disjoint regions while backward
runs. They could be two regions inside one larger allocation. Tiny
AgenC uses two separately owned allocations, so clearing the complete
gradient allocation cannot touch any saved forward value.

The value arena is the model's complete capacity-sized forward
workspace and saved-forward storage. [Chapter
11](11-wiring-the-model-forward.md#the-caller-cannot-own-the-saved-ids)
constructs the rule that says which call those saved contents belong
to. The arena contains the embedded residual stream, every output
grouped by
[Chapter 9's
`BlockTensors`](09-parameters-and-the-blueprint.md#why-the-private-blueprint-must-be-exact),
four saved layernorm statistics per block, the final normalized values
and statistics, logits, and probabilities.

Some of those values are destinations that backward does not reread.
For example, the projected and down outputs have already been combined
into residual results. Keeping one regular `BlockTensors` layout is
less error-prone than giving forward destinations and backward caches
unrelated storage plans.

The gradient arena contains the derivative counterpart of each
`BlockTensors` field, plus gradients for the embedded stream, final
normalized values, and logits. It has no gradient arrays for
layernorm means or reciprocal standard deviations. Layernorm backward
reads them as saved coefficients; no other operation accumulates a
gradient into them. It also has no probability-gradient matrix:
cross-entropy reads the saved probabilities and writes `d_logits`
directly.

Parameter storage is separate. Each `Param` owns one allocation split
into its values, gradient, first AdamW moment, and second AdamW moment.
There is no global parameter arena.

Token and target ids occupy two integer allocations. The complete
ownership picture is:

```text
Model owns
|
+-- params list ------> Param 0 owns [value | grad | m1 | m2]
|                      Param 1 owns [value | grad | m1 | m2]
|                      ...
+-- blocks array
+-- values_arena ------ Mat and float-pointer fields borrow regions
+-- gradient_arena ---- gradient Mat fields borrow regions
+-- tokens
+-- targets
```

The compact `m1` and `m2` labels mean first and second AdamW moment.

`emalloc` does not clear the two arenas or the id arrays. That is safe
only because a forward route writes every slot it later reads and
copies the active token ids first. The upper triangle of each causal
score grid is neither written nor read. Other unused capacity may also
remain uninitialized.

Backward is different. Its operations accumulate with `+=`, because
several graph paths can meet at one gradient. Reading an old
uninitialized float there would corrupt the first correction.
Chapter 9's `model_zero_gradients` clears every parameter gradient and
the whole activation-gradient arena before a fresh backward cycle.
Construction alone does not establish that zero.

## Measure and place with one cursor

A tempting design has one function add shapes into `needed` and
another assign pointers into the allocation:

```c
/* A tempting design, not Tiny AgenC source. */
needed += (size_t)rows * channels;
needed += (size_t)score_rows * time;

bt->normed1 = take_region(arena, rows, channels);
bt->scores  = take_region(arena, score_rows, time);
```

Add a new activation to the second list and forget the first. The
allocator returns too few floats, yet every `Mat` still carries a
plausible shape. The eventual out-of-bounds write may occur far from
the missing count.

The repair is compact enough to state in one sentence:

> Run the same layout function once with no storage to measure, then
> again with storage to place.

Tiny AgenC gives both passes one cursor and one placement sequence.
The cursor counts floats, not bytes. During the measuring pass there
is no allocation. During the placement pass the same cursor becomes
an offset into the allocation.

### The two placement helpers

The exact helpers in `model_memory.c` are:

```c
static Mat place(float *base, size_t *offset, int rows, int cols)
{
    Mat m = mat_make(base == NULL ? NULL : base + *offset, rows, cols);

    *offset += mat_size(m);
    return m;
}

static float *place_floats(float *base, size_t *offset, size_t count)
{
    float *start = base == NULL ? NULL : base + *offset;

    *offset += count;
    return start;
}
```

Read `place` in source order. The conditional expression first tests
`base`. With a real base, `base + *offset` finds the next float. With
`NULL`, the result pointer is `NULL`.

That conditional is a C sharp edge, not decoration. Pointer arithmetic
is defined only within an actual array, plus its one-past address.
`NULL + 0` is not a harmless way to say "no storage." The unselected
side of `?:` is not evaluated, so the measuring pass performs no null
pointer arithmetic.

`mat_make` records the pointer and shape. `mat_size` needs only the
shape, so a `NULL`-backed description can still advance the cursor.
The dry-run description must never reach `mat_row` or any operation
that reads its storage.

`place_floats` applies the same rule to unshaped arrays. Layernorm keeps
one mean and one reciprocal standard deviation per row, so those
regions need a pointer and a count rather than a two-dimensional
description.

For the seven-slot example, both helpers begin with offset 0. Placing
`2 x 2` returns the current start and advances to 4. Placing `1 x 3`
returns the next start and advances to 7. A dry run and a real run
therefore finish at the same count because they execute the same two
calls.

Running one fixed placement sequence first to count and again to
assign addresses is a **two-pass arena layout**. The name labels the
count-then-place construction above; it does not add another routine.

### One value layout, run twice

[Chapter 1's
`ModelConfig`](01-the-map.md#when-copying-the-description-is-useful)
is the six-integer capacity description. The whole value sequence
reads those capacities in this exact source:

```c
static size_t lay_out_values(Model *m, float *base)
{
    ModelConfig cfg    = m->cfg;
    int         rows   = cfg.batch_size * cfg.block_size;
    size_t      offset = 0;

    m->embedded = place(base, &offset, rows, cfg.d_model);
    for (int layer = 0; layer < cfg.layer_count; layer++) {
        Block *b = &m->blocks[layer];

        place_block_tensors(&b->acts, base, &offset, cfg);
        b->means1 = place_floats(base, &offset, (size_t)rows);
        b->rstds1 = place_floats(base, &offset, (size_t)rows);
        b->means2 = place_floats(base, &offset, (size_t)rows);
        b->rstds2 = place_floats(base, &offset, (size_t)rows);
    }
    m->final_normed = place(base, &offset, rows, cfg.d_model);
    m->final_means  = place_floats(base, &offset, (size_t)rows);
    m->final_rstds  = place_floats(base, &offset, (size_t)rows);
    m->logits       = place(base, &offset, rows, cfg.vocab_size);
    m->probs        = place(base, &offset, rows, cfg.vocab_size);
    return offset;
}
```

The function resets its local cursor to zero. It places the embedded
stream, then every block's regular tensors and four statistic arrays.
The final normalized values, two final statistic arrays, logits, and
probabilities finish the sequence.

During `lay_out_values(m, NULL)`, these assignments briefly put
`NULL` into the model's view fields. The caller uses only the returned
count, allocates that many floats, and immediately calls the same
function with the real base. The second pass replaces every dry-run
description.

The gradient sequence reuses the block shape order:

```c
static size_t lay_out_gradients(Model *m, float *base)
{
    ModelConfig cfg    = m->cfg;
    int         rows   = cfg.batch_size * cfg.block_size;
    size_t      offset = 0;

    m->d_embedded = place(base, &offset, rows, cfg.d_model);
    for (int layer = 0; layer < cfg.layer_count; layer++)
        place_block_tensors(&m->blocks[layer].grads, base, &offset, cfg);
    m->d_final_normed = place(base, &offset, rows, cfg.d_model);
    m->d_logits       = place(base, &offset, rows, cfg.vocab_size);
    return offset;
}
```

There are no statistic arrays and no `d_probs`, matching the lifetime
argument above.

The shared layout functions prevent their own measured cursor from
drifting away from their placements. They do not produce a
caller-visible byte total. A separate checked calculation will do
that after the arena counts are known.

## Tiny model, every slot

Use a compact geometry whose tables and score square are still
nontrivial:

```text
B=1, T=2, C=2, H=1, L=1, V=3
```

One batch of two positions makes `1*2 = 2` rows. A regular
channel-sized grid is therefore `2 x 2 = 4` floats. The exceptional
shapes are still small enough to count:

```text
packed qkv       2 rows * (3*2) columns = 12 floats
scores           (2*1) rows * 2 columns =  4 floats
up, activated    2 rows * (4*2) columns = 16 floats each
one statistic    2 rows                  =  2 floats
logits, probs    2 rows * 3 columns      =  6 floats each
```

Start the cursor at zero and place those shapes in source order. In
`0..4`, slot 0 is included and 4 is the next free slot, so the region
contains four floats.

```text
value region             slots       count
embedded                  0..4          4
block.normed1             4..8          4
block.qkv                 8..20        12
block.scores             20..24         4
block.attended           24..28         4
block.projected          28..32         4
block.after_attention    32..36         4
block.normed2            36..40         4
block.up                 40..56        16
block.activated          56..72        16
block.down               72..76         4
block.after_mlp          76..80         4
block.means1             80..82         2
block.rstds1             82..84         2
block.means2             84..86         2
block.rstds2             86..88         2
final_normed             88..92         4
final_means              92..94         2
final_rstds              94..96         2
logits                   96..102        6
probs                   102..108        6
```

**Predict:** where does the value cursor finish?

The final next-free offset is 108, so the value arena needs 108
floats.

The gradient cursor follows the same block-tensor order but omits
statistics and probabilities:

```text
gradient region          slots       count
d_embedded                0..4          4
block.normed1             4..8          4
block.qkv                 8..20        12
block.scores             20..24         4
block.attended           24..28         4
block.projected          28..32         4
block.after_attention    32..36         4
block.normed2            36..40         4
block.up                 40..56        16
block.activated          56..72        16
block.down               72..76         4
block.after_mlp          76..80         4
d_final_normed           80..84         4
d_logits                 84..90         6
```

Its final next-free offset is 90, so the gradient arena needs 90
floats.

**Predict:** where would a second block begin in the value arena?

It would begin at slot 88, immediately after the first block's four
statistic arrays. The final fields would move later. The loop in
`lay_out_values` creates exactly that order.

## Count one block before the whole model

The tiny walk now gives the repeated sizes something concrete to
name. [Chapter 1's configuration
predicate](01-the-map.md#build-checkpoint-specify-the-machine) requires
positive bounded dimensions, `B*T` within its row ceiling, and `C`
divisible by `H`. For any configuration that passes those geometry
checks:

```text
R = B*T       active or capacity rows
N = R*C       floats in one channel-sized grid
S = R*H*T     floats in one attention-score grid
Q = R*V       floats in one logit-sized grid
```

The tiny values were `R=2`, `N=4`, `S=4`, and `Q=6`. The exact block
placement generalizes the same sequence:

```c
static void place_block_tensors(BlockTensors *bt, float *base, size_t *offset,
                                ModelConfig cfg)
{
    int rows = cfg.batch_size * cfg.block_size;
    int wide = MODEL_MLP_WIDENING * cfg.d_model;

    bt->normed1         = place(base, offset, rows, cfg.d_model);
    bt->qkv             = place(base, offset, rows,
                                QKV_STREAMS * cfg.d_model);
    bt->scores          = place(base, offset, rows * cfg.head_count,
                                cfg.block_size);
    bt->attended        = place(base, offset, rows, cfg.d_model);
    bt->projected       = place(base, offset, rows, cfg.d_model);
    bt->after_attention = place(base, offset, rows, cfg.d_model);
    bt->normed2         = place(base, offset, rows, cfg.d_model);
    bt->up              = place(base, offset, rows, wide);
    bt->activated       = place(base, offset, rows, wide);
    bt->down            = place(base, offset, rows, cfg.d_model);
    bt->after_mlp       = place(base, offset, rows, cfg.d_model);
}
```

The products stored in `int` here are bounded by the configuration
predicate. Capacity rows satisfy:

```text
rows       = B*T       <= 1,048,576
wide       = 4*C       <= 65,536
qkv width  = 3*C       <= 49,152
score rows = rows*H    <= 268,435,456
```

[Chapter 3](03-data.md#store-one-owned-sequence-of-ids) named
`INT_MAX`, the largest signed `int`. On the supported four-byte-`int`
builds, it is 2,147,483,647, so every bound above fits. These bounds
protect individual matrix dimensions. They do not prove that the sum
of every matrix or its byte count fits; the later `size_t` report
handles those larger totals.

`normed1` contributes `N`. Packed query, key, and value channels
contribute `3N`. The scores contribute `S`. Four channel-sized grids
from `attended` through `normed2` contribute `4N`. The widened `up`
and `activated` grids contribute `4N` each. `down` and `after_mlp`
contribute the final `2N`.

**Predict:** what coefficient of `N` do those pieces produce?

```text
1 + 3 + 4 + 4 + 4 + 2 = 18
```

One `BlockTensors` therefore occupies:

```text
18N + S floats
```

A value block adds four `R`-float layernorm statistic arrays:

```text
block values       = 18N + S + 4R
block gradients    = 18N + S
```

Across the whole model, let `A` be value-arena floats and `G` be
gradient-arena floats:

```text
A = 2N + L(18N + S + 4R) + 2R + 2Q
G = 2N + L(18N + S)      + Q
```

The `2N` terms are embedded plus final normalized values, and their
two gradient counterparts. The value-only `2R` holds final means and
reciprocal standard deviations. Values keep both logits and
probabilities, while gradients keep only `d_logits`.

Substituting the tiny counts gives `A=108` and `G=90`, matching both
cursor walks. The formulas have now earned their names from the
individual slots.

Chapter 9 derived the learned-scalar count:

```text
P = (V + T)C + L(12C*C + 4C) + 2C
```

Each learned scalar has four `Param` buffer slots. The formula sums
those slots across separately owned `Param` allocations. The tiny
configuration has `P=70`.

## Attention owns the expensive square

For batch `B`, time `T`, and `H` heads, attention scores need:

```text
B * H * T * T floats per layer
```

One sequence with one head and two positions needs `1*1*2*2 = 4`
scores. Keep batch and heads fixed, then double time to four.
**Predict:** how many score floats are needed now?

The score area becomes `1*1*4*4 = 16`, four times larger.

Most activation shapes scale with `B*T*C`. Scores scale with `T*T`
because every query position reserves a row with room for every key
position. [Chapter 1's no-peeking
rule](01-the-map.md#no-peeking-at-the-answer) prevents later keys from
contributing, but the stored score grid still has the full square
shape.

That is why independently reasonable dimension ceilings can combine
into an absurd allocation. Doubling `T` doubles ordinary
channel-sized grids and quadruples score storage.

## Check arithmetic before performing it

The formulas are trustworthy only when their arithmetic is. `size_t`
is unsigned, so a result beyond `SIZE_MAX` wraps into the representable
range. Testing the wrapped result afterward is too late.

Pretend for one worked example that `SIZE_MAX` is 15.
**Predict:** which operations fit, and what result does each fitting
operation produce?

```text
9 + 6
9 + 7
3 * 5
4 * 4
```

Both `9 + 6` and `3 * 5` fit exactly at 15. The mathematical result
of each remaining operation is 16, which this pretend `size_t` cannot
represent.

The exact helpers in `model.c` test before calculating:

```c
static int checked_add(size_t left, size_t right, size_t *result)
{
    if (left > SIZE_MAX - right)
        return 0;
    *result = left + right;
    return 1;
}

static int checked_multiply(size_t left, size_t right, size_t *result)
{
    if (left != 0 && right > SIZE_MAX / left)
        return 0;
    *result = left * right;
    return 1;
}

static int checked_scaled_add(size_t *total, size_t count, size_t scale)
{
    size_t term;

    return checked_multiply(count, scale, &term)
        && checked_add(*total, term, total);
}
```

For addition, `SIZE_MAX - right` is the largest safe left operand.
With the pretend maximum, `15 - 6 = 9`, so `9 + 6` is admitted.
For `9 + 7`, the largest safe left is 8, so the helper returns zero
without adding.

For multiplication, zero needs no division: `0 * right` is always
representable as zero. The left side of `&&` is false when `left` is
zero, so C skips `SIZE_MAX / left` and avoids division by zero.

A nonzero `left` can multiply at most `SIZE_MAX / left`. The safe
bound for left 3 is `15 / 3 = 5`.
The bound for left 4 is `15 / 4 = 3`, where integer division discards
the remainder. Four is above that bound, so `4 * 4` is rejected before
it wraps.

`checked_scaled_add` first constructs `count * scale`, then adds the
term into a running total. With maximum 15, total 7, count 2, and
scale 4, the term is 8 and the new total is 15. Starting at total 8
would reject `8 + 8`. The `&&` stops before the addition if the
multiplication failed.

These helpers return one for success and zero for failure. They do not
allocate, print, or choose a resource policy.

Together, these bounds-before-operation rules are **checked
memory-size arithmetic**. The calculation reports failure instead of
letting an unrepresentable byte request wrap into a smaller number.

## The public memory report is an independent calculation

The layout cursor can measure an arena only after a `Model` and its
block array exist. [Checkpoint loading in Chapter
13](13-durable-checkpoints.md#count-bytes-and-memory-separately) and
[command setup in Chapter
14](14-the-command-line.md#read-data-before-allocating-the-model) need
an answer before constructing either.
`model_memory_requirements` therefore recomputes the shape formulas
with checked arithmetic.

This is intentionally an independent calculation written directly
from the shapes. It is not the cursor returned by `lay_out_values` or
`lay_out_gradients`, and `model_new` does not allocate the arenas from
its byte fields. A new view must update both the placement sequence
and this report.

The caller receives five byte counts in this exact public value:

```c
typedef struct {
    size_t parameter_bytes;
    size_t activation_bytes;
    size_t gradient_bytes;
    size_t token_bytes;
    size_t total_bytes;
} ModelMemory;
```

`parameter_bytes` covers all four buffers inside every `Param`.
`activation_bytes` is the value arena, and `gradient_bytes` is the
activation-gradient arena. `token_bytes` covers both integer id
caches. The last field adds those four subtotals.

This five-field value is the **model memory report**. Each field is a
`size_t` because it counts bytes in an object-size type, not model
scalars in `int`.

The report needs the learned-scalar count `P`, but even that formula
can overflow. This exact helper checks each group:

```c
static int parameter_floats(ModelConfig cfg, size_t *result)
{
    size_t width      = (size_t)cfg.d_model;
    size_t embeddings;
    size_t vector_floats;
    size_t square_floats;
    size_t total = 0;

    if (!checked_add((size_t)cfg.vocab_size, (size_t)cfg.block_size,
                     &embeddings)
        || !checked_multiply(embeddings, width, &embeddings)
        || !checked_multiply((size_t)cfg.layer_count, 4 * width,
                             &vector_floats)
        || !checked_add(vector_floats, 2 * width, &vector_floats)
        || !checked_multiply(width, width, &square_floats)
        || !checked_multiply(square_floats, 12, &square_floats)
        || !checked_multiply(square_floats, (size_t)cfg.layer_count,
                             &square_floats)
        || !checked_add(total, embeddings, &total)
        || !checked_add(total, vector_floats, &total)
        || !checked_add(total, square_floats, &total))
        return 0;
    *result = total;
    return 1;
}
```

The first two operations compute `(V + T)C` for the token and
position tables. The next two compute `4LC + 2C` for block layernorm
vectors and final layernorm. The next three compute `12*L*C*C` for
the block weight matrices. The last three additions combine those
groups.

That geometry predicate already caps `C`, so the local `4 * width`
and `2 * width` scales fit before reaching a helper. Every growing
product and total is checked. `*result` is written only after the
complete count succeeds.

One exact wrapper exposes that checked count to later model code:

```c
size_t model_parameter_float_count(ModelConfig cfg)
{
    size_t result;

    return parameter_floats(cfg, &result) ? result : 0;
}
```

On success, the conditional expression returns the count written to
`result`. On failure, it returns zero. The Chapter 10 witness does not
call this wrapper, but Chapter 13's checkpoint code will. Leaving it
out can therefore pass this chapter's lab and still leave the model
module incomplete.

`model_memory_requirements` then combines every checked group:

```c
int model_memory_requirements(ModelConfig cfg, ModelMemory *memory)
{
    if (memory == NULL || !model_config_valid(cfg))
        return 0;

    size_t rows;
    size_t channels;
    size_t scores;
    size_t logits;
    size_t block_values = 0;
    size_t block_gradients = 0;
    size_t value_floats = 0;
    size_t gradient_floats = 0;
    size_t parameter_count;

    if (!checked_multiply((size_t)cfg.batch_size, (size_t)cfg.block_size,
                          &rows)
        || !checked_multiply(rows, (size_t)cfg.d_model, &channels)
        || !checked_multiply(rows, (size_t)cfg.head_count, &scores)
        || !checked_multiply(scores, (size_t)cfg.block_size, &scores)
        || !checked_multiply(rows, (size_t)cfg.vocab_size, &logits)
        || !parameter_floats(cfg, &parameter_count)
        || !checked_scaled_add(&block_values, channels, 18)
        || !checked_add(block_values, scores, &block_values)
        || !checked_scaled_add(&block_values, rows, 4)
        || !checked_scaled_add(&block_gradients, channels, 18)
        || !checked_add(block_gradients, scores, &block_gradients)
        || !checked_scaled_add(&value_floats, channels, 2)
        || !checked_scaled_add(&value_floats, block_values,
                               (size_t)cfg.layer_count)
        || !checked_scaled_add(&value_floats, rows, 2)
        || !checked_scaled_add(&value_floats, logits, 2)
        || !checked_scaled_add(&gradient_floats, channels, 2)
        || !checked_scaled_add(&gradient_floats, block_gradients,
                               (size_t)cfg.layer_count)
        || !checked_add(gradient_floats, logits, &gradient_floats)
        || !checked_multiply(parameter_count, 4 * sizeof(float),
                             &memory->parameter_bytes)
        || !checked_multiply(value_floats, sizeof(float),
                             &memory->activation_bytes)
        || !checked_multiply(gradient_floats, sizeof(float),
                             &memory->gradient_bytes)
        || !checked_multiply(rows, 2 * sizeof(int), &memory->token_bytes))
        return 0;

    memory->total_bytes = 0;
    return checked_add(memory->total_bytes, memory->parameter_bytes,
                       &memory->total_bytes)
        && checked_add(memory->total_bytes, memory->activation_bytes,
                       &memory->total_bytes)
        && checked_add(memory->total_bytes, memory->gradient_bytes,
                       &memory->total_bytes)
        && checked_add(memory->total_bytes, memory->token_bytes,
                       &memory->total_bytes);
}
```

The first guard rejects a missing output pointer and invalid geometry.
The first five checked operations construct `R`, `N`, `S`, and `Q`.
`parameter_floats` constructs `P` from Chapter 9's formula with the
same checked helpers.

The next group builds one value block and one gradient block. The
following group scales those by `L` and adds the model-wide fields.
Only then are float and integer counts converted to bytes with
`sizeof`.

The final four additions form the total. If any operation fails, the
function returns zero. A caller must ignore every output field after
failure because earlier fields may already have been written.

The arena cursors themselves use unchecked `size_t` additions, and
`model_new` later multiplies their final counts by `sizeof(float)`
without another guard. The successful report makes those operations
safe in the current source because it enumerates exactly the same
`A` and `G` shapes. It proves each final float-to-byte product fits
`size_t`. Every cursor increment is nonnegative, so every prefix is
no greater than that proven final float count.

This agreement is a source-maintained invariant, not a runtime
comparison. The program does not compare a dry-run cursor with the
corresponding report field. If a future field changes one calculation
without changing the other, this proof no longer holds. That is why a
new view must update and review both routes.

Configuration validity and representable arithmetic answer different
questions. Neither enforces the one-GiB policy used by the command line
and checkpoint loader. A valid, representable report may exceed that
policy. `model_memory_requirements` still succeeds and reports the
number; those callers decide whether to reject it.

The report covers four payload families:

```text
all Param value, gradient, and two-moment float buffers
the value-arena float buffer
the activation-gradient-arena float buffer
the two cached-id integer buffers
```

It excludes the `Model`, `Block`, `Param`, and `Mat` structures, the
parameter pointer registry, the temporary construction RNG, allocator
metadata, the tokenizer and corpus, and caller-owned arrays. The total
is an exact contract for the listed buffers on the current platform.
It is not every byte requested from the allocator or the running
process's total memory in RAM.

## Three sizes, from hand-checkable to the banner

On the supported builds used by the committed evidence,
`sizeof(float)` and `sizeof(int)` are both four bytes. The tiny
geometry's `P=70`, `A=108`, `G=90`, and `R=2` become:

```text
family                 calculation                  bytes
Param payloads         4*70 floats * 4              1,120
value arena            108 floats * 4                 432
gradient arena          90 floats * 4                 360
token and target ids   2*2 ints * 4                    16
total                                                1,928
```

The Chapter 10 lab uses:

```text
V=5, T=4, C=8, H=2, L=1, B=2
R=8, N=64, S=64, Q=40
P=888, A=1,472, G=1,384
```

**Predict:** what total do the first four lines produce?

```text
Param payloads         4*888*4       = 14,208 bytes
value arena            1,472*4       =  5,888 bytes
gradient arena         1,384*4       =  5,536 bytes
token and target ids   2*8*4         =     64 bytes
total                                  25,696 bytes
```

Finally use the Chapter 0 defaults:

```text
V=80, T=128, C=128, H=4, L=4, B=32
R=4,096
N=524,288
S=2,097,152 per layer
Q=327,680
P=815,360
A=47,915,008
G=47,513,600
```

Their byte report is:

```text
Param payloads          13,045,760 bytes    12.44140625 MiB
value arena            191,660,032 bytes   182.78125000 MiB
gradient arena         190,054,400 bytes   181.25000000 MiB
token and target ids        32,768 bytes     0.03125000 MiB
total                  394,792,960 bytes   376.50390625 MiB
```

Printed to one decimal place, the total is `376.5 MiB`. The
[committed development
log](logs/dev-measurements.md#training-memory-default-model-10-steps)
records the same arena float counts, four 815,360-float parameter
buffers, and rounded total. That historical run also records
379.5 MiB peak resident memory, the running process's total memory in
RAM at that peak. It is larger because the public report intentionally
omits the other process allocations named above.

## Capacity is not the current shape

The arenas are sized for configured `batch_size` and `block_size`. A
generation call may use one sequence of three tokens. Validation may
use the full batch. Both share the allocation.

Most activation matrices can use [Chapter 4's leading-row
view](04-poor-mans-tensors.md#expose-fewer-rows-without-copying).
Their column count remains tied to channels, widened channels, or
vocabulary. Attention scores need more care. Their logical row width
is the current `time`, not the maximum block size.

Use the lab's configured `B=2`, `T=4`, and `H=2`. At full capacity,
there are `B*H*T = 2*2*4 = 16` score rows. Each row has four columns:

```text
full score shape    16 x 4 = 64 floats
```

Now call the same model with one sequence, two heads, and three
positions. **Predict:** how many score rows and columns does that call
need?

Each head has one query row per position, so the current call needs
`1*2*3 = 6` rows with three columns:

```text
current score shape    6 x 3 = 18 floats
```

The tempting `mat_first_rows(bt->scores, 6)` would preserve the
capacity width. It would produce `6 x 4`, expose 24 floats, and move
from one logical row to the next with stride 4. Attention needs stride
3. The current call instead needs a new `6 x 3` description over the
first 18 slots of the same storage.

That is [Chapter 4's reshape
view](04-poor-mans-tensors.md#changing-width-needs-a-new-description),
not a leading-row view. The exact constructor applies it:

```c
BlockTensors model_block_views(const BlockTensors *bt, int rows, int time,
                               int head_count)
{
    BlockTensors view;

    view.normed1         = mat_first_rows(bt->normed1, rows);
    view.qkv             = mat_first_rows(bt->qkv, rows);
    view.scores          =
        mat_make(bt->scores.vals, (rows / time) * head_count * time, time);
    view.attended        = mat_first_rows(bt->attended, rows);
    view.projected       = mat_first_rows(bt->projected, rows);
    view.after_attention = mat_first_rows(bt->after_attention, rows);
    view.normed2         = mat_first_rows(bt->normed2, rows);
    view.up              = mat_first_rows(bt->up, rows);
    view.activated       = mat_first_rows(bt->activated, rows);
    view.down            = mat_first_rows(bt->down, rows);
    view.after_mlp       = mat_first_rows(bt->after_mlp, rows);
    return view;
}
```

Every ordinary field keeps its capacity column count and exposes the
first `rows`. `scores` instead makes a new shape over the same first
address. The expression `rows / time` recovers the current batch
count. Its callers have already established positive `time` and
`rows = batch*time`. Each sequence has `head_count * time` score rows,
each with `time` columns.

No floats move, and no allocation occurs. The returned
`BlockTensors` is a by-value collection of descriptions for this call.
The capacity descriptions in the block remain unchanged.

`model_block_views` is an internal helper, so it relies on its caller's
already-checked facts: `time` is positive, `rows` equals
`batch * time`, and the requested batch and time fit the configured
capacity. The helper does not check them again. Without those facts,
`rows / time` could divide by zero or truncate a non-whole batch.

## Select the stream without copying it

Backward needs the residual stream entering any block and the matching
gradient destination. Layer 0 receives `embedded`. Every later layer
receives the previous block's `after_mlp`.

The two exact selectors mirror each other:

```c
Mat model_stream_into(const Model *m, int layer, int rows)
{
    Mat full =
        layer == 0 ? m->embedded : m->blocks[layer - 1].acts.after_mlp;

    return mat_first_rows(full, rows);
}

Mat model_d_stream_into(const Model *m, int layer, int rows)
{
    Mat full =
        layer == 0 ? m->d_embedded : m->blocks[layer - 1].grads.after_mlp;

    return mat_first_rows(full, rows);
}
```

For layer 0, both selectors choose the embedded pair. For layer 1,
they choose block 0's ending pair. Passing `L` selects the last block's
output, which is the stream entering final layernorm. `rows` then
narrows the capacity description without copying.

These selectors likewise rely on an internal layer from 0 through `L`,
inclusive, and a row count no greater than capacity. They choose and
narrow storage; they do not validate a public call.

## Construction, in order

`model_new` performs a one-way ownership handoff. Its exact source is:

```c
Model *model_new(ModelConfig cfg, unsigned long long seed)
{
    ModelMemory memory;
    int memory_ok = model_memory_requirements(cfg, &memory);

    assert(model_config_valid(cfg));
    assert(memory_ok);
    if (!memory_ok)
        die("invalid or unrepresentable model configuration");

    Model *m   = ecalloc(1, sizeof *m);
    Rng   *rng = rng_new(seed);
    int max_tokens = cfg.batch_size * cfg.block_size;

    m->cfg    = cfg;
    m->blocks = ecalloc((size_t)cfg.layer_count, sizeof *m->blocks);
    model_create_parameters(m, rng);
    rng_free(rng);

    size_t value_floats = lay_out_values(m, NULL);

    m->values_arena = emalloc(value_floats * sizeof *m->values_arena);
    lay_out_values(m, m->values_arena);

    m->gradient_floats = lay_out_gradients(m, NULL);
    m->gradient_arena =
        emalloc(m->gradient_floats * sizeof *m->gradient_arena);
    lay_out_gradients(m, m->gradient_arena);

    m->tokens  = emalloc((size_t)max_tokens * sizeof *m->tokens);
    m->targets = emalloc((size_t)max_tokens * sizeof *m->targets);
    return m;
}
```

The checked public report runs before the assertions. Debug builds
also assert the geometry and report result. When assertions are
disabled, the explicit `if` still stops an invalid or unrepresentable
configuration before allocation.

`ecalloc` zeroes the `Model` and block structures. Parameter
construction creates each separately owned `Param`, initializes its
values, and leaves its gradient and moments zero. The temporary RNG is
freed as soon as the parameter sequence is complete.

The value layout then runs with `NULL`, `emalloc` obtains its
uninitialized float block, and the layout repeats with the real base.
The gradient layout follows the same two-pass pattern and saves its
float count for `model_zero_gradients`. Its `emalloc` result is also
uninitialized. Token and target caches are the final two uninitialized
allocations.

The constructor uses `ModelMemory` as a checked preflight. Because the
current report and layouts enumerate the same shapes, `memory_ok`
establishes representability before the unchecked dry runs. The report
does not place views or supply their allocation counts; the two
dry-run layouts size the arenas.

Allocation failure follows Chapter 2's fatal allocator policy.
`model_new` neither returns `NULL` nor unwinds a partly built model.
After success, `Model` owns every allocation shown in the ownership
diagram. `model_free` releases each `Param`, both arena bases, both id
buffers, the registry, the block array, and finally the model. It does
not free any borrowed view pointer.

No training step allocates memory after that. Forward writes the values
its route needs, backward accumulates into cleared gradients, and the
next cycle reuses the same addresses.
[Chapter 12](12-wiring-the-model-backward.md#the-last-block-unwinds-first)
uses the stream selectors to hand each block's returning gradient to the
block below it.

## Build checkpoint: place the storage

**Build.** Chapter 9 deliberately left two files unfinished. In
`model.c`, implement `checked_add`, `checked_multiply`,
`checked_scaled_add`, `parameter_floats`,
`model_parameter_float_count`, and `model_memory_requirements`. Keep
the report independent of placement and preserve its failure
contract.

In `model_memory.c`, implement `place`, `place_floats`, block placement,
both arena layout passes, current-shape block views, the two stream
selectors, and the complete `model_new` order. Use the supplied
`model_internal.h` without changing its field order. Do not zero
`emalloc` buffers during construction or allocate one block per view.

**Verify.**

```sh
make -C labs check-10
# answer key: make -C labs WORK=../src check-10
```

**Expected.** [`labs/check10.c`](../labs/check10.c) reports:

```text
check-10: all 892 arena checks passed
```

Four checks establish that the memory report succeeds, the arenas
dominate the id caches, and full `2 x 4` and short `1 x 3` forward and
backward calls produce finite positive losses. After the short call,
the remaining 888 checks inspect every parameter-gradient scalar for
a finite value.

That witness does not inspect arena offsets, exact byte fields,
activation-gradient entries, whether a training call allocated, or
`model_parameter_float_count`. It also clears the full-call gradients
before checking the short call, so only the short call's 888
parameter-gradient entries are examined. The hand calculations,
source order, and later whole-model checks cover different parts of
the contract.

**Common failures.**

- An undefined `model_memory_requirements` means Chapter 10's
  `model.c` work was mistaken for placement work in one file.
- A crash during the measuring pass usually performed arithmetic on
  `NULL` instead of selecting `NULL` before adding the cursor.
- A correct arena with an incorrect public total means the independent
  report formula drifted from the placement shapes.
- A full batch that works while a short batch fails usually kept the
  maximum score width instead of reshaping it to current `time`.
- Finite results on the first cycle but stale or nonfinite results on
  the next usually left the activation-gradient arena uncleared.
- Corruption after adding one block field means its placement or the
  report was updated on only one side.
- A double free or invalid free usually treated a borrowed view pointer
  as another owner.
- Reading inconsistent first-step values usually assumed `emalloc`
  returned zeroed arena or id storage.

The model now has learned numbers and temporary numbers.
[Chapter 11](11-wiring-the-model-forward.md#one-block-has-two-highway-states)
connects them in the forward direction.

---

[Previous: Parameters and the Private Blueprint](09-parameters-and-the-blueprint.md) | [Contents](README.md) | [Next: Wiring the Model Forward](11-wiring-the-model-forward.md)
