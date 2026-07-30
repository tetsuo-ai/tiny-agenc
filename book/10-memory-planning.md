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

### One cursor takes every region

An offset alone cannot say how much storage remains. The source carries
the base, next-free offset, and capacity together:

```c
typedef struct {
    float  *base;
    size_t  next;
    size_t  capacity;
} ArenaCursor;

typedef struct {
    size_t values;
    size_t gradients;
} ArenaLayout;
```

`ArenaCursor` is the state of one walk. `ArenaLayout` is the pair of
final counts produced by the two measuring walks. The exact cursor
helpers in `model_memory.c` are:

```c
static ArenaCursor arena_cursor(float *base, size_t capacity)
{
    ArenaCursor cursor = { base, 0, capacity };

    return cursor;
}

static float *arena_take(ArenaCursor *cursor, size_t count)
{
    if (cursor->next > cursor->capacity
        || count > cursor->capacity - cursor->next)
        die("model arena layout exceeded its measured capacity");

    float *start = NULL;

    if (cursor->base != NULL)
        start = cursor->base + cursor->next;
    cursor->next += count;
    return start;
}

static Mat arena_place_mat(ArenaCursor *cursor, int rows, int cols)
{
    size_t count = (size_t)rows * (size_t)cols;

    return mat_make(arena_take(cursor, count), rows, cols);
}

static float *arena_place_floats(ArenaCursor *cursor, size_t count)
{
    return arena_take(cursor, count);
}
```

`arena_cursor` starts `next` at zero. `arena_take` first proves that
the requested count fits in the remaining capacity. The subtraction
occurs only after proving `next <= capacity`, so it cannot wrap. A
placement bug stops at the point that asks for too much storage.

The pointer starts as `NULL`. With a real base, the `if` advances to
the current slot. With a null base, the addition is not executed and
the returned start remains `NULL`. This is a C sharp edge, not
decoration. Pointer arithmetic is defined only within an actual array,
plus its one-past address. `NULL + 0` is not a harmless way to say
"no storage."

`arena_place_mat` turns rows and columns into a float count, takes that
region, then records its pointer and shape. `arena_place_floats`
applies the same operation to unshaped arrays. Layernorm keeps one mean
and one reciprocal standard deviation per row, so those regions need a
pointer and a count rather than a two-dimensional description.

For the seven-slot example, a cursor begins at zero with capacity
seven. Placing `2 x 2` returns the current start and advances to 4.
Placing `1 x 3` returns the next start and advances to 7. Asking for
one more float would fail the remaining-capacity check.

Running one fixed placement sequence first to count and again to
assign addresses is a **two-pass arena layout**. The name labels the
count-then-place construction above; it does not add another routine.

### One value layout, run twice

[Chapter 1's
`ModelConfig`](01-the-map.md#when-copying-the-description-is-useful)
is the six-integer capacity description. One value block combines its
regular tensors with four row-sized statistic arrays:

```c
static void place_block_statistics(Block *block, ArenaCursor *cursor,
                                   size_t rows)
{
    block->means1 = arena_place_floats(cursor, rows);
    block->rstds1 = arena_place_floats(cursor, rows);
    block->means2 = arena_place_floats(cursor, rows);
    block->rstds2 = arena_place_floats(cursor, rows);
}

static void place_block_values(Block *block, ArenaCursor *cursor,
                               ModelConfig cfg, size_t rows)
{
    place_block_tensors(&block->acts, cursor, cfg);
    place_block_statistics(block, cursor, rows);
}
```

`place_block_values` keeps those two groups adjacent for every layer.
The whole value sequence then reads the capacities in this exact
source:

```c
static void place_value_views(Model *m, ArenaCursor *cursor)
{
    ModelConfig cfg  = m->cfg;
    int         rows = cfg.batch_size * cfg.block_size;

    m->embedded = arena_place_mat(cursor, rows, cfg.d_model);
    for (int layer = 0; layer < cfg.layer_count; layer++)
        place_block_values(&m->blocks[layer], cursor, cfg, (size_t)rows);
    m->final_normed = arena_place_mat(cursor, rows, cfg.d_model);
    m->final_means = arena_place_floats(cursor, (size_t)rows);
    m->final_rstds = arena_place_floats(cursor, (size_t)rows);
    m->logits = arena_place_mat(cursor, rows, cfg.vocab_size);
    m->probs = arena_place_mat(cursor, rows, cfg.vocab_size);
}
```

The caller supplies the cursor, already reset to zero. The function
places the embedded stream, then every block's regular tensors and four
statistic arrays through `place_block_values`.
The final normalized values, two final statistic arrays, logits, and
probabilities finish the sequence.

The gradient sequence reuses the block shape order:

```c
static void place_gradient_views(Model *m, ArenaCursor *cursor)
{
    ModelConfig cfg  = m->cfg;
    int         rows = cfg.batch_size * cfg.block_size;

    m->d_embedded = arena_place_mat(cursor, rows, cfg.d_model);
    for (int layer = 0; layer < cfg.layer_count; layer++)
        place_block_tensors(&m->blocks[layer].grads, cursor, cfg);
    m->d_final_normed = arena_place_mat(cursor, rows, cfg.d_model);
    m->d_logits = arena_place_mat(cursor, rows, cfg.vocab_size);
}
```

There are no statistic arrays and no `d_probs`, matching the lifetime
argument above.

The measuring pass gives each route a null cursor whose capacity is the
largest `size_t`:

```c
static ArenaLayout measure_arena_layout(Model *m)
{
    ArenaCursor values = arena_cursor(NULL, SIZE_MAX);
    ArenaCursor gradients = arena_cursor(NULL, SIZE_MAX);
    ArenaLayout layout;

    place_value_views(m, &values);
    place_gradient_views(m, &gradients);
    layout.values = values.next;
    layout.gradients = gradients.next;
    return layout;
}
```

These two calls briefly install `NULL`-backed descriptions in the
model. No operation reads them. `arena_take` advances only the counts
and performs no null pointer arithmetic. The later placement passes
replace every description with a view into allocated storage.

The shared placement functions prevent a measured cursor from drifting
away from its placements. `ArenaLayout` holds float counts, not the
caller-visible byte total. A separate checked calculation produces
that report before the model exists, and construction later compares
the two routes.

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
`place_value_views` creates exactly that order.

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
static void place_block_tensors(BlockTensors *bt, ArenaCursor *cursor,
                                ModelConfig cfg)
{
    int rows = cfg.batch_size * cfg.block_size;
    int wide = MODEL_MLP_WIDENING * cfg.d_model;

    bt->normed1 = arena_place_mat(cursor, rows, cfg.d_model);
    bt->qkv = arena_place_mat(cursor, rows, QKV_STREAMS * cfg.d_model);
    bt->scores =
        arena_place_mat(cursor, rows * cfg.head_count, cfg.block_size);
    bt->attended = arena_place_mat(cursor, rows, cfg.d_model);
    bt->projected = arena_place_mat(cursor, rows, cfg.d_model);
    bt->after_attention = arena_place_mat(cursor, rows, cfg.d_model);
    bt->normed2 = arena_place_mat(cursor, rows, cfg.d_model);
    bt->up = arena_place_mat(cursor, rows, wide);
    bt->activated = arena_place_mat(cursor, rows, wide);
    bt->down = arena_place_mat(cursor, rows, cfg.d_model);
    bt->after_mlp = arena_place_mat(cursor, rows, cfg.d_model);
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
from the shapes. It is not the `ArenaLayout` returned by
`measure_arena_layout`, and `model_new` does not allocate the arenas
from its byte fields. A new view must update both the placement
sequence and this report. Construction will compare the two answers
before allocating either arena.

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
can overflow. One large routine could mix three independent parameter
families and the final sum. The source gives each family one job:

```c
static int embedding_parameter_floats(ModelConfig cfg, size_t *result)
{
    size_t table_rows;

    return checked_add((size_t)cfg.vocab_size, (size_t)cfg.block_size,
                       &table_rows)
        && checked_multiply(table_rows, (size_t)cfg.d_model, result);
}

static int vector_parameter_floats(ModelConfig cfg, size_t *result)
{
    size_t width      = (size_t)cfg.d_model;
    size_t block_vectors;

    return checked_multiply((size_t)cfg.layer_count, 4 * width,
                            &block_vectors)
        && checked_add(block_vectors, 2 * width, result);
}

static int matrix_parameter_floats(ModelConfig cfg, size_t *result)
{
    size_t square;

    return checked_multiply((size_t)cfg.d_model, (size_t)cfg.d_model,
                            &square)
        && checked_multiply(square, 12, &square)
        && checked_multiply(square, (size_t)cfg.layer_count, result);
}

static int measure_parameter_floats(ModelConfig cfg, size_t *result)
{
    size_t embeddings;
    size_t vectors;
    size_t matrices;
    size_t total;

    if (!embedding_parameter_floats(cfg, &embeddings)
        || !vector_parameter_floats(cfg, &vectors)
        || !matrix_parameter_floats(cfg, &matrices)
        || !checked_add(embeddings, vectors, &total)
        || !checked_add(total, matrices, &total))
        return 0;
    *result = total;
    return 1;
}
```

`embedding_parameter_floats` computes `(V + T)C` for the two tables.
`vector_parameter_floats` computes `4LC + 2C` for block and final
layernorm vectors. `matrix_parameter_floats` computes `12*L*C*C` for
the block matrices. The last helper obtains those three counts, then
adds them.

That geometry predicate already caps `C`, so the local `4 * width`
and `2 * width` scales fit before reaching a helper. Every growing
product and total is checked. `*result` is written only after the
complete count succeeds.

One exact wrapper exposes that checked count to later model code:

```c
size_t model_parameter_float_count(ModelConfig cfg)
{
    size_t result;

    return model_config_valid(cfg)
        && measure_parameter_floats(cfg, &result) ? result : 0;
}
```

The wrapper now requires valid geometry as well as representable
arithmetic. On success, the conditional expression returns the count
written to `result`. On either failure, it returns zero. Chapter 13's
checkpoint code uses this boundary.

The rest of the report follows the hand calculation's three levels.
These exact private records name them:

```c
typedef struct {
    size_t rows;
    size_t channels;
    size_t scores;
    size_t logits;
} ShapeCounts;

typedef struct {
    size_t values;
    size_t gradients;
} BlockFloatCounts;

typedef struct {
    size_t parameters;
    size_t values;
    size_t gradients;
} ModelFloatCounts;
```

`ShapeCounts` holds `R`, `N`, `S`, and `Q`. `BlockFloatCounts` holds
one block's value and gradient totals. `ModelFloatCounts` adds
parameters and scales the two block totals across the model. The names
label the levels already constructed from the tiny slot walk.

The first stage builds the four shapes:

```c
static int measure_shape_counts(ModelConfig cfg, ShapeCounts *result)
{
    ShapeCounts counts;

    if (!checked_multiply((size_t)cfg.batch_size, (size_t)cfg.block_size,
                          &counts.rows)
        || !checked_multiply(counts.rows, (size_t)cfg.d_model,
                             &counts.channels)
        || !checked_multiply(counts.rows, (size_t)cfg.head_count,
                             &counts.scores)
        || !checked_multiply(counts.scores, (size_t)cfg.block_size,
                             &counts.scores)
        || !checked_multiply(counts.rows, (size_t)cfg.vocab_size,
                             &counts.logits))
        return 0;
    *result = counts;
    return 1;
}
```

The products are `B*T`, then `R*C`, `R*H*T`, and `R*V`. The function
uses a local record and publishes it only after every product fits.

The second stage applies the two one-block formulas:

```c
static int measure_block_float_counts(ShapeCounts shape,
                                      BlockFloatCounts *result)
{
    BlockFloatCounts counts = {0};

    if (!checked_scaled_add(&counts.values, shape.channels, 18)
        || !checked_add(counts.values, shape.scores, &counts.values)
        || !checked_scaled_add(&counts.values, shape.rows, 4)
        || !checked_scaled_add(&counts.gradients, shape.channels, 18)
        || !checked_add(counts.gradients, shape.scores,
                        &counts.gradients))
        return 0;
    *result = counts;
    return 1;
}
```

The value side becomes `18N + S + 4R`. The gradient side becomes
`18N + S`. Starting the local record at zero makes each
`checked_scaled_add` a direct term in those formulas.

The third stage adds the fields outside the repeated blocks:

```c
static int measure_model_float_counts(ModelConfig cfg, ShapeCounts shape,
                                      BlockFloatCounts block,
                                      ModelFloatCounts *result)
{
    ModelFloatCounts counts = {0};

    if (!measure_parameter_floats(cfg, &counts.parameters)
        || !checked_scaled_add(&counts.values, shape.channels, 2)
        || !checked_scaled_add(&counts.values, block.values,
                               (size_t)cfg.layer_count)
        || !checked_scaled_add(&counts.values, shape.rows, 2)
        || !checked_scaled_add(&counts.values, shape.logits, 2)
        || !checked_scaled_add(&counts.gradients, shape.channels, 2)
        || !checked_scaled_add(&counts.gradients, block.gradients,
                               (size_t)cfg.layer_count)
        || !checked_add(counts.gradients, shape.logits,
                        &counts.gradients))
        return 0;
    *result = counts;
    return 1;
}
```

The value route computes `2N + L*block.values + 2R + 2Q`. The
gradient route computes `2N + L*block.gradients + Q`. The same stage
also obtains `P`. Again, a failed operation leaves its caller's result
untouched.

Only the next stage converts complete float counts to bytes. These
exact remaining stages from `model.c` use the already-shown checked
helpers:

```c
static int total_memory_bytes(ModelMemory *memory)
{
    size_t total;

    return checked_add(memory->parameter_bytes, memory->activation_bytes,
                       &total)
        && checked_add(total, memory->gradient_bytes, &total)
        && checked_add(total, memory->token_bytes, &memory->total_bytes);
}

static int build_memory_report(ShapeCounts shape, ModelFloatCounts floats,
                               ModelMemory *result)
{
    ModelMemory memory;

    if (!checked_multiply(floats.parameters, 4 * sizeof(float),
                          &memory.parameter_bytes)
        || !checked_multiply(floats.values, sizeof(float),
                             &memory.activation_bytes)
        || !checked_multiply(floats.gradients, sizeof(float),
                             &memory.gradient_bytes)
        || !checked_multiply(shape.rows, 2 * sizeof(int),
                             &memory.token_bytes)
        || !total_memory_bytes(&memory))
        return 0;
    *result = memory;
    return 1;
}

static int measure_model_memory(ModelConfig cfg, ModelMemory *result)
{
    ShapeCounts shape;
    BlockFloatCounts block;
    ModelFloatCounts floats;

    return measure_shape_counts(cfg, &shape)
        && measure_block_float_counts(shape, &block)
        && measure_model_float_counts(cfg, shape, block, &floats)
        && build_memory_report(shape, floats, result);
}

int model_memory_requirements(ModelConfig cfg, ModelMemory *memory)
{
    ModelMemory result;

    if (memory == NULL || !model_config_valid(cfg)
        || !measure_model_memory(cfg, &result))
        return 0;
    *memory = result;
    return 1;
}
```

`build_memory_report` multiplies `P` by four float buffers, converts
`A` and `G` to float bytes, converts `R` to two integer caches, and
asks `total_memory_bytes` to add the four families. That helper keeps
the first two partial sums in local `total`; only the final checked
addition writes `total_bytes`. `build_memory_report` also owns a local
`ModelMemory`, so none of its fields reach the caller unless all five
fields succeed.

`measure_model_memory` is the short stage coordinator. The `&&`
operators stop at the first failure. The public function adds the
missing-output and geometry checks, computes into another local
record, and performs one final assignment. Therefore any failure
leaves the caller's `ModelMemory` unchanged. That behavior is part of
the declaration in `model.h`, so callers may keep a previous report
without receiving a mixture of old and partial fields.

### Compare the independent answers at runtime

An independent calculation can drift from placement. Construction now
checks every reported storage family after parameters and block records
exist. This exact pair performs the comparison:

```c
static int memory_total_matches_components(ModelMemory memory)
{
    size_t remainder = memory.total_bytes;

    if (memory.parameter_bytes > remainder)
        return 0;
    remainder -= memory.parameter_bytes;
    if (memory.activation_bytes > remainder)
        return 0;
    remainder -= memory.activation_bytes;
    if (memory.gradient_bytes > remainder)
        return 0;
    remainder -= memory.gradient_bytes;
    return remainder == memory.token_bytes;
}

static int memory_report_matches_layout(const Model *m, ModelMemory memory,
                                        ArenaLayout layout)
{
    size_t parameter_unit = 4 * sizeof(float);
    size_t token_unit = 2 * sizeof(int);
    size_t max_tokens =
        (size_t)m->cfg.batch_size * (size_t)m->cfg.block_size;

    return memory.parameter_bytes % parameter_unit == 0
        && memory.parameter_bytes / parameter_unit
            == model_parameter_count(m)
        && memory.activation_bytes % sizeof(float) == 0
        && memory.gradient_bytes % sizeof(float) == 0
        && memory.activation_bytes / sizeof(float) == layout.values
        && memory.gradient_bytes / sizeof(float) == layout.gradients
        && memory.token_bytes % token_unit == 0
        && memory.token_bytes / token_unit == max_tokens
        && memory_total_matches_components(memory);
}
```

The first pair checks the four `Param` buffers against the constructed
parameter registry. The next four checks compare the reported arena
bytes with both measured cursor totals. The token pair checks two
integer caches for every capacity token. The last helper subtracts
each component from `total_bytes` only after proving it fits, then
requires the remainder to equal `token_bytes`.

`create_model_storage` runs that comparison before arena allocation:

```c
static void create_model_storage(Model *m, ModelMemory memory)
{
    ArenaLayout layout = measure_arena_layout(m);

    require_matching_memory_report(m, memory, layout);
    create_value_arena(m, layout.values);
    create_gradient_arena(m, layout.gradients);
    create_token_caches(m);
}
```

A mismatch terminates with
`model storage layout disagrees with its memory report`. The real
placement cursors receive the measured count as capacity.
`arena_take` rejects overflow during placement, and
`require_full_arena` rejects a pass that finishes short. The
independent routes still need to be updated together, but a drift no
longer survives model construction silently.

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
V=5, T=4, C=8, H=2, L=2, B=2
R=8, N=64, S=64, Q=40
P=1,688, A=2,720, G=2,600
```

**Predict:** what total do the first four lines produce?

```text
Param payloads         4*1,688*4     = 27,008 bytes
value arena            2,720*4       = 10,880 bytes
gradient arena         2,600*4       = 10,400 bytes
token and target ids   2*8*4         =     64 bytes
total                                  48,352 bytes
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

`model_new` performs a one-way ownership handoff. The allocation work
is split into helpers named for the storage they create. These exact
allocation helpers appear in `model_memory.c`:

```c
static Model *allocate_model_record(ModelConfig cfg)
{
    Model *m = ecalloc(1, sizeof *m);

    m->cfg = cfg;
    return m;
}

static void create_parameter_storage(Model *m, Rng *rng)
{
    m->blocks =
        ecalloc((size_t)m->cfg.layer_count, sizeof *m->blocks);
    model_create_parameters(m, rng);
}

static void create_value_arena(Model *m, size_t value_floats)
{
    m->values_arena =
        emalloc(value_floats * sizeof *m->values_arena);

    ArenaCursor cursor =
        arena_cursor(m->values_arena, value_floats);

    place_value_views(m, &cursor);
    require_full_arena(cursor);
}

static void create_gradient_arena(Model *m, size_t gradient_floats)
{
    m->gradient_floats = gradient_floats;
    m->gradient_arena =
        emalloc(gradient_floats * sizeof *m->gradient_arena);

    ArenaCursor cursor =
        arena_cursor(m->gradient_arena, gradient_floats);

    place_gradient_views(m, &cursor);
    require_full_arena(cursor);
}

static void create_token_caches(Model *m)
{
    size_t max_tokens =
        (size_t)m->cfg.batch_size * (size_t)m->cfg.block_size;

    m->tokens = emalloc(max_tokens * sizeof *m->tokens);
    m->targets = emalloc(max_tokens * sizeof *m->targets);
}
```

`allocate_model_record` zeroes the model and records its configuration.
`create_parameter_storage` zeroes the block records, then runs Chapter
9's parameter construction. Each arena helper allocates one measured
float count, places all its views, and requires the final cursor to
equal its capacity. `create_gradient_arena` also saves that exact count
for `model_zero_gradients`. The last helper allocates the two
full-capacity id caches.

With those jobs named, the exact constructor is short:

```c
Model *model_new(ModelConfig cfg, unsigned long long seed)
{
    ModelMemory memory;
    int memory_ok = model_memory_requirements(cfg, &memory);

    assert(model_config_valid(cfg));
    assert(memory_ok);
    if (!memory_ok)
        die("invalid or unrepresentable model configuration");

    Model *m = allocate_model_record(cfg);
    Rng *rng = rng_new(seed);

    create_parameter_storage(m, rng);
    rng_free(rng);
    create_model_storage(m, memory);
    return m;
}
```

The checked public report runs before the assertions. Debug builds
also assert the geometry and report result. When assertions are
disabled, the explicit `if` still stops an invalid or unrepresentable
configuration before allocation.

The model record comes first, followed by the temporary RNG, block
records, and separately owned `Param` allocations. Parameter values
are initialized while gradients and moments start at zero. The RNG is
freed as soon as that ordered parameter sequence is complete.

`create_model_storage`, shown in the runtime-comparison section, then
measures both null-backed layouts. It compares parameter, value,
gradient, token, and total counts with the checked report before
allocating arenas. The value and gradient passes repeat with their
real bases and exact measured capacities. Their `emalloc` results
remain uninitialized. Token and target caches are the final two
uninitialized allocations.

The constructor uses `ModelMemory` as a checked preflight, then uses
`ArenaLayout` as the placement counts. The runtime reconciliation
requires the independent calculations to agree before those roles
separate.

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
`checked_scaled_add`, the three parameter-family counts, the shape,
block, and model count stages, `model_parameter_float_count`, and
`model_memory_requirements`. Keep the report independent of placement,
publish it only on success, and preserve its failure contract.

In `model_memory.c`, implement `ArenaCursor`, `arena_take`, matrix and
float placement, both arena layout passes, report reconciliation,
named allocation helpers, current-shape block views, the two stream
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
check-10: all 1874 arena checks passed
```

The witness constructs the two-layer geometry counted above. It checks
that every reported byte family agrees with constructed storage. It
walks every matrix view and statistic span in placement order, checking
matrix shapes, exact next addresses, capacity bounds, and final arena
boundaries. It fills the complete gradient arena with ones, calls
`model_zero_gradients`, and checks that every reported float became
zero. Full `2 x 4` and short `1 x 3` forward/backward calls must still
produce finite positive losses. After the short call, all 1,688
parameter-gradient scalars must be finite.

The broader integration witness separately fixes the five exact
Chapter 0 showcase byte fields, exercises inclusive and rejected
configuration boundaries, requires a representable maximum report,
checks that a failed report leaves a sentinel record unchanged, and
matches constructed parameter storage to the architecture formula.
Those checks do not have a chapter-local count.

**Common failures.**

- An undefined `model_memory_requirements` means Chapter 10's
  `model.c` work was mistaken for placement work in one file.
- A crash during the measuring pass usually performed arithmetic on
  `NULL` instead of selecting `NULL` before adding the cursor.
- `model storage layout disagrees with its memory report` means an
  independent report formula drifted from constructed storage.
- `model arena placement did not fill its measured capacity` means a
  real placement pass ended before or after its measured boundary.
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
