# Chapter 9: Parameters and the Private Blueprint

Chapter 0 trained one visible model. One call cleared its old
corrections, one call applied the adjustment rule across all 815,360
learned numbers, and one call saved them. Chapter 8 has now built the
smaller object underneath those calls: one `Param` can hold a matrix
of learned values, its gradient, and its AdamW history.

The build track still needs a way to reach every `Param`. The model
has readable names such as `token_table` and `norm1_gain`, but
whole-model chores cannot be handwritten as a different list of names
every time. One omitted object would never learn. One duplicated
object could be adjusted or freed twice.

Start with a paper model containing four `Param *` values:

```text
named fields                  separately written work list

token  ---------> [ A ]       params[0] ---------> [ A ]
pos    ---------> [ B ]       params[1] ---------> [ B ]
gain   ---------> [ C ]       params[2] ---------> [ C ]
bias   ---------> [ D ]       params[3] ---------> [ C ]
```

Predict which learned object a loop over `params` will miss. It misses
`D`, while visiting `C` twice. A zeroing loop leaves `D`'s old
gradient in place. An update loop moves `C` twice. A teardown loop
tries to free the same allocation twice.

The named fields are not the problem. They let later code say which
parameter performs which job. The separately maintained work list is
the problem. We need the names and the list to receive the same
pointers as each object is created.

## One assignment creates both views

Suppose `make_param()` returns the address of a newly allocated
`Param`. Put that address into the list and its readable field in one
statement:

```c
m->token_table = m->params[at++] = make_param();
```

This is the first **chained assignment** in the book. C parses it as:

```text
token_table = (params[at++] = make_param())
```

The inner assignment stores the returned pointer in the selected list
slot. An assignment expression has the value it stored, so the outer
assignment stores that same pointer in `token_table`. Chapter 3 used
`written++`: the expression selects the old index, and `at` is one
larger by the end of this full statement. Chapter 2 introduced a
pointer to a pointer with `char **`. Here `Param **params` means that
each list slot holds a `Param *`.

After four correct assignments, both views reach the same objects:

```text
                    one object per slot

token  --------------> [ A ] <-------------- params[0]
pos    --------------> [ B ] <-------------- params[1]
gain   --------------> [ C ] <-------------- params[2]
bias   --------------> [ D ] <-------------- params[3]

named route: says what an object does
list route:  visits every object exactly once
```

Correct coverage is not enough. Suppose a seeded toy constructor
provides the draws `0.10` and `-0.30`. One run constructs token then
position:

```text
token =  0.10
pos   = -0.30
```

Another run constructs position then token:

```text
pos   =  0.10
token = -0.30
```

Both lists contain each object once. Predict whether the same seed now
gives the token object the same value. It does not, because the two
jobs consumed the seeded draws in a different order. Loading later
bytes through a different order would put values into the wrong jobs
for the same reason.

The flat, exactly-once list is the model's **parameter registry**. The
one fixed sequence of its entries is the **canonical parameter
order**. Coverage and order are separate promises:

- coverage lets zeroing, updating, counting, and freeing visit every
  object once;
- stable order makes seeded construction repeatable and gives later
  checkpoint bytes one unambiguous sequence.

The second promise does not follow from the first. A list can contain
every object once and still put position parameters before token
parameters on one run and after them on another.

The real constructor allocates enough pointer slots before filling
them:

```c
m->param_count =
    MODEL_TENSORS_ELSEWHERE + MODEL_TENSORS_PER_BLOCK * cfg.layer_count;
m->params = emalloc((size_t)m->param_count * sizeof *m->params);

int at = 0;
```

These are exact lines from `model_create_parameters`. The constants
say there are eight objects per block and four elsewhere. The cast to
`size_t` uses Chapter 1's object-size type before the multiplication.
`sizeof *m->params` asks for the size of one list entry, a `Param *`,
without repeating its type. `at` begins at slot zero.

## One block, all its learned tensors

Now scale the four-object construction to the transformer. Each
pre-norm block owns eight `Param *` values:

```text
within one block     shape

norm1 gain           [1, C]
norm1 bias           [1, C]
QKV weights          [3C, C]
projection           [C, C]
norm2 gain           [1, C]
norm2 bias           [1, C]
MLP up               [4C, C]
MLP down             [C, 4C]
```

Four more objects sit outside the blocks:

```text
outside the blocks   shape

token table          [V, C]
position table       [T, C]
final gain           [1, C]
final bias           [1, C]
```

For two blocks, predict the number of registry entries:

```text
2 opening objects + 2 * 8 block objects + 2 final objects
= 4 + 8 * 2
= 20 Param objects
```

Their order is:

```text
index       object

0           token table
1           position table
2..9        block 0, in the eight-row order above
10..17      block 1, in the same order
18          final gain
19          final bias
```

In general, `L` blocks produce:

```text
Param object count = 4 + 8L
```

This is not the number of learned scalar values. A `Param` can contain
one `1 x C` row or a much larger matrix. Use the Chapter 1 shapes with
`V = 5`, `T = 3`, `C = 4`, and `L = 2`:

```text
token and position tables
    5*4 + 3*4                                      = 32

one block
    4 + 4 + 3*4*4 + 4*4 + 4 + 4 + 4*4*4 + 4*4*4 = 208

two blocks
    2*208                                          = 416

final gain and bias
    4 + 4                                          = 8

learned scalar count
    32 + 416 + 8                                   = 456
```

The toy therefore has 20 `Param` objects containing 456 learned
scalars. The bundled configuration has:

```text
Param objects = 4 + 8*4
              = 36

learned scalars = 26,624 + 4*197,120 + 256
                = 815,360
```

Chapter 1 derived the general scalar formula:

```text
P = (V + T)C + L(12C^2 + 4C) + 2C
```

Batch size `B` and head count `H` do not appear in either count.
They change how the existing parameters are used, not how many are
created.

The public accessors preserve the distinction:

```c
ModelParams model_params(const Model *m)
{
    ModelParams view = { m->params, m->param_count };

    return view;
}

size_t model_parameter_count(const Model *m)
{
    size_t total = 0;

    for (int i = 0; i < m->param_count; i++)
        total += mat_size(param_values(m->params[i]));
    return total;
}
```

`model_params` returns a borrowed view of the pointer list and its
object count. The model still owns the list and every object in it.
`model_parameter_count` walks the same registry, asks each value
matrix for its scalar count, and adds those counts. For the two-block
toy, these functions report 20 and 456 respectively.

Predict the result if one registry entry were duplicated and another
omitted while their matrix sizes happened to match. The scalar total
could still be 456. Count alone cannot prove identity or order. The
Chapter 9 witness checks every named pointer, list position, and shape
on the toy model as separate facts.

## One table, two jobs

The final score for a character needs one learned row per vocabulary
entry. A first attempt could allocate a second table for that job and
copy the token table into it:

```text
opening lookup                    final scoring

token table [V, C] --copy-------> output table [V, C]
```

The two arrays begin equal, but equality at construction is not shared
identity. Consider three two-entry token rows and one final note:

```text
token table                 final note x = [0.5, 1.0]

A  [ 1.0, 2.0 ]             A score = 0.5*1 + 1*2  = 2.5
B  [-1.0, 1.0 ]             B score = 0.5*-1 + 1*1 = 0.5
C  [ 0.0, 3.0 ]             C score = 0.5*0 + 1*3  = 3.0
```

Suppose training changes `A`'s first lookup value from `1.0` to
`2.0`. Predict `A`'s new score. If scoring reads the same row, the
score becomes:

```text
0.5*2 + 1*2 = 3.0
```

A copied output table would still produce `2.5` until some separate
update happened to move it. It would also add `V*C` learned values,
gradients, and optimizer histories. At the bundled `V = 80` and
`C = 128`, that is 10,240 unnecessary learned values.

Use one object instead:

```text
                       one token-table Param
                      +----------------------+
token ids ----------> | row lookup           |
                      | values [V, C]         |
final notes --------> | row-by-row scoring   |
                      +----------------------+
                              |
                              v
                       one gradient region
```

The scoring operation reads each stored token row as a weight row.
The transpose in the mathematical spelling describes that role; the
program does not allocate or store a transposed copy.

This one-object arrangement is **weight tying**. The final scoring
stage is the **tied head**. Weight tying is not two arrays that happen
to start equal. It is one `Param *` used twice.

The source proves the identity from both directions. These selected
exact lines from `model_forward.c` pass the same
`m->token_table` values to opening lookup and final scoring:

```c
embedding_forward(stream, m->tokens, param_values(m->token_table),
                  param_values(m->position_table), time);

matmul_forward(logits, final_normed, param_values(m->token_table));
```

These selected exact lines from `model_backward.c` send both
contributions to the same gradient region:

```c
matmul_backward(d_final_normed, param_gradient(m->token_table), d_logits,
                final_normed, param_values(m->token_table));

embedding_backward(param_gradient(m->token_table),
                   param_gradient(m->position_table),
                   mat_first_rows(m->d_embedded, rows),
                   m->tokens, m->time);
```

Chapter 11 walks the complete forward wiring and
[Chapter 12](12-wiring-the-model-backward.md#one-tied-object-receives-two-returns)
walks the two tied-gradient returns in the complete backward wiring.
The architectural fact needed here is already visible: there is no
output-head field, allocation, registry entry, or second gradient.
Both jobs meet in `token_table`.

## Initialization is architecture

[Chapter 1](01-the-map.md#how-the-learned-numbers-start) constructed
the model's starting recipe. Chapter 2 constructed the seeded Gaussian
draws it uses. This section applies that already-owned recipe while
building the canonical registry.

The token table, position table, QKV matrix, and MLP-up matrix use
spread `0.02`. The attention projection and MLP-down matrix are the
two residual write-backs in each block. The source gives those two
groups a depth-dependent spread:

```text
residual spread = 0.02 / sqrt(2L)
```

For `L = 2`, predict the result:

```text
0.02 / sqrt(2*2)
= 0.02 / sqrt(4)
= 0.02 / 2
= 0.01
```

The factor tempers the starting scale of the `2L` writes that will be
added into the residual stream. Reading it as an exact promise about
the later variance would require assumptions about equal scales and
independence. Here it is the architecture's chosen starting rule.

The constructor begins with these exact lines:

```c
static const float INIT_STDDEV = 0.02f;

void model_create_parameters(Model *m, Rng *rng)
{
    ModelConfig cfg  = m->cfg;
    int         wide = MODEL_MLP_WIDENING * cfg.d_model;
    float residual_stddev =
        INIT_STDDEV / sqrtf(2.0f * (float)cfg.layer_count);
```

`cfg` is a local copy of the public dimensions. `wide` computes `4C`
once. `sqrtf` performs the float square root in the residual formula.
For the two-layer toy, the stored float is approximately `0.01`.

The opening assignments fill registry slots zero and one:

```c
    m->token_table = m->params[at++] =
        param_new_gaussian(cfg.vocab_size, cfg.d_model, INIT_STDDEV, rng);
    m->position_table = m->params[at++] =
        param_new_gaussian(cfg.block_size, cfg.d_model, INIT_STDDEV, rng);
```

The right side constructs a Gaussian `Param`. The middle assignment
puts its address into the next registry slot. The left assignment puts
the same address into the readable field. Both tables use spread
`0.02`.

The loop then performs all eight assignments for each block:

```c
    for (int layer = 0; layer < cfg.layer_count; layer++) {
        Block *b = &m->blocks[layer];

        b->norm1_gain = m->params[at++] =
            param_new_constant(1, cfg.d_model, 1.0f);
        b->norm1_bias = m->params[at++] =
            param_new_constant(1, cfg.d_model, 0.0f);
        b->qkv_weights = m->params[at++] =
            param_new_gaussian(QKV_STREAMS * cfg.d_model, cfg.d_model,
                               INIT_STDDEV, rng);
        b->proj_weights = m->params[at++] =
            param_new_gaussian(cfg.d_model, cfg.d_model,
                               residual_stddev, rng);
        b->norm2_gain = m->params[at++] =
            param_new_constant(1, cfg.d_model, 1.0f);
        b->norm2_bias = m->params[at++] =
            param_new_constant(1, cfg.d_model, 0.0f);
        b->up_weights = m->params[at++] =
            param_new_gaussian(wide, cfg.d_model, INIT_STDDEV, rng);
        b->down_weights = m->params[at++] =
            param_new_gaussian(cfg.d_model, wide, residual_stddev, rng);
    }
```

`&m->blocks[layer]` takes the address of the current block, and `b`
keeps the eight field names shorter. The two gain rows start at one
and the two bias rows start at zero. Those constant constructors do
not draw from `rng`. QKV and MLP-up use `0.02`; projection and MLP-down
use the residual spread. The loop repeats that exact order for every
layer.

The final two constant rows finish the registry:

```c
    m->final_gain = m->params[at++] =
        param_new_constant(1, cfg.d_model, 1.0f);
    m->final_bias = m->params[at++] =
        param_new_constant(1, cfg.d_model, 0.0f);
    assert(at == m->param_count);
}
```

The assertion checks that construction consumed the promised number
of slots. It cannot detect a swap, a duplicate pointer, or a wrong
shape that leaves the count unchanged. The lab's identity-and-shape
ledger checks those stronger promises.

The seed enters once. Gaussian construction advances one `Rng` in
canonical order:

```text
token, position,
block 0 QKV, projection, up, down,
block 1 QKV, projection, up, down,
...
```

Constant gains and biases consume no draws. Reordering Gaussian
objects assigns different consecutive parts of the stream to the moved
objects and any objects between them, even if every shape and count
remains valid. After the reordered interval consumes the same total
number of draws, later objects can resume at the same stream position.

The `Rng` and its saved Box-Muller result still persist across
constructor calls; the stream does not restart per `Param`. If a
Gaussian matrix consumes an odd number of results, its saved partner
becomes the next matrix's first result. Every Gaussian matrix in the
lab's toy and showcase geometries has an even scalar count, so that
particular boundary case is not exercised there. The witness does
replay the full initialization stream bit for bit within the same
build. It checks that the same seed reproduces every initialized
parameter value and a different seed changes the parameter values.

## Why the private blueprint must be exact

Chapter 1 made `Model *` an opaque public handle. Code calling the
model sees this declaration:

```c
typedef struct Model Model;
```

It can hold the address, but it cannot reach the hidden fields.
Implementation files such as `model_parameters.c`, `model_forward.c`,
and `model_backward.c` do need those fields. They are compiled
separately.

Imagine that one file believes `token_table` is the first pointer
after `cfg`, while another inserts a different pointer before it:

```text
file A's offsets                 file B's offsets

cfg             byte 0          cfg             byte 0
token_table     byte K          extra           byte K
position_table byte K+P         token_table     byte K+P
```

Both files accept a `Model *`. The linker cannot see that their field
offsets disagree. File A reads byte `K` as `token_table`; file B wrote
an unrelated pointer there. The shared internal definition must fix
the type and order of every field.

The header first fixes the repeated counts and the temporary slots
inside one block:

```c
enum {
    MODEL_MLP_WIDENING      = 4,
    MODEL_TENSORS_PER_BLOCK = 8,
    MODEL_TENSORS_ELSEWHERE = 4,
};

typedef struct {
    Mat normed1;
    Mat qkv;
    Mat scores;
    Mat attended;
    Mat projected;
    Mat after_attention;
    Mat normed2;
    Mat up;
    Mat activated;
    Mat down;
    Mat after_mlp;
} BlockTensors;
```

The enum gives one spelling to `4`, `8`, and `4` wherever separately
compiled files need them. Each `Mat` is a Chapter 4 view slot, not a
new allocation by itself. Chapter 10 will give these temporary value
slots storage. Chapters 11 and 12 will use them for forward values and
their gradient counterparts.

The exact `Block` declaration groups its eight learned objects with
those future slots:

```c
typedef struct {
    Param *norm1_gain;
    Param *norm1_bias;
    Param *qkv_weights;
    Param *proj_weights;
    Param *norm2_gain;
    Param *norm2_bias;
    Param *up_weights;
    Param *down_weights;

    BlockTensors acts;
    BlockTensors grads;
    float *means1;
    float *rstds1;
    float *means2;
    float *rstds2;
} Block;
```

The first eight fields are additional named pointers to the same
objects entered into the registry. `acts` and `grads` reserve matching
groups of views.
Layernorm needs the four saved-statistic pointers. Their computation
does not belong to this chapter; their positions in the shared
structure do.

The model begins with its configuration, readable parameter fields,
and registry:

```c
struct Model {
    ModelConfig cfg;

    Param  *token_table;
    Param  *position_table;
    Block  *blocks;
    Param  *final_gain;
    Param  *final_bias;

    Param **params;
    int     param_count;
```

`blocks` points to the array whose named parameter fields the
constructor fills. `params` points to the canonical registry. The
named pointers point to those same objects; they do not own second
`Param` objects.

The rest of the exact structure reserves storage and cached state for
later chapters:

```c
    float  *values_arena;
    float  *gradient_arena;
    size_t  gradient_floats;

    Mat     embedded;
    Mat     d_embedded;
    Mat     final_normed;
    Mat     d_final_normed;
    Mat     logits;
    Mat     d_logits;
    Mat     probs;
    float  *final_means;
    float  *final_rstds;

    int    *tokens;
    int    *targets;
    int     batch;
    int     time;
    int     has_targets;
};
```

The literal `arena` field names preview Chapter 10's storage plan.
The `Mat` fields are top-level views. The final fields remember the
most recent call's ids and shape; Chapter 11 constructs the
[latest-forward record](11-wiring-the-model-forward.md#the-caller-cannot-own-the-saved-ids).
For now, the point is structural: every model implementation file
calculates the same offsets from this one declaration.

This exact shared declaration is the model's **private blueprint**.
The lab setup supplies `model_internal.h`. `check-blueprint` compares
it byte for byte with both the reference header and the learner's
copy. Do not reorder, rename, add, or resize its fields. Chapter 10
and later checks deliberately combine separately compiled model files,
so matching declarations are a correctness condition, not a style
preference.

## Apply one operation to the whole model

The registry now gives whole-model work one trustworthy route. First,
keep gradient clearing separate from the update. The training call
order is:

```text
zero old gradients
forward and compute the current loss
backward and fill current gradients
step using those gradients
```

If `model_step` cleared gradients itself, it would erase the very
correction it was asked to apply.

The clearing function walks every persistent `Param` first:

```c
void model_zero_gradients(Model *m)
{
    for (int i = 0; i < m->param_count; i++)
        param_zero_gradient(m->params[i]);
    memset(m->gradient_arena, 0,
           m->gradient_floats * sizeof *m->gradient_arena);
}
```

The loop clears one parameter-gradient region per registry entry.
`memset` then clears the future temporary gradient storage from
Chapter 10. It does not clear learned values, AdamW moments, or the
forward-value storage.

Chapter 8 built global gradient clipping. Applying it through the
registry matters. Put gradient `[3]` in one `Param` and `[4]` in
another. Clipping each object to length one would produce `[1, 1]`,
changing the ratio from `3:4` to `1:1`.

One registry-wide measurement instead sees:

```text
sqrt(3^2 + 4^2) = 5
shared factor    = 1/5
result           = [0.6, 0.8]
```

Global clipping preserves direction in real arithmetic because every
entry receives the same positive scale. Stored float results are
rounded, so the source produces approximately `0.600000024` and
`0.800000012` for this witness.

The file fixes the cap at one:

```c
static const float GRADIENT_CLIP_NORM = 1.0f;
```

Chapter 2 introduced file-private `static` and read-only `const`.
The `f` suffix makes the literal a `float`, matching the gradients.

Before measuring, the model rejects any nonfinite gradient:

```c
static int clip_gradient_norm(Model *m)
{
    for (int i = 0; i < m->param_count; i++)
        if (!param_gradient_is_finite(m->params[i]))
            return -1;

    double norm_squared = 0.0;

    for (int i = 0; i < m->param_count; i++)
        norm_squared += (double)param_gradient_norm_squared(m->params[i]);
```

The first loop finishes across the whole registry before any scaling.
The second loop asks each `Param` for its squared-norm subtotal.
Chapter 8 showed that helper summing products in `double` and returning
the subtotal as `float`. `model.c` converts each float-rounded subtotal
back to `double` for this outer sum.

The ordinary path takes the square root, narrows it to `float`, and
uses one float factor:

```c
    float norm = (float)sqrt(norm_squared);

    if (finite_float(norm)) {
        if (norm <= GRADIENT_CLIP_NORM)
            return 0;
        for (int i = 0; i < m->param_count; i++)
            param_scale_gradient(m->params[i], GRADIENT_CLIP_NORM / norm);
        return 0;
    }
```

A finite norm at or below one needs no scale. A finite norm above one
uses `1.0f / norm` for every `Param`. The shared factor, rather than a
separate factor per object, is the model-wide invariant.

Very large finite gradients can make a per-`Param` subtotal overflow
when it is narrowed to float. The resulting nonfinite norm takes a
second path. It needs a sum and a scaling loop that never narrow an
intermediate total or factor to `float`:

```c
static double exact_gradient_norm_squared(const Model *m)
{
    double total = 0.0;

    for (int i = 0; i < m->param_count; i++) {
        Mat gradient = param_gradient(m->params[i]);

        for (size_t at = 0; at < mat_size(gradient); at++)
            total += (double)gradient.vals[at] * (double)gradient.vals[at];
    }
    return total;
}

static void scale_gradients_exactly(Model *m, double factor)
{
    for (int i = 0; i < m->param_count; i++) {
        Mat gradient = param_gradient(m->params[i]);

        for (size_t at = 0; at < mat_size(gradient); at++)
            gradient.vals[at] =
                (float)((double)gradient.vals[at] * factor);
    }
}
```

The first function begins a `double` total. Its outer loop visits every
registry entry. `param_gradient` obtains that object's matrix view.
The inner loop visits every scalar, converts both factors to `double`
before multiplication, and adds the product without an intervening
float subtotal. It returns the complete squared norm as `double`.

The second function receives a `double` factor. Its two loops visit
the same registry entries and scalars. Each stored float is converted
to `double`, multiplied by the factor, then converted back to `float`
for storage. No allocation or separate per-object scale appears.

With those helpers built, the tail of `clip_gradient_norm` is:

```c
    double exact_norm = sqrt(exact_gradient_norm_squared(m));

    if (exact_norm <= GRADIENT_CLIP_NORM)
        return 0;
    scale_gradients_exactly(m, (double)GRADIENT_CLIP_NORM / exact_norm);
    return 0;
}
```

The square root stays `double`. A result at or below one needs no
scaling. A larger result produces one double factor for the helper.
The name `exact` in these private helpers distinguishes this path from
the float-rounded subtotals. It does not remove final floating-point
rounding.

The lab puts `FLT_MAX/4` and `FLT_MAX/8` in one gradient. Their
float subtotal overflows, so the double rescan finds their common
scale. Predict the ratio after scaling. It remains approximately
`2:1`, with stored values near `0.89442718` and `0.44721359`.

`model_step` puts validation, clipping, and AdamW in the required
order:

```c
int model_step(Model *m, AdamW opt, int step)
{
    if (!param_adamw_recipe_valid(opt, step))
        return -1;
    if (clip_gradient_norm(m) != 0)
        return -1;
    for (int i = 0; i < m->param_count; i++) {
        if (param_adamw_step(m->params[i], opt, step) != 0)
            return -1;
    }
    return 0;
}
```

An invalid recipe returns before the gradient scan. A nonfinite
gradient returns before scaling or parameter updates. A valid model
then applies AdamW in canonical order.

Each `Param` update preflights its own entries, as Chapter 8 showed.
The whole model is not a transaction. Clipping may already have
changed gradients, and corrupted state in a later `Param` may be
found after earlier objects have moved. The function reports failure;
it does not roll those earlier objects back. A successful step also
leaves the current gradients present, possibly clipped. The next
training cycle calls `model_zero_gradients` explicitly.

The same exactly-once coverage closes the ownership loop:

```c
void model_free(Model *m)
{
    for (int i = 0; i < m->param_count; i++)
        param_free(m->params[i]);
    free(m->params);
    free(m->blocks);
    free(m->values_arena);
    free(m->gradient_arena);
    free(m->tokens);
    free(m->targets);
    free(m);
}
```

Each registry entry owns one `Param`, so the loop frees each once.
The pointer list and block array are separate allocations and follow.
The later storage and id buffers follow those. The `Model` allocation
is last. Named fields such as `token_table` point to objects already
freed through the registry, so they are not freed separately. The tied
head adds no second object to free.

## The file split is now earned

A transformer is not a new mathematical operation hiding behind a
dramatic name. It is an arrangement of the operations already tested,
plus an agreement about which arrays persist and how every file
reaches them.

The responsibilities can now be separated without hiding their
connections:

```text
model_parameters.c   create every Param in canonical order
model.c              validate, expose, count, zero, step, and free
model_internal.h     fix the shared private field layout
model_memory.c       measure and place temporary storage       Chapter 10
model_forward.c      connect the complete score route          Chapter 11
model_backward.c     connect the complete correction route     Chapter 12
checkpoint.c         serialize the persistent model            Chapter 13
```

Configuration validity is a Chapter 1 contract applied by `model.c`.
Its exact predicate requires positive bounded `V`, `T`, `C`, `H`, and
`L`; positive `B`; `B*T` no greater than 1,048,576 after widening the
multiplication to `long long`; and `C % H == 0`. Passing that geometry
predicate does not promise that the requested storage fits. Chapter 10
performs the checked byte calculation and completes `model_new`.

## Build checkpoint: construct the blueprint

**Build.** Use the supplied `model_internal.h` without changing one
byte. Implement configuration validation and the shared accessors in
`model.c`. In `model_parameters.c`, allocate the pointer registry and
construct every `Param` with one shared `at` cursor in the exact order
walked above. Implement model-wide gradient clearing, clipping,
stepping, counting, and teardown. Do not place temporary storage or
wire forward and backward yet.

**Verify.**

```sh
make -C labs check-09
# answer key: make -C labs WORK=../src check-09
```

**Expected.** `check-blueprint` first reports that the shared private
layout is exact. The Chapter 9 witness then reports 60 model-contract
checks. On its two-block toy, it verifies 20 registry entries, every
named-pointer identity and shape, and the exact seeded initialization
stream. Those checked shapes imply the 456 learned scalars calculated
in the chapter. On its showcase parameter geometry, it directly checks
36 objects and 815,360 learned scalars. The ordinary clipping witness
turns `(3, 4)` into approximately `(0.6, 0.8)`. The large finite
witness exercises the elementwise double-rescan fallback within
tolerance. An invalid recipe leaves the `(3, 4)` gradient unclipped,
and an infinite gradient is rejected.

The witness does not exercise the tied head's complete forward and
backward routes, temporary-storage placement, or
`model_zero_gradients`. Chapters 10 through 12 add the storage and
wiring checks. The source predicate and the broader model integration
checks establish the configuration boundary.

**Common failures.**

- A correct scalar total with a wrong named pointer means count was
  mistaken for identity and order.
- A missing or duplicated update means named fields and the registry
  were filled in separate statements.
- A seeded replay mismatch often means Gaussian constructors moved out
  of canonical order or a constant constructor consumed RNG draws.
- Projection or MLP-down values at spread `0.02` missed the smaller
  residual scale.
- A separate output table turned one tied object into two untied
  parameters.
- A `(3, 4)` result of `(1, 1)` clipped each `Param` separately.
- A failed step that already scaled a nonfinite gradient skipped the
  whole-registry finite preflight.
- A later training cycle using old corrections forgot that
  `model_step` does not clear gradients.
- A blueprint comparison failure means the supplied private header was
  edited.
- Training that continues after `model_step` returns failure discarded
  the status at the caller boundary.

The blueprint now owns every learned object and one canonical route
through them. Chapter 10 gives all the temporary numbers somewhere to
live.

---

[Previous: AdamW](08-adamw.md) | [Contents](README.md) | [Next: Memory Planning and Arenas](10-memory-planning.md)
