# Chapter 8: AdamW

Chapter 0's loss fell because every training round eventually changed
the model's adjustable numbers. Chapter 7 gave us a referee for the
gradients from Chapter 6. A gradient now tells us the direction in
which the loss increases nearby. Its negative gives the direction in
which a small change could lower the loss. It does not say how large
the actual change should be.

Start with one adjustable number `p` and this made-up loss:

```text
loss = (p - 1)^2
```

At `p = 3`, the loss is:

```text
loss = (3 - 1)^2 = 4
```

[Chapter 6](06-backprop-by-hand.md) built the derivative of a square.
The gradient here is `2(p - 1)`, so it is `4`. Its positive sign says
that increasing `p` would increase the loss. We should move `p`
downward.

The most literal update subtracts the gradient:

```text
p = 3 - 4 = -1
```

Predict the new loss before reading on. At `p = -1`, it is:

```text
loss = (-1 - 1)^2 = 4
```

The sign told us to move downward near `3`, but a four-unit leap went
past the best value at `1`. The loss did not improve.

## Give the gradient a distance control

Put a nonnegative scale in front of the gradient. With a scale of
`0.25`, the same update becomes:

```text
change = 0.25 * 4 = 1
p      = 3 - 1    = 2
loss   = (2 - 1)^2 = 1
```

Subtracting the gradient still chooses the move's direction. The new
scale controls its distance. That scale is the **learning rate**.

Written for any parameter value `theta`, learning rate `lr`, and
gradient `g`, the rule is:

```text
theta = theta - lr * g
```

This is **stochastic gradient descent**, usually shortened to **SGD**.
The gradient points toward the local increase in loss. Subtracting it
moves in the opposite, locally downhill direction. It is stochastic
because each training batch comes from randomly selected corpus
windows, so its gradient is an estimate based on that batch. The
default command uses `32 * 128 = 4,096` prediction positions, but both
dimensions are configurable.

The general mechanism that turns gradients into parameter changes is
an **optimizer**. SGD is the first optimizer we could have invented
from the failed raw subtraction.

The learning rate is not a promise that every step lowers the next
reported loss. A gradient describes nearby behavior, and another
random batch may give a different grade. A rate that is too large can
cross the useful region. A rate that is very small can require many
steps. Tiny AgenC defaults to `0.001`; the larger rates in this chapter
keep the hand calculations visible.

Before moving on, predict the SGD result for:

```text
theta = [2.0, -1.0]
g     = [0.4, -0.2]
lr    = 0.5
```

Both entries use the same rule:

```text
change    = [0.2, -0.1]
new theta = [1.8, -0.9]
```

That rule remembers nothing about earlier batches. The next problem
comes from that missing memory.

## Keep a fading history

Suppose three batches produce these gradients for one parameter:

```text
batch         1      2      3
gradient    2.0   -2.0    2.0
```

SGD follows every sign reversal at full strength. We want recent
batches to vote together without storing every gradient ever seen.

An ordinary average keeps all old batches equally. That becomes slow
to change after thousands of steps. Instead, retain a fraction `beta`
of one running value and give the remaining fraction to the new
gradient:

```text
history = beta * old_history + (1 - beta) * gradient
```

Use `beta = 0.5` and begin at zero. The first update is:

```text
history[1] = 0.5 * 0 + 0.5 * 2 = 1
```

Predict the history after the second gradient, `-2`:

```text
history[2] = 0.5 * 1 + 0.5 * -2 = -0.5
```

Then the third gradient gives:

```text
history[3] = 0.5 * -0.5 + 0.5 * 2 = 0.75
```

Each new batch has the largest single vote. Older votes remain, but
their coefficients acquire another factor of `beta` at every step.
This construction is an **exponential moving average**, or **EMA**.
Using a gradient EMA to carry direction between steps is called
**momentum**.

In this construction, a momentum step uses the history where SGD used
the current gradient:

```text
theta = theta - lr * history
```

For `theta = 1`, `lr = 0.1`, and the second history value `-0.5`,
predict the sign of the move and the new value. The step gives
`1 - 0.1 * -0.5 = 1.05`. The negative recent vote moves the value
upward, but by less than using the current gradient `-2` directly.

Tiny AgenC calls this history the **first moment**:

```text
m = beta1 * old_m + (1 - beta1) * g
```

`beta1` is the retained fraction, not the new-gradient fraction. The
source uses `beta1 = 0.9`, so it retains 90 percent of the previous
history and gives 10 percent to the current gradient.

This history is a weighted record of observed batch gradients. It
does not prove that the noise has vanished, and a current gradient of
zero does not erase old momentum. The stored `m` fades rather than
stopping in one step.

## Give each coordinate its own scale

Momentum remembers direction, but one learning rate still acts on
coordinates whose gradients can have very different sizes. Consider:

```text
theta = [1.0, 1.0]
g     = [0.1, 10.0]
lr    = 0.1
```

An SGD step changes the first coordinate by `0.01` and the second by
`1.0`. The second change is one hundred times larger.

We need a size history for each coordinate. Squaring removes the sign
while preserving magnitude:

```text
g * g = [0.01, 100.0]
```

Keep a fading history of those squares:

```text
v = beta2 * old_v + (1 - beta2) * g * g
```

This is the **second raw moment**, an EMA of squared gradients. It is
not the variance from Chapter 2: the code does not subtract a mean
before squaring.

For a first look, take `v = g * g`. Its square root restores the units
of the original gradient:

```text
sqrt(v) = [0.1, 10.0]
```

Before reading the division, calculate `0.1 / 0.1` and `10 / 10`.
Both coordinates produce:

```text
g / sqrt(v) = [1.0, 1.0]
```

The `10.0` coordinate no longer receives a change one hundred times
larger merely because its recent gradient scale is larger. This is
**adaptive scaling**: every scalar coordinate has its own divisor.
The divisor comes from squared-gradient history, not from one shared
scale for an entire matrix.

Two failures remain. Both histories begin at zero, so their first
values are pulled toward zero. A coordinate with no gradient history
also tries to divide zero by zero.

## Repair the zero start

Return to a constant gradient of `2` and use `beta = 0.5`. Starting
the first-moment history at zero gives:

```text
m[1] = 0.5 * 0 + 0.5 * 2 = 1
m[2] = 0.5 * 1 + 0.5 * 2 = 1.5
```

The constant input is `2`, yet the stored values are smaller. The
weights placed on the observed gradients add to only:

```text
step 1: 0.5              = 1 - 0.5^1
step 2: 0.25 + 0.5 = 0.75 = 1 - 0.5^2
```

Before reading the corrected values, predict what a successful
rescaling should recover from the constant input `2`. Divide by the
accumulated coefficient mass so those weights sum to one:

```text
m_hat[1] = 1.0 / 0.5  = 2
m_hat[2] = 1.5 / 0.75 = 2
```

The squared-gradient history has the same zero-start problem. Predict
the corrected first value for a constant squared gradient of `4`
before reading it:

```text
v[1]     = 0.5 * 0 + 0.5 * 4 = 2
v_hat[1] = 2 / (1 - 0.5^1)   = 4
```

The two corrected histories are therefore:

```text
m_hat = m / (1 - beta1^step)
v_hat = v / (1 - beta2^step)
```

This is **bias correction**. More exactly, the division renormalizes
the coefficient mass that has accumulated so far. Starting from zero
made that total less than one. Calling the corrected first history an
unbiased estimate assumes a stable average gradient. Making the same
claim for the corrected second history separately assumes a stable
average squared gradient. Real training gradients change, so the
correction does not promise knowledge of a future batch.

The step number must begin at one. At step zero,
`1 - beta^0 = 0`, so both corrections would divide by zero.
[Appendix A](appendix-a-derivations.md#adams-cold-start-correction)
derives the general coefficient sum.

There is one more zero case. If `m_hat = 0` and `v_hat = 0`, the
adaptive fraction would be `0 / 0`. Add a small positive value to the
divisor:

```text
adaptive direction = m_hat / (sqrt(v_hat) + epsilon)
```

The source uses `epsilon = 0.00000001`, or `1e-8`. It sits outside the
square root. This is a separate guard from
[layernorm's epsilon](05-forward-pass.md#keep-one-rows-scale-from-controlling-the-next-operation),
which is `1e-5` and sits inside a different calculation.

We have now built a first-moment history, a second raw moment,
zero-start corrections, per-coordinate scaling, and a finite zero
case. This complete update is **Adam**, short for adaptive moment
estimation:

```text
m       = beta1 * old_m + (1 - beta1) * g
v       = beta2 * old_v + (1 - beta2) * g * g
m_hat   = m / (1 - beta1^step)
v_hat   = v / (1 - beta2^step)
theta   = theta - lr * m_hat / (sqrt(v_hat) + epsilon)
```

## Pull selected values directly toward zero

The implementation adds one policy to Adam. It applies a small pull
toward zero to selected learned values.

Suppose `theta = 1` on step one, with both histories still zero. The
data gradient is zero, the learning rate is `0.1`, and the pull
coefficient is `0.01`. Predict the direct change by multiplying those
three numbers:

```text
decay change = lr * decay * theta
             = 0.1 * 0.01 * 1
             = 0.001
new theta    = 1 - 0.001
             = 0.999
```

A naive attempt adds `decay * theta = 0.01` to the gradient before
Adam sees it. Predict whether per-coordinate adaptive division will
preserve that `0.01` size. On the first step, Adam's corrected
histories give:

```text
m_hat = 0.01
v_hat = 0.01^2 = 0.0001
m_hat / sqrt(v_hat) = 1
```

The adaptive fraction turns the intended `0.01` pull into a direction
near `1`. With learning rate `0.1`, the value moves by about `0.1`
instead of `0.001`.

Keep the pull outside the adaptive fraction:

```text
theta = theta
        - lr * (adaptive_direction + decay * theta)
```

The direct pull is **weight decay**. Keeping it separate from the
adaptive gradient calculation makes it **decoupled weight decay**.
Adam with that policy is **AdamW**, the optimizer used by Tiny AgenC.

The source does not decay values according to their architectural
meaning. Its exact stored-shape rule is:

```text
decay applies when rows > 1 and cols > 1
```

Both dimensions must exceed one. Under the default configuration,
transformation matrices and the token and position tables qualify.
The `1 x C` layernorm gains and biases do not. A valid width-one shape
can behave differently from that architectural summary; the C
predicate, not the label, decides.

## One scalar update under glass

Now put every part into one checkable step. Take a matrix entry with
value `1.0`, gradient `0.2`, learning rate `0.1`, `beta1 = 0.9`,
`beta2 = 0.999`, epsilon `1e-8`, and decay `0.01`.

Before reading the final value, predict whether it will end above or
below `0.9`. The adaptive direction is near `1`, and decay adds a
second positive direction, so it must end below `0.9`.

The histories at step one are:

```text
m = 0.9 * 0 + 0.1 * 0.2
  = 0.02

v = 0.999 * 0 + 0.001 * 0.2^2
  = 0.00004
```

The correction divisors are:

```text
1 - 0.9^1   = 0.1
1 - 0.999^1 = 0.001
```

Correcting the histories gives:

```text
m_hat = 0.02 / 0.1       = 0.2
v_hat = 0.00004 / 0.001 = 0.04
```

Then:

```text
adaptive = 0.2 / (sqrt(0.04) + 0.00000001)
         = 0.2 / 0.20000001
         = about 0.99999995

change = 0.1 * (0.99999995 + 0.01 * 1.0)
       = about 0.100999995

new value = 1.0 - 0.100999995
          = about 0.8990
```

A `1 x C` entry follows the same Adam arithmetic but receives zero
decay under the shape rule, so it ends near `0.9000`. The committed C
path uses `float`, so last digits can vary with compiler
floating-point choices. The rounded hand results are the contract.
The `0.1` learning rate is for this example; the command-line default
remains `0.001`.

## Four aligned buffers make one learnable object

The update needs four persistent numbers for every coordinate:

```text
value       used by the forward pass
gradient    accumulated by the backward pass
first       gradient EMA
second      squared-gradient EMA
```

For a `2 x 2` learned matrix, four separate shapes would be needless
bookkeeping. One backing float allocation can hold four equal regions:

```text
float offset     0   1   2   3 |  4   5   6   7
region         values           | gradient

float offset     8   9  10  11 | 12  13  14  15
region         first moment     | second moment
```

The C object that owns one matrix-shaped set of learned values and
these three same-shaped companions is a **`Param`**. This is different
from one scalar **parameter**, the Chapter 0 name for one adjustable
number. A `Param` can contain many scalar parameters.

The public header keeps the fields private:

```c
typedef struct Param Param;
```

Callers can hold a `Param *`, but only `param.c` sees the definition:

```c
struct Param {
    Mat    values;
    float *gradient;
    float *first_moment;    /* running average of gradients */
    float *second_moment;   /* running average of squared gradients */
};

enum {
    PARAM_VALUE_BUFFER,
    PARAM_GRADIENT_BUFFER,
    PARAM_FIRST_MOMENT_BUFFER,
    PARAM_SECOND_MOMENT_BUFFER,
    PARAM_BUFFER_COUNT,
};
```

`values` is a `Mat` because forward code needs its rows and columns.
The other three regions have the same number of entries, so pointers
are enough inside this private module. The enum names each region's
offset multiplier; its final member counts the four buffers.

Here is the complete private constructor from
[`param.c`](../src/param.c):

```c
static Param *param_new(int rows, int cols)
{
    if (rows < 1 || cols < 1
        || (size_t)rows > SIZE_MAX / (size_t)cols)
        return NULL;

    size_t count = (size_t)rows * (size_t)cols;

    if (count > SIZE_MAX / PARAM_BUFFER_COUNT
        || PARAM_BUFFER_COUNT * count > SIZE_MAX / sizeof(float))
        return NULL;

    Param *p = emalloc(sizeof *p);
    float *store = ecalloc(PARAM_BUFFER_COUNT * count, sizeof *store);

    p->values = mat_make(store + PARAM_VALUE_BUFFER * count, rows, cols);
    p->gradient = store + PARAM_GRADIENT_BUFFER * count;
    p->first_moment = store + PARAM_FIRST_MOMENT_BUFFER * count;
    p->second_moment = store + PARAM_SECOND_MOMENT_BUFFER * count;
    return p;
}
```

The first guard rejects nonpositive shapes and protects `rows * cols`.
The second protects the four-region count and byte multiplication.
Chapter 2 built the bound-before-add pattern. Chapter 3 applied the
corresponding bound-before-product guard.
[Chapter 10](10-memory-planning.md#check-arithmetic-before-performing-it)
turns the pattern into reusable checked addition and multiplication
for the whole model.

`emalloc` allocates the small `Param` structure. `ecalloc` makes a
separate backing float allocation and fills it with zero bytes. The
public Gaussian and constant constructors overwrite the value region;
the gradient and both histories therefore begin at zero. Allocation
exhaustion remains fatal through those utility functions. Returning
`NULL` here is specifically the invalid-shape result.

The four assignments carve the float allocation with
[Chapter 4's pointer offsets](04-poor-mans-tensors.md). Adding
`count` to a `float *` advances by `count` float objects, not by
`count` bytes. The destructor later frees `p->values.vals`, the start
of that float allocation, and then frees the separate `Param`.

Accessors return two `Mat` views:

```c
Mat param_values(const Param *p)
{
    return p->values;
}

Mat param_gradient(const Param *p)
{
    return mat_make(p->gradient, p->values.rows, p->values.cols);
}
```

Returning a `Mat` by value copies its pointer and shape, not the
floats. Writing through the gradient view therefore writes into the
`Param`'s gradient region.

Now that the `Param` fields are visible, here is the source's exact
decay policy:

```c
/* Shapes with more than one row and column get weight decay; gains,
 * biases, and other vectors do not. */
static int parameter_has_matrix_shape(const Param *p)
{
    return p->values.rows > 1 && p->values.cols > 1;
}
```

The helper reads the stored row and column counts and returns true only
when both comparisons succeed. Its name records that this is a shape
test used by the project's decay policy; it does not redefine Chapter
1's mathematical meaning of matrix.

## Check the recipe before changing history

The caller packages the five optimizer settings in this source-exact
structure:

```c
typedef struct {
    float learning_rate;
    float beta1;          /* decay of the gradient average */
    float beta2;          /* decay of the squared-gradient average */
    float epsilon;        /* keeps the divide finite */
    float weight_decay;   /* pull toward zero for eligible matrix shapes */
} AdamW;
```

Passing this small structure by value gives the function its own copy
of the five floats.

The recipe must reject infinities and `NaN`. Chapter 5 constructed
those nonfinite values; this is the first time the book opens the
source's exact bit test:

```c
static int finite_float(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof bits);
    return (bits & FLOAT_EXPONENT_MASK) != FLOAT_EXPONENT_MASK;
}
```

`memcpy` copies the float's 32 stored bits into the fixed-width
unsigned integer without asking C to reinterpret one pointer type as
another. The file-level `FLOAT_EXPONENT_MASK` is `0x7F800000u`; it
selects all eight exponent bits in the float representation supported
by this program. An
all-ones exponent encodes either an infinity or `NaN`. Bitwise `&`
keeps those exponent bits, and the comparison returns false when they
are all one.

The recipe and step number are valid only under these exact tests:

```c
int param_adamw_recipe_valid(AdamW opt, int step)
{
    return step >= 1
        && finite_float(opt.learning_rate) && opt.learning_rate >= 0.0f
        && finite_float(opt.beta1) && opt.beta1 >= 0.0f && opt.beta1 < 1.0f
        && finite_float(opt.beta2) && opt.beta2 >= 0.0f && opt.beta2 < 1.0f
        && finite_float(opt.epsilon) && opt.epsilon > 0.0f
        && finite_float(opt.weight_decay) && opt.weight_decay >= 0.0f;
}
```

The learning rate and decay may be zero. Each beta includes zero but
not one. Epsilon must be positive. Every setting must be finite.
The `finite_float` helper above supplies each finite-value test.

The corrections, shape-selected decay, and bounds are the same for
every entry. Passing each one as a separate helper argument would make
it easy to swap two values. Recomputing them inside the entry loop
would do needless work. The source gives the values one private
record:

```c
typedef struct {
    AdamW  opt;
    size_t count;
    float  correction1;
    float  correction2;
    float  decay;
    double safe;
    double inverse_correction1;
    double inverse_correction2;
    double inverse_epsilon;
} PreparedAdamW;
```

`opt` is the checked recipe. `count` is the number of entries. The
next three floats are the two bias corrections and the selected
decay. The four doubles belong only to the conservative arithmetic
proof. `PreparedAdamW` is the prepared step record, one read-only
bundle shared by the scan and the later update.

This source-exact helper constructs it:

```c
static int adamw_prepare(PreparedAdamW *prepared, const Param *p,
                         AdamW opt, int step)
{
    if (!param_adamw_recipe_valid(opt, step))
        return 0;

    prepared->opt = opt;
    prepared->count = mat_size(p->values);
    prepared->decay =
        parameter_has_matrix_shape(p) ? opt.weight_decay : 0.0f;
    prepared->correction1 = 1.0f - powf(opt.beta1, (float)step);
    prepared->correction2 = 1.0f - powf(opt.beta2, (float)step);

    if (!adamw_corrections_valid(prepared))
        return 0;
    prepared->safe = (double)FLT_MAX / ADAMW_SAFE_RANGE_DIVISOR;
    prepared->inverse_correction1 = 1.0 / (double)prepared->correction1;
    prepared->inverse_correction2 = 1.0 / (double)prepared->correction2;
    prepared->inverse_epsilon = 1.0 / (double)prepared->opt.epsilon;
    return 1;
}
```

The first guard checks the public recipe before filling the record.
The next five assignments copy the recipe, count the entries, choose
the decay, and compute both corrections. `powf(base, exponent)` is the
float version of raising a number to a power. The cast supplies `step`
in the float type that `powf` expects. The file-level
`ADAMW_SAFE_RANGE_DIVISOR` names the factor of four used to leave room
below `FLT_MAX`.

The correction helper accepts two finite positive values:

```c
static int adamw_corrections_valid(const PreparedAdamW *step)
{
    return finite_float(step->correction1) && step->correction1 > 0.0f
        && finite_float(step->correction2) && step->correction2 > 0.0f;
}
```

A correction that rounded to zero or became nonfinite stops
preparation. Only after that check does `adamw_prepare` divide by the
corrections and epsilon. `<float.h>` supplies `FLT_MAX`, the largest
finite `float`; one quarter of it is the proof's conservative ceiling.

A later entry could still contain a nonfinite value or history. If the
function changed entry zero before discovering that bad entry, one
`Param` would be left half updated. The source therefore scans every
entry without writing to the `Param`.

Two small helpers check stored state and the candidate moments:

```c
static int adamw_stored_entry_valid(const Param *p, size_t i,
                                    float gradient)
{
    return finite_float(gradient) && finite_float(p->values.vals[i])
        && finite_float(p->first_moment[i])
        && finite_float(p->second_moment[i])
        && p->second_moment[i] >= 0.0f;
}

static int adamw_candidate_moments_valid(const PreparedAdamW *step,
                                         float first, float second)
{
    return finite_float(first) && finite_float(second) && second >= 0.0f
        && fabs((double)first) <= step->safe
        && (double)second <= step->safe;
}
```

The first rejects a nonfinite gradient, value, or old history. A
squared-gradient history cannot be negative. The second applies the
same checks to the two moments that this step would produce and keeps
their magnitudes below the prepared ceiling. Casting `first` to
`double` makes `fabs` compute its absolute value in double precision;
Chapter 7's `fabsf` was the float version.

The next helper proves that correction, division, decay, and the final
subtraction stay within that ceiling:

```c
static int adamw_update_bounds_valid(const Param *p,
                                     const PreparedAdamW *step, size_t i,
                                     float first, float second)
{
    double variance_bound = (double)second * step->inverse_correction2;
    double smoothed_bound =
        fabs((double)first) * step->inverse_correction1;
    double ratio_bound = smoothed_bound * step->inverse_epsilon;
    double decay_bound =
        (double)step->decay * fabs((double)p->values.vals[i]);
    double direction_bound = ratio_bound + decay_bound;
    double change_bound =
        (double)step->opt.learning_rate * direction_bound;

    return (double)step->opt.epsilon
               <= step->safe / ADAMW_EPSILON_RANGE_DIVISOR
        && variance_bound <= step->safe
        && smoothed_bound <= step->safe
        && ratio_bound <= step->safe
        && decay_bound <= step->safe
        && direction_bound <= step->safe
        && change_bound <= step->safe
        && fabs((double)p->values.vals[i]) + change_bound <= step->safe;
}
```

Each local variable bounds the next operation in source order:
corrected variance, corrected first moment, division by epsilon,
decay, their sum, and the learning-rate-scaled change. The final test
also includes the old value. These double calculations prove that the
original float update will stay finite. The file-level
`ADAMW_EPSILON_RANGE_DIVISOR` names the factor of two in the epsilon
bound. These calculations do not replace that update with double
arithmetic.

The complete read-only scan is now a short coordinator:

```c
static int adamw_inputs_valid(const Param *p, const PreparedAdamW *step)
{
    for (size_t i = 0; i < step->count; i++) {
        float gradient = p->gradient[i];
        float first = step->opt.beta1 * p->first_moment[i]
                    + (1.0f - step->opt.beta1) * gradient;
        float second = step->opt.beta2 * p->second_moment[i]
                     + (1.0f - step->opt.beta2) * gradient * gradient;

        if (!adamw_stored_entry_valid(p, i, gradient)
            || !adamw_candidate_moments_valid(step, first, second)
            || !adamw_update_bounds_valid(p, step, i, first, second))
            return 0;
    }
    return 1;
}
```

The loop computes the exact candidate moments for one entry, then
asks the three helpers about stored state, candidate state, and update
bounds. Logical `||` returns failure when any check fails. Reaching
the final `return 1` means every entry passed and nothing in the
`Param` changed.

Only then may one entry commit:

```c
static void adamw_apply_entry(Param *p, const PreparedAdamW *step, size_t i)
{
    float gradient = p->gradient[i];

    p->first_moment[i]  = step->opt.beta1 * p->first_moment[i]
                        + (1.0f - step->opt.beta1) * gradient;
    p->second_moment[i] = step->opt.beta2 * p->second_moment[i]
                        + (1.0f - step->opt.beta2) * gradient * gradient;

    float smoothed = p->first_moment[i] / step->correction1;
    float spread   = sqrtf(p->second_moment[i] / step->correction2);

    p->values.vals[i] -= step->opt.learning_rate
                       * (smoothed / (spread + step->opt.epsilon)
                          + step->decay * p->values.vals[i]);
}
```

The helper reads one gradient, updates its two histories, corrects
them, and then changes the matching value. `sqrtf` restores the scale
of the squared-gradient history. Epsilon is outside that square root.
Decay is outside the adaptive fraction but still multiplied by the
learning rate. The gradient is not cleared here.

The commit loop has one job and preserves ascending entry order:

```c
static void adamw_apply(Param *p, const PreparedAdamW *step)
{
    for (size_t i = 0; i < step->count; i++)
        adamw_apply_entry(p, step, i);
}
```

The public function now shows the whole transaction:

```c
int param_adamw_step(Param *p, AdamW opt, int step)
{
    PreparedAdamW prepared;

    if (!adamw_prepare(&prepared, p, opt, step)
        || !adamw_inputs_valid(p, &prepared))
        return -1;
    adamw_apply(p, &prepared);
    return 0;
}
```

Preparation and the complete scan both precede the only call that can
mutate the `Param`. A failure returns `-1`; a successful commit returns
zero.

`param_adamw_step` is all-or-nothing for this one `Param`: invalid
input returns before any entry commits. That guarantee does not extend
across the complete model. Chapter 9's `model_step` visits `Param`
objects in order. Corrupt internal state in a later object can be
discovered after earlier objects moved, and clipping may already have
changed gradients. The model-wide operation is deliberately not
advertised as a transaction.

## Put one limit around all gradients

One batch can occasionally produce a much larger gradient vector than
nearby batches. If that spike enters the histories unchanged, its
square enters the second moment too. The model therefore measures all
parameter gradients together before any AdamW moment update.

Construct the measurement from the two-entry vector `[3, 4]`:

```text
squared length = 3^2 + 4^2
               = 9 + 16
               = 25

length = sqrt(25)
       = 5
```

This square root of the sum of squared entries is the **gradient
norm**. Tiny AgenC's maximum is `1`.

Predict the one shared scale needed to turn norm `5` into norm `1`.
It is `1 / 5 = 0.2`:

```text
[3, 4] * 0.2 = [0.6, 0.8]
sqrt(0.6^2 + 0.8^2) = sqrt(1) = 1
```

Multiplying every coordinate by the same positive number preserves
direction: `0.6 / 0.8` is still `3 / 4`. This operation is **global
gradient clipping**. If the norm is already at most one, including a
zero norm, the code returns without division or scaling. Only a norm
above one uses `1 / norm`.

Clipping each `Param` independently would change the whole vector.
For two `Param` gradients `[3, 0]` and `[0, 4]`, separate unit caps
produce `[1, 0, 0, 1]`. The original nonzero ratio `3:4` becomes
`1:1`. One global scale instead produces `[0.6, 0, 0, 0.8]`.

The local helpers in `param.c` expose the required pieces:

```c
float param_gradient_norm_squared(const Param *p)
{
    size_t count = mat_size(p->values);
    double total = 0.0;

    for (size_t i = 0; i < count; i++)
        total += (double)p->gradient[i] * (double)p->gradient[i];
    return (float)total;
}

void param_scale_gradient(Param *p, float factor)
{
    size_t count = mat_size(p->values);

    for (size_t i = 0; i < count; i++)
        p->gradient[i] *= factor;
}
```

The first loop casts before multiplying so its running sum is
`double`, then returns the source's float result. The second loop
applies exactly one factor to every entry in the `Param`.

Chapter 9 constructs the canonical list needed to combine these local
pieces. Its model-level path performs:

```text
reject an invalid optimizer recipe
reject a nonfinite gradient
measure one norm across every Param
if norm > 1, scale every gradient by the same factor
visit each Param with AdamW
```

The ordinary path preserves the source's float-rounded per-`Param`
sums. If that norm becomes nonfinite, `model.c` rescans every element
into one `double` sum and uses a double scale before casting each
result back to float. This lets very large finite gradients be clipped.
No committed evidence records how often either path runs.

Clipping bounds the gradient vector entering the moment updates. It
does not by itself bound the final parameter change. The learning
rate, existing histories, epsilon, and decay still participate.
`param_adamw_step` separately preflights the candidate arithmetic.

## Save values, not the training history

Chapter 3 constructed exact binary reads and writes. A `Param` uses
the same boundary for its value region:

```c
int param_write(const Param *p, FILE *stream)
{
    size_t count = mat_size(p->values);

    if (!parameter_values_are_finite(p))
        return -1;
    return fwrite(p->values.vals, sizeof *p->values.vals, count, stream)
               == count
        ? 0 : -1;
}

int param_read(Param *p, FILE *stream)
{
    size_t count = mat_size(p->values);

    if (fread(p->values.vals, sizeof *p->values.vals, count, stream) != count)
        return -1;
    return parameter_values_are_finite(p) ? 0 : -1;
}
```

The writer first rejects a nonfinite value, then succeeds only if
`fwrite` reports every requested float. The reader likewise requires a
complete payload and finite resulting values.

These functions transfer values only. They do not write the gradient,
either moment, the optimizer step, or the shape. A complete finite
payload round-trips the value bits. A failed read is not
transactional: `fread` may already have changed part or all of the
destination. Chapter 13 builds the containing checkpoint boundary by
loading into a candidate model that can be discarded on failure.

Omitting moments means a saved model can generate text, but it cannot
resume the exact in-memory AdamW history from that checkpoint. That is
the deliberate format implemented by this repository.

## Why the optimizer needs a different witness

Chapter 7 compared a derivative with a measured slope. AdamW is not a
new derivative. It is a stateful update rule whose history changes at
every step.

The Chapter 8 witness therefore maintains an independent
double-precision transcription of the published update. It supplies
changing gradients for several steps, asks the real float code to
advance, and compares the resulting values. It runs one `2 x 3`
matrix with decay and one `1 x 3` row without decay. Separate checks
exercise zeroing, scaling, value-only I/O, invalid shapes, invalid
recipes, nonfinite input, and a later unsafe entry that must prevent
an earlier entry from moving. The witness then repairs that unsafe
entry and compares its next update with an untouched control. The
comparison would fail if a rejected attempt had committed candidate
moments that changed the next update. The read-only source scan
establishes the stronger claim that neither moment array is written.

Predict what would happen if both the implementation and its witness
folded decay into the adaptive gradient. They could agree with each
other while both tested the wrong rule. Independence here means the
reference separately transcribes the intended equations and keeps
decay outside the adaptive fraction.

## Build checkpoint: move one parameter

**Build.** Implement `Param` allocation and accessors, gradient zeroing
and scaling, the local squared-norm helper, finite-gradient detection,
value-only `param_write` and `param_read`, recipe validation, and
fallible AdamW in `param.c`. Start the optimizer step at one. Put
epsilon outside the square root and shape-selected decay outside the
adaptive fraction. Preflight every entry before committing one.

The global clipping idea is complete here, but its loop needs the
model-wide parameter list. Chapter 9 implements that list and
`model_step`.

**Verify.**

```sh
make -C labs check-08
# answer key: make check-optimizer
```

**Expected.** The lab prints
`check-08: all 44 optimizer checks passed`. Several steps agree with an
independent double-precision reference. A `2 x 3` matrix decays and a
`1 x 3` row does not. Gradient zeroing and uniform scaling affect every
entry without changing shape. Finite parameter values round-trip bit
for bit, short reads fail, and nonfinite writes are rejected.
Nonpositive shapes return `NULL`; the constructor code also guards
unrepresentable shape arithmetic, though this chapter's two advertised
commands do not exercise that separate case. Invalid recipes and
unsafe candidate arithmetic leave one `Param`'s values and histories
unchanged. A zero current gradient after earlier nonzero steps remains
finite; history may still move its value.

**Common failures.**

- A nonfinite first step often began at step zero and divided by
  `1 - beta^0`.
- Decay that changes the moments was folded into the gradient instead
  of applied outside the adaptive fraction.
- A `1 x C` row shrinking under decay means the literal two-dimension
  shape test was replaced by a semantic guess.
- A rejected later entry leaving earlier entries changed means
  validation and commit happened in the same loop.
- A short read expected to preserve old values assumes a transaction
  that `param_read` does not provide.
- A zero current gradient expected to freeze the value forgot the
  first-moment history.

One `Param` can now turn a checked gradient into a guarded update.
Chapter 9 constructs every `Param` in one canonical model-wide order,
then supplies the loop that clips and visits them all.

---

[Previous: Trust, but Verify](07-trust-but-verify.md) | [Contents](README.md) | [Next: Parameters and the Private Blueprint](09-parameters-and-the-blueprint.md)
