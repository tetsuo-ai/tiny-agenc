# Chapter 7: Trust, but Verify

Chapter 6 wrote the operation gradients by hand. Here is the
uncomfortable truth about hand-written backprop: **a wrong gradient can
still reduce the loss.**

Take one adjustable number `p` and this toy loss:

```text
loss = (p - 1)^2
```

At `p = 2`, the loss is `1`. [Chapter 6's power and chain
rules](06-backprop-by-hand.md#carry-one-change-through-the-next) give
a loss slope of `2`. Suppose a missing factor makes the backward code
claim that the slope is `1`.

Before reading the new loss, predict whether subtracting this
incomplete but same-sign slope can still lower it.

Now make a toy move that subtracts `0.1` times that claimed slope:

```text
p before      = 2
claimed slope = 1
p after       = 2 - 0.1*1
              = 1.9

loss before   = (2.0 - 1)^2
              = 1
loss after    = (1.9 - 1)^2
              = 0.81
```

Predict the verdict from a check that asks only whether the loss fell.
It says the code worked. The claimed slope is still wrong by a factor
of two.

The falling loss reveals a useful direction here. It does not verify
the direction's exact size, and a larger model can contain mistakes
that affect only some inputs or cancel for a while. Training behavior
cannot referee each handwritten formula.

We need a second route to the slope, one that runs the forward
arithmetic and never reads the backward formula.

## Measure a slope from nearby values

Use a function whose exact slope is already known from Chapter 6:

```text
f(x) = x^3
```

At `x = 2`, the power rule says the slope is

```text
3*x^2 = 3*2^2 = 12
```

Pretend that answer is hidden. The forward function can still be run
near `2`. Start with a nudge `h = 0.1`:

```text
f(2)       = 2^3
           = 8

f(2 + 0.1) = 2.1^3
           = 9.261
```

Predict whether the interval's slope will be above or below the
one-point slope `12`.

The observed change per unit nudge is

```text
(9.261 - 8) / 0.1 = 12.61
```

This estimate uses the value at the point and one value on one side.
It is a **one-sided difference**. Its `12.61` is in the neighborhood
of `12`, but the curved function became steeper over the interval from
`2` to `2.1`. The estimate includes that bend.

Measure equally far on both sides instead:

```text
f(2 + 0.1) = 2.1^3 = 9.261
f(2 - 0.1) = 1.9^3 = 6.859
```

Predict whether the average slope across this balanced interval will
be above or below `12.61`. The calculation is

```text
(9.261 - 6.859) / (2*0.1)

= 2.402 / 0.2

= 12.01
```

The balanced estimate is much closer to `12`. Now shrink the nudge by
a factor of ten. Predict whether the remaining error will fall by a
factor nearer to ten or one hundred:

```text
f(2 + 0.01) = 2.01^3 = 8.120601
f(2 - 0.01) = 1.99^3 = 7.880599

(8.120601 - 7.880599) / (2*0.01)

= 0.240002 / 0.02

= 12.0001
```

The error fell from `0.01` to `0.0001`. The algebra shows why. Expand
the two sides without skipping a term:

```text
(2 + h)^3 = 8 + 12*h + 6*h^2 + h^3
(2 - h)^3 = 8 - 12*h + 6*h^2 - h^3
```

The one-sided calculation keeps every extra term:

```text
((2 + h)^3 - 2^3) / h

= (12*h + 6*h^2 + h^3) / h

= 12 + 6*h + h^2
```

Its error is `6*h + h^2`. For small `h`, the `6*h` part dominates.
Dividing `h` by ten therefore divides that leading error by about ten.

The balanced subtraction cancels the shared constant and the matching
`6*h^2` terms:

```text
((2 + h)^3 - (2 - h)^3) / (2*h)

= ((8 + 12*h + 6*h^2 + h^3)
   - (8 - 12*h + 6*h^2 - h^3)) / (2*h)

= (24*h + 2*h^3) / (2*h)

= 12 + h^2
```

Its leftover error is exactly `h^2` for this cubic. Reducing `h` by
ten reduces `h^2` by one hundred:

```text
0.1^2  = 0.01
0.01^2 = 0.0001
```

The shorthand `O(h^2)` records the scale derived for this cubic: for
sufficiently small `h`, the error is bounded by some constant times
`h^2`. It does not say that every function has an error equal to
`h^2`. Another function needs its own error analysis.

An error caused by using a nonzero interval in place of the limiting
slope is **truncation error**. An estimate whose leading truncation
error scales as `h^2` has **second-order error**. The balanced formula
is the **central difference**:

```text
numeric slope = (f(x + h) - f(x - h)) / (2*h)
```

Using nearby forward values to estimate a derivative is the method of
**finite differences**. The test will compare this independently
measured slope with the slope returned by backpropagation.

## A smaller nudge eventually loses information

The last table suggests that `h` should keep shrinking forever.
Finite-precision storage stops that argument.

Use a toy machine that keeps seven significant decimal digits. Around
`x = 2`, take `h = 0.00000001`. The exact cubic values begin as

```text
f(2 + h) = 8.000000120000...
f(2 - h) = 7.999999880000...
```

After this toy machine rounds both values to seven significant digits,
it stores

```text
f(2 + h) = 8.000000
f(2 - h) = 8.000000
```

Predict the measured numerator. It is zero, so this machine reports a
slope of zero instead of twelve.

[Chapter 2 built the corresponding limit for the source's binary
`float`](02-foundations.md#turn-32-bits-into-a-fraction). When two
nearby rounded values are subtracted, their shared leading digits
disappear. The small remainder carries fewer reliable digits. With a
still smaller nudge, `x + h` or `x - h` can even round back to `x`
before the function runs.

This loss of useful digits when nearly equal floating-point values are
subtracted is **floating-point cancellation**. The nudge now has two
opposing jobs:

```text
h too large  -> truncation error from measuring across a wide interval
h too small  -> rounding and cancellation erase the small difference
```

There is no universally best `h`. It depends on the precision, the
calculation, and the scale of the values. The source settles on

```c
static const float NUDGE = 1e-2f;   /* h: clears float noise */
```

for these `float` operation fixtures. This is the same `0.01` that
gave `12.0001` for the cubic. That example explains the scale; the
test's varied operation inputs determine whether it is adequate.

## Turn a vector output into one measurable number

Central differences measure one scalar output. Most model operations
return a matrix. We need one scalar that depends on every output
entry, with a gradient that can be supplied to the backward function.

Start with a two-entry output:

```text
out = [3, 4]
```

Choose another fixed row:

```text
u = [2, -1]
```

Before reading the result, multiply matching entries and add, using
[Chapter 5's dot
product](05-forward-pass.md#the-two-loops-under-the-machine):

```text
measured value = out dot u
               = 3*2 + 4*(-1)
               = 6 - 4
               = 2
```

Nudge only `out[0]` by `0.1`:

```text
new value = 3.1*2 + 4*(-1)
          = 2.2

change per unit nudge = (2.2 - 2) / 0.1
                      = 2
```

Nudge only `out[1]` by `0.1`:

```text
new value = 3*2 + 4.1*(-1)
          = 1.9

change per unit nudge = (1.9 - 2) / 0.1
                      = -1
```

Predict the gradient with respect to `out`. It is `[2, -1]`, exactly
`u`. Therefore one choice solves both sides of the check:

```text
forward measurement:     value = out dot u
backward arriving value:  d_out = u
```

The source fills `u` with [Chapter 2's Gaussian
draws](02-foundations.md#turn-two-flat-draws-into-a-bell). Varied
signs and sizes avoid a fixture where every output is asked to move in
the same direction. `CHECK_SEED` is fixed at `42`, and each case
allocates its `u` once. The values are random when the fixture is
constructed and fixed while every `x + h` and `x - h` pair is
measured.

Mapping a vector or matrix to one scalar by its dot product with a
fixed random vector is a **random projection**. Here the projection is
a test instrument. It does not alter the operation being tested.

The source implementation is:

```c
static float projected(Mat out, Mat u)
{
    float sum = 0.0f;

    for (size_t i = 0; i < mat_size(out); i++)
        sum += out.vals[i] * u.vals[i];
    return sum;
}
```

[Chapter 2's file-private
`static`](02-foundations.md#build-a-generator-from-state) keeps
`projected` private to `gradcheck.c`. Its two [Chapter 4 `Mat`
views](04-poor-mans-tensors.md#put-the-shape-beside-the-address) are
copied descriptions of existing storage. `sum` starts the dot-product
accumulator at zero. The loop visits every output entry, multiplies it
by the matching fixed projection entry, and adds the product. The
return line hands the one scalar to the nudge driver. The case builders
provide `out` and `u` with matching sizes.

A projection can hide an error that happens to cancel in its chosen
direction. For example, an output error `[1, 2]` has dot product zero
with `[2, -1]`. Gaussian entries make exact accidental cancellation
unlikely for ordinary mistakes, but one projection is evidence rather
than a proof about every possible arriving output gradient.

Cross-entropy needs no projection. Its forward function already
returns one scalar loss, so its measurement function returns that loss
directly.

## Let one driver call different operations

A first nudge driver could call `gelu_forward` directly. Matmul,
layernorm, and attention would then need copies of the same save,
nudge, measure, restore, and compare loop. Copies invite drift in the
referee itself.

The common loop instead receives a small operation-specific function.
Each such function reruns one forward operation and returns its scalar
measurement.

GELU's measurement needs an output matrix, the fixed projection, and
the input being nudged. Those values live in `Case`. The complete
source struct also carries fields used by the other operations. This
shortened view shows only GELU's fields:

```c
/* Shortened from Case: only the fields used by measure_gelu. */
typedef struct {
    Mat out, u;
    Mat x;
} Case;
```

`out` is writable forward storage. `u` is the fixed projection, and
`x` is the target input view. The GELU case builder then fills those
three named fields:

```c
/* Shortened from check_gelu: construction of its Case. */
enum { ROWS = 4, COLS = 5 };
Case c = { .out = mat_new_zeros(ROWS, COLS),
           .u   = mat_new_gaussian(rng, ROWS, COLS),
           .x   = mat_new_gaussian(rng, ROWS, COLS) };
```

The [Chapter 2 `enum`
pattern](02-foundations.md#read-bytes-without-trusting-the-file) gives
`ROWS` and `COLS` local integer names. `Case c` creates one struct
value. Each `.field = value` entry says which field receives the value,
so this list need not follow the declaration order. `out` receives
zeroed output storage. `u` and `x` receive varied Gaussian values.
Fields from the complete `Case` that are not listed begin with zero
values and are not used by this GELU fixture.

This field-named brace list is a **designated struct initializer**.
The leading dot selects a field, echoing [Chapter 2's struct member
access](02-foundations.md#fixed-integers-and-a-clock-that-does-not-turn-back),
while `=` supplies its initial value.

The real measurement function is:

```c
static float measure_gelu(void *context)
{
    Case *c = context;

    gelu_forward(c->out, c->x);
    return projected(c->out, c->u);
}
```

The parameter type `void *` is [Chapter 2's generic object
address](02-foundations.md#a-block-of-bytes-and-one-owner). It lets
the common driver carry an address without knowing the case's fields.
Inside this GELU-specific function, `Case *c = context` recovers the
known pointer type. C permits this conversion from `void *` without an
explicit cast. The next line reruns the real forward operation with the
current, possibly nudged `x`. The return line projects the new output
against the same `u`.

The driver receives that function through this parameter:

```c
float (*measure)(void *)
```

Read it from the name outward:

```text
measure             is
*measure            a pointer to a function
(*measure)(void *)  taking one void pointer
float (...)         and returning a float
```

A value that stores which function to call is a **function pointer**.
When the common driver later invokes that supplied function, the
function is serving as a **callback**.

GELU connects its backward result, target input, callback, and case in
two source lines:

```c
gelu_backward(d_x, c.u, c.x);
nudge_all("gelu d_x", c.x, d_x, measure_gelu, &c);
```

The first line computes the analytic input gradient using `u` as
`d_out`. In this source, **analytic** means computed from the
handwritten backward formula. It does not mean known correct. The
second line asks the common driver to nudge `c.x` and compare against
`d_x`. `measure_gelu` appears without parentheses, so C passes its
function address instead of calling it there. `&c` is the address of
the case object. The driver later calls `measure_gelu(&c)` through the
function pointer.

The same `context` address must remain valid for every callback. Here
`c` lives in `check_gelu` until all of its nudges finish, so its
[Chapter 2 lifetime](02-foundations.md#a-block-of-bytes-and-one-owner)
covers the calls.

## Decide when two float answers agree

Exact equality would reject harmless rounding. A single fixed margin
also fails across scales. Consider a margin of `0.001`.

For slopes near zero, an absolute margin is useful:

```text
analytic = 0.0040
numeric  = 0.0048
difference = 0.0008
```

The values differ by less than `0.001`. Now consider slopes near
three:

```text
analytic = 3.00
numeric  = 3.05
difference = 0.05
```

The fixed `0.001` margin is tiny relative to either value even though
their proportional disagreement is about two percent.

Using only a proportional margin has the opposite failure near zero:
two tiny float-noise values can differ by a large fraction while both
remain close to the intended zero.

The source adds both allowances. Its exact constants are:

```c
static const float RELATIVE_TOLERANCE = 2e-2f;
static const float ABSOLUTE_TOLERANCE = 1e-3f;
```

For analytic answer `a` and numeric answer `n`, it constructs

```text
allowed = 0.001 + 0.02*max(abs(a), abs(n))
```

Return to the two examples. Near zero:

```text
allowed = 0.001 + 0.02*0.0048
        = 0.001096

difference 0.0008 <= 0.001096, so it passes
```

Near three:

```text
allowed = 0.001 + 0.02*3.05
        = 0.062

difference 0.05 <= 0.062, so it passes
```

Predict the effect of doubling both answers in the second example.
The relative part doubles while the `0.001` floor stays fixed.

An absolute floor plus an allowance proportional to the values'
magnitude is a **mixed tolerance**. The acceptance rule is exactly

```text
abs(a - n) <= absolute tolerance
              + relative tolerance*max(abs(a), abs(n))
```

Here is the complete source comparison:

```c
static void compare(const char *label, float analytic, float numeric)
{
    float allowed = ABSOLUTE_TOLERANCE
                  + RELATIVE_TOLERANCE * fmaxf(fabsf(analytic), fabsf(numeric));

    checks++;
    if (fabsf(analytic - numeric) <= allowed)
        return;
    printf("FAIL %-28s analytic % .6f  numeric % .6f\n",
           label, (double)analytic, (double)numeric);
    failures++;
}
```

The signature receives a diagnostic label and the two slopes.
`fabsf` takes a `float` absolute value. `fmaxf` selects the larger
magnitude. The first statement implements the exact sum above.

`checks++` records that one comparison ran. The `if` computes the
absolute disagreement; an accepted comparison returns without
printing. A rejected comparison reaches `printf`. The `%-28s` field
left-aligns the label in 28 columns, and `% .6f` prints a sign space
and six digits after the decimal point. The explicit casts match the
`double` values consumed under [Chapter 2's variadic call
rule](02-foundations.md#one-place-decides-how-to-stop). Finally,
`failures++` records the failed comparison.

The tolerance creates a resolution limit. A bug whose effect stays
inside the allowance can pass. Tightening either constant can instead
reject correct formulas because of central-difference and float error.
The chosen values are part of this fixture's engineering contract,
not a theorem about every model.

## Save, nudge, restore, compare

All pieces now fit in the common driver:

```c
static void nudge_all(const char *label, Mat target, Mat analytic,
                      float (*measure)(void *), void *context)
{
    for (size_t i = 0; i < mat_size(target); i++) {
        float saved = target.vals[i];

        target.vals[i] = saved + NUDGE;
        float above = measure(context);

        target.vals[i] = saved - NUDGE;
        float below = measure(context);

        target.vals[i] = saved;
        compare(label, analytic.vals[i], (above - below) / (2.0f * NUDGE));
    }
}
```

Read the signature one piece at a time. `label` names failures.
`target` is a matrix view of the values to perturb. `analytic` is the
same-shaped gradient produced by backward. `measure` is the callback
whose function-pointer shape was built above. `context` is the case
address to pass back to that callback.

The loop visits every target entry. `mat_size(target)` supplies its
flattened element count. A one-element spot check could miss a wrong
row index or a failed accumulation path; this loop gives each entry a
turn.

`saved` captures the exact original `float`. Assigning through
`target.vals[i]` changes the borrowed backing storage seen by the
forward operation.

The first assignment writes the original value plus `NUDGE`.
`measure(context)` then runs the operation and returns the projected
forward value named `above`.

The second assignment starts from `saved` again and writes the
original value minus `NUDGE`. It does not subtract one nudge from the
already raised value. The callback returns the matching `below`
measurement.

The restore assignment puts back the exact saved bits before any other
entry is tested. Without it, later comparisons would measure a
function with several inputs changed.

The final call forms the central difference and compares it with the
matching analytic entry. Parentheses around `2.0f * NUDGE` make the
full separation between `x + h` and `x - h` the denominator.

Predict what happens if the projection vector is regenerated inside
the callback. `above` and `below` would be values of two different
scalar functions, so their difference would no longer estimate either
function's derivative.

## Labels for a later parameter loop

The same source file contains a C idiom that first becomes active in
this book's whole-model gradient checkpoint in Chapter 12. That loop
needs a different diagnostic label for each numbered parameter. Its
relevant source lines are:

```c
enum { LABEL_CAPACITY = 32 };

char label[LABEL_CAPACITY];

snprintf(label, sizeof label, "model param[%d]", i);
```

The `enum` gives the capacity a compile-time integer name. The second
line allocates a 32-character array for one label. The third formats
the current integer `i` into text such as `model param[3]`.

[Chapter 2's
`sizeof`](02-foundations.md#a-block-of-bytes-and-one-owner) reports
the whole array's byte capacity because `label` is still an array in
this scope. `snprintf` writes no more than that [Chapter 3 buffer
capacity](03-data.md#encode-without-inventing-a-token), including the
terminating zero byte that marks the end of a C string. If the
formatted result is too long, it is truncated; this fixed format and
an `int` index fit the chosen capacity.

Producing formatted text with a supplied destination capacity is
**bounded formatted output**. The C library function used here is
`snprintf`. Chapter 7 owns this first syntax contact because the line
is visible while reading `tests/gradcheck.c`; Chapter 12 owns the
whole-model test that uses the label.

## The operation checks

One backward call with fixed `u` produces every analytic input slope
for an operation. `nudge_all` then asks the forward route for those
slopes one target entry at a time.

The operation group in `tests/gradcheck.c` covers these pairings:

| Forward measurement | Nudged target | Analytic result |
|---|---|---|
| matmul projection | input `x` | `d_x` |
| matmul projection | weights | `d_w` |
| layernorm projection | input `x` | `d_x` |
| layernorm projection | gain | `d_gain` |
| layernorm projection | bias | `d_bias` |
| attention projection | packed QKV | `d_qkv` |
| GELU projection | input `x` | `d_x` |
| residual projection | input `a` | `d_a` |
| residual projection | input `b` | `d_b` |
| embedding projection | token table | `d_tokens` |
| embedding projection | position table | `d_positions` |
| cross-entropy scalar loss | logits | `d_logits` |

Layernorm and attention run their measurement callback once before
backward so the backward call sees means, reciprocal standard
deviations, or attention weights from the matching forward
calculation. Cross-entropy does the same to fill its probabilities.
The later nudge measurements may overwrite this scratch, but the
analytic gradient has already been computed.

Repeated embedding ids, causal attention prefixes, and shared
layernorm gain and bias make the fixtures exercise accumulation. The
fixed Gaussian inputs also give the comparisons mixed signs and
scales.

The Chapter 7 lab command selects only this `backward` operation group:

```text
./labs/build/check-06 backward
```

The lab's Makefile invokes that group for you. The independent test for
AdamW, the parameter updater previewed in Chapter 1, waits for
[Chapter 8](08-adamw.md). The whole-model parameter test, including the
`snprintf` labels above, waits for [Chapter
12](12-wiring-the-model-backward.md) after the model's backward wiring
exists. Chapter 7 makes no claim from either deferred group.

## Two referees watch different calculations

Suppose the intended forward operation is

```text
y = x^2
```

At `x = 2`, a hand-calculated forward example expects `y = 4`. Now
imagine that the implementation accidentally computes

```text
y = x^3
```

and its backward function consistently claims the cubic slope
`3*x^2`. At `x = 2`, the implementation returns `8`, so the
hand-calculated expected value catches it.

What does the central-difference check see? It measures the cubic
forward implementation around `2` and gets a slope near `12`. The
backward implementation also returns `12`. That derivative comparison
passes even though both pieces implement the wrong equation.

Reverse the mistake. If forward correctly computes `x^2` while
backward returns the cubic slope, the expected forward value `4`
passes and the derivative comparison fails near `4` versus `12`.

The first witness is a **known-answer forward test**: it checks whether
forward computes an independently worked expected value. The second is
a **derivative check**: it checks whether backward differentiates the
forward calculation that actually ran. Each can pass while the other
contract is broken.

The answer-key commands keep the witnesses separate:

```sh
make check-forward
make check-backward
```

`make check-forward` runs the hand-calculated operation witnesses from
Chapter 5. `make check-backward` runs the operation finite differences
constructed here.

## What agreement licenses

When the operation group passes, the analytic gradients agree with
central-difference measurements of the implemented forward operations,
at the fixed finite fixtures, along the fixed projection directions,
within the mixed tolerance.

Every qualifier matters. The check does not establish any of these
broader claims:

- that forward implements the intended equation;
- that the formulas agree for every finite input or invalid input;
- that differences smaller than the tolerance are absent;
- that every possible output-gradient direction agrees;
- that a value disconnected from both forward and backward was meant
  to be disconnected;
- that the operation kernels are wired correctly through the whole
  model;
- that AdamW updates parameters correctly.

A disconnected target can produce numeric zero and analytic zero. A
separate test must establish that an intended path exists. One random
projection can also miss an error whose dot product with its chosen
direction is zero. More seeds and directions can strengthen that
evidence, while never turning finite fixtures into a proof for all
inputs.

[Chapter 8](08-adamw.md) supplies the independent AdamW witness.
[Chapter 12](12-wiring-the-model-backward.md) supplies the whole-model
derivative check and constructs a separate
[parameter-connectivity
witness](12-wiring-the-model-backward.md#find-a-completely-unused-parameter-object)
after the forward and backward assembly is available.

## Read a failure as a pattern

One failed number names an entry. Several failures often expose the
shape of the defect:

| Pattern | First place to inspect |
|---|---|
| analytic and numeric values have opposite signs | a local sign |
| one repeated table row is too small | a lost `+=` contribution |
| every cross-entropy entry has one scale error | the mean over rows |
| only attention query entries fail | the query score path and scale |
| early attention keys fail | accumulation from later visible queries |
| small values fail while large ones pass | the absolute allowance |
| every deliberate mutation passes | tolerance or target connection |
| repeated runs change | RNG ownership or a parallel data race |

The pattern narrows the next source line to inspect. It does not select
an edit by itself. Return to the matching forward equation and trace
the analytic and measured paths separately.

## Build checkpoint: make the referee object

**Build.** Read the supplied random-projection driver in
[`tests/gradcheck.c`](../tests/gradcheck.c). Trace GELU from its fixed
`u`, through `gelu_backward`, through `measure_gelu`, and into
`nudge_all`. Before running it, write down which layernorm comparisons
you expect a missing centering or spread term to break. The referee
stays separate from learner-owned `ops.c`, so the witness does not
reuse its derivative formulas. Chapter 7 builds a test technique
rather than a runtime module and intentionally reuses the Chapter 6
lab target.

**Verify.**

```sh
make -C labs check-06
# answer key: make check-forward check-backward
```

Then, in your learner workspace, temporarily remove
`- d_norm_mean` from the `d_input[c]` update in
`layernorm_backward`. The edit still compiles. Run `check-06` again
and predict that the `layernorm d_x` comparisons will fail. Restore
the term immediately. As an extension, write a new finite-difference
driver for residual addition before reading that operation's supplied
measurement callback.

**Expected.** The intact operation group passes. Removing
`- d_norm_mean` makes `layernorm d_x` fail, and restoring the term
restores the suite. `check-forward` answers the known-value question;
`check-backward` answers the derivative question. No fixed aggregate
count is part of this chapter's contract. AdamW and the whole-model
parameter group are not required here.

**Common failures.**

- Perturbing `x + h` and then subtracting `h` once measures `x + h`
  against `x`, not the balanced pair `x + h` and `x - h`.
- Regenerating the projection between `above` and `below` changes the
  scalar function being differentiated.
- Calling backward again after each nudge can accumulate extra
  analytic gradients unless their destinations are cleared.
- Forgetting to restore the exact saved value contaminates later
  comparisons.
- Testing one convenient entry leaves row indexing and accumulation
  paths unchecked.
- Comparing floats for exact equality mistakes rounding noise for a
  derivative error.
- Loosening the tolerance until a mutation passes removes the
  referee's ability to object.

---

[Previous: Backprop by Hand](06-backprop-by-hand.md) | [Contents](README.md) | [Next: AdamW](08-adamw.md)
