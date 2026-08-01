# Appendix A: The Derivations Under the Loops

Chapter 6 gives the useful formulas and the code that implements them.
This appendix opens the formulas and leaves every gear on the table.
Nothing new is being added to the model. We are only slowing the chain
rule down enough to watch it become C.

You do not need to carry all of this algebra in your head while reading
the main story. The point is more practical: when a gradient looks like
magic, you should be able to come here, walk from the forward equation
to the backward loop, and find that the magic was bookkeeping.

The main chapters already built the needed pieces:

```text
Chapter 2   exp, log, mean, and variance
Chapter 5   the forward equations
Chapter 6   slopes, gradients, and the local backward rules
Chapter 7   an independent finite-difference referee
Chapter 8   the parameter update
Chapter 12  the order that joins all local rules into one model
```

This appendix is a slower second route through the local arithmetic. It
does not replace any of those owner chapters, and it is not a
prerequisite for continuing through the book.

The source routes covered here are:

| Route | Source function |
|---|---|
| residual addition | [`residual_backward`](../src/ops.c) |
| token and position lookup | [`embedding_backward`](../src/ops.c) |
| matrix multiplication | [`matmul_backward`](../src/ops.c) |
| GELU | [`gelu_backward`](../src/ops.c) |
| softmax | [`softmax_backward_in_place`](../src/ops.c) |
| cross-entropy | [`crossentropy_backward`](../src/ops.c) |
| layer normalization | [`layernorm_backward`](../src/ops.c) |
| causal attention | [`attention_head_backward`](../src/ops.c) |
| AdamW update | [`param_adamw_step`](../src/param.c) |

The recurring route is:

```text
small numbers -> one local slope -> every returning path -> one C loop
```

## Read the notation from a calculation

### One scalar path

Start with numbers rather than symbols:

```text
x = 2
y = 3*x = 6
L = y*y = 36
```

If `y` rises by a small amount, `L` rises at about `2*y = 12`
times that amount. If `x` rises by a small amount, `y` rises at
`3` times that amount. The two rates multiply:

```text
loss rate at y     = 12
y rate at x        =  3
loss rate at x     = 12 * 3 = 36
```

Chapter 6 built and named this multiplication the
[chain rule](06-backprop-by-hand.md#carry-one-change-through-the-next).
The compact gradient spelling is:

```text
d_y = partial L / partial y = 12
d_x = partial L / partial x = d_y * partial y / partial x
                            = 12 * 3
                            = 36
```

The `d_` prefix is not a different kind of number. `d_x` records how
fast the final loss would change if `x` changed. Backward starts with
`d_L = 1` because raising `L` by one raises `L` by one.

### Several paths meet

Now let two outputs use the same two inputs:

```text
x[0] = 1
x[1] = 2

y[0] =  2*x[0] + x[1]   = 4
y[1] =   -x[0] + 3*x[1] = 5

L = 4*y[0] - 2*y[1] = 6
```

The loss rates arriving at the two outputs are:

```text
d_y[0] =  4
d_y[1] = -2
```

Each output has one local slope for each input:

```text
                         input changed
                       x[0]    x[1]
output y[0]              2       1
output y[1]             -1       3
```

Predict the sign of `d_x[0]`. Raising `x[0]` raises `y[0]`, which the
loss rewards, and lowers `y[1]`, whose arriving rate is negative. Both
paths raise the loss, so the result must be positive:

```text
d_x[0] = 4*2 + (-2)*(-1) = 10
d_x[1] = 4*1 + (-2)*3    = -2
```

Chapter 6 names a complete table of local slopes a
[Jacobian](06-backprop-by-hand.md#a-softmax-row-moves-together). The
source rarely stores such a table. It walks the entries that matter and
adds every returning path into the destination.

The memory rule follows from the arithmetic:

```text
                    +--> one use -----+
shared value -------+                 +--> later loss
                    +--> another use -+

d_shared = contribution from one use
         + contribution from another use
```

This is why public input and parameter gradients use `+=`. Assignment
would erase a contribution that arrived earlier. Chapter 6 states the
full [accumulation
contract](06-backprop-by-hand.md#a-method-you-can-reuse).

## Residual addition: the smallest fork

Forward residual addition is:

```text
out[i] = a[i] + b[i]
```

Changing either input entry by one changes the matching output by one.
Both local slopes are therefore `1`. A backward pass that sends the
arriving gradient to only one input loses a real path. A backward pass
that assigns into a destination loses anything already there.

Use destinations that already contain contributions:

```text
d_out       = [ 2, -3]
d_a before  = [10,  1]
d_b before  = [-4,  5]
```

**Predict:** what must the two destinations contain after this
residual route returns?

```text
d_a after = [12, -2]
d_b after = [-2,  2]
```

The symbolic route has no hidden step:

```text
partial out[i] / partial a[i] = 1
partial out[i] / partial b[i] = 1

d_a[i] += d_out[i] * 1
d_b[i] += d_out[i] * 1
```

The source is the same arithmetic:

```c
void residual_backward(Mat d_a, Mat d_b, Mat d_out)
{
    size_t count = mat_size(d_out);

    for (size_t i = 0; i < count; i++) {
        d_a.vals[i] += d_out.vals[i];
        d_b.vals[i] += d_out.vals[i];
    }
}
```

`mat_size` flattens the arriving matrix into one element count. The
loop reads each arriving entry once and adds it to both destinations.
Neither forward input is needed because addition's slope never depends
on its input value.

The function does not assert matching shapes. Its caller must provide
valid destinations with the same element count. Chapter 6 first walks
this [residual
route](06-backprop-by-hand.md#residual-the-highway-works-in-reverse).

## Embedding: reverse a row selection

Embedding forward selects one token row and one position row, then adds
them. Assignment seems plausible in backward until a row is selected
twice.

Use two sequences of length `T = 2`, one channel per table row, and
four flattened output gradients:

```text
flattened row       0   1   2   3
sequence            0   0   1   1
position            0   1   0   1
token id            0   1   0   0
d_out               2  -1   3   4
```

Token row `0` served flattened rows `0`, `2`, and `3`.

**Predict:** what total returns to token row `0`?

```text
d_token_table[0] += 2 + 3 + 4 = 9
d_token_table[1] += -1
```

Positions repeat once per sequence:

```text
d_position_table[0] += 2 + 3  = 5
d_position_table[1] += -1 + 4 = 3
```

For any table cell, the rule is now visible: add the matching channel
from every output row that selected that table row. Token ids and
position numbers select routes. They are integers, not float values
whose slopes are being measured.

The complete source is:

```c
void embedding_backward(Mat d_token_table, Mat d_position_table, Mat d_out,
                        const int *tokens, int time)
{
    for (int row = 0; row < d_out.rows; row++) {
        const float *d_output = mat_row(d_out, row);

        add_scaled(mat_row(d_token_table, tokens[row]), 1.0f, d_output, d_out.cols);
        add_scaled(mat_row(d_position_table, row % time), 1.0f, d_output, d_out.cols);
    }
}
```

The loop first locates one arriving row. The first `mat_row` destination
uses the same `tokens[row]` selection as forward. The second uses
`row % time`, so positions restart at the next sequence. `add_scaled`
with scale `1.0f` adds every channel without erasing a repeated use.

The loop is serial because different flattened rows may choose the
same token or position destination. Running those additions
concurrently without coordination would create a data race. Chapter 6
builds this [add-back
behavior](06-backprop-by-hand.md#lookups-return-corrections-to-table-rows);
Chapter 5 owns the [forward
lookup](05-forward-pass.md#give-each-id-a-row-of-opening-notes).

## Matrix multiplication: where the two sums come from

Chapter 5 constructs Tiny AgenC's
[matrix multiplication](05-forward-pass.md#make-every-output-from-one-input-row)
with inputs shaped `R x I`, weights shaped `O x I`, and output shaped
`R x O`:

```text
out[r, o] = sum over k of x[r, k] * weights[o, k]
```

The weight rows are output channels, so this is the storage form of
`out = X * W^T`.

Copying `d_out` into `d_x` cannot be the general backward rule. Their
column counts are `O` and `I`, which may differ, and the weights decide
how strongly each input affected each output. Updating a weight from
only one input row also fails because the same weight served every row.
The backward pass therefore needs two different collections of
returning paths.

Use the same two-by-two check as Chapter 6:

```text
X = [ 2  -1 ]       W = [  3  4 ]
    [ 1   3 ]           [ -2  1 ]

out = [ 2  -5 ]
      [15   1 ]

d_out = [ 5  -3 ]
        [ 2   4 ]
```

First follow one input entry, `x[1,0] = 1`. It helped make both output
channels in row `1`. Their local slopes are the matching weights:

```text
d_x[1,0] = d_out[1,0] * weights[0,0]
           + d_out[1,1] * weights[1,0]

         = 2*3 + 4*(-2)
         = -2
```

Now follow one weight, `weights[0,1] = 4`. It served output channel `0`
for both input rows:

```text
d_weights[0,1] = d_out[0,0] * x[0,1]
                 + d_out[1,0] * x[1,1]

               = 5*(-1) + 2*3
               = 1
```

**Predict:** which direction supplies each remaining sum. An input
entry collects over output channels. A weight entry collects over input
rows. Completing those sums gives:

```text
d_x = [21  17]
      [-2  12]

d_weights = [12   1]
            [-2  15]
```

Now hold the indices as symbols. For a fixed `x[r,k]`, outputs in other
rows do not depend on it. Every output channel in row `r` does:

```text
partial out[r,o] / partial x[r,k] = weights[o,k]

d_x[r,k]
    += sum over o of d_out[r,o] * weights[o,k]
```

For a fixed `weights[o,k]`, every input row uses it in output channel
`o`:

```text
partial out[r,o] / partial weights[o,k] = x[r,k]

d_weights[o,k]
    += sum over r of d_out[r,o] * x[r,k]
```

Those are the two row stages and their public loops:

```c
static void matmul_accumulate_input_row(Mat d_x, Mat d_out, Mat weights,
                                        int row)
{
    const float *d_output = mat_row(d_out, row);
    float       *d_input  = mat_row(d_x, row);

    for (int o = 0; o < weights.rows; o++)
        add_scaled(d_input, d_output[o], mat_row(weights, o), weights.cols);
}

static void matmul_accumulate_weight_row(Mat d_weights, Mat d_out, Mat x,
                                         Mat weights, int output)
{
    float *d_neuron = mat_row(d_weights, output);

    for (int row = 0; row < x.rows; row++) {
        const float *d_output = mat_row(d_out, row);

        add_scaled(d_neuron, d_output[output], mat_row(x, row), weights.cols);
    }
}

void matmul_backward(Mat d_x, Mat d_weights, Mat d_out, Mat x, Mat weights)
{
    /* d_x[r] += sum_o d_out[r][o] * weights[o]: rows are independent. */
    #pragma omp parallel for if(x.rows >= PARALLEL_THRESHOLD)
    for (int row = 0; row < x.rows; row++)
        matmul_accumulate_input_row(d_x, d_out, weights, row);

    /* d_weights[o] += sum_r d_out[r][o] * x[r]: output channels are
     * independent, so this loop parallelizes without collisions. */
    #pragma omp parallel for if(weights.rows >= PARALLEL_THRESHOLD)
    for (int o = 0; o < weights.rows; o++)
        matmul_accumulate_weight_row(d_weights, d_out, x, weights, o);
}
```

The first helper owns one `d_x` row. Each output channel adds its
arriving rate times one weight row. The second owns one `d_weights`
row and walks every input row that used that output channel. The
public loops assign those independent rows to OpenMP iterations.

The backward function does not repeat forward's shape assertions. The
caller must supply:

```text
x and d_x                 R x I
weights and d_weights     O x I
d_out                     R x O
```

It must also supply non-overlapping storage with valid capacities. The
numeric construction and code receive their first full walk in Chapter
6's [matmul backward
section](06-backprop-by-hand.md#matmul-sends-contributions-to-inputs-and-weights).

## GELU: unwind the nested scalar

GELU treats every matrix entry independently. Its forward approximation
uses:

```text
K = 0.7978845608
A = 0.044715

inner = K * (x + A*x^3)
t      = tanh(inner)
y      = 0.5*x*(1 + t)
```

Passing `d_out` through unchanged would claim the curve has slope one
everywhere.

**Predict:** at `x = 0`, should the GELU slope be `0`, `0.5`, or `1`?
Commit to one before following the two product paths:

```text
inner = 0
tanh(inner) = 0
y = 0
```

The factor `x` supplies one path with slope `0.5*(1 + 0) = 0.5`.
The path through `t` is multiplied by `x = 0`, so it contributes zero.
The total GELU slope at zero is exactly `0.5`.

At `x = 1`, rounded intermediate values are:

```text
inner          = about 0.833562
t              = about 0.682384
inner slope    = about 0.904917
y              = about 0.841192
```

The number to explain is a GELU slope of about `1.082964`. With an
arriving `d_out = 0.5`, backward must add about `0.541482` to `d_x`.

Chapter 6 constructs the
[product, power, and `tanh` slope
rules](06-backprop-by-hand.md#the-gelu-bend-changes-the-returning-slope).
Apply them one local path at a time. The inside is:

```text
inner = K * (x + A*x^3)

partial inner / partial x
    = K * (1 + A*3*x^2)
    = K * (1 + 3*A*x^2)
```

The `tanh` path is:

```text
partial t / partial inner = 1 - t^2

partial t / partial x
    = (1 - t^2) * K * (1 + 3*A*x^2)
```

The final product `0.5*x*(1+t)` has two paths:

```text
partial y / partial x
    = 0.5*(1 + t)
      + 0.5*x * partial t / partial x

    = 0.5*(1 + t)
      + 0.5*x*(1 - t^2)*K*(1 + 3*A*x^2)
```

Substitute the `x = 1` values:

```text
slope
    = 0.5*(1 + 0.68238398)
      + 0.5*1*(1 - 0.68238398^2)*0.90491679

    = about 1.08296408
```

The complete source keeps exactly those intermediates:

```c
void gelu_backward(Mat d_x, Mat d_out, Mat x)
{
    size_t count = mat_size(x);

    for (size_t i = 0; i < count; i++) {
        float value   = x.vals[i];
        float inner   = GELU_SQRT_2_OVER_PI * (value + GELU_CUBIC_COEFF * value * value * value);
        float tanh_of = tanhf(inner);
        float d_inner =
            GELU_SQRT_2_OVER_PI
            * (1.0f + CUBIC_DERIVATIVE_FACTOR * GELU_CUBIC_COEFF
                          * value * value);
        float slope = GELU_HALF * (1.0f + tanh_of)
                    + GELU_HALF * value * (1.0f - tanh_of * tanh_of)
                          * d_inner;

        d_x.vals[i] += slope * d_out.vals[i];
    }
}
```

`count` flattens the matrix because no entry consults another entry.
`value`, `inner`, and `tanh_of` replay the forward calculation.
`CUBIC_DERIVATIVE_FACTOR` names the power-rule factor three, and
`GELU_HALF` names the repeated half. `d_inner` is the first local slope
derived above. The two terms in `slope` are the two product paths. The
final line multiplies by the arriving loss rate and accumulates.

For large negative ordinary finite inputs, the slope approaches zero.
For large positive ordinary finite inputs, it approaches one.
Extremely large finite arithmetic can overflow, and this kernel does
not validate nonfinite inputs. Chapter 5 owns the
[forward curve](05-forward-pass.md#put-a-bend-between-widen-and-narrow).

## Softmax backward without a square table

Softmax makes every output depend on every input score through one
shared total. Sending each arriving gradient only to the matching score
would miss that coupling.

Start with two scores whose exponentials are easy to check:

```text
score          = [0, log(3)]
exp(score)     = [1, 3]
shared total   = 4
weight         = [0.25, 0.75]
```

Chapter 2 builds [`exp` and
`log`](02-foundations.md#growth-and-its-undo). Raising `score[0]`
changes its exponential share at rate `1`. It also changes the shared
total at rate `1`. The quotient rule gives:

```text
partial weight[0] / partial score[0]
    = (1*4 - 1*1) / 4^2
    =  0.1875

partial weight[1] / partial score[0]
    = (0*4 - 3*1) / 4^2
    = -0.1875
```

Raising `score[1]` changes its exponential share at rate `3`:

```text
partial weight[0] / partial score[1] = -0.1875
partial weight[1] / partial score[1] =  0.1875
```

The complete local-slope table is:

```text
                              input score
                              0        1
output weight[0]         0.1875  -0.1875
output weight[1]        -0.1875   0.1875
```

Let the later operation return:

```text
d_weight = [2, -2]
```

Multiply the arriving rates by each input column:

```text
d_score[0] = 2*0.1875 + (-2)*(-0.1875) =  0.75
d_score[1] = 2*(-0.1875) + (-2)*0.1875 = -0.75
```

The square table helped reveal the coupling, but allocating one for
every softmax row would waste work. The same result comes from one
shared number:

```text
coupled = 0.25*2 + 0.75*(-2)
        = -1

d_score[0] = 0.25 * ( 2 - (-1)) =  0.75
d_score[1] = 0.75 * (-2 - (-1)) = -0.75
```

Now derive that collapse for any row. Let:

```text
Z         = sum over k of exp(score[k])
weight[i] = exp(score[i]) / Z
```

Fix one input `score[j]`. There are two cases. For its matching output,
both the numerator and `Z` change:

```text
partial weight[j] / partial score[j]

    = (exp(score[j])*Z - exp(score[j])*exp(score[j])) / Z^2

    = weight[j] * (1 - weight[j])
```

For a different output `i`, its numerator does not change, but `Z`
still does:

```text
partial weight[i] / partial score[j]

    = (0*Z - exp(score[i])*exp(score[j])) / Z^2

    = -weight[i] * weight[j]
```

The gradient for `score[j]` collects the matching path and every
different-output path:

```text
d_score[j]

    = d_weight[j] * weight[j] * (1 - weight[j])
      + sum over i != j of
        d_weight[i] * (-weight[i] * weight[j])

    = weight[j]
      * (d_weight[j]
         - d_weight[j]*weight[j]
         - sum over i != j of d_weight[i]*weight[i])

    = weight[j]
      * (d_weight[j] - sum over i of weight[i]*d_weight[i])
```

The final sum is the `coupled` value:

```text
coupled    = dot(weight, d_weight)
d_score[j] = weight[j] * (d_weight[j] - coupled)
```

This slope table is the
[Jacobian Chapter 6
constructs](06-backprop-by-hand.md#a-softmax-row-moves-together). The
source evaluates its effect without storing the table:

```c
static void softmax_backward_in_place(float *d_weights, const float *weights, int count)
{
    float coupled = dot(weights, d_weights, count);

    for (int i = 0; i < count; i++)
        d_weights[i] = weights[i] * (d_weights[i] - coupled);
}
```

On entry, `weights` holds the saved softmax probabilities and
`d_weights` holds the arriving gradient. `dot` computes `coupled`
while every arriving entry is still intact. The loop then overwrites
that private scratch row with `d_score`. Moving the dot product inside
the overwrite loop would mix old and new values.

**Predict:** what should all entries of `d_score` add to? Use the
general formula:

```text
sum over j of d_score[j]

    = sum over j of weight[j]*d_weight[j]
      - coupled * sum over j of weight[j]

    = coupled - coupled*1
    = 0
```

Adding the same amount to every input score changes no softmax
probability, so the loss has zero rate in that direction. Float
arithmetic makes the observed sum approximately zero. A weight that
has rounded to exactly zero also receives an exactly zero score
gradient in this float calculation. Chapter 5 owns the
[stable forward
softmax](05-forward-pass.md#turn-arbitrary-scores-into-usable-shares).

## Cross-entropy: the target path and the shared total

Cross-entropy does not grade each logit independently. Raising any
logit changes the shared softmax total, while raising the target logit
also changes the term that rewards the known next token.

Start from probabilities so the signs are visible. Use two rows and
three choices:

```text
probabilities              target
[0.2, 0.3, 0.5]               2
[0.6, 0.1, 0.3]               0
```

Before averaging the rows, each non-target logit receives its
probability. The target logit receives its probability minus one:

```text
row 0   [ 0.2, 0.3, -0.5]
row 1   [-0.4, 0.1,  0.3]
```

**Predict:** what changes when forward returns the mean of `R = 2`
row losses? Every entry is divided by two:

```text
d_logits =
[ 0.10, 0.15, -0.25]
[-0.20, 0.05,  0.15]
```

Each row still sums to zero. The target entry is negative unless its
probability has reached one. Following the negative gradient during
an update therefore tends to raise the target score and lower the
others.

Now derive the two paths. For one row, write:

```text
Z = sum over c of exp(logit[c])

row_loss = -log(exp(logit[target]) / Z)
         = -log(exp(logit[target])) + log(Z)
         = -logit[target] + log(Z)
```

For any fixed channel `c`, the shared-total path is:

```text
partial log(Z) / partial logit[c]

    = (1/Z) * partial Z / partial logit[c]
    = exp(logit[c]) / Z
    = probability[c]
```

The target path has two cases:

```text
partial (-logit[target]) / partial logit[c]
    = -1    when c == target
    =  0    otherwise
```

Add the shared and target paths:

```text
partial row_loss / partial logit[c]
    = probability[c] - 1    when c == target
    = probability[c]        otherwise
```

Tiny AgenC returns:

```text
L = (1/R) * sum over rows of row_loss[row]
```

so the final source rule is:

```text
d_logits[row,c]
    += (probability[row,c] - target_mark[c]) / R

target_mark[c] is 1 when c is the target, and 0 otherwise.
```

That construction maps directly to one row helper and its caller:

```c
void crossentropy_backward(Mat d_logits, Mat probs, const int *targets)
{
    float mean_scale = 1.0f / (float)probs.rows;

    for (int row = 0; row < probs.rows; row++)
        crossentropy_backward_row(mat_row(d_logits, row),
                                  mat_row(probs, row), targets[row],
                                  probs.cols, mean_scale);
}

static void crossentropy_backward_row(float *d_logits, const float *probs,
                                      int target, int count,
                                      float mean_scale)
{
    for (int channel = 0; channel < count; channel++) {
        float target_probability = channel == target ? 1.0f : 0.0f;

        d_logits[channel] +=
            (probs[channel] - target_probability) * mean_scale;
    }
}
```

`mean_scale` is `1/R`. The outer loop selects matching probability,
target, and gradient rows. The row helper's conditional constructs the
`target_mark` value under the name `target_probability`. The final line
performs the two-path subtraction, mean scaling, and accumulation.
Chapter 6 first builds this [bet-sheet
grade](06-backprop-by-hand.md#send-the-bet-sheet-grade-back-to-logits).

### Why forward does not take `log(probability[target])`

The derivation used the unshifted total `Z` because it makes the paths
easy to see. Computing that form in `float` can overflow. Forward first
subtracts the largest row logit `maximum`:

```text
shifted_total
    = sum over c of exp(logit[c] - maximum)

row_loss
    = log(shifted_total) + maximum - logit[target]
```

Every shifted exponential is at most one. A stored target probability
can still underflow to zero, but the loss does not take `log(0)`.
This route stays finite when the resulting mean loss is representable
as a `float`; an extraordinarily large finite penalty can still
overflow during the final conversion. Chapter 5 gives the full
[stable forward
boundary](05-forward-pass.md#cross-entropy-keeping-score).

The maximum shift changes the arithmetic route, not the real-valued
function. Its backward rule remains the two cases above.

### The longer cancellation through probabilities

There is a second way to reach the same result. It is useful for
checking the coupling, but it is exact-real algebra only. Real-number
softmax probabilities are positive. A stored float probability can be
zero, and the C backward path never divides by it.

For the exact-real route:

```text
d_probability[target] = -1 / probability[target]
d_probability[c]      =  0 for every other c
```

The softmax `coupled` value is:

```text
coupled
    = probability[target] * (-1 / probability[target])
    = -1
```

At the target:

```text
d_logit[target]
    = probability[target]
      * (-1/probability[target] - (-1))

    = probability[target] - 1
```

At every other channel:

```text
d_logit[c]
    = probability[c] * (0 - (-1))
    = probability[c]
```

The cancellation confirms the fused result. The implementation uses
the direct `probability - indicator` loop because it remains meaningful
when a stored probability has rounded to zero.

## Layernorm backward, one path at a time

Layernorm works independently on each row, but not independently on
each channel. A naive backward rule,

```text
d_x[c] = d_out[c] * gain[c] * rstd
```

keeps only the direct path. It misses that changing one input also
changes the row mean and the row spread, which affect every channel.

Start with the numeric row from Chapter 6:

```text
x          = [1, 2, 3]
mean       = 2
variance   = 2/3
epsilon    = 0.00001
rstd       = about 1.2247357
norm       = about [-1.2247357, 0, 1.2247357]
gain       = [1, 1, 1]
d_out      = [1, -1, 2]
d_norm     = [1, -1, 2]
```

Chapter 2 constructs
[mean and variance](02-foundations.md#measure-a-sets-center-and-spread).
The two shared averages needed by backward are:

```text
mean(d_norm)
    = (1 + -1 + 2) / 3
    = 0.6666667

mean(d_norm ⊙ norm)
    = (-1.2247357 + 0 + 2.4494714) / 3
    = about 0.4082452
```

The `⊙` symbol is Chapter 6's
[Hadamard-product
reminder](06-backprop-by-hand.md#layernorm-couples-every-channel-in-a-row):
multiply matching entries and keep every result.

The complete input answer is:

```text
d_x[c] = rstd
         * (d_norm[c]
            - mean(d_norm)
            - norm[c] * mean(d_norm ⊙ norm))

d_x = about [1.0206039, -2.0412261, 1.0206223]
```

**Predict:** what should the three entries sum to? A common shift of
the whole input row is removed by its mean, so the answer should be
zero. The unrounded values sum to about `0.0000000`.

The symbolic derivation below explains every term in that answer.

### Draw the three routes

For one row with `C` channels, forward saves `mean` and `rstd`:

```text
mean        = (1/C) * sum over c of x[c]
centered[c] = x[c] - mean
variance    = (1/C) * sum over c of centered[c]^2
rstd        = 1 / sqrt(variance + epsilon)
norm[c]     = centered[c] * rstd
out[c]      = gain[c] * norm[c] + bias[c]
```

One input reaches the output through three routes:

```text
x[c] --------+------> centered values ------> normalized values --> out
             |              |                         ^
             +--> mean -----+                         |
                            +--> variance --> rstd ----+
```

Backward unwinds the gain and bias first, then `rstd`, variance, and
centering.

### Gain and bias

The final forward line uses matching channels:

```text
partial out[c] / partial gain[c] = norm[c]
partial out[c] / partial bias[c] = 1
partial out[c] / partial norm[c] = gain[c]
```

Therefore:

```text
d_gain[c] += d_out[c] * norm[c]
d_bias[c] += d_out[c]
d_norm[c]  = d_out[c] * gain[c]
```

The same gain and bias serve every row, so their gradients accumulate.
For the numeric row:

```text
d_gain added = about [-1.2247357, 0, 2.4494714]
d_bias added = [1, -1, 2]
```

Call `d_norm[c]` by the shorter name `g[c]` during the remaining
algebra.

### Through reciprocal standard deviation

The direct path through

```text
norm[c] = centered[c] * rstd
```

contributes:

```text
d_centered_direct[c] = rstd * g[c]
```

Every normalized channel also uses the shared `rstd`, so:

```text
d_rstd = sum over c of g[c] * centered[c]
```

To find how `rstd` changes with variance, write:

```text
s = variance + epsilon
rstd * rstd * s = 1
```

Differentiate the left side with respect to `variance`. Either `rstd`
factor can change, and `s` changes at rate one:

```text
rstd*s * partial rstd / partial variance
+ rstd*s * partial rstd / partial variance
+ rstd*rstd*1
= 0
```

Combine the first two terms:

```text
2*rstd*s * partial rstd / partial variance + rstd^2 = 0
```

Move `rstd^2` to the other side and divide:

```text
partial rstd / partial variance
    = -rstd^2 / (2*rstd*s)
    = -rstd / (2*s)
```

Because `rstd^2*s = 1`, `1/s = rstd^2`:

```text
partial rstd / partial variance
    = -(1/2) * rstd^3
```

That shared route gives:

```text
d_variance
    = -(1/2) * rstd^3
      * sum over j of g[j]*centered[j]
```

Variance uses every centered channel:

```text
variance = (1/C) * sum over j of centered[j]^2

partial variance / partial centered[c]
    = (2/C) * centered[c]
```

Add its contribution to the direct path:

```text
d_centered[c]

    = rstd*g[c]
      - (centered[c]*rstd^3/C)
        * sum over j of g[j]*centered[j]
```

Use:

```text
norm[c] = centered[c]*rstd

mean(g ⊙ norm)
    = (rstd/C) * sum over j of g[j]*centered[j]
```

to obtain:

```text
d_centered[c]
    = rstd
      * (g[c] - norm[c]*mean(g ⊙ norm))
```

### Through centering

Every centered entry shares the row mean:

```text
centered[j] = x[j] - mean
mean        = (1/C) * sum over k of x[k]
```

Fix one input `x[c]`. Its matching `centered[c]` has a direct slope of
one and a mean slope of `-1/C`:

```text
partial centered[c] / partial x[c] = 1 - 1/C
```

Every different `centered[j]` receives only the mean path:

```text
partial centered[j] / partial x[c] = -1/C, when j != c
```

Collect both cases:

```text
d_x[c]

    = d_centered[c]*(1 - 1/C)
      + sum over j != c of d_centered[j]*(-1/C)

    = d_centered[c]
      - (1/C) * sum over j of d_centered[j]

    = d_centered[c] - mean(d_centered)
```

The centered row has mean zero, so the normalized row also has mean
zero:

```text
mean(norm) = rstd * mean(centered) = 0
```

Average the earlier `d_centered` formula:

```text
mean(d_centered)

    = rstd
      * (mean(g) - mean(norm)*mean(g ⊙ norm))

    = rstd * mean(g)
```

Substitute it:

```text
d_x[c]

    = rstd
      * (g[c]
         - mean(g)
         - norm[c]*mean(g ⊙ norm))
```

Restore `g = d_norm`. This is the complete compact formula:

```text
d_x[c]

    = rstd
      * (d_norm[c]
         - mean(d_norm)
         - norm[c]*mean(d_norm ⊙ norm))
```

For channel zero in the numeric row:

```text
d_x[0]

    = 1.2247357
      * (1 - 0.6666667 - (-1.2247357*0.4082452))

    = about 1.0206039
```

### Epsilon and the constant-row edge

Epsilon remains inside the saved `rstd`; it did not disappear from the
derivative. The normalized row satisfies:

```text
mean(norm^2) = variance / (variance + epsilon)
```

This value is below one. It is close to one only when variance is much
larger than epsilon. A constant row has:

```text
variance       = 0
norm           = [0, 0, ..., 0]
mean(norm^2)   = 0
rstd           = 1/sqrt(0.00001)
               = about 316.2278
```

The finite `rstd` lets an ordinary finite gradient pass through the
centered route. This function does not validate nonfinite or
overflowing inputs.

### The compact formula in C

The source locals map one for one:

| Derivation | C local |
|---|---|
| `norm[c]` | `norm` |
| `g[c] = d_norm[c]` | `d_norm` |
| `mean(g)` | `d_norm_mean` |
| `mean(g ⊙ norm)` | `d_norm_norm_mean` |
| `rstd` | `rstds[row]` |

The following shortened excerpt joins the backward records and row
stages. Forward-only layernorm code between these pieces in `ops.c` is
omitted:

```c
typedef struct {
    Mat          d_x;
    float       *d_gain;
    float       *d_bias;
    Mat          d_out;
    Mat          x;
    const float *gain;
    const float *means;
    const float *rstds;
} LayernormBackward;

typedef struct {
    const float *input;
    const float *d_output;
    float       *d_input;
    float       *d_gain;
    float       *d_bias;
    const float *gain;
    int          channels;
    float        mean;
    float        rstd;
} LayernormBackwardRow;

typedef struct {
    float d_norm_mean;
    float d_norm_norm_mean;
} LayernormGradientMeans;

static LayernormGradientMeans layernorm_gradient_means(
    const LayernormBackwardRow *row)
{
    LayernormGradientMeans means = {
        .d_norm_mean = 0.0f,
        .d_norm_norm_mean = 0.0f,
    };

    for (int c = 0; c < row->channels; c++) {
        float norm   = (row->input[c] - row->mean) * row->rstd;
        float d_norm = row->d_output[c] * row->gain[c];

        means.d_norm_mean      += d_norm;
        means.d_norm_norm_mean += d_norm * norm;
    }
    means.d_norm_mean      /= (float)row->channels;
    means.d_norm_norm_mean /= (float)row->channels;
    return means;
}

static void layernorm_accumulate_row_gradients(
    const LayernormBackwardRow *row, LayernormGradientMeans means)
{
    for (int c = 0; c < row->channels; c++) {
        float norm   = (row->input[c] - row->mean) * row->rstd;
        float d_norm = row->d_output[c] * row->gain[c];

        row->d_input[c] +=
            row->rstd
            * (d_norm - means.d_norm_mean
               - norm * means.d_norm_norm_mean);
        row->d_gain[c] += row->d_output[c] * norm;
        row->d_bias[c] += row->d_output[c];
    }
}

static void layernorm_backward_row(const LayernormBackward *backward,
                                   int row)
{
    LayernormBackwardRow backward_row = {
        .input = mat_row(backward->x, row),
        .d_output = mat_row(backward->d_out, row),
        .d_input = mat_row(backward->d_x, row),
        .d_gain = backward->d_gain,
        .d_bias = backward->d_bias,
        .gain = backward->gain,
        .channels = backward->x.cols,
        .mean = backward->means[row],
        .rstd = backward->rstds[row],
    };
    LayernormGradientMeans means =
        layernorm_gradient_means(&backward_row);

    layernorm_accumulate_row_gradients(&backward_row, means);
}

void layernorm_backward(Mat d_x, float *d_gain, float *d_bias, Mat d_out,
                        Mat x, const float *gain,
                        const float *means, const float *rstds)
{
    LayernormBackward backward = {
        .d_x = d_x,
        .d_gain = d_gain,
        .d_bias = d_bias,
        .d_out = d_out,
        .x = x,
        .gain = gain,
        .means = means,
        .rstds = rstds,
    };

    for (int row = 0; row < x.rows; row++)
        layernorm_backward_row(&backward, row);
}
```

The gradient-means stage reconstructs `norm` and `d_norm`, then turns
their sums into means. The accumulation stage reconstructs the same
two locals and applies the three returning routes. The row coordinator
binds one row to its saved statistics before it runs those stages.

The row loop stays serial because every row adds into the same gain and
bias destinations. Chapter 6 first walks the
[numeric layernorm
routes](06-backprop-by-hand.md#layernorm-couples-every-channel-in-a-row);
Chapter 5 owns the [forward
operation](05-forward-pass.md#keep-one-rows-scale-from-controlling-the-next-operation).

## Causal attention: unwind the consultation

Attention has three forward stages for each query:

```text
query and visible keys  -> scaled scores
scaled scores           -> softmax weights
weights and values      -> output mixture
```

Backward must visit all three in reverse. Stopping after the value
mixture would leave queries and keys unchanged. Stopping after the
score route would miss how the values made the output.

### Two positions before symbols

Reuse Chapter 6's two-position, one-head fixture. The head size is
`D = 2`, so:

```text
scale = 1/sqrt(2) = about 0.7071068
```

The head slices are:

```text
position    query      key        value
   0        [1, 0]    [1, 0]     [2, 1]
   1        [0, 1]    [0, 1]     [4, 3]
```

Position `0` sees only itself, so its weight is `[1]`. At position `1`,
the scaled scores are:

```text
score[0] = 0.7071068 * dot([0,1], [1,0]) = 0
score[1] = 0.7071068 * dot([0,1], [0,1]) = 0.7071068
```

**Predict:** which value will get more influence at position `1`?
Compare the two scaled scores before reading the weights.

Softmax gives:

```text
weight = about [0.330238, 0.669762]
```

Value row `1` gets the larger weight because its key matches the query.

Let the arriving output gradients be:

```text
d_out[0] = [1,  0]
d_out[1] = [1, -0.5]
```

Start at the final weighted mixture for query position `1`. Each
weight's slope is its value dotted with `d_out[1]`:

```text
d_weight[0] = dot([2,1], [1,-0.5]) = 1.5
d_weight[1] = dot([4,3], [1,-0.5]) = 2.5
```

Each value receives the arriving output gradient scaled by its saved
weight:

```text
d_value[0] += 0.330238 * [1,-0.5]
            = about [0.330238, -0.165119]

d_value[1] += 0.669762 * [1,-0.5]
            = about [0.669762, -0.334881]
```

Now pass `d_weight` through the coupled softmax rule:

```text
coupled
    = 0.330238*1.5 + 0.669762*2.5
    = about 2.169762

d_score[0]
    = 0.330238 * (1.5 - 2.169762)
    = about -0.221181

d_score[1]
    = 0.669762 * (2.5 - 2.169762)
    = about 0.221181
```

**Predict:** what should the two score gradients sum to? They cancel,
as every softmax score-gradient row must.

The score already included the head scale. Returning through

```text
score = scale * dot(query, key)
```

adds one factor of `scale`:

```text
d_dot = scale * d_score
      = about [-0.156399, 0.156399]
```

The query receives each dot coefficient times the corresponding key:

```text
d_query[1]
    = -0.156399*[1,0] + 0.156399*[0,1]
    = about [-0.156399, 0.156399]
```

Each key receives its coefficient times query `[0,1]`:

```text
d_key[0] += about [0, -0.156399]
d_key[1] += about [0,  0.156399]
```

Query position `0` has a one-entry softmax. Its only weight is always
one, so changing its only score changes no output weight. Its score,
query, and key gradients from that route are zero. Its value route
still contributes:

```text
d_value[0] += 1 * [1,0]
```

Value row `0` served both queries. Add both contributions:

```text
d_value[0] = about [1.330238, -0.165119]
d_value[1] = about [0.669762, -0.334881]
```

This is accumulation made visible: an earlier value can matter to
several later outputs.

### General stage 1: the value mixture

Fix one sequence, one head, and query position `t`. Chapter 1's
[no-peeking
rule](01-the-map.md#no-peeking-at-the-answer) permits positions
`t2 = 0` through `t`. Chapter 5 constructs the
[forward consultation](05-forward-pass.md#let-one-position-consult-the-visible-past):

```text
out[t,d]
    = sum over t2=0..t of weight[t2] * value[t2,d]
```

For one visible value channel:

```text
partial out[t,d] / partial value[t2,d] = weight[t2]

d_value[t2,d] += weight[t2] * d_out[t,d]
```

One weight affects every output channel:

```text
partial out[t,d] / partial weight[t2] = value[t2,d]

d_weight[t2]
    = sum over d of d_out[t,d] * value[t2,d]
    = dot(d_out[t], value[t2])
```

`d_value` uses `+=` because later query positions may return to the
same earlier value. `d_weight` lives in private scratch with one
producer for this query, so assignment is safe.

### General stage 2: softmax

The preceding section derived:

```text
coupled
    = sum over j=0..t of weight[j]*d_weight[j]

d_score[t2]
    = weight[t2] * (d_weight[t2] - coupled)
```

Only the visible prefix participates. Future scratch entries are
untouched and have no meaning for this query.

### General stage 3: the scaled dot

For head width `D`:

```text
scale = 1/sqrt(D)

score[t2]
    = scale
      * sum over d of query[t,d]*key[t2,d]
```

Each score sends:

```text
d_query[t,d]
    += scale * d_score[t2] * key[t2,d]

d_key[t2,d]
    += scale * d_score[t2] * query[t,d]
```

The query collects over all visible keys. An earlier key collects from
every later query that consulted it. There is no future-token gradient
to mask away: positions after `t` never entered the forward loop, so no
returning path exists.

### Match the source names and loop ownership

The mathematical and C names differ at one point:

| Meaning | Name here | C storage or local |
|---|---|---|
| saved probability | `weight` | `weights`, inside `scores` |
| gradient for a weight | `d_weight` | `d_weights`, inside `d_scores` |
| gradient for a scaled score | `d_score` | `d_weights` after softmax backward |
| coefficient on the unscaled dot | `d_dot` | local `d_raw` |

After forward, the visible part of the arena named `scores` contains
softmax weights, not pre-softmax scores. During backward,
`softmax_backward_in_place` replaces the visible `d_scores` prefix with
`d_score`. The source then writes:

```c
float d_raw = position->scale * position->d_weights[source];
```

Despite the local name, this value is the coefficient called `d_dot`
above. It already includes the one required scale. The two following
`add_scaled` calls send it to the query and key. Applying another scale
would be wrong.

The outer
[`attention_backward`](../src/ops.c) wrapper assigns disjoint channel
slices to different `(sequence, head)` pairs. With OpenMP enabled, it
may run those pairs concurrently when
`sequence_count * head_count >= 64`. Query positions within one pair
stay serial because they add into shared key and value rows. Chapter 6
walks the complete
[`attention_head_backward`
source](06-backprop-by-hand.md#attention-unwinds-the-consultation).

## Adam's cold-start correction

AdamW does not send a gradient through a forward operation. It uses the
gradients already collected to adjust learned values. One part of that
update still needs algebra: a history that starts at zero initially has
less than a full unit of coefficient weight.

### Watch the missing weight at three steps

Use one history, `beta = 0.5`, a constant gradient of `2`, and
`m[0] = 0`:

```text
m[1] = 0.5*0   + 0.5*2 = 1
m[2] = 0.5*1   + 0.5*2 = 1.5
m[3] = 0.5*1.5 + 0.5*2 = 1.75
```

The history has not reached `2` because some coefficient weight still
belongs to the zero start. Open the third step:

```text
m[3] = 0.125*g[1] + 0.25*g[2] + 0.5*g[3]
```

Its coefficient total is:

```text
0.125 + 0.25 + 0.5 = 0.875
```

**Predict:** what restores the constant gradient `2`? Divide by the
coefficient total:

```text
m_hat[3] = 1.75 / 0.875 = 2
```

For changing gradients, the correction does not make them equal. It
turns the three coefficients into:

```text
0.125/0.875 = 1/7
0.25 /0.875 = 2/7
0.5  /0.875 = 4/7
```

Those weights sum to one while retaining the preference for recent
gradients.

### Derive the coefficient total

For one parameter, the first history follows:

```text
m[t] = beta1*m[t-1] + (1 - beta1)*g[t]
m[0] = 0
```

The first three steps are:

```text
m[1] = (1 - beta1)*g[1]

m[2] = beta1*(1 - beta1)*g[1]
       +      (1 - beta1)*g[2]

m[3] = beta1^2*(1 - beta1)*g[1]
       + beta1*(1 - beta1)*g[2]
       +       (1 - beta1)*g[3]
```

At step `t`:

```text
m[t]
    = (1 - beta1)
      * sum over i=1..t of beta1^(t-i)*g[i]
```

Now derive the coefficient total instead of quoting it. Let:

```text
S       = 1 + beta + beta^2 + ... + beta^(t-1)
beta*S  =     beta + beta^2 + ... + beta^(t-1) + beta^t
```

Subtract the second line from the first. Every middle term cancels:

```text
S - beta*S = 1 - beta^t

(1 - beta)*S = 1 - beta^t
```

Chapter 8 validates `0 <= beta < 1`, so the right side is positive for
`step >= 1`. The coefficients in `m[t]` therefore sum to:

```text
1 - beta1^t
```

Divide away only the mass missing because the history began at zero:

```text
m_hat[t] = m[t] / (1 - beta1^t)
```

The second history repeats the same arithmetic with squared gradients:

```text
v[t] = beta2*v[t-1] + (1 - beta2)*g[t]^2
v[0] = 0

v_hat[t] = v[t] / (1 - beta2^t)
```

At step one:

```text
m[1]     = (1 - beta1)*g[1]
m_hat[1] = g[1]

v[1]     = (1 - beta2)*g[1]^2
v_hat[1] = g[1]^2
```

That also shows why `step` begins at one. At step zero, both correction
denominators would be zero. The correction removes deliberate
zero-start shrinkage. It does not claim that old and future gradients
come from an unchanging process.

### The corrected update

For parameter value `theta`, the normal finite update is:

```text
correction1 = 1 - beta1^step
correction2 = 1 - beta2^step

first  = beta1*first  + (1 - beta1)*gradient
second = beta2*second + (1 - beta2)*gradient^2

smoothed = first / correction1
spread   = sqrt(second / correction2)

theta -= learning_rate
         * (smoothed / (spread + epsilon)
            + decay*theta)
```

Epsilon is outside the square root. The decay term is outside the
adaptive fraction, so it does not enter either history. The literal
source policy is:

```text
decay = weight_decay when rows > 1 and cols > 1
decay = 0 otherwise
```

A `1 x C` gain or bias follows Adam's adaptive arithmetic but receives
no decay.

### The fallible source shell

The formula is not the whole
[`param_adamw_step`](../src/param.c) contract. Current source prevents
an invalid entry near the end of a parameter from leaving earlier
entries partly updated.

The route is:

```text
zero histories allocated by param_new
    -> validate the recipe
    -> compute and validate both corrections
    -> preflight every entry without mutation
    -> commit every entry
    -> return 0

any failed gate -> return -1 before the commit loop
```

`param_new` obtains the values, gradient, and two history regions from
one `ecalloc`, so both histories begin at zero. The value constructors
then fill the value region.

`param_adamw_recipe_valid` requires:

```text
step >= 1
finite learning_rate >= 0
finite 0 <= beta1 < 1
finite 0 <= beta2 < 1
finite epsilon > 0
finite weight_decay >= 0
```

The updater computes both `1 - beta^step` values and rejects a
nonfinite or nonpositive result. `adamw_inputs_valid` then reads every
value, gradient, and history entry. It checks their finite ranges, the
candidate histories, and conservative bounds for every later
division, decay term, change, and result. It does not mutate the
parameter.

Only after the full preflight succeeds does the commit loop execute the
displayed update. The function returns `0` after success and `-1` after
rejection. Chapter 8 constructs the
[zero-start repair](08-adamw.md#repair-the-zero-start) and then walks
the complete optimizer policy.

## From local equations to a whole-model check

Each local backward route has a precise boundary:

| Route | Reads saved from forward | Arriving gradient | Destination |
|---|---|---|---|
| cross-entropy | probabilities, targets | scalar loss rate `1` | logits |
| matmul | input, weights | output | input, weights |
| residual | no numeric forward value | output | both inputs |
| GELU | input | output | input |
| layernorm | input, gain, mean, rstd | output | input, gain, bias |
| attention | QKV, saved weights | output | packed QKV |
| embedding | token ids, time | output | token and position tables |

Softmax backward is private attention scratch: it reads saved weights
and overwrites one arriving `d_weight` row with `d_score`. AdamW begins
after all parameter gradients have returned; it is an update, not
another backward route.

The table stops at local contracts. Chapter 12 owns the
[whole-model reverse
order](12-wiring-the-model-backward.md#walk-the-whole-model-in-source-order),
including the tied token-table meeting and the saved-value lifetime.
Its [two independent
witnesses](12-wiring-the-model-backward.md#two-witnesses-make-different-claims)
separate local derivative correctness from assembly correctness.

Chapter 7's finite-difference referee provides a different route to the
same local slopes. It changes one source input up and down, measures
the resulting loss change, and compares that measurement with the
hand-derived gradient. See [Measure a slope from nearby
values](07-trust-but-verify.md#measure-a-slope-from-nearby-values) and
[What agreement
licenses](07-trust-but-verify.md#what-agreement-licenses).

The symbolic derivation checks the intended equation. The
finite-difference test checks whether the forward and backward source
agree at the tested values. The whole-model witness checks whether
individually correct functions were joined in the right order. None of
the three claims substitutes for the other two.

## Build

Reconstruct the local routes from their forward equations before
looking at the linked function bodies:

1. Send residual addition's arriving gradient to both inputs with
   accumulation.
2. Scatter each embedding output row back to its selected token and
   position rows.
3. Give matmul one sum over output channels for `d_x` and a different
   sum over input rows for `d_weights`.
4. Recompute GELU's local intermediates and multiply its slope by
   `d_out`.
5. Compute softmax's `coupled` dot before overwriting private scratch.
6. Subtract one only at each cross-entropy target, then divide every row
   by `R`.
7. Collect layernorm's two shared means before applying its compact
   input formula.
8. Unwind attention through the value mixture, softmax, and scaled dot
   in that order.
9. Implement AdamW's corrected normal update inside its validate,
   preflight, commit, and result boundary.

Keep public input and parameter gradient writes as `+=`. Assignment is
reserved for private scratch with one producer, such as the visible
`d_scores` prefix before and during softmax backward.

## Verify

From the repository root, run the focused backward and optimizer
checks, then the combined serial gradient and integration checks:

```sh
make OPENMP=0 check-backward
make OPENMP=0 check-optimizer
make -C labs WORK=../src OPENMP=0 check-08
make OPENMP=0 check
```

[`tests/gradcheck.c`](../tests/gradcheck.c) checks residual, embedding,
matmul, GELU, softmax through attention, cross-entropy, and layernorm
against measured finite differences. It also checks additive
destinations from nonzero starting values and preserves sentinels
outside attention's causal scratch prefixes. Its AdamW case compares
several updates with an independent double-precision transcription, the
[different optimizer
witness](08-adamw.md#why-the-optimizer-needs-a-different-witness)
Chapter 8 constructs.

The optimizer integration cases also check model-level rejection before
parameter updates. The staged
[`check08.c`](../labs/check08.c) case puts unsafe arithmetic in a later
entry and verifies that no earlier value moves. That is the direct
witness for the source's per-parameter preflight.

For pencil-and-paper checks, recompute:

```text
residual destinations              [12,-2] and [-2,2]
embedding token rows               [9,-1]
embedding position rows            [5,3]
matmul d_x                          [[21,17],[-2,12]]
matmul d_weights                    [[12,1],[-2,15]]
GELU slope at x=0                   0.5
GELU slope at x=1                   about 1.082964
softmax d_score                     [0.75,-0.75]
cross-entropy row-gradient sums     0
layernorm d_x row sum               about 0
Adam step-one m_hat and v_hat       g and g^2
```

## Expected

Each focused command should finish with its pass line and no `FAIL`
diagnostic. The combined serial command should report successful
gradient and integration suites. Exact total counts are not frozen
here; the tests themselves are the maintained source of that total.

The two-position attention example should reproduce, to the shown
precision:

```text
weight at position 1   [ 0.330238,  0.669762]
d_weight               [ 1.500000,  2.500000]
d_score                [-0.221181,  0.221181]
d_dot                  [-0.156399,  0.156399]
d_query[1]             [-0.156399,  0.156399]
d_value[0]             [ 1.330238, -0.165119]
d_value[1]             [ 0.669762, -0.334881]
```

Small float differences are expected inside finite-difference
comparisons, which use both absolute and relative tolerances. A printed
`FAIL` is not an accepted rounding artifact.

## Common failures

- **Replacing an accumulated gradient.** Residual inputs, table rows,
  matmul destinations, keys, values, and parameters may already contain
  another valid path.
- **Reversing the matmul layout.** Tiny AgenC stores weights as
  `O x I`; forward computes `X * W^T`.
- **Dropping one GELU product path.** The slope changes both through
  the leading `x` and through `tanh(inner)`.
- **Overwriting before softmax coupling.** `coupled` needs the complete
  incoming `d_weights` row.
- **Forgetting the mean-loss scale.** Cross-entropy divides every logit
  contribution by `probs.rows`.
- **Dividing by a stored target probability.** The fused C backward
  uses `probability - indicator`; it never evaluates `-1/p`.
- **Using `C - 1` in layernorm.** Forward and backward use the
  population variance divisor `C`.
- **Dropping gain from `d_norm`.** The input route begins with
  `d_out * gain`.
- **Reading attention `scores` as raw scores.** After forward, its
  visible prefixes contain weights.
- **Applying the attention scale twice or not at all.** The scaled dot
  contributes exactly one `1/sqrt(D)` during backward.
- **Letting attention visit `t2 > t`.** Forward and backward must use
  the same causal prefix.
- **Parallelizing query positions within one head.** They add into
  shared key and value rows.
- **Starting AdamW at step zero.** Both correction denominators would
  be zero.
- **Sharing one correction between histories.** The first uses
  `beta1`; the second uses `beta2`.
- **Treating the normal update as the full source contract.** Recipe
  checks, correction checks, read-only preflight, and the fallible
  result are part of `param_adamw_step`.
- **Using a nearby AdamW policy.** Epsilon placement, decoupled decay,
  and the literal `rows > 1 && cols > 1` decay predicate must match the
  implementation under test.

---

[Previous: Epilogue](18-epilogue.md) | [Contents](README.md) | [Next: Appendix B](appendix-b-sources.md)
