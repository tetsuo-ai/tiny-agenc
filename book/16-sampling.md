# Chapter 16: Sampling

Training taught the model to assign probabilities. Chapter 13's
[checkpoint](13-durable-checkpoints.md#state-the-version-1-promise-exactly)
now contains the learned numbers that can produce those probabilities,
but it does not contain a finished continuation.

Those learned numbers still leave one missing operation. Chapter 15
ended with learned parameter values and left `model_sample` as an
opaque handoff.
A target-free forward call does not leave a chosen character behind.
It leaves raw scores. This chapter constructs the choice, appends it,
and asks again.

## Generation begins after adjustment stops

An `id` is still Chapter 3's
[compact number for one vocabulary byte](03-data.md#give-each-byte-a-compact-number).
Keep the two routes separate before joining any code:

```text
TRAINING

corpus ids + known answers
          |
          v
forward -> grade -> backward -> update
                                  |
                           parameters change

ONE GENERATED ID

fixed parameter values + visible ids
          |
          v
target-free forward -> newest scores -> choose -> append

no known answer, grade, backward, or update
```

The word `fixed` applies to parameter values. A target-free forward
still overwrites activation storage, copied token ids, saved
statistics, logits, and Chapter 11's
[latest-forward record](11-wiring-the-model-forward.md#only-the-newest-forward-record-survives).
The choice operation also advances Chapter 2's
[seeded generator](02-foundations.md#enter-the-sequence-through-a-seed),
writes an output id, and uses one probability row as scratch. The
`Model` object is working; its learned numbers are not moving.

Chapter 15's progress hook runs only after the current update. It
discards its generated ids before the next corpus batch. The standalone
`sample` command has no training loop to return to. Neither route feeds
generated text through backward or AdamW.

The parameter-fixed route is:

```text
visible ids + learned parameters
               |
               v
       target-free forward
               |
               v
       newest raw score row
               |
               v
      fresh probability sheet
               |
               v
       one weighted choice
               |
               v
          append the id
               |
               +----> run again
```

Every box after the forward call is the missing Chapter 16 mechanism.

## The loop

Chapter 11 constructed
[inference mode](11-wiring-the-model-forward.md#stop-when-there-is-no-answer-sheet):
`model_forward` receives `NULL` targets, runs the shared score route,
and returns the `0.0f` no-loss sentinel. It does not choose a token.
It also does not refresh `m->probs`; those bytes may belong to an older
graded call.

For a batch of one and three visible ids, the active logit shape is
`3 x V`. Chapter 5 named each row's `V` raw scores
[logits](05-forward-pass.md#grade-one-next-token-bet). Chapter 1's
[causality rule](01-the-map.md#no-peeking-at-the-answer) gives the rows
different questions:

```text
visible ids             A       C       D

logit row 0          after A
logit row 1          after A C
logit row 2          after A C D   <- current question
```

**Predict:** which row describes the unknown id after the complete
visible text `A C D`?

Row 2 does. More generally, a call with `window` visible positions must
read row `window - 1`. Row zero asks what follows only the first id.
Reading every row would make several choices for one empty output slot.

Suppose that newest row contains three scores:

```text
id          0   1   2
logit       2   1   0
```

Chapter 5's
[softmax](05-forward-pass.md#turn-arbitrary-scores-into-usable-shares)
can turn that row into non-negative shares that sum to one. The
target-free forward did not do so, because inference stops at logits.
Chapter 16 must make a fresh sheet and use it.

## A largest-only rule throws information away

One possible chooser would always take id 0 because `2` is the largest
score. That rule would ignore every smaller nonzero share. The RNG seed
would have no effect while the score order stayed the same.

Tiny AgenC instead gives each id a region whose width equals its
probability. Begin with an easy sheet:

```text
id                    0          1          2
probability          0.5        0.3        0.2
running boundary     0.5        0.8        1.0

number line       0          0.5       0.8       1.0
                  |--- id 0 ---|-- id 1 --|-- id 2 --|
```

The regions are half open:

```text
id 0 owns [0.0, 0.5)
id 1 owns [0.5, 0.8)
id 2 owns [0.8, 1.0)
```

Chapter 2 built the seeded
[uniform draw](02-foundations.md#turn-32-bits-into-a-fraction) between
zero inclusive and one exclusive.

**Predict:** which ids receive the exact draws `0.25`, `0.75`, and
`0.9375`?

`0.25` lies in id 0's region. `0.75` passes the first boundary and lies
below `0.8`, so id 1 wins. `0.9375` lies in id 2's region. All three
values are points on the implementation's 24-bit grid.

Choosing one token id according to these probability widths is
**sampling**. It is a weighted draw from Chapter 3's finite
[vocabulary](03-data.md#give-each-byte-a-compact-number), not a vote
for every id and not another training step.

Sampling is the payoff: turn probabilities back into text, one
character at a time.

## Drawing from the distribution

With probabilities in hand, one uniform draw picks the character by
walking a running total.

For the divisor-one sheet produced from `[2, 1, 0]`, Chapter 5's
arithmetic gives:

```text
probability          [0.6652, 0.2447, 0.0900]
running boundary     [0.6652, 0.9099, about 1]
```

**Predict:** where does the exact draw `0.75` land?

It is not below `0.6652`. Adding id 1's share moves the boundary to
`0.9099`, which is above the draw, so id 1 is returned.

The comparison is `<`, not `<=`. In the easy sheet, the exact draw
`0.5` passes id 0 and belongs to id 1. The source walks ids in numeric
order, so that boundary rule is deterministic.

### The draw is finite

Chapter 2's generator exposes `2^24` possible uniform values, not every
real number between zero and one. Shrink that grid to eight points for
a hand check:

```text
possible draws     0, 1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8
decimal form       0, .125, .250, .375, .500, .625, .750, .875
```

**Predict:** how many of the eight draws fall below `0.6`?

If two intended shares are `[0.6, 0.4]`, the first region contains five
of those eight points and the second contains three:

```text
realized grid shares     5/8 = 0.625     3/8 = 0.375
```

The real 24-bit grid makes this discrepancy much smaller. It does not
erase it. An extremely narrow nonzero region may contain no available
draw, and a softmax entry that underflows to zero has no region.

Float addition creates one more edge. Imagine stored probabilities
rounded to the four decimals printed above:

```text
0.6652 + 0.2447 + 0.0900 = 0.9999
```

The largest 24-bit draw is
`16,777,215 / 16,777,216`, about `0.99999994`. That draw would pass all
three rounded boundaries. The source therefore returns the final id
after the loop. The last region owns any float-rounding gap, and the
fallback consumes no second draw.

## Change the gaps without changing their order

The scores produced by the learned parameters determine one
distribution, but a caller may need a more concentrated or more
spread-out draw without retraining.

Changing all probabilities by one shared multiplier fails. For the
easy sheet:

```text
original              [0.5, 0.3, 0.2]
multiply by 2         [1.0, 0.6, 0.4]
sum                    2.0
divide by the sum     [0.5, 0.3, 0.2]
```

Adding one shared value to every logit also fails. Chapter 5 showed that
softmax subtracts the largest score, so `[2, 1, 0]` and `[7, 6, 5]`
both become gaps `[0, -1, -2]`.

The useful control must change the gaps themselves. Start by
subtracting the largest logit:

```text
logits              [ 2,  1,  0]
subtract 2          [ 0, -1, -2]
```

Now divide each gap by one positive number.

**Predict:** does dividing by `0.5` move the losing gaps toward zero or
farther away?

It moves them farther away:

```text
[0, -1, -2] / 0.5 = [0, -2, -4]
```

Work the resulting softmax without skipping a step:

```text
exponentials     [1.0000000, 0.1353353, 0.0183156]
sum               1.0000000 + 0.1353353 + 0.0183156
                = 1.1536509
divide by sum     1.0000000 / 1.1536509 = 0.8668133
                  0.1353353 / 1.1536509 = 0.1173104
                  0.0183156 / 1.1536509 = 0.0158762
probabilities    [0.8668133, 0.1173104, 0.0158762]
```

**Predict:** when the divisor changes from `0.5` to `2`, should the
three shares move closer together or farther apart?

Dividing by `2` moves the gaps toward zero:

```text
[0, -1, -2] / 2 = [0, -0.5, -1]

exponentials     [1.0000000, 0.6065307, 0.3678794]
sum               1.0000000 + 0.6065307 + 0.3678794
                = 1.9744101
divide by sum     1.0000000 / 1.9744101 = 0.5064804
                  0.6065307 / 1.9744101 = 0.3071959
                  0.3678794 / 1.9744101 = 0.1863237
probabilities    [0.5064804, 0.3071959, 0.1863237]
```

The middle case divides by one:

```text
[0, -1, -2] / 1 = [0, -1, -2]

exponentials     [1.0000000, 0.3678794, 0.1353353]
sum               1.0000000 + 0.3678794 + 0.1353353
                = 1.5032147
divide by sum     1.0000000 / 1.5032147 = 0.6652410
                  0.3678794 / 1.5032147 = 0.2447285
                  0.1353353 / 1.5032147 = 0.0900306
probabilities    [0.6652410, 0.2447285, 0.0900306]
```

The three cases line up:

| Positive divisor | Scaled gaps | Probabilities |
|---:|---|---|
| `0.5` | `[0, -2, -4]` | `[0.8668, 0.1173, 0.0159]` |
| `1.0` | `[0, -1, -2]` | `[0.6652, 0.2447, 0.0900]` |
| `2.0` | `[0, -0.5, -1]` | `[0.5065, 0.3072, 0.1863]` |

This positive divisor on centered logit gaps is **temperature**. Below
one, it stretches the gaps and concentrates the sheet. One leaves the
gaps unchanged. Above one, it shrinks the gaps and moves the sheet
toward equal shares.

![For fixed logits, positive temperature changes probability spread
without changing rank.](figures/16-temperature-distribution.svg)

The [figure script](../scripts/figures/16_temperature_distribution.py)
calculates every point from the same illustrative logits `[2, 1, 0]`.
The plotted values are deterministic arithmetic, not a measured model
run.

Positive division preserves ordering. If `2 > 1` and `temperature > 0`,
then `2 / temperature > 1 / temperature`. Equal logits also stay tied.
At extreme values, float precision can collapse tiny differences into
ties even though the mathematical order remains.

As positive temperature gets closer to zero, losing gaps can become so
negative that their exponentials round to zero. As temperature grows,
all finite gaps move toward zero and their shares move toward equal.
Zero itself is invalid because the division has no usable result.
Negative and nonfinite values are invalid too.

Chapter 14's
[sample-command boundary](14-the-command-line.md#sampling-crosses-the-boundary-in-the-other-direction)
accepts a finite positive value no larger than `1,000,000`. The
low-level sampler trusts its caller to meet the positive-finite
precondition. The training progress hook uses the source policy
`0.8`, which multiplies centered gaps by `1 / 0.8 = 1.25`. That value
is a default, not proof of an optimal writing setting.

Temperature changes the scratch probability sheet for this draw. It
does not edit the input logits or any parameter.

## Four small score-to-id stages

Finding the largest score, constructing the probability sheet, and
drawing from that sheet have different jobs. Keeping each job in a
private helper gives the names a direct meaning in the coordinator.
These helpers trust that the supplied logit row is finite. Chapter
5's [softmax boundary](05-forward-pass.md#turn-arbitrary-scores-into-usable-shares)
places nonfinite inputs outside its guarantee; this code neither
validates nor repairs such a row.

The first helper in
[`model_sampling.c`](../src/model_sampling.c) scans the logits in
ascending id order:

```c
static float maximum_logit(const float *logits, int vocab)
{
    float maximum = logits[0];

    for (int id = 1; id < vocab; id++)
        if (logits[id] > maximum)
            maximum = logits[id];
    return maximum;
}
```

`maximum` begins at id zero, and the loop visits the remaining ids.
Every valid model has at least one vocabulary id, so the first read
exists. Returning the value leaves the logit row unchanged.

The second helper uses that value to overwrite every entry in the
scratch sheet:

```c
static void build_distribution(float *distribution, const float *logits,
                               int vocab, float temperature)
{
    float maximum = maximum_logit(logits, vocab);

    for (int id = 0; id < vocab; id++)
        distribution[id] = (logits[id] - maximum) / temperature;
    softmax_in_place(distribution, vocab);
}
```

Subtracting the maximum before division keeps the winning entry at
exactly zero even when a tiny positive temperature would overflow a
positive scaled logit. A losing gap may become negative infinity at an
extreme. With a finite maximum still at zero, Chapter 5's softmax gives
that entry zero share.

`softmax_in_place` repeats its own maximum subtraction. The scratch row
already has maximum zero, so this second stability step subtracts zero
and changes no mathematical value.

The third helper owns the one random operation:

```c
static int draw_from_distribution(const float *distribution, int vocab,
                                  Rng *rng)
{
    float draw       = rng_uniform(rng);
    float cumulative = 0.0f;

    for (int id = 0; id < vocab; id++) {
        cumulative += distribution[id];
        if (draw < cumulative)
            return id;
    }
    return vocab - 1;
}
```

`rng_uniform` advances the supplied generator once. The final loop adds
one probability at a time and returns at the first boundary above the
draw. If rounded additions never cross it, `vocab - 1` is the final
valid id and owns the remainder. The draw happens before the loop, so a
one-id vocabulary also advances the generator exactly once.

The coordinator provides storage and connects the two transformations:

```c
static int sample_from_logits(Model *m, const float *logits, Rng *rng,
                              float temperature)
{
    float *distribution = mat_row(m->probs, 0);
    int    vocab        = m->cfg.vocab_size;

    build_distribution(distribution, logits, vocab, temperature);
    return draw_from_distribution(distribution, vocab, rng);
}
```

`mat_row(m->probs, 0)` takes Chapter 4's
[row address](04-poor-mans-tensors.md#turn-two-coordinates-into-one-offset).
The target-free forward left this storage stale, so the coordinator
treats row zero as scratch. `build_distribution` overwrites all
`vocab` entries. The path allocates no new probability array.

`vocab` comes from the model configuration. Valid geometry permits a
one-id vocabulary, in which case that id receives every draw. The
coordinator's two calls say what happens without mixing either loop
into the storage choice.

The stages read the logit row. They never write through the `const`
pointer, so temperature scaling cannot change the model's stored
scores.

## One choice must become the next question

One weighted choice gives one id. Text needs the choice appended and
then included in the next question. Chapter 1 named that
[one-token-at-a-time route](01-the-map.md#generation-uses-what-training-learned)
**autoregressive**.

Feed the model everything written so far while the prefix fits. If
`block_size = 3` and the known ids are `A B`, the next call can see both:

```text
known = 2
window = 2
context = ids + 2 - 2 = ids + 0
visible ids = [A B]
```

After the chosen `C` is stored, the next call sees:

```text
known = 3
window = 3
context = ids + 3 - 3 = ids + 0
visible ids = [A B C]
```

Appending another id creates a problem. The model has capacity for only
three positions. Passing all four known ids would violate Chapter 1's
[context-length ceiling](01-the-map.md#capacity-and-the-active-call).
Keeping the oldest three would discard the id the model wrote most
recently:

```text
wrong oldest context     [A B C] D
right newest context      A [B C D]
```

The starting address must move. Work four states:

| `known` | `window` | `known - window` | Visible ids | Slot written |
|---:|---:|---:|---|---:|
| 2 | 2 | 0 | `[A B]` | 2 |
| 3 | 3 | 0 | `[A B C]` | 3 |
| 4 | 3 | 1 | `[B C D]` | 4 |
| 5 | 3 | 2 | `[C D E]` | 5 |

**Predict:** after `E` fills slot 4, which address and three ids must
the `known = 5` call use?

`window` remains three. The start is `5 - 3 = 2`, so `ids + 2` exposes
`[C D E]`. Chapter 3's
[pointer arithmetic](03-data.md#draw-every-legal-start) changes the
starting address; it does not copy those ids or allocate a new array.

Moving the left edge once the visible prefix reaches capacity is a
**sliding context window**. It always exposes the newest
`block_size` ids and no older id.

## The complete generation loop

The low-level function trusts a programming contract. `m` and `rng`
must be valid. The first `prompt_count` array entries must be valid ids,
and the array must have room for `total_count`. Counts must satisfy
`1 <= prompt_count <= total_count`, and temperature must be finite and
positive. An equal count is a safe no-op; the CLI always requests a
positive tail.

Only `prompt_count >= 1` has an assertion inside this function. Chapter
14's checked command path supplies the other conditions. Like
`mat_row`, this compute kernel does not convert bad direct-call
arguments into recoverable CLI errors.

Here is the complete exact loop:

```c
void model_sample(Model *m, Rng *rng, int *ids, int prompt_count,
                  int total_count, float temperature)
{
    assert(prompt_count >= 1);

    for (int known = prompt_count; known < total_count; known++) {
        int        window  =
            known < m->cfg.block_size ? known : m->cfg.block_size;
        const int *context = ids + known - window;

        model_forward(m, context, NULL, 1, window);
        ids[known] = sample_from_logits(m, mat_row(m->logits, window - 1),
                                        rng, temperature);
    }
}
```

`known` is both the number of filled ids and the index of the first
empty slot. Beginning at `prompt_count` preserves the entire supplied
prefix. The loop stops before `total_count`, so it fills exactly
`total_count - prompt_count` entries.

The Chapter 3
[conditional expression](03-data.md#keep-complete-scenes-out-of-training)
selects the smaller of `known` and `block_size`. Before capacity is
full, the whole prefix is visible. Afterward, `window` stays at
capacity.

`context = ids + known - window` points at the suffix we built by hand.
The call to `model_forward` fixes `batch = 1`, uses the active
`window` as time, and passes `NULL` targets. It produces scores without
a loss or backward-eligible record.

`mat_row(m->logits, window - 1)` selects the row that saw the entire
active suffix. The helper returns one id, and `ids[known]` appends it.
The `for` increment then makes that id part of the next visible suffix.

The effects are exact:

| State | Effect of `model_sample` |
|---|---|
| parameter values | read, unchanged |
| optimizer moments and gradients | unchanged |
| activations, copied ids, statistics, logits | overwritten |
| latest-forward record | replaced by target-free calls |
| probability row zero | overwritten as choice scratch |
| RNG | advanced once per generated id |
| caller's `ids` array | generated suffix filled |

There is no stop id. A generated newline is an ordinary next context
id, not a request to leave the loop. The function writes the exact
requested count.

### Why the source recomputes the suffix

When the context moves from `[A B C]` to `[B C D]`, the retained ids
receive local positions zero through two again:

```text
first call       A at 0   B at 1   C at 2
next call        B at 0   C at 1   D at 2
```

Chapter 5's learned
[position table](05-forward-pass.md#give-each-id-a-row-of-opening-notes)
therefore gives `B` and `C` different position rows on the second call.
Tiny AgenC recomputes the whole suffix through the shared forward path
so that reindexing is exact.

With a three-position causal window, one head computes one query-key
dot product for row zero, two for row one, and three for row two: six
dot products even though only the final logit row chooses an id. Larger
inference engines can retain earlier key and value calculations before
an eviction. Retaining them across Tiny AgenC's position reset would
change this model's route.
[Appendix C](appendix-c-capstone-labs.md#5-cache-keys-and-values) turns
that different state-management problem into a capstone.

Such retained calculations would not be Chapter 11's latest-forward
record. That record describes the newest complete call and whether
backward may replay it; generated inference calls must never feed
backward.

## Seeding the writer

Both entry points prepend a newline to the known id sequence. This is a
token seed, meaning a known starting id. It is not the numeric seed that
starts the RNG. A long prompt can push that newline outside the first
active context.

Chapter 14 already constructed and walked the command boundary:

```text
checkpoint -> matched model and tokenizer
                     |
newline + encoded prompt -> known ids
                     |
--seed -> fresh text RNG
                     |
model_sample -> known prefix plus generated tail
                     |
decode ids[1..total-1] -> append one presentation newline
```

The private newline supplies a familiar boundary before a later corpus
line. The very first corpus line has no preceding byte, so saying every
line literally began after newline would be false.

Chapter 3's
[encoder](03-data.md#encode-without-inventing-a-token) skips prompt
bytes outside the checkpoint vocabulary. They do not receive a made-up
id and do not appear in output. The array is sized for the unfiltered
prompt, so skipped bytes leave harmless spare capacity.

All surviving prompt ids are decoded and printed, regardless of prompt
length. The first generated choice sees the newest `block_size` ids
from the private newline followed by those surviving prompt ids. If
fewer than `block_size` prompt ids survive, that active context still
includes the private newline. `print_text` omits the private `ids[0]`,
decodes the visible prompt and requested tail, then prints one extra
presentation newline.

The public command creates a new RNG from `--seed` for each invocation.
Chapter 15's
[progress hook](15-the-training-loop.md#schedule-side-effects-after-the-update)
instead keeps one RNG created from the training seed plus one. Each
progress sample starts a new text context at newline, but the RNG stream
does not restart. The step-500 sample begins after the 200 draws
consumed by the step-250 sample.

## Determinism is testable

The numeric seed controls the draw stream. Temperature controls the
region boundaries. Keep the draw fixed at the exact grid value `0.75`:

```text
temperature 1.0 boundaries     [0.6652, 0.9099, about 1]
draw 0.75                       passes id 0, selects id 1

temperature 0.5 boundaries     [0.8668, 0.9841, about 1]
draw 0.75                       selects id 0
```

The seed did not change. The sheet did.

One changed id can alter every later question:

```text
same prefix + draw 0 -> chosen id
                          |
                          v
                  next prefix changes
                          |
                          v
                   next logits change
                          |
                          v
                draw 1 meets new boundaries
```

The same checkpoint bits, tokenizer, surviving prompt ids, requested
length, temperature, initial RNG seed, executable behavior, and
floating-point environment replay the same ids and therefore the same
text. Prompt spelling matters only through the ids that survive
encoding.

Changing one input permits a different result; it does not guarantee
one. Two seeds can land in the same regions. Two temperatures can
select the same id for a particular draw. Equal logits produce equal
shares at every valid temperature. A longer request preserves the
generated tail from the shorter request as its prefix when all earlier
replay inputs match. The command prints its presentation newline after
that requested tail.

Training-time progress does not restart from the standalone CLI stream.
It uses seed plus one and retains its RNG position across reports.

## Close the first-light path

Return to the checkpoint Chapter 15 created:

```sh
./tiny-agenc sample \
    --model first-light.bin \
    --prompt "RAZR:" \
    --length 80 \
    --temperature 0.8 \
    --seed 1337
```

Use `labs/build/tiny-agenc` in place of `./tiny-agenc` on the learner
track. This 200-step checkpoint was built to prove movement and
serialization, not polished prose. The command closes the route from
learned parameters through a loaded checkpoint to generated ids.

The bundled 5,000-step checkpoint has committed replay evidence. Run:

```sh
./tiny-agenc sample \
    --model tiny-agenc.bin \
    --prompt "=== TRANSMISSION 0999 // SECTOR 9: " \
    --length 400 \
    --temperature 0.8 \
    --seed 1337
```

And the writer writes. This is a shortened opening from
[`showcase-sample.txt`](logs/showcase-sample.txt):

```text
=== TRANSMISSION 0999 // SECTOR 9: ICE MARKET ===
DOC: What's the pad heat? Fast that corpo labyrief if we're late.
RAZR: GHOST, you sure the is sector at the Neon Docks' woun'rate out out there?
GHOST: WIRES, stay frosty; the corpo langes in the Grid.
```

`make check-evidence` reruns the exact command and compares all 400
requested characters with the committed file. That witness establishes
replay for the recorded artifact. It does not establish originality,
truth, understanding, held-out quality, or an ideal temperature.
[Chapter 17](17-the-training-run.md#one-generated-sample-is-one-trace)
interprets what the recorded samples can and cannot show.

## Know what the Chapter 16 witness proves

The focused witness in [`tests/sampling.c`](../tests/sampling.c) splits
the mechanism into four claims.

First, it zeros every parameter in a five-id model. Equal logits produce
five equal regions. From one prompt id it requests eleven new ids, and
an independent copy of the RNG performs the same manual running-total
walk. Each generated position gets one valid-id check and one exact-id
check:

```text
11 generated positions * 2 checks = 22 checks
```

Second, a one-id model must choose id zero while still consuming exactly
one uniform draw. An independent RNG advances once, and the next values
from the two generators must match. Two checks cover the id and the RNG
position.

Third, a controlled nonuniform model receives the same first draw
twice. Temperature `0.05` concentrates the sheet enough to select id
zero; temperature `100` flattens it enough to select another id. Three
checks establish the cold result, hot result, and difference.

Fourth, the prefix `[0, 0, 0, 1, 1, 0]` with `block_size = 3` produces
the same next id as an explicit `[1, 1, 0]` prefix under the same RNG.
The controlled model also makes that choice depend on all three
retained ids, not only the newest one. Two checks cover this comparison.

The total is:

```text
22 equal-logit + 2 one-id + 3 temperature + 2 suffix checks = 29
```

The dependent integration witness trains a tiny model, saves and loads
it, then compares a 32-slot sequence containing two prompt ids and
thirty generated ids. Original, repeated, and loaded-model runs agree
under equal seeds, and every generated id is in the vocabulary.

The lab adds command-level comparisons. Two identical invocations must
write byte-identical output. It also compares prompts `RAZR:` and
`RAZR:@` against a fixture vocabulary without `@`; their output matches
because the unknown byte is skipped before generation.

These checks do not grade prose quality, force the float-rounding
fallback, inspect allocation calls, compare parameter bits around
sampling, or promise cross-platform bit identity. The source walk
establishes the no-allocation and no-update paths. Different witnesses
carry different claims.

## Build checkpoint: make it write

### Build

Keep Chapter 14's complete command path and Chapter 15's training loop.
Replace the borrowed sampling answer with `model_sampling.c`. Implement
the reusable scratch row, maximum-centered temperature scaling, the
running-total draw with final-id fallback, final-logit-row selection,
and the newest-suffix loop.

Reuse Chapter 5's `softmax_in_place`, Chapter 2's `rng_uniform`, and the
model's existing arenas. Do not allocate once per generated id.

### Verify

Run the learner checkpoint:

```sh
make -C labs check-16
```

Then inspect the answer key:

```sh
make -C labs WORK=../src check-16
```

The focused root witness is also available:

```sh
make OPENMP=0 check-sampling
```

Run the first-light command twice with the same replay inputs and
compare the complete standard output.

### Expected

Equal logits follow the independently reproduced uniform draws.
Controlled low and high temperatures can choose different ids from the
same draw. An evicted prefix agrees with its explicit newest block.
Original and loaded models replay the same seeded suffix, repeated CLI
commands match byte for byte, and unknown prompt bytes are skipped.

The first-light output may remain rough. Success here means the exact
requested suffix is produced through the mechanism above, not that a
small checkpoint writes well.

### Common failures

- Reading `m->probs` after inference uses stale bytes. Rebuild the
  scratch sheet from the newest logits.
- Always taking the largest logit discards the weighted draw. Reading
  row zero instead of `window - 1` answers an earlier prefix.
- Scaling probabilities by one common factor changes nothing after
  normalization. Center and divide logits before softmax.
- Dividing raw positive logits before maximum subtraction can overflow
  at a tiny positive temperature.
- A direct caller that supplies zero, negative, or nonfinite
  temperature violates the compute-kernel contract.
- Using `<=` changes boundary ownership. Omitting the final-id fallback
  leaves a float-rounding path without a valid result.
- Keeping the oldest full window makes the model ignore what it wrote
  most recently. An off-by-one suffix address drops or repeats an id.
- Allocating a context or probability array for every id hides
  ownership work inside the hottest loop.
- Calling backward or update after generated inference would violate
  the fixed-parameter route. Reusing the batch RNG would let inspection
  steer later training batches.
- Printing `ids[0]` exposes the private token seed. Assuming a numeric
  seed alone guarantees replay ignores the checkpoint, filtered prompt,
  temperature, length, and floating-point environment.

The choice mechanism is now visible from score row to output byte.
[Chapter 17](17-the-training-run.md#progress-samples-change-two-things)
can use it as a fixed instrument while reading the recorded training
run.

---

[Previous: The Training Loop](15-the-training-loop.md) | [Contents](README.md) | [Next: The Training Run](17-the-training-run.md)
