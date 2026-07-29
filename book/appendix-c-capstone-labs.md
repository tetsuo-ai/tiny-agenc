# Appendix C: Capstone Labs

A useful exercise changes one thing and makes success observable.
Otherwise it is a feature request wearing a mortarboard.

The labs below are ordered by conceptual reach, not merely lines of C.
Each gives you a goal, likely files, acceptance criteria, hints, and the
observation the work is meant to expose. Commit a known-passing starting
point before beginning. Keep the existing checks as witnesses while
you change their world.

These are eight independent experiments. They are not eight patches to
stack into one model. Start each lab from the same passing reference
implementation unless the lab explicitly says otherwise.

## How to use a capstone

The first two labs change only what you observe. The next two challenge
or add one calculation. The fifth changes inference execution while
preserving the model equation. The sixth and seventh change contracts
that reach from input or geometry to checkpoint bytes. The last lab
measures the assembled system.

```text
observe -> control -> challenge -> compare
                                    |
                                    v
                         execution -> architecture -> measurement
```

Use the smallest lab that asks the question you care about:

| Lab | Return path through the book | Reach |
|---:|---|---|
| 1 | Chapter 16 | saved outputs only |
| 2 | Chapters 3, 15, and 17 | data and evaluation |
| 3 | Chapters 6, 7, and 12 | one temporary wrong derivative |
| 4 | Chapter 17 | one new comparator |
| 5 | Chapters 10, 11, and 16 | inference state |
| 6 | Chapters 2, 5-7, 9, and 13 | geometry and format |
| 7 | Chapters 3, 9, 10, 13, 14, and 17 | text unit and format |
| 8 | Chapters 1, 10, 17, and 18 | controlled measurement |

The core result in each lab is bounded. Stretch work is labeled. A
working cache before the context fills is a complete Lab 5 result.
Solving rolling eviction is not required. A correct rotary operation
and migrated checkpoint contract complete Lab 6. Proving that the new
model is better would be a different experiment.

## Write the experiment record first

Chapter 18 ended with four lines for a controlled extension. A capstone
needs a few more because it may leave the reference architecture:

```text
baseline commit:
question:
one change:
prediction:
contract that changes:
mistake an old witness could miss:
new witness:
unchanged witnesses:
resource ceiling:
output paths:
stop when:
claim still not earned:
```

Fill this record before editing. The `prediction` must name an
observable result. "The cache will be faster" is too loose. "For prefix
lengths 1 through T, cached and full-prefix logits agree within the
declared tolerance" can be tested.

The final line prevents a passing test from growing into a larger claim.
A short position-rotation run can show that the new path trains without
an immediate failure. It cannot show that the new mechanism is better
than learned position rows.

## Protect the passing baseline

Begin with an empty `git status --short` and record the starting commit.
Run the complete serial gate before changing the source:

```sh
git status --short
git rev-parse HEAD
make OPENMP=0 check-all
```

Create a separate branch or a second checkout made with `git worktree`
for the experiment. Put generated checkpoints, logs, splits, and tables
under a unique path such as `build/capstones/temperature/`. Do not
overwrite `tiny-agenc.bin`, the committed corpus, committed logs, or
`EVIDENCE.sha256`.

Record the compiler, build flags, thread count, seeds, and input hashes
for any timing or model comparison. Set a memory and time ceiling before
a full-corpus run. Start with a hand-checkable fixture and the smallest
model that can expose the contract.

`EVIDENCE.sha256` pins the reference source, tests, Makefile, corpus,
and model artifact. A legitimate source experiment therefore makes
`make check-evidence` object to the changed identity. That objection is
not a math failure, and it is not permission to replace a hash. A patch
intended for the upstream project must deliberately migrate every
affected lab, artifact, measurement, and evidence record before the
complete gate can pass again.

## 1. Temperature safari

### Goal

Separate confidence, diversity, and correctness by sampling one
checkpoint across temperatures and seeds.

Here, confidence means the concentration of one fixed logit row after
temperature changes it. The numeric calculation and Chapter 16's
probability witness measure that direction. Once sampled characters
diverge, saved text measures output variation and format instead. It has
no correctness answer key.

### Need

One attractive sample cannot establish a sampling policy. It does not
say how often the format survives, how much text repeats, or how much a
different random draw changes the continuation. "Reliably valid" also
has no meaning until `valid` and the number of trials are written down.

Chapter 16 constructed
[temperature](16-sampling.md#change-the-gaps-without-changing-their-order)
as a positive divisor on the gaps between logits. It changes the bet
sheet used for a draw. It does not change the learned parameters.

### Construct the measurement

Use the fixed temperatures `0.2`, `0.5`, `0.8`, `1.1`, and `1.5`.
Choose one prompt, one requested length, and twenty seeds before seeing
any output.

Write the format rule first. One suitable rule reuses Chapter 3's
[NIGHT GRID grammar](03-data.md#cleaning-as-a-grammar-not-a-mop): ignore
the final partial line, then count how many complete nonempty lines
match either the header rule or the dialogue rule in
[`clean-corpus.sh`](../scripts/clean-corpus.sh). Record the raw count
instead of declaring a continuous temperature range between the five
measured points.

Choose one repetition measure too. For example:

```text
numerator
    complete dialogue-line occurrences whose exact text already
    appeared earlier in this same sample

denominator
    all complete dialogue-line occurrences in this sample
```

The first occurrence of one line is not repeated; its second and later
occurrences enter the numerator. If a sample has no complete dialogue
line, record `0/0: no dialogue lines` rather than inventing a fraction.
Keep each sample's raw numerator and denominator. For a sweep-wide
fraction, sum its numerators and denominators separately, excluding
`0/0` samples; do not average the per-sample fractions. If the summed
denominator remains zero, report that the sweep produced no complete
dialogue lines. Keep the five-temperature and twenty-seed sweeps
separate.

This fraction and the format count can be recomputed from saved output.
Keep every sample, not only the ones you would show another person.

First compare all five temperatures with seed `1337`. The random source
then replays the same draw sequence while the probability intervals
move. Next hold one temperature fixed and compare the twenty seeds.
These answer different questions.

### Predict

Suppose one step has logits `[2, 1, 0]`.

```text
temperature below 1  -> gaps widen
temperature above 1  -> gaps narrow
```

Before running the command, rank the two cases by how concentrated you
expect the largest share to be. Then predict why changing the first
sampled character can make every later character differ even though
the later RNG draws still replay.

### Files

No source file needs to change. Read
[`model_sampling.c`](../src/model_sampling.c) and the checked sample
boundary in [`main.c`](../src/main.c). Store the experiment outputs
under `build/capstones/temperature/`.

### Build

Build the current executable and record the two input identities:

```sh
make OPENMP=0 tiny-agenc
mkdir -p build/capstones/temperature
sha256sum tiny-agenc tiny-agenc.bin \
    > build/capstones/temperature/identity.sha256
```

Generate the paired, same-seed traces:

```sh
for sample_temperature in 0.2 0.5 0.8 1.1 1.5; do
    ./tiny-agenc sample \
        --model tiny-agenc.bin \
        --prompt 'RAZR:' \
        --length 200 \
        --temperature "$sample_temperature" \
        --seed 1337 \
        > "build/capstones/temperature/paired-${sample_temperature}.txt"
done
```

Then hold temperature at `0.8` and sweep the predeclared seeds.
`seq 1337 1356` supplies twenty consecutive values:

```sh
for sample_seed in $(seq 1337 1356); do
    ./tiny-agenc sample \
        --model tiny-agenc.bin \
        --prompt 'RAZR:' \
        --length 200 \
        --temperature 0.8 \
        --seed "$sample_seed" \
        > "build/capstones/temperature/seed-${sample_seed}.txt"
done
```

### Verify

Run the existing probability and sliding-window witnesses:

```sh
make OPENMP=0 check-sampling check-evidence
```

Repeat one command into a second file and compare the bytes:

```sh
./tiny-agenc sample \
    --model tiny-agenc.bin \
    --prompt 'RAZR:' \
    --length 200 \
    --temperature 0.8 \
    --seed 1337 \
    > build/capstones/temperature/replay.txt
cmp build/capstones/temperature/paired-0.8.txt \
    build/capstones/temperature/replay.txt
```

Your result table must contain the checkpoint hash, executable hash,
prompt, length, temperature, seed count, format count, and repetition
fraction. The same program, checkpoint, arguments, and platform should
replay the saved bytes. Chapter 16 explains the
[limits of that replay claim](16-sampling.md#determinism-is-testable).

### Expected

The logit-gap direction follows the prediction. Format and repetition
need not change monotonically across the five temperature traces. The
twenty-seed sweep describes variation at temperature `0.8`; seed order
has no warmer-to-colder direction. Report a flat or surprising result
if that is what the saved outputs show. No exact sample text is
required.

### Common failures

- **Changing temperature and seed together.** A changed result then has
  two possible causes.
- **Writing the rubric after sampling.** The outputs can steer the rule
  toward the desired conclusion.
- **Keeping selected examples only.** Selection hides the denominator.
- **Calling a colder sample more correct.** Sampling has no answer key.
- **Inferring unmeasured temperatures.** Five points do not establish a
  continuous safe range.

### Stop when

Stop when the paired temperature traces, the fixed-temperature seed
sweep, and one bounded claim are recorded. Finding the best temperature
is not required. Chapter 17's warning still applies: [one generated
sample is one trace](17-the-training-run.md#one-generated-sample-is-one-trace).

**Carry away.** Lower temperature means narrower sampling, not truer
knowledge.

## 2. Validation experiments

### Goal

Learn to read training and held-out loss as two measurements with
different permissions.

### Need

Suppose the reported validation loss changes from one step to the next.
If the model changed and the validation windows also changed, the
difference has two possible causes. A longer experiment does not repair
that ambiguity.

The original version of this lab changed split percentage, model size,
training duration, and validation sampling policy together. Its listed
choices already make at least twenty-four conditions. The capstone rule
is one contrast at a time.

Chapter 3 built the [whole-scene training and validation
split](03-data.md#keep-complete-scenes-out-of-training). Chapter 15
deals [four fixed batches once](15-the-training-loop.md#deal-held-out-windows-once)
and grades them without
[sending corrections back](15-the-training-loop.md#grade-without-sending-corrections-back).
Keep those boundaries visible.

### Construct the core experiment

Start with one 90/10 split, one model, one seed, and one training run.
The repeated reports from that run are the first result. They show what
happens as the model changes while the held-out questions stay fixed.

After that core result, you may choose one optional contrast:

| Contrast | Hold fixed | Change |
|---|---|---|
| split policy | model, steps, seed, rate, threads | 90/10 versus 80/20 |
| capacity | split, `B`, `T`, steps, seed, rate | either `L` or `C` |
| duration | split, model, seed, rate | stopping step |
| question sampling | checkpoint and split | fixed versus fresh windows |

Do not run every combination. Changing the split ratio changes both the
training records and the held-out questions, so the resulting losses are
not two grades on one matched test. That contrast measures a policy
change, not a direct contest.

For a capacity comparison, keep the number of target slots equal:

```text
target slots = steps * B * T
```

For a duration study, one longer run already supplies a sequence of loss
reports. Separate shorter runs are needed only if the earlier
checkpoints themselves are part of the question.

Fresh validation windows are stretch work. The current CLI does not
offer that mode. Implement it either in [`main.c`](../src/main.c) with a
separate validation RNG or in a standalone evaluator that loads saved
checkpoints. Never let observation consume the batch RNG or the progress
sample RNG.

### Predict

Before the core run, sketch the direction you expect for training and
validation loss. Do not predict exact values.

Then answer two questions:

1. Which curve is easier to compare across steps, one graded on fixed
   windows or one graded on fresh windows?
2. If validation reporting is observation-only, should enabling
   `--val-data` change the final checkpoint bytes?

The first answer is fixed windows. Fresh windows can cover more of the
held-out file across repeated measurements, but they also change the
questions. The second answer is no. Chapter 15's RNG separation and
forward-only grading are meant to preserve the training path.

### Files

The core experiment changes no source. Read
[`split-corpus.sh`](../scripts/split-corpus.sh),
[`evaluation.h`](../src/evaluation.h), and the validation route in
[`main.c`](../src/main.c). A resampling extension must change
`src/main.c` or add an evaluator and a direct test for its RNG and
no-update contracts.

### Build

Build the splitter and executable, then put the derived files away from
the canonical split:

```sh
make OPENMP=0 build/split-order tiny-agenc
mkdir -p build/capstones/validation

VALIDATION_PERCENT=10 SPLIT_SEED=1337 \
    bash scripts/split-corpus.sh \
    data/cyberpunk.txt \
    build/capstones/validation/90.train.txt \
    build/capstones/validation/90.val.txt
```

If split policy is your chosen contrast, create the second pair without
overwriting the first:

```sh
VALIDATION_PERCENT=20 SPLIT_SEED=1337 \
    bash scripts/split-corpus.sh \
    data/cyberpunk.txt \
    build/capstones/validation/80.train.txt \
    build/capstones/validation/80.val.txt
```

Run the core model with the serial build. This small configuration
grades `1,000 * 4 * 64 = 256,000` training target slots:

```sh
./tiny-agenc train \
    --data build/capstones/validation/90.train.txt \
    --val-data build/capstones/validation/90.val.txt \
    --out build/capstones/validation/core.bin \
    --steps 1000 \
    --layers 2 \
    --heads 2 \
    --width 64 \
    --block 64 \
    --batch 4 \
    --lr 0.001 \
    --seed 1337 \
    > build/capstones/validation/core.log
```

Record the starting commit, both split hashes, scene and byte counts,
compiler, flags, thread count, and full command beside the loss table.
Before an optional contrast, copy the table and mark the one field that
will change.

### Verify

Run the existing data, command, model, and staged training-loop
witnesses:

```sh
make OPENMP=0 check-data check-cli check-model
make -C labs WORK=../src OPENMP=0 check-15
```

The existing split witness checks deterministic replay, expected record
counts, and preservation of every line with its repetition count. Those
are its exact claims. If a changed splitter needs a stronger
record-integrity witness, give every record in a tiny fixture one unique
body line and verify that its header and body reach the same output.

`make check-cli` also runs a tiny pair with and without held-out
reporting and compares the resulting checkpoint bytes. If you add a
resampling mode, extend that isolation witness. The evaluator must call
forward only. No validation result may flow to `model_backward` or
`model_step`.

Inspect `run_train` as well. It must build the tokenizer from training
bytes before it reads the validation file, and it must reject a
validation byte absent from that fixed training vocabulary. Otherwise
the held-out side can shape the input mapping or disappear silently.

One detail matters when reading the log. The training loss on a report
line was measured before that step's update. The validation loss was
measured after it. Use trends across reports, not a claim that the two
numbers came from an identical parameter instant.

### Expected

Fixed held-out questions make report points comparable. Freshly sampled
questions can estimate a wider part of the validation file across many
sets, with added question-selection variation. No split ratio, capacity,
or stopping step is promised to win.

When the two grades separate persistently, Chapter 15 names the pattern
[overfitting](15-the-training-loop.md#when-the-two-grades-part-company).
One noisy gap is not enough.

### Common failures

- **Running every contrast together.** The result no longer identifies
  which change mattered.
- **Letting validation consume the batch RNG.** Observation then steers
  later training examples.
- **Comparing different held-out questions as matched grades.** A
  changed split changes the test.
- **Selecting repeatedly on the same validation set.** The held-out
  answers become inputs used to choose settings, so they no longer grade
  an untouched decision.
- **Calling one gap a trend.** Question selection and batch loss both
  vary.

### Stop when

Stop after the core run supports one bounded statement. Run at most one
optional controlled contrast. Other axes remain separate experiments.

**Carry away.** A falling training loss proves optimization on allowed
data. Generalization needs a boundary the optimizer cannot cross.

## 3. Break a gradient, watch the referee

### Goal

Experience a model that can still train with a wrong derivative, then
watch a direct measurement expose it.

### Need

A wrong derivative is not required to point uphill at every step. It
can retain the correct sign for many inputs, send a smaller or larger
correction, and still make a short loss move down. Training output alone
therefore makes a weak referee.

Chapter 6 showed that layernorm's input gradient contains two shared
return paths because changing one channel changes the row's mean and
variance. The final source line is:

```c
            d_input[c] += rstd * (d_norm - d_norm_mean - norm * d_norm_norm_mean);
```

The last product carries the variance path. This capstone halves its
coefficient while leaving the forward calculation intact.

### Construct the challenge

Use a disposable branch or second checkout. Record the inverse edit
before making the fault:

```text
broken:
rstd * (d_norm - d_norm_mean
        - 0.5f * norm * d_norm_norm_mean)

restore:
rstd * (d_norm - d_norm_mean - norm * d_norm_norm_mean)
```

Commit the mutation by itself. After the observation, an inverse commit
can restore it without erasing the experiment's history.

Three witnesses see different boundaries:

```text
forward known answers     backward did not run
operation finite differences
                          compare layernorm's local slope
whole-model differences   compare the assembled returning paths
```

Chapter 7 constructed the [local numeric
referee](07-trust-but-verify.md#two-referees-watch-different-calculations).
Chapter 12 explains why the [whole-model witness makes a different
claim](12-wiring-the-model-backward.md#two-witnesses-make-different-claims).

### Predict

Before editing, write pass or fail beside each command:

```text
make check-forward
make check-backward
make check-model
```

The forward check should still pass. The operation derivative check
should report layernorm input-gradient disagreements. The model check
should find the wrong return after it flows through the assembled
network.

Now predict the 50-step loss. "It may fall, stall, or rise" is the only
claim the witnesses justify. The point is to see whether a plausible
training trace can coexist with a measured derivative error.

### Files

Change only [`ops.c`](../src/ops.c). Read
[`gradcheck.c`](../tests/gradcheck.c), but do not weaken its tolerance
or expected behavior.

### Build

Establish the local baseline:

```sh
make OPENMP=0 check-forward check-backward check-model
mkdir -p build/capstones/gradient
```

Insert only the `0.5f` coefficient shown in the broken form. Inspect the
one-file diff and commit that deliberate fault. Run the three predicted
commands separately so one expected failure does not prevent the next
command from running.

After recording those results, build the broken executable and make one
short observation:

```sh
make OPENMP=0
./tiny-agenc train \
    --data labs/tiny-corpus.txt \
    --out build/capstones/gradient/broken.bin \
    --steps 50 \
    --layers 1 \
    --heads 1 \
    --width 8 \
    --block 8 \
    --batch 1 \
    --lr 0.001 \
    --seed 1337 \
    > build/capstones/gradient/broken.log
```

Do not use the broken checkpoint for any later lab. Apply the recorded
inverse edit, commit the restoration, and confirm that `src/ops.c`
matches the baseline commit.

### Verify

The broken phase is successful when its failure pattern matches the
prediction:

```text
forward witness       passes
local derivative      objects
whole-model derivative objects
```

The final state is successful only when the exact reference source has
been restored and all gates pass:

```sh
make OPENMP=0 check-forward check-backward check-model
make OPENMP=0 check-all
make check-sanitizers
```

Compare the broken loss log with the derivative failures. Loss movement
does not overrule the independent finite-difference measurement.

### Expected

No direction is promised for the short loss. A falling trace is
especially useful because it demonstrates the gap between "the
optimizer found some downhill motion" and "the implemented derivative
matches the forward calculation."

### Common failures

- **Changing forward and backward together.** The forward check no
  longer isolates the mutation.
- **Changing more than one term.** The resulting failures cannot be
  attributed to one coefficient.
- **Loosening the tolerance.** That silences the referee instead of
  repairing the slope.
- **Treating a failed build as a derivative finding.** The intended
  fault must compile before the numeric test can judge it.
- **Leaving the fault in the branch.** The capstone ends with the exact
  passing source restored.

### Stop when

Stop when the predicted failure pattern is recorded, the loss
observation is interpreted narrowly, and the restored source passes the
complete gate.

**Carry away.** "It trains" is one of the weakest correctness claims in
machine learning.

## 4. Beat the bigram honestly

### Goal

Build a stronger non-neural
[baseline](17-the-training-run.md#loss-needs-a-landmark), a simpler
comparison rule graded on the same held-out positions.

The rule remembers one more byte than the bigram. The title is a
challenge, not an acceptance threshold. A correctly measured loss that
is higher than the bigram is a complete result.

### Need

Consider two tiny training fragments:

```text
a b a
c b d
```

A predictor that remembers only `b` combines the two cases:

```text
after b:  a appeared once, d appeared once
```

The two preceding bytes separate them:

```text
after a b:  a appeared once
after c b:  d appeared once
```

More context creates a new problem. A two-byte history absent from
training has no count row. Assigning probability zero to its next byte
would make one held-out mistake produce infinite loss.

### Construct the comparator

Keep Chapter 17's
[add-one bigram](17-the-training-run.md#keep-every-answer-possible) as
the shorter-context rule. For a two-byte history seen during training,
use add-one counts in that row. With the four-byte vocabulary
`{a, b, c, d}`, the history `a b` has:

```text
row total                   1
provisional count for a     1 + 1 = 2
denominator                 1 + 4 = 5
p(a | a,b)                  2/5
each other next byte        1/5
```

For an unseen two-byte history, use the add-one row for its final byte.
This move to a shorter available context is called **backoff**. Fix that
rule before looking at validation loss.

An ordered run of three adjacent tokens is a **trigram**. The first two
tokens name the context and the third is the answer being counted. The
new comparator is therefore:

```text
two visible preceding bytes and a seen row -> add-one trigram
one visible byte or an unseen pair         -> add-one bigram
```

This rule also fixes a subtle fairness boundary. At local time zero of
one transformer validation window, only the first input byte is
visible. The trigram evaluator must use the bigram there. Reading the
byte before the window would give the comparator information denied to
the transformer.

### Predict

For the tiny fragments above, the add-one bigram gives:

```text
p(a | b) = (1 + 1) / (2 + 4) = 1/3
```

The observed trigram row gives `p(a | a,b) = 2/5`. Predict which rule
assigns less loss to answer `a` after `a b`.

Now make the harder prediction: must the trigram beat the bigram on
held-out text?

No. Its rows contain fewer observations, so sparse counts can outweigh
the extra context. Do not adjust the fallback after seeing validation
results. Choosing among fallback rules requires a third data split for
that choice, leaving validation untouched.

### Files

Read [`bigram.c`](../tests/bigram.c) and
[`evaluation.h`](../src/evaluation.h). Keep the first experiment in a
separate source such as `experiments/trigram.c`. If you later integrate
it as `tests/trigram.c`, the Makefile and evidence manifest become part
of the deliberate project change.

### Build

Begin with two program-level fixtures:

1. the four-symbol example above, with probabilities checked by hand;
2. an unseen two-byte history that must return a finite bigram-backed
   probability.

Then count from training bytes only. Grade two routes:

```text
full validation file
    bigram versus trigram on every adjacent answer each can grade

fixed validation windows
    bigram versus trigram on the same target slots
```

Reuse the fixed-window seed and loops in `tests/bigram.c`. At time zero,
use the bigram. From time one onward, the two bytes must both come from
inside the active window.

A transformer comparison is optional. The clean held-out checkpoint
behind Chapter 17's [recorded validation
run](17-the-training-run.md#read-the-held-out-run) is not bundled.
`tiny-agenc.bin` was trained on the complete corpus, including the
validation scenes, so it cannot replace that checkpoint. Either compare
with the recorded fixed-window result and retain its hashes and command,
or train a new model on the training split and grade the exact same
windows. Do not make the scratch trigram program silently load the
bundled model.

A dense table indexed by all byte triples contains
`256 * 256 * 256` counters. With eight-byte counters, predict its byte
count before revealing the multiplication:

```text
256 * 256 * 256 * 8 = 134,217,728 bytes = 128 MiB
```

A compact training-vocabulary table or an observed-context table can be
much smaller. Whichever representation you choose, bound every size
calculation before allocation and report allocated bytes plus the
number of observed two-byte contexts.

One possible scratch build is:

```sh
mkdir -p build/capstones/trigram
cc -std=c11 -O3 -Wall -Wextra -Werror -Isrc \
    -o build/capstones/trigram/trigram \
    experiments/trigram.c src/rng.c src/util.c -lm

./build/capstones/trigram/trigram \
    data/cyberpunk.train.txt \
    data/cyberpunk.val.txt
```

### Verify

Run the reference comparator first:

```sh
make OPENMP=0 bigram-baseline
```

The new witness must establish all of these contracts:

- the hand-counted row matches;
- an unseen pair receives the declared bigram-backed probability;
- no count comes from validation bytes;
- time zero of a fixed window cannot read outside the window;
- full-file and fixed-window losses use natural logarithms;
- every direct transformer comparison uses the exact same target slots;
- memory counts are checked before allocation.

Preserve raw totals as well as averages:

```text
summed negative log-probability / graded target count
```

That pair exposes an accidental change in the denominator.

### Expected

The new evaluator produces finite full-file and matched-window losses.
It may beat, tie, or lose to the add-one bigram. The result earns
meaning from the matched questions and fixed rule, not from its rank.

### Common failures

- **Reading before the window.** The comparator receives hidden context.
- **Counting validation text.** Held-out answers then shape the rule.
- **Returning zero for an unseen row.** One surprise creates infinite
  loss.
- **Tuning until the trigram wins.** Validation becomes training by
  repeated selection.
- **Comparing only the same file.** The target positions and visible
  histories can still differ.
- **Allocating `256^3` counters without a bound.** A derived size still
  needs checked arithmetic.

### Stop when

Stop when the tiny fixtures pass and one honest matched report is
recorded. Winning is not the stopping condition.

**Carry away.** A transformer deserves credit only for predictive value
beyond simpler context models measured under the same rules.

## 5. Cache keys and values

### Goal

Avoid recomputing the complete prefix for every generated token while
preserving the current logits before the context window fills.

### Need

Generation currently runs the complete visible suffix for each new
token. With a three-token capacity, the growing calls are:

```text
[A]       -> score one attention row
[A B]     -> score three causal pairs
[A B C]   -> score six causal pairs
```

Across the three calls, the naive route visits `1 + 3 + 6 = 10`
query-key pairs. If each call computes only its new query row, it visits
`1 + 2 + 3 = 6`. It also avoids repeating the earlier normalization,
projection, residual, and MLP rows.

Causality makes this possible before eviction. The representation at
`A` cannot depend on later `B`, so appending `B` does not change the key
or value already calculated for `A`.

### Construct the saved state

Store each completed layer's earlier key and value rows:

```text
after A       keys [KA]          values [VA]
after B       keys [KA KB]       values [VA VB]
after C       keys [KA KB KC]    values [VA VB VC]
```

The new query still compares with every visible key. Its attention
weights still mix every visible value. Only calculations for rows that
already existed are retained.

Storage that keeps those rows for later inference calls is a
**key-value cache**, shortened to **KV cache**.

Keep its owner narrow. A separate private inference state can be created
for one `model_sample` call, filled from that prompt, and freed before
return. It owns:

```text
K[layer][position][channel]
V[layer][position][channel]
active length and checked geometry
```

It does not own model parameters, optimizer moments, training
activations, or the caller's ids. For `L` layers, capacity `T`, and width
`C`, its main payload is:

```text
cache floats = 2 * L * T * C
```

At the default shape, that is:

```text
2 * 4 * 128 * 128 = 131,072 floats
131,072 * 4       = 524,288 bytes
```

Heads partition the `C` channels; they do not multiply this count.
Check the size before allocation and report it separately from the
current `ModelMemory`, which describes storage owned by `Model`. If you
instead place the cache inside `Model`, then construction, reset, free,
the public memory report, and resource preflight all become part of the
change.

### Push one token through the blocks

For a new token at local position `p`, preserve the existing operation
order:

```text
token row + position row p
    -> layer 0 normalization and QKV projection
    -> store K0[p] and V0[p]
    -> Q0[p] attends over K0[0..p]
    -> weights mix V0[0..p]
    -> projection, residual, MLP
    -> repeat for each later layer
    -> final normalization and tied output head
    -> one logit row
```

The existing [`model_forward.c`](../src/model_forward.c) route computes
the same equations for every row. This route changes which rows execute.

Keep the cache separate from Chapter 11's
[latest-forward record](11-wiring-the-model-forward.md#only-the-newest-forward-record-survives).
Incremental inference must mark that record target-free so
`model_backward` cannot replay stale training activations. Cached rows
also become invalid after a parameter update. The bounded sampler avoids
that case by doing no update during its cache lifetime.

Feed the visible prompt through the cache one token at a time for the
first implementation. Loading several prompt rows at once is stretch
work.

### The eviction boundary

The cache is exact while a local prefix grows from one through `T`.
After that, current Tiny AgenC changes more than the visible left edge:

```text
first window       A at 0   B at 1   C at 2
next window        B at 0   C at 1   D at 2
```

Chapter 16 explains why the source
[recomputes this suffix](16-sampling.md#why-the-source-recomputes-the-suffix).
`B` and `C` receive new learned position rows after the shift. Their
hidden states, later-layer keys, and later-layer values must therefore
be recomputed.

Use one bounded policy:

```text
visible prefix length <= T  append to the cache
oldest token is evicted     clear and rebuild the newest T-token suffix
```

This preserves current semantics. A rolling cache that retains old rows
after their local positions change is a different model and belongs to
a separate experiment.

### Predict

For `[A B]` becoming `[A B C]`, predict whether `KA` and `KB` change.
Then predict again for `[A B C]` becoming `[B C D]`.

The first answer is no. The second answer is yes because local positions
change.

Predict the more dangerous test too: can equal sampled ids prove that
all logits match?

No. Two different probability rows can place one RNG draw in the same
interval. Compare logits before sampling.

### Files

The likely impact reaches:

```text
model_sampling.c
    own, load, use, rebuild, and free inference state

model_forward.c
    provide the operation order for one-row execution

ops.c and ops.h
    one-query attention over cached K and V rows

model_internal.h
    private state and internal route

tests/integration.c
    numeric cached-versus-naive logits

tests/sampling.c
    ids, prompt isolation, and eviction rebuild
```

If cache storage becomes model-owned, add `model.c` and
`model_memory.c`. If new source files are introduced, the Makefile also
changes. Checkpoint bytes and the parameter registry should not change;
the cache is temporary working state.

### Build

Keep the original full-prefix calculation available to the test. Use a
fixed token sequence. For every prefix length from one through `T`:

1. run ordinary `model_forward`;
2. copy its final logit row;
3. push the same newest token through the cached route;
4. compare all `V` logits with one declared mixed tolerance.

Only after numeric agreement should a second fixture use the same seed
to compare sampled ids. Test a second prompt with a fresh state, an
explicit reset if the API exposes one, and the first eviction.

For timing, pin one thread and warm both routes. Measure each prefix
length an odd number of times, sort those times, and take the middle
value. That middle value is the **median**. Time score computation, not
checkpoint loading or output printing.

### Verify

Run the focused serial paths:

```sh
make OPENMP=0 check-forward
make OPENMP=0 check-backward
make OPENMP=0 check-model
make OPENMP=0 check-sampling
```

The new witnesses must establish:

```text
all logits agree for prefix lengths 1 through T
the controlled same-seed fixture chooses the same ids
a second prompt cannot see state from the first
inference cannot leave backward eligible
eviction rebuild matches an explicit newest-T suffix
all allocated cache storage is released
```

Then run:

```sh
make check-sanitizers
```

Wire the new cache witness into a sanitizer path or run its
sanitizer-built executable directly. Old sanitized tests do not cover a
new route automatically.

### Expected

Before eviction, cached and naive logits agree within the declared
tolerance at every vocabulary entry. The eviction rebuild agrees with
the existing explicit-suffix behavior.

The timing record should expose less growing-prefix work. No fixed
speedup is promised. Width, thread overhead, allocation policy,
compiler, and prefix length all belong to that measured result.

### Common failures

- **Caching a combined QKV row without a reason.** Queries are consumed
  once; only keys and values need to survive.
- **Reusing rows after local positions shift.** The cache then describes
  different hidden states.
- **Testing sampled ids only.** Equal choices can hide unequal logits.
- **Leaving the last forward target-bearing.** Backward can replay stale
  activations.
- **Reusing a cache after `model_step`.** Its rows came from older
  parameter values.
- **Timing I/O and printing.** The measurement no longer isolates
  prefix work.
- **Adding allocation without a free path.** The execution optimization
  introduces an ownership bug.

### Stop when

Stop when prefixes through `T` match numerically, prompt isolation and
eviction fallback pass, sanitizer coverage includes the new route, and
the timing claim remains tied to its environment. Rolling eviction and
cross-request caches are outside this lab.

**Carry away.** A cache changes execution and state management even when
the intended attention equation stays the same.

## 6. Replace learned positions with RoPE

### Goal

Move position information from an added embedding table into rotations
applied to queries and keys.

### Need

The current model adds a learned row before the first block:

```text
token notes + position notes -> block input
```

Removing that table without a replacement would make two occurrences of
one token enter with no explicit position difference. The replacement
must affect which positions match during attention while leaving the
value cargo alone.

Start with a two-number row:

```text
before rotation       [1, 0]
quarter turn          [0, 1]
```

Its squared length stays one. Rotating a query and key by the same angle
also preserves their dot product. Rotating them by different
position-dependent angles makes the dot product depend on their
position difference.

Use `query = [1, 0]` and `key = [1, 0]`. Predict the dot product when
both receive a quarter turn, then when only the key does:

```text
before                    [1, 0] dot [1, 0] = 1
both turn                 [0, 1] dot [0, 1] = 1
only key turns            [1, 0] dot [0, 1] = 0
```

The equal turn preserves the match. The unequal turn changes it.

### Construct the production rule

Work inside each attention head. Let `D = C/H` be the head size and
pair adjacent channels:

```text
(0, 1), (2, 3), ... (D-2, D-1)
```

For pair index `i` and local position `p`, fix this variant:

```text
frequency(i) = 10000^(-2i/D)
angle(p, i)  = p * frequency(i)

x'[2i]   = x[2i]   * cos(angle) - x[2i+1] * sin(angle)
x'[2i+1] = x[2i]   * sin(angle) + x[2i+1] * cos(angle)
```

For head size `D = 4`, there are two pairs. Work their frequencies:

```text
i = 0    10000^(-2*0/4) = 10000^0      = 1
i = 1    10000^(-2*1/4) = 10000^(-1/2) = 1/100 = 0.01

at position p = 3:
pair 0 angle = 3 * 1    = 3 radians
pair 1 angle = 3 * 0.01 = 0.03 radians
```

Positions start at zero for every sequence in a batch. A rebuilt sliding
window also starts at zero, preserving Tiny's current local-position
policy. Apply the rule to query and key slices after their projection.
Do not rotate values.

This position-dependent query/key transformation is **rotary position
embedding**, or **RoPE**. Read the
[RoFormer paper](https://arxiv.org/abs/2104.09864) after the Tiny rule is
fixed if you want the longer construction.

Every head must contain complete pairs:

```text
C = 6, H = 2 -> D = 3 -> invalid
```

Checking even `C` is insufficient. The configuration predicate must
require `(C/H) % 2 == 0` after establishing that `C` is divisible by
`H`.

### Construct the backward route

A bounded implementation can rotate Q and K in the packed `qkv` buffer
in place after `matmul_forward` and before `attention_forward`.
Backward receives gradients with respect to those rotated values. Before
the QKV projection backward, rotate the Q and K gradient pairs by the
negative angle:

```text
raw QKV
    -> rotate Q and K
    -> attention
    -> attention backward gives d_rotated_Q and d_rotated_K
    -> rotate those gradients by -angle
    -> QKV projection backward
```

The inverse rotation is the transpose of the forward rotation. Values
follow the identity path.

The quarter-turn fixture makes that inverse visible:

```text
forward          [1, 0] rotated by  pi/2 -> [0, 1]
backward         [0, 1] rotated by -pi/2 -> [1, 0]
```

Keep rotation as a separate forward/backward operation so Chapter 7's
numeric referee can reach it directly. If the operation mutates its
input in place, a finite-difference fixture must restore the unrotated
input before every nudged forward call.

### Predict the reach

The default learned position table owns:

```text
T * C = 128 * 128 = 16,384 learned values
```

Predict the new parameter count before editing:

```text
old count       815,360
position rows  - 16,384
new count       798,976
```

Each removed value owned a value, gradient, and two AdamW moment slots:

```text
16,384 * 4 buffers * 4 bytes = 262,144 bytes
```

That byte subtraction assumes the in-place design adds no new arena
buffer. The independent formula in `model.c` must reach the same count.

Now predict the nonnumeric changes. Removing the second parameter shifts
the registry order. It also shifts the seeded initialization stream for
every later random parameter unless the design deliberately consumes
equivalent discarded draws. State which policy the fork uses. Old
known-answer losses cannot be copied into the new tests.

### Files and contracts

Group the impact before touching individual lines:

| Contract | Likely files |
|---|---|
| geometry and parameter count | `src/model.h`, `src/model.c` |
| registry and initialization | `src/model_internal.h`, `src/model_parameters.c` |
| token-only embedding and rotation | `src/ops.h`, `src/ops.c` |
| forward and backward wiring | `src/model_forward.c`, `src/model_backward.c` |
| public geometry diagnostic | `src/main.c`, `scripts/smoke-cli.sh` |
| saved architecture identity | `src/checkpoint.c` |
| local and whole-model math | `tests/gradcheck.c`, `tests/integration.c` |
| sampling and release-mode geometry | `tests/sampling.c`, `tests/model_invalid.c` |
| staged blueprints and fixtures | affected files under `labs/` |

`model_memory.c` needs a change only if the chosen design adds or
removes arena storage; the learned position table itself is a `Param`
already counted by `model.c`.

The format must change. A version 1 checkpoint contains learned position
rows and has no exact conversion to this RoPE model. The bounded fork
writes version 2 and rejects version 1. Loading version 1 and discarding
its position rows would pretend that a different model was restored.

The old bundled checkpoint, parameter counts, hashes, samples, and
recorded losses remain evidence for version 1. Do not overwrite them
during the capstone.

### Build

Begin with a scalar rotation fixture. Test a supplied angle of zero and
`pi/2` so the results are checkable:

```text
[1, 0] rotated by 0       -> [1, 0]
[1, 0] rotated by pi/2    -> [0, 1]
[3, 4] squared length     -> 25 before and after
```

Add a fixture that applies the production frequency schedule, respects
head boundaries, leaves values unchanged, and resets position at each
batch row. Then add finite differences for Q and K inputs.

Only after the operation passes should you:

1. remove the learned position parameter and its gradient route;
2. make embedding token-only;
3. wire forward rotation and backward inverse rotation;
4. update geometry, its CLI diagnostic, registry order, count, and
   memory calculations;
5. introduce checkpoint version 2 and its exact payload;
6. migrate affected known answers and invalid-construction fixtures.

Write experimental checkpoints under `build/capstones/rope/`.

### Verify

Run the affected functional gates:

```sh
make OPENMP=0 check-forward
make OPENMP=0 check-backward
make OPENMP=0 check-model
make OPENMP=0 check-sampling
make OPENMP=0 check-cli
make OPENMP=0 check-ndebug
make OPENMP=0 smoke
make OPENMP=0 overfit
make check-sanitizers
```

The new witnesses must cover:

```text
hand-calculated forward rotations
finite differences for Q and K
unchanged V channels
no pair crossing a head boundary
odd head-size rejection
CLI diagnostic for the even-head-size rule
parameter and buffer counts
version 2 round trip
version 1 rejection
truncated or wrong-sized version 2 payload rejection
```

The current oversized-construction fixtures use one-channel heads. Those
shapes become invalid before they reach the memory ceiling. Replace each
with an even-head-size shape that still exercises the intended resource
preflight.

### Expected

Known rotations match their hand calculations, the new backward route
passes finite differences, and every retained model parameter has a
connected finite gradient. A small run remains finite and can memorize
the fixed batch.

No old loss, checkpoint hash, sample, or timing is promised. Those
belong to the learned-position architecture. `make check-all` remains a
mainline migration gate until the affected labs, bundled artifact,
documentation, and evidence have been deliberately rebuilt.

### Common failures

- **Pairing across a head boundary.** Independent heads become mixed.
- **Checking even width instead of head size.** Some channels remain
  unpaired.
- **Rotating values.** The change reaches beyond positional Q/K
  matching.
- **Using the forward rotation in backward.** The gradient needs the
  transpose, or negative angle.
- **Reusing a rotated finite-difference input.** Each nudge measures a
  different function.
- **Removing the parameter from one count only.** Registry, independent
  formula, and checkpoint disagree.
- **Keeping checkpoint version 1.** The same bytes acquire a different
  meaning.
- **Reusing old known losses.** Initialization and architecture have
  changed.

### Stop when

Stop when known answers, finite differences, geometry rejection,
parameter counts, and version 2 round trips pass, and version 1 is
rejected explicitly. A quality comparison and bundled-model migration
are stretch work.

**Carry away.** "Position encoding" names a role. Learned addition and
rotary geometry are different mechanisms filling it.

## 7. Learn a BPE vocabulary

### Goal

Replace one-byte tokens with a learned variable-length token vocabulary
and compare fairly.

### Need

One byte per token makes each decision visible, but it spends one
context position on every byte. Begin with:

```text
a b a b a
```

Its adjacent pair counts are:

```text
pair       a b    b a
count       2      2
```

A repeatable trainer needs a tie rule. Base ids follow byte value, so
choose the smaller ordered id pair, `a b`. Replace non-overlapping
occurrences from left to right:

```text
before       a  b  a  b  a       5 tokens
merge a+b   [ab] [ab] a          3 tokens
```

The new id covers two bytes. Recount the new token sequence and repeat.
This builds a shorter sequence without losing the original bytes.

### Construct the merge program

Begin with ids `0` through `255`, one for every possible byte. A byte
absent from training remains representable later. Each successful merge
appends one id.

Fix all choices before training:

```text
winning pair   highest adjacent count
tie break      smaller left id, then smaller right id
replacement    non-overlapping, left to right
stop           512 merges or winning count below 2
```

With 512 successful merges:

```text
base ids          256
new merge ids     512
maximum vocab     768
```

Repeatedly merging the most frequent adjacent token pair is
**byte-pair encoding**, or **BPE**. Chapter 3 already constructed why a
[subword token](03-data.md#give-each-byte-a-compact-number) can cover
more text per position. The
[Sennrich, Haddow, and Birch paper](https://aclanthology.org/P16-1162/)
is further reading after this concrete byte construction.

### Merge order is tokenizer state

Token byte strings alone do not settle how new text is split. Consider:

```text
rank 0       a + b  -> ab
rank 1      ab + a  -> aba
```

Applied in rank order, `aba` becomes one token. If `b + a -> ba` ran
first, the same input could become `a, ba`. A vocabulary can contain all
of `a`, `b`, `ab`, `ba`, and `aba` while segmentation still depends on
the learned order.

Serialize the merge program, not an unordered bag of strings. Base ids
can be implicit. Each record then stores:

```text
new id       left id       right id
256             97             98
257            256             97
```

Both child ids must be lower than the new id. That rule keeps the merge
graph from containing a dependency loop: no id can eventually depend on
itself. Decoding can therefore reconstruct the complete byte sequence
under every token. Encoding begins with base-byte ids and replays the
stored merges in rank order using the same non-overlap rule.

The checkpoint loader must bound the merge count, child ids, token
lengths, and total serialized size before allocation. The current
preflight assumes exactly one tokenizer byte per vocabulary row; that
assumption must disappear under a new checkpoint version.

### Change the API honestly

The current encoder uses `ids written == source bytes` as evidence that
no byte was skipped. Merging breaks that equality:

```text
source bytes       a b a b a       count 5
encoded ids        ab ab a         count 3
all bytes consumed                  yes
```

The new API must report both the number of ids written and whether every
input byte was consumed. Validation setup checks complete consumption,
not equality between token and byte counts. Because merging never
increases the count, `Dataset` can still allocate one integer per source
byte as a safe upper bound.

Decoding changes too. One id can no longer return one `char`. It must
return a byte span or write to a caller buffer with an explicit
capacity. CLI output writes the returned byte count, rather than calling
`putchar` once.

An API test can round-trip an embedded zero byte. A shell command-line
prompt cannot contain that byte because `argv` strings and the current
`strlen` boundary end at zero. Keep the tokenizer buffer contract and
the public shell-input contract separate. Generated output may also
contain control bytes from base tokens that never appeared as training
targets, so write experimental samples to a file and inspect their
bytes rather than assuming terminal text.

### Compare in one unit

One token can now cover several bytes. A lower mean token loss may only
reflect the larger answer unit.

Suppose two target tokens cover two bytes and one byte, with penalties
`0.4` and `0.8`:

```text
mean token loss       (0.4 + 0.8) / 2 = 0.6
predicted bytes        2 + 1           = 3
loss per byte          (0.4 + 0.8) / 3 = 0.4
```

Because the penalties use natural logarithms, the last quantity is
**nats per byte**. Accumulate total target-token penalty and total bytes
represented by those target tokens, then divide once. If a global mean
loss and a global mean token length cover the same tokens with the same
weighting, their ratio gives this same answer. Totals make that
denominator visible and avoid a different mistake: averaging
per-window ratios as if windows with different byte counts had equal
weight.

The same file alone does not give the two tokenizers the same graded
bytes. Predict the targets for this file:

```text
held-out bytes       a b c
byte tokens          [a] [b] [c]    graded bytes b and c
BPE tokens           [ab] [c]       graded byte c
```

Neither model can grade its first token because no earlier context
exists. The BPE model's first token may cover more than one byte.

Use one exact shared target interval. Tokenize the complete held-out file
with the frozen BPE program. Let `start` be the byte offset immediately
after its first token. Grade bytes from `start` through the end of the
file under both tokenizers:

```text
byte model     keep the prefix as context, omit its targets, then grade
               every byte from start
BPE model      leave its first token as context, then grade every
               later token and all bytes covered by those tokens
denominator    file byte count - start
```

Run this as one sequential sliding-context evaluation. For each target,
give the model at most its newest `T` earlier tokens. Do not reset at
arbitrary file chunks. Record `start`, the final byte offset, and the
exact common denominator. A shared non-mergeable beginning token could
recover the excluded prefix, but that changes the tokenizer and model
and is outside the bounded comparison.

Context needs both units too. A four-token window with token lengths
`[3, 1, 2, 4]` covers ten bytes. Another four-token window may cover a
different number. Report the observed mean and range of byte coverage
instead of calling `T = 128` the same text span in both models.

### Predict the reach

Before integration, predict how a chosen vocabulary size changes:

```text
tied token-table values       V * C
logits and probability rows   B * T * V
checkpoint tokenizer payload  ranked merge records
generated length              tokens, not bytes
```

Raising `MODEL_MAX_VOCAB_SIZE` is only one edit. Audit every
vocabulary-sized count, multiplication, allocation, file-size check,
and output loop.

| Contract | Likely files |
|---|---|
| training, encode, decode, serialization | `src/tokenizer.h`, `src/tokenizer.c` |
| token buffer and counts | `src/dataset.c` |
| vocabulary limits and memory | `src/model.h`, `src/model.c`, `src/model_memory.c` |
| validated payload and format version | `src/checkpoint.c` |
| validation completeness and output | `src/main.c` |
| build wiring and dedicated fixtures | Makefile and tests |

The parameter constructor already sizes the tied table from `V`, but
its limits and every surrounding buffer still need witnesses.

### Build

Train merges from training bytes only, then freeze that ranked list
before encoding validation. Use four phases:

1. merge trainer with deterministic tiny fixtures;
2. length-aware encode and decode API;
3. bounded tokenizer serialization and checkpoint version;
4. model, metric, and CLI integration.

The first fixtures should include:

```text
ababa        tie break and repeated replacement
aaaa         overlapping pair handling
all 256      every base byte survives
a 0x00 b     byte-buffer round trip
```

Add tokenizer round trips before changing the model. Then raise the
vocabulary ceiling, replace the validation completeness check, update
printing, and migrate the checkpoint. Write experimental checkpoints
under `build/capstones/bpe/`; do not replace `tiny-agenc.bin`.

### Verify

Run the affected functional gates:

```sh
make OPENMP=0 check-data
make OPENMP=0 check-model
make OPENMP=0 check-sampling
make OPENMP=0 check-cli
make OPENMP=0 smoke
make OPENMP=0 overfit
make check-sanitizers
```

The current sanitizer target does not run `check-cli`. Wire a dedicated
variable-length output witness into its sanitized targets, or add the
affected CLI fixture there, before counting the generated-byte path as
covered.

The new tokenizer and checkpoint witnesses must cover:

```text
all 256 byte values, including the zero byte
deterministic ties and overlapping replacements
identical merge replay after serialization
training-only merge learning
complete validation-byte consumption
vocabulary sizes above 256
malformed child ids, ranks, lengths, and truncated payloads
checked serialized-size arithmetic before allocation
multi-byte generated-token output
one shared target-byte interval under both tokenizers
newest-`T` context with no arbitrary chunk resets
```

For the model comparison, record the training and validation hashes,
merge count, vocabulary size, parameter count, token count, shared
starting byte offset, total target bytes, nats per byte, and
token-window byte coverage.

### Expected

The same training bytes and budget reproduce the same ranked merge
list. Every byte buffer round-trips through the tokenizer API.
Validation uses the frozen training tokenizer and consumes every byte.
A versioned checkpoint reconstructs the merge program and parameter
rows exactly.

No lower held-out loss is promised. Mean token loss is not a fair
cross-tokenizer result; nats per byte over the declared common target
interval supplies the shared unit. Both records must name the same
starting and final byte offsets and the same target-byte denominator.
`make check-all` remains a mainline migration gate until the byte-token
labs, bundled checkpoint, documentation, and recorded evidence have
been deliberately migrated.

### Common failures

- **Serializing token strings without merge ranks.** Encoding behavior
  is lost.
- **Learning merges from validation.** Held-out structure shapes the
  tokenizer.
- **Treating fewer ids than bytes as failure.** Correct merges shorten
  the sequence.
- **Keeping a one-character decoder.** Multi-byte tokens are truncated.
- **Using `strlen` for arbitrary token bytes.** An embedded zero ends the
  string early.
- **Raising the vocabulary limit alone.** Logits, preflight, and file
  sizes retain narrowing assumptions.
- **Comparing token loss across tokenizers.** The answer units differ.
- **Calling the same file the same targets.** Different first tokens can
  exclude different leading bytes.
- **Resetting at arbitrary chunks.** Each reset removes another target
  and changes the visible context.
- **Calling equal token windows equal context.** Their byte coverage
  differs.
- **Keeping checkpoint version 1.** The tokenizer payload has changed
  meaning.

### Stop when

Stop when merge training is deterministic, arbitrary byte buffers
round-trip, validation uses training merges only, the new checkpoint
round-trips, the model and CLI accept the larger vocabulary, and the
comparison reports the shared target interval, nats per byte, and byte
context coverage. Faster merge training is a separate project.

**Carry away.** Tokenization changes the units seen by every later
dimension, metric, and context claim.

## 8. Scale study

### Goal

Measure how capacity, data, memory, and wall time trade against one
another in this implementation.

### Need

If layer count and width both increase, a changed loss cannot identify
which change mattered. A different batch, context, corpus, thread count,
or training budget would add more possible causes.

This core study changes width only:

```text
vocabulary V       80
context T          64
heads H             4
layers L            2
batch B             8
steps             500
seed             1337
width C        32, 64, 96
```

Every run sees the same number of target slots:

```text
500 steps * 8 rows * 64 targets = 256,000 target slots
```

The corpus, split, target positions, random starting procedure, build,
and training budget stay fixed. Width changes parameter capacity,
channel work, and buffers.

An equal-wall-time comparison asks a different question and is stretch
work. Do not mix its result into the equal-target-slot table.

### Predict the allocation

Chapter 1's [parameter
count](01-the-map.md#every-learned-number) is:

```text
P = (V + T)C + (4L + 2)C + 12LC^2
```

For this sweep:

```text
P = (80 + 64)C + (4*2 + 2)C + 12*2*C^2
  = 154C + 24C^2
```

Chapter 10's [model memory
report](10-memory-planning.md#the-public-memory-report-is-an-independent-calculation)
counts four parameter buffers, the value arena, the
activation-gradient arena, and two id buffers. With four-byte `float`
and `int` values:

```text
R = B*T             = 512
N = R*C
S = R*H*T           = 131,072
Q = R*V             = 40,960

A = 2N + L(18N + S + 4R) + 2R + 2Q
G = 2N + L(18N + S)      + Q

total bytes = 16P + 4A + 4G + 8R
```

Work the first row by hand, then complete the table before starting any
training process:

| `C` | `P` | parameter bytes | value bytes | gradient bytes | id bytes | total bytes |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 29,504 | 472,064 | 3,887,104 | 3,702,784 | 4,096 | 8,066,048 |
| 64 | 108,160 | 1,730,560 | 6,377,472 | 6,193,152 | 4,096 | 14,305,280 |
| 96 | 235,968 | 3,775,488 | 8,867,840 | 8,683,520 | 4,096 | 21,330,944 |

The totals are about `7.7`, `13.6`, and `20.3` MiB. All are below the
CLI's one-GiB model-buffer ceiling.

This is an exact calculation for the buffers named by `ModelMemory`.
It is not peak process memory. The tokenizer, corpus, small structures,
allocator metadata, executable, and other process storage remain
outside that report.

### Predict the measurements

Which quantities must rise with width: parameter count, reported model
memory, training time, or held-out quality?

Parameter count and the buffer report must rise because their formulas
contain `C`. Training time is measured on this build and machine.
Held-out quality is an experimental result, not a consequence of making
one dimension larger.

Use two timing columns:

```text
reported ms/step    completed training work only
shell real time     complete command, including reports and saves
```

Chapter 15 constructed the
[training timer boundary](15-the-training-loop.md#time-only-completed-learning-work).
Chapter 17 explains why [timings belong to one
build](17-the-training-run.md#timings-belong-to-one-build). Do not call
either number floating-point operations. An operation count would need
its own declared convention.

### Files

The core study changes no tracked file. Read the formulas in
[`model.c`](../src/model.c) and the CLI preflight in
[`main.c`](../src/main.c). Store all outputs under
`build/capstones/scale/`.

### Build

Build one portable serial executable and record the fixed inputs:

```sh
make OPENMP=0 NATIVE=0 tiny-agenc
mkdir -p build/capstones/scale

cp build/openmp-0/.build-config \
    build/capstones/scale/build-config.txt
sha256sum tiny-agenc \
    > build/capstones/scale/executable.sha256
sha256sum data/cyberpunk.train.txt data/cyberpunk.val.txt \
    > build/capstones/scale/inputs.sha256
```

The copied build configuration records Make's effective compiler,
preprocessor flags, C flags, linker flags, libraries, OpenMP mode,
native-code mode, and compiler version. It is stronger evidence than
running `cc --version` separately, because `CC` may select another
compiler.

Run the smallest width first:

```sh
for sweep_width in 32 64 96; do
    {
        /usr/bin/time -p ./tiny-agenc train \
            --data data/cyberpunk.train.txt \
            --val-data data/cyberpunk.val.txt \
            --out "build/capstones/scale/width-${sweep_width}.bin" \
            --steps 500 \
            --layers 2 \
            --heads 4 \
            --width "$sweep_width" \
            --block 64 \
            --batch 8 \
            --lr 0.001 \
            --seed 1337
    } > "build/capstones/scale/width-${sweep_width}.log" \
      2> "build/capstones/scale/width-${sweep_width}.time"
done
```

For each width, record:

```text
predicted and printed parameter count
predicted and printed model-buffer total
step-500 training loss
step-500 held-out loss
reported ms/step
shell real time
```

Three raw points fit in a table. A plot is optional.

### Keep data scale separate

Corpus growth is a different experiment. Do not change data during the
width sweep.

If you later extend NIGHT GRID, copy the raw input to an ignored path
first. `gen-corpus.sh` appends to its output:

```sh
SCALE_DATA_DIR=build/capstones/scale-data
mkdir -p "$SCALE_DATA_DIR"
cp -- data/cyberpunk.raw.txt "$SCALE_DATA_DIR/night-grid.raw.txt"

RAW_FILE="$SCALE_DATA_DIR/night-grid.raw.txt" \
TARGET_BYTES=1300000 \
    bash scripts/gen-corpus.sh

bash scripts/clean-corpus.sh \
    "$SCALE_DATA_DIR/night-grid.raw.txt" \
    "$SCALE_DATA_DIR/night-grid.txt"

make OPENMP=0 build/split-order

VALIDATION_PERCENT=10 SPLIT_SEED=1337 \
    bash scripts/split-corpus.sh \
    "$SCALE_DATA_DIR/night-grid.txt" \
    "$SCALE_DATA_DIR/night-grid.train.txt" \
    "$SCALE_DATA_DIR/night-grid.val.txt"

sha256sum "$SCALE_DATA_DIR"/night-grid*.txt \
    > "$SCALE_DATA_DIR/inputs.sha256"
```

The generator requires Bash, Ollama, `curl`, `jq`, and the GNU/Linux
`timeout`, `shuf`, and `stat` utilities used by the script. Its added
text is a new stochastic result, not a replay of the historical corpus.
Use the same new train and validation hashes for every model in that
separate study. Do not attribute a
canonical-versus-extended-data difference to width.

### Verify

Inspect the architecture banners and final reports:

```sh
grep -Eh 'parameters|step +500/500|checkpoint saved' \
    build/capstones/scale/width-*.log
```

The parameter counts must match the table. The banner rounds MiB to one
decimal place, so its buffer totals should display `7.7`, `13.6`, and
`20.3` MiB. Every loss must be finite, every command must write its own
checkpoint, and the recorded inputs and settings must match.

Because the core capstone changes no tracked source or data, the
reference identity gate must still pass:

```sh
make OPENMP=0 check-all
```

### Expected

All three configurations pass geometry because every width is divisible
by four. Parameter counts and model-buffer reports rise exactly as
predicted.

Loss and timing order remain measurements. Three points establish three
results for this corpus, budget, seed, build, and machine. They do not
show that wider models are always better.

An empirical relationship fitted across model, data, or training scales
is called a **scaling law**. Three tiny points neither establish nor test
one.

### Common failures

- **Changing width and depth together.** The comparison no longer
  isolates one dimension.
- **Changing `B` or `T` silently to fit memory.** Target slots and
  activation work change too.
- **Mixing equal-target and equal-time results.** The budgets answer
  different questions.
- **Treating `ModelMemory` as peak process memory.** It names a narrower
  set of buffers.
- **Reading lower training loss as better generalization.** Held-out
  loss is the separate measurement.
- **Comparing `ms/step` with wall time as one quantity.** Their timed
  regions differ.
- **Extending the canonical raw corpus in place.** A pinned evidence
  input is lost.
- **Fitting a law to three points.** The claim outruns the sweep.

### Stop when

Stop when one predeclared width sweep finishes within its resource
ceiling, calculated counts match program reports, and one narrow result
is tied to the recorded environment.

**Carry away.** Scaling laws are empirical relationships under
controlled budgets, not a spell cast by making `--width` larger.

## The final acceptance criterion

Finishing an experiment can mean three different things:

```text
baseline identity
    the committed program and evidence are unchanged

functional fork
    the new behavior works and preserved contracts still pass

upstream-ready fork
    every affected test, lab, artifact, and evidence record has moved
```

Do not report the third when you have reached only the second. A
tokenizer or architecture fork can be functionally correct while the
old checkpoint, lab copies, bundle metadata, and evidence hashes are
correctly rejecting its new identity.

### Build

Start from the clean serial baseline recorded at the beginning of this
appendix. Add the smallest focused witness that would fail if the new
mechanism were wrong. Wire that witness into the normal build, including
the sanitizer build, rather than leaving it as a command only you know.

If the capstone was observational, or if you restored a deliberate
mutation, return to baseline identity:

```sh
make OPENMP=0 check-all
```

If the capstone created a functional fork, list every old contract that
should still hold and every identity contract that the change
invalidated. Run the focused new witness and the old focused witnesses
for preserved behavior before attempting the broad suite.

Write a short experiment report answering:

1. What contract changed?
2. Which existing test would have missed the mistake you feared most?
3. What new evidence says the feature works?
4. What did the experiment still not prove?
5. Which previous evidence became inapplicable, and why?

### Verify

For a functional fork, run the relevant focused witness first. Then ask
Make to keep running independent targets after one fails. Its `-k`
option makes the broad serial suite a more useful impact audit:

```sh
make -k OPENMP=0 check-all
make OPENMP=0 check-install
```

`check-install` is the final recipe of `check-all`, so an earlier failed
prerequisite can prevent it from running even with `-k`. Invoke it
separately to expose the installation boundary.

Read every failure. A source hash mismatch after changing source is not
a numerical failure. A checkpoint rejection after changing parameter
geometry may be the intended compatibility boundary. A lab failure can
mean that the teaching copy has not moved with the answer key. Name each
one; an unexplained failure is still a defect.

Do not silence `check-evidence` by replacing a hash. Evidence may move
only after its complete recorded procedure has been rerun with the new
source, inputs, seed, flags, and environment. Architecture and tokenizer
changes can also require a new checkpoint version, migrated fixtures,
updated labs, bundle metadata, installer checks, and new measured logs.

An upstream-ready fork passes the repository's complete final gate:

```sh
make OPENMP=0 check-all
make OPENMP=1 check-parallel
make check-sanitizers
```

The OpenMP run checks behavior that a serial build cannot exercise. The
sanitizer run checks invalid memory use and undefined behavior; it is a
separate target and is not included in `check-all`.

### Expected

An observational capstone or restored mutation reaches baseline
identity: the broad serial suite passes without changing recorded
evidence.

A functional fork has a passing new witness, passing witnesses for every
preserved contract, and a written explanation for each old identity
check that no longer applies. It is a sound experiment, not yet an
upstream contribution.

An upstream-ready fork has migrated every affected contract and passes
all three final commands. Its report states a narrow result in the unit
actually measured.

### Common failures

- **Calling one generated sample proof.** Sampling demonstrates a
  possible outcome, not a general property.
- **Updating expected bytes until a test passes.** Compatibility bytes
  are a contract; explain why they changed before accepting new ones.
- **Replacing an evidence hash without rerunning the procedure.** The
  record no longer identifies the measurement it claims.
- **Ignoring a failed lab because the source test passes.** The lab is
  a second implementation and must teach the same contract.
- **Leaving a focused witness outside the build.** The next change can
  break the feature without the repository noticing.
- **Calling a functional fork merge-ready.** Known artifact, evidence,
  lab, or installation failures still mark unfinished migration work.

The report is part of the implementation. Code records what the machine
does. The report records what you asked and what the result can support.

---

[Previous: Appendix B](appendix-b-sources.md) | [Contents](README.md)
