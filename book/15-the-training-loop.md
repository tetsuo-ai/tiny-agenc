# Chapter 15: The Training Loop

We have spent fourteen chapters constructing a function, its derivative,
its memory, its file format, and the boundary around them.
Training is the moment they acquire a purpose.

Chapter 14 deliberately stopped one operation early. Its
[learner-track seam](14-the-command-line.md#stop-at-an-honest-learner-track-seam)
accepted a valid command, constructed an initialized model, and saved
that unchanged model while saying that no updates had run. The
checkpoint proved that the command boundary worked. It did not prove
that the model could learn.

This chapter replaces that honest temporary ending with the repeated
work that changes the parameters.

## Begin where the boundary stopped

Do not rebuild command parsing, corpus loading, tokenization, geometry
checks, or model-memory checks here. Chapter 14 already admitted a
checked set of objects:

```text
checked TrainOptions
        |
        +-> training Dataset
        +-> optional validation Dataset
        +-> Tokenizer
        +-> initialized Model
        |
        v
Chapter 14: save unchanged parameters and say "no updates yet"
Chapter 15: draw batches, learn, observe, and save changed parameters
```

The temporary ending can produce two checkpoints from the same seed
that are byte-for-byte equal even when one request asks for one step and
the other asks for fifty. No operation between construction and saving
uses the corpus to change a parameter.

**Predict:** what minimum learning event can the fifty-step run contain
that the one-step run cannot?

It needs at least one additional completed update, beginning with step
2. That update requires a target-bearing forward pass followed by its
matching backward pass and a successful parameter update. Repeating
only the forward pass would calculate more grades while leaving the
adjustable numbers untouched.

The progress-text generator must not consume a batch draw. Chapter 14
already built the public seed policy. The shared evaluation header gives
progress text its own offset:

```c
/* Keep progress-sample draws from changing which corpus windows train next. */
static const unsigned long long SAMPLE_SEED_OFFSET =
    TINY_AGENC_SAMPLE_SEED_OFFSET;
```

The completed `run_train` keeps Chapter 14's checked setup and replaces
the provisional direct save with two long-lived RNG handles and a call
to `train_loop`. The final lines of the setup helper create those three
owners:

```c
resources.model      = model_new(options->cfg, options->seed);
resources.batch_rng  = rng_new(options->seed);
resources.sample_rng = rng_new(options->seed + SAMPLE_SEED_OFFSET);
```

`run_train` itself only coordinates the named phases:

```c
static int run_train(TrainOptions options)
{
    reject_training_path_collisions(&options);

    TrainingResources resources = prepare_training_resources(&options);

    print_training_summary(&resources, &options);
    train_loop(&resources, &options);
    printf("tiny-agenc: checkpoint saved to %s\n", options.out_path);
    free_training_resources(resources);
    return EXIT_SUCCESS;
}
```

This is the complete [`run_train`](../src/main.c). `model_new`
temporarily creates and consumes its own Chapter 9
[initialization generator](09-parameters-and-the-blueprint.md#initialization-is-architecture),
then frees it. `resources.batch_rng` is a different handle created
later with the same numeric seed. The two handles begin from the same
PCG state, so calling them statistically independent would be false.
Their state is separate: Chapter 2's
[seeded-handle rule](02-foundations.md#enter-the-sequence-through-a-seed)
means draws made during construction cannot advance the later batch
handle.

The progress-text handle starts at `seed + 1`. The temporary validation
handle built later starts at `seed + 2`. Advancing either one cannot
consume the next batch draw. Changing the one public seed still changes
both initialization and batch selection; this CLI does not expose a way
to vary those two jobs separately.

The loop returns only after the final checkpoint has been saved. The
success line therefore says `checkpoint saved`, not `initialized
checkpoint saved`. The final helper releases the six long-lived owners
in their required order:

```c
static void free_training_resources(TrainingResources resources)
{
    rng_free(resources.sample_rng);
    rng_free(resources.batch_rng);
    model_free(resources.model);
    if (resources.validation_data != NULL)
        dataset_free(resources.validation_data);
    dataset_free(resources.training_data);
    tokenizer_free(resources.tokenizer);
}
```

## One cycle must close before another begins

The model already knows every individual operation. It does not know
when the caller should perform them.

Use the two-row fixture from
[`check15-model.c`](../labs/check15-model.c). Its tokenizer assigns ids
in sorted byte order:

```text
id       0        1  2  3
byte     newline  a  b  c
```

With `B = 2` and `T = 4`, both fixed rows contain the same shifted
question-and-answer sequence:

```text
input row 0    newline  a  b  c       ids 0 1 2 3
target row 0   a        b  c newline  ids 1 2 3 0

input row 1    newline  a  b  c       ids 0 1 2 3
target row 1   a        b  c newline  ids 1 2 3 0
```

Chapter 3 built why each target sits
[one token ahead](03-data.md#one-extra-token-supplies-every-answer).
The eight target ids now grade eight next-character bets together.

Suppose an old batch left these two illustrative gradient entries:

```text
old gradient entries       [ 0.30, -0.20 ]
after clearing             [ 0.00,  0.00 ]
new batch's backward pass  [ 0.08, -0.05 ]
```

The actual model has many more entries. These two are a control-flow
ledger, not a recorded model measurement.

**Predict:** if the clear is omitted, which entries would the updater
receive?

They would be `[0.38, -0.25]`, the old and new corrections added
together. That would silently make one update describe two batches.
Chapter 6 constructed
[gradient accumulation](06-backprop-by-hand.md#add-every-returning-path);
it is required where several graph paths return to one value. Carrying
an earlier training batch into a later one is a different accumulation,
and this program does not request it.

The order for one cycle is therefore forced:

```text
draw one shifted batch
          |
          v
erase parameter and activation gradients
          |
          v
target-bearing forward -> one mean loss and one latest-forward record
          |
          v
matching backward -> gradients for this batch
          |
          v
checked model update -> changed parameters
```

Moving backward above forward would leave no matching saved values.
Putting another forward between them would replace Chapter 11's
[latest-forward record](11-wiring-the-model-forward.md#only-the-newest-forward-record-survives).
Updating before backward would lack this batch's gradients. Weight
decay or stored AdamW history could still move some values, but that
move would not describe the loss the current batch produced.

Some placements have room. Gradient clearing could happen after forward
but before backward because forward does not write gradients. The
source puts it at the cycle entrance so the new batch begins at a
visible clean boundary. Drawing at the top likewise keeps one batch
attached to one cycle. The immediate forward-to-backward pairing and
backward-before-update order are the load-bearing constraints.

That safe, visible cycle is Chapter 1's
[training step](01-the-map.md#from-known-snippets-to-one-adjustment).
Repeating it, with observation work only after each completed update,
forms the program's **training loop**.

The ceremony is shorter than the preparation:

```text
┌──► draw a batch
│    clear old gradients
│    predict and measure the loss
│    run the chain rule backward
│    clip and update the parameters
│    perform any due observation work
└──── repeat from the top
```

That is not a summary hiding another system. It is the system.

## The complete learning step

The update has its own helper in [`main.c`](../src/main.c). The function
is complete:

```c
static float run_training_step(TrainingResources *resources,
                               TrainingStepState *state,
                               const TrainOptions *options, int step)
{
    dataset_batch(resources->training_data, resources->batch_rng,
                  state->inputs, state->targets,
                  options->cfg.batch_size, options->cfg.block_size);
    model_zero_gradients(resources->model);

    float loss =
        model_forward(resources->model, state->inputs, state->targets,
                      options->cfg.batch_size, options->cfg.block_size);

    model_backward(resources->model);
    if (model_step(resources->model, state->optimizer, step) != 0)
        die("non-finite gradient or invalid optimizer update at step %d",
            step);
    return loss;
}
```

`resources` supplies objects that live for the whole command. `state`
supplies the batch buffers and optimizer recipe owned by the loop. Read
the calls in order.

`dataset_batch` fills `B*T` input ids and `B*T` target ids. Chapter 3's
[window draw](03-data.md#draw-every-legal-start) samples each row with
replacement, so a later step may revisit a corpus position.

`model_zero_gradients` clears both parameter gradients and the reusable
activation-gradient arena. Chapter 12 established why the two
[destination families start clean](12-wiring-the-model-backward.md#start-from-clean-destinations).

`model_forward` receives targets, so this is not Chapter 11's inference
mode. It runs the shared score route, grades all `B*T` positions, caches
the values backward needs, and returns Chapter 5's
[mean cross-entropy loss](05-forward-pass.md#cross-entropy-keeping-score).
The local variable `loss` belongs to the parameters as they existed
before this step's update.

`model_backward` immediately replays that target-bearing record in
reverse. Its exact assembly is
[Chapter 12's route](12-wiring-the-model-backward.md#walk-the-whole-model-in-source-order).

`model_step` asks the Chapter 9
[parameter registry](09-parameters-and-the-blueprint.md#apply-one-operation-to-the-whole-model)
to validate the recipe, clip the model-wide gradient when required,
and visit each `Param` with Chapter 8's AdamW updater. A zero return
means the update completed. A nonzero return means the command cannot
claim a healthy next step, so the CLI stops with the step number.

The failure boundary is exact. Invalid optimizer settings and
nonfinite gradients are rejected before parameter updates begin.
Corruption discovered inside a later `Param` update is not rolled back
across earlier parameters. `model_step` is a checked update, not a
transaction.

## Allocate repeated storage once

Putting allocation inside the loop would request and release the same
two batch buffers on every step. Their capacity depends only on the
checked configuration, so the loop can reserve them once.

For `B = 2` and `T = 3`:

```text
span = B*T = 2*3 = 6 token positions

inputs   needs 6 int slots
targets  needs 6 int slots
```

The buffers and the optimizer recipe travel together through each
update helper, so the source gives them one small record:

```c
typedef struct {
    int   *inputs;
    int   *targets;
    AdamW  optimizer;
} TrainingStepState;
```

One helper prepares that record before the loop:

```c
static TrainingStepState prepare_training_step_state(
    const TrainOptions *options)
{
    int span = options->cfg.batch_size * options->cfg.block_size;
    TrainingStepState state;

    state.inputs     = emalloc((size_t)span * sizeof *state.inputs);
    state.targets    = emalloc((size_t)span * sizeof *state.targets);
    state.optimizer  = adamw_with_rate(options->learning_rate);
    return state;
}
```

`span` is safe because Chapter 14 already checked the combined `B*T`
ceiling. The two `emalloc` calls turn six positions into a byte request
with the Chapter 2
[`sizeof` pattern](02-foundations.md#a-block-of-bytes-and-one-owner).
The returned record copies two owning pointers and one small optimizer
value. The allocations live until the loop ends.

Only the learning rate is exposed as a train flag. The other recipe
values remain the Chapter 8 policy:

```c
/* Keep the remaining AdamW constants fixed so the teaching CLI stays
 * focused; Chapter 8 records the policy. */
static const float ADAM_BETA1   = 0.9f;
static const float ADAM_BETA2   = 0.999f;
static const float ADAM_EPSILON = 1e-8f;
static const float WEIGHT_DECAY = 0.01f;
```

`adamw_with_rate` copies the requested learning rate beside those fixed
values:

```c
static AdamW adamw_with_rate(float learning_rate)
{
    AdamW opt = { learning_rate, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, WEIGHT_DECAY };

    return opt;
}
```

Chapter 4 introduced this
[positional struct initializer](04-poor-mans-tensors.md#put-the-shape-beside-the-address).
Its fields match the public `AdamW` record in `param.h`: learning rate,
first-history fraction, second-history fraction, denominator guard, and
direct pull toward zero. Chapter 8 constructed their meanings and
[validates the recipe before changing history](08-adamw.md#check-the-recipe-before-changing-history).
This adapter changes only the exposed learning rate.

The optimizer field is therefore created once before the first
iteration:

```c
state.optimizer  = adamw_with_rate(options->learning_rate);
```

No core training step allocates memory. The model arenas, batch buffers,
and optimizer moments all exist before the first draw. Later sections
will construct the optional observation storage before showing the
complete function.

## Repeat and count from one

The loop counter itself has this shape. This is a schematic with the
body omitted, not a source excerpt:

```text
for (int step = 1; step <= options->steps; step++) {
    /* one update and its due observation work */
}
```

For `--steps 3`, predict the three values passed to `model_step`.

The initializer stores `1`, the condition admits `1`, `2`, and `3`,
and the increment runs after each body:

```text
requested steps       3
step values           1, 2, 3
number of updates     3
```

Starting at one is mathematical, not cosmetic. Chapter 8's
[bias correction](08-adamw.md#repair-the-zero-start) contains
`1 - beta^step`. At step zero that denominator is zero. The public
updater rejects nonpositive step numbers.

A rejected update never reaches the observation work for that step.

## Make one tiny batch impossible to ignore

A complete model should first prove that it can memorize one batch.
This is not a quality goal. It is a wiring test.

Fresh random batches can make a broken run hard to read. One batch may
be easier than the next, so loss can move down by chance even when an
update did little. Reusing the exact `\nabc` fixture removes that source
of change:

```text
step 1   same 8 input ids -> same 8 answers
step 2   same 8 input ids -> same 8 answers
step 3   same 8 input ids -> same 8 answers
...      only parameters and optimizer history change
```

**Predict:** if the loss falls sharply while the input and answers stay
fixed, what changed?

Only the model state and optimizer history can account for the
improvement. That makes the fixed batch a useful assembled-path
witness.

Reusing one unchanged question-and-answer batch through many steps and
requiring a large finite loss drop is a **fixed-batch memorization
test**. It tests the assembled learning path, not prediction on unseen
text.

The focused lab performs sixty one-based updates:

```c
for (int step = 1; step <= STEPS; step++) {
    model_zero_gradients(model);
    (void)model_forward(model, inputs, targets, BATCH, TIME);
    model_backward(model);
    expect(model_step(model, optimizer, step) == 0,
           "the fixed-batch optimizer step stays finite");
}

float trained_loss =
    model_forward(model, inputs, targets, BATCH, TIME);
```

This exact excerpt from
[`check15-model.c`](../labs/check15-model.c) uses the four-call contract
we constructed. `(void)` says the loop intentionally discards each
intermediate returned loss. After all updates, one final forward pass
measures the trained model on the unchanged fixture.

The witness requires both losses to remain finite and requires the
trained loss to be less than half the initial loss. It then saves and
reloads the checkpoint, compares every parameter bit, and requires the
loaded model to reproduce the same loss bits.

The repository also has a stricter memorization witness. Its acceptance
condition is source policy rather than an unrecorded output number:

```c
int passed = finite_float(initial_loss)
          && finite_float(final_loss)
          && final_loss < 0.10f
          && final_loss < initial_loss * 0.10f;
```

If an illustrative initial loss were `1.6`, the two upper bounds would
be `0.10` and `0.16`, so the stricter requirement would be `0.10`.
The real command prints the losses produced by the current compiler and
machine, then applies the conditions above.

The result proves that the assembled forward, backward, and update path
can drive its own grade down on one unchanging problem. It does not
prove that clipping activated, that every derivative is correct, or
that the model predicts unseen text. The
[Chapter 7 referee](07-trust-but-verify.md#the-operation-checks),
Chapter 12
[connectivity witness](12-wiring-the-model-backward.md#two-witnesses-make-different-claims),
and held-out measurement answer those different questions.

## Training loss is not a report card

The loop's `loss` comes from one random training batch. It answers:

> How surprised was the current model by these windows from material it
> is allowed to learn?

It does not answer:

> How well does the model predict material it never trained on?

Suppose two reports happen to draw batches with these losses:

```text
step 50 random training batch     1.20
step 100 random training batch    0.90
```

The second batch might contain easier local patterns. Those two numbers
alone cannot separate a better model from an easier draw.

Chapter 3 already constructed the
[whole-transmission split](03-data.md#keep-complete-scenes-out-of-training).
The training file supplies examples allowed to change parameters. The
validation file holds complete transmissions behind the no-peeking
boundary. Chapter 14 then checked that both files use the training
tokenizer and can supply full windows.

The missing mechanism is a stable way to grade held-out examples while
training proceeds.

## Deal held-out windows once

Drawing a new validation batch at every report would add the same
moving-target problem:

```text
report 50    model A on held-out windows Q
report 100   model B on held-out windows R
```

If the grade changes, both the model and the questions changed. Keep the
questions fixed:

```text
before step 1   draw held-out batches Q0, Q1, Q2, Q3
report 1        grade current model on Q0, Q1, Q2, Q3
report 50       grade current model on Q0, Q1, Q2, Q3
report 100      grade current model on Q0, Q1, Q2, Q3
```

For a hand-sized configuration with `B = 2`, `T = 3`, and four held-out
batches:

```text
span per batch       B*T = 2*3 = 6 positions
stored positions     4*6 = 24 inputs and 24 targets
flat batch offsets   0, 6, 12, 18
```

Held-out input-and-target windows selected once by a dedicated seeded
handle and reused at every report are **fixed validation batches**.

The validation count and seed offset are file policy, shared through
`evaluation.h`. `main.c` includes that header, then gives the imported
count a local name. This is the exact relevant line from the larger
policy enum:

```c
    VALIDATION_BATCHES  = TINY_AGENC_VALIDATION_BATCHES,
```

The offset declaration is also exact:

```c
/* Held-out windows are fixed independently of both training and sampling. */
static const unsigned long long VALIDATION_SEED_OFFSET =
    TINY_AGENC_VALIDATION_SEED_OFFSET;
```

The copied learner headers already include `evaluation.h`; the learner's
`main.c` must include it too. The shared header sets the count to four
and the offset to two.

The source record stores the two flat arrays and the number of batches:

```c
typedef struct {
    int *inputs;
    int *targets;
    int  count;
} ValidationBatches;
```

Here is the complete constructor:

```c
static ValidationBatches prepare_validation(const Dataset *ds,
                                            const TrainOptions *options)
{
    ValidationBatches validation = { 0 };

    if (ds == NULL)
        return validation;

    int    span  = options->cfg.batch_size * options->cfg.block_size;
    size_t count = (size_t)VALIDATION_BATCHES * (size_t)span;
    Rng   *rng   = rng_new(options->seed + VALIDATION_SEED_OFFSET);

    validation.inputs  = emalloc(count * sizeof *validation.inputs);
    validation.targets = emalloc(count * sizeof *validation.targets);
    validation.count   = VALIDATION_BATCHES;

    for (int batch = 0; batch < validation.count; batch++)
        dataset_batch(ds, rng,
                      validation.inputs + (size_t)batch * (size_t)span,
                      validation.targets + (size_t)batch * (size_t)span,
                      options->cfg.batch_size, options->cfg.block_size);
    rng_free(rng);
    return validation;
}
```

`ValidationBatches validation = { 0 };` explicitly initializes the
first field to zero; C zero-initializes the omitted fields too. The
absent record is therefore `{ NULL, NULL, 0 }`. If Chapter 14 supplied
no validation dataset, the function returns that record by value. C
copies its three fields back to the caller, as it did for Chapter 4's
[small `Mat` descriptions](04-poor-mans-tensors.md#put-the-shape-beside-the-address).

With a dataset, `span` repeats the checked `B*T` count. The source-wide
constant `VALIDATION_BATCHES` is four. `count` is therefore
`4*span`, calculated as `size_t` before allocation.

The temporary RNG begins at `seed + 2`. It draws all four batches once
and is then freed. It cannot advance `batch_rng`, which is a different
object.

Inside the loop, Chapter 3's
[pointer arithmetic](03-data.md#draw-every-legal-start) selects each
flat destination. With `span = 6`, batch two begins at
`validation.inputs + 12`; `dataset_batch` fills slots 12 through 17.
The target array uses the matching offset.

Returning this populated record copies the two pointers and count, not
the allocated arrays. Ownership of those arrays passes to the caller.

## Grade without sending corrections back

Each fixed batch already has the same size, so its forward call returns
a mean over the same number of positions. Averaging the four batch
means therefore gives the mean over all `4*B*T` held-out positions.

Use four small batch losses:

```text
batch losses     1.2, 0.8, 1.0, 1.4
sum              1.2 + 0.8 + 1.0 + 1.4 = 4.4
held-out mean    4.4 / 4 = 1.1
```

**Predict:** which additional calls would let those held-out answers
change parameters?

Backward would create gradients from them, and `model_step` before the
next clear would apply those gradients. Backward alone does not change
a parameter; the next `model_zero_gradients` would discard its result.
This measurement performs neither call, leaving no correction path from
the held-out answers to an update.

Measuring mean loss on examples withheld from updates, using
target-bearing forwards but no backward or update, is **held-out
evaluation**.

The helper is valid only for a populated record with `count > 0`.
Dividing the absent record's zero total by its zero count would be
invalid. The report branch checks `validation.count > 0` before calling
it.

The complete measurement helper stops after forward:

```c
static float measure_validation(Model *m, ValidationBatches validation,
                                const TrainOptions *options)
{
    int    span  = options->cfg.batch_size * options->cfg.block_size;
    double total = 0.0;

    for (int batch = 0; batch < validation.count; batch++)
        total += model_forward(m,
                               validation.inputs + (size_t)batch * (size_t)span,
                               validation.targets + (size_t)batch * (size_t)span,
                               options->cfg.batch_size, options->cfg.block_size);
    return (float)(total / validation.count);
}
```

`total` uses the wider `double` type while adding the four returned
`float` values. Chapter 2 introduced that
[wider type](02-foundations.md#fixed-integers-and-a-clock-that-does-not-turn-back).
The pointer offsets select batches 0 through 3 from the flat arrays.

Every call supplies targets. Validation is therefore not inference
mode: it computes probabilities and a real loss. It does not learn
because no validation call is followed by `model_backward` or
`model_step`.

The last validation forward overwrites the model's latest-forward
record. That is safe only because the training backward and update for
this step have already completed. The next iteration's target-bearing
training forward replaces the validation record before the next
backward. This evaluation observes whether improvements cross the split
without sending held-out corrections back through the model.

Cleanup is unconditional:

```c
static void free_validation(ValidationBatches validation)
{
    free(validation.inputs);
    free(validation.targets);
}
```

When validation is absent, both pointers are `NULL`. Chapter 2's
[allocation contract](02-foundations.md#a-block-of-bytes-and-one-owner)
owns the release rule. One C edge appears here for the first time:
`free(NULL)` performs no operation, so the same cleanup works for both
record states.

## When the two grades part company

Now two grades can answer two different questions. Consider this
illustrative report sequence:

| Step | Training batch | Fixed held-out batches |
|---:|---:|---:|
| 1 | 2.0 | 2.0 |
| 50 | 1.4 | 1.5 |
| 100 | 0.9 | 1.1 |

Both columns fall. Corrections learned from update-eligible examples
also improve prediction on examples kept behind the no-peeking line.
That transfer is **generalization**.

Continue the example:

| Step | Training batch | Fixed held-out batches |
|---:|---:|---:|
| 100 | 0.9 | 1.1 |
| 150 | 0.6 | 1.1 |
| 200 | 0.4 | 1.2 |

**Predict:** did steps 100 through 200 make the model better at the
held-out questions?

No. The training grade improved while the held-out grade stalled and
then worsened. Continued fitting of training examples that no longer
transfers to held-out examples is **overfitting**.

The stricter memorization target's shell name now has context:
`make overfit`. The fixed batch test deliberately asks the model to
memorize one tiny stack of cards as an implementation stress test.
Overfitting during an actual run describes an unwanted gap between
training progress and held-out progress. The same word points at
intentional behavior in the test and a warning sign when deciding how
long to train.

The repository's committed validation run supplies a real pair of
curves. Its exact configuration, data hashes, and selected reports live
in
[`validation-cyberpunk-5000-summary.log`](logs/validation-cyberpunk-5000-summary.log).
[Chapter 17](17-the-training-run.md#read-the-held-out-run) will
interpret the run in full.

![Training and fixed held-out losses converge, then separate.](figures/15-validation-losses.svg)

At step 1,000 the recorded values are nearly equal: training `1.1218`
and held out `1.1182`. By step 5,000, training has reached `0.7638`
while held out is `0.9023`. The widening gap shows that later training
improvements transfer less fully to these fixed held-out batches. It
does not show held-out loss stalling or rising, so these selected
reports do not by themselves witness the overfitting pattern
constructed above. They also do not identify one universally correct
stopping step. Four fixed batches are a stable sample of the validation
split, not complete knowledge of every held-out window.

## Reporting without steering

Adding observation creates a new failure mode. If reporting consumes
the batch RNG, lets validation gradients reach `model_step`, or
replaces the current latest-forward record before its backward pass,
turning reporting on changes later computation.

Tiny AgenC places every observation after the current update:

```text
core step: batch -> zero -> forward -> backward -> update
                                                    |
                                                    v
                 report due? -> progress text due? -> save due?
                                                    |
                                                    v
                                             next batch draw
```

Placing reports, held-out grades, progress text, and saves after the
current update while keeping them away from the batch RNG and updater
is **observation isolation**. The program can inspect a completed step
without steering the following one.

Validation consumes no batch RNG draws and applies no update. Progress
text receives `sample_rng`, not `batch_rng`. Saving reads model and
tokenizer state without changing the next batch choice. The next
section adds the timer to this arrangement before opening the complete
report branch.

The Chapter 15 lab tests the strongest inexpensive consequence. It runs
the same two-step command twice, once without validation and once with
an identical validation corpus, then compares the checkpoints byte for
byte. The equality proves that configuration, vocabulary, and saved
parameter values match after those two runs. Chapter 13 established
that the file omits AdamW moments and RNG state, so this comparison
alone cannot prove that every unsaved state byte matched. Source order
and the separate handles supply that stronger non-steering argument.
The copied validation bytes intentionally make this an isolation
witness, not a held-out-quality or generalization experiment.

## Time only completed learning work

A timer around the whole command would mix training with corpus setup,
validation, terminal output, generated text, and filesystem writes.
The printed `ms/step` is meant to describe the core cycles.

Work a small timing window by hand:

```text
clock at window start          10.00 seconds
now after 5 completed steps    10.25 seconds
elapsed                         0.25 seconds
```

**Predict:** what mean number of milliseconds belongs to each completed
step?

```text
1000 * 0.25 / 5 = 50 milliseconds per step
```

Suppose validation and printing then consume another `0.08` seconds.
Resetting `clock` after them starts the next window at `10.33`, so that
observation time is not charged to the next set of training steps.

This mean elapsed duration per completed core cycle is the program's
**training-step timing**. It is a local performance estimate, not a
promise that another compiler or machine will produce the same number.

Two exact file policies give the reporting interval and unit
conversion:

```c
    LOSS_INTERVAL       = 50,      /* steps between loss reports */
```

```c
static const double             MILLISECONDS_PER_SECOND = 1e3;
```

The fixed validation batches and timer travel through the reporting,
sampling, and saving helpers. The source keeps them in one record:

```c
typedef struct {
    ValidationBatches validation;
    double            clock;
    int               timed_steps;
} TrainingObservationState;
```

One helper prepares it:

```c
static TrainingObservationState prepare_training_observation(
    const TrainingResources *resources, const TrainOptions *options)
{
    TrainingObservationState state;

    state.validation =
        prepare_validation(resources->validation_data, options);
    state.clock       = time_seconds();
    state.timed_steps = 0;
    return state;
}
```

`state.clock` receives Chapter 2's
[monotonic elapsed-time clock](02-foundations.md#fixed-integers-and-a-clock-that-does-not-turn-back).
The counter begins at zero. Each successful update then runs:

```c
observation.timed_steps++;
```

A rejected update never enters the denominator. Before reading the
report code, follow step 1 in source order: training `loss` already
exists, backward and update complete, then optional held-out forwards
run inside the branch.

**Predict:** which side of update 1 does each printed grade describe?

Printing the line and deciding whether it is due are separate jobs. The
first helper chooses the line shape and performs optional validation:

```c
static void print_training_report(TrainingResources *resources,
                                  TrainingObservationState *state,
                                  const TrainOptions *options,
                                  int step, float loss, double milliseconds)
{
    if (state->validation.count > 0) {
        float validation_loss =
            measure_validation(resources->model, state->validation, options);

        printf("step %5d/%d | loss %.4f | val %.4f | %6.1f ms/step\n",
               step, options->steps, (double)loss,
               (double)validation_loss, milliseconds);
    } else {
        printf("step %5d/%d | loss %.4f | %6.1f ms/step\n",
               step, options->steps, (double)loss, milliseconds);
    }
}
```

The second helper owns the schedule and timer boundary:

```c
static void report_training_if_due(TrainingResources *resources,
                                   TrainingObservationState *state,
                                   const TrainOptions *options,
                                   int step, float loss)
{
    if (step == 1 || step % LOSS_INTERVAL == 0) {
        double now = time_seconds();
        double milliseconds =
            MILLISECONDS_PER_SECOND
            * (now - state->clock) / state->timed_steps;

        print_training_report(resources, state, options, step, loss,
                              milliseconds);
        restart_training_timer(state);
    }
}
```

The Chapter 1
[remainder operator](01-the-map.md#make-the-geometry-safe) makes the
second condition true at exact multiples of 50. Step 1 is reported
separately, so a run produces early feedback without waiting.

`now` is captured before validation and printing. The numerator
therefore includes batch selection, gradient clearing, forward,
backward, clipping, and updating, and stops before report work.
`state->validation.count > 0` is the guard that protects
`measure_validation` from its absent record.

The training loss was measured before this step's update. The optional
validation loss is measured inside the branch after that update. At
step 1, the first value grades the initialized model and the second
grades the once-updated model. The two columns on one line straddle one
update.
[Chapter 17](17-the-training-run.md#read-the-held-out-run) preserves
this distinction.

The format strings control alignment, not arithmetic. `%5d` prints an
integer in a field at least five characters wide. `%.4f` prints four
digits after the decimal point. `%6.1f` prints one digit after the
decimal in a field at least six characters wide. Wider values expand
the field rather than being truncated.

`restart_training_timer` runs after `print_training_report` returns, so
validation and output are not billed to the next window:

```c
static void restart_training_timer(TrainingObservationState *state)
{
    state->clock       = time_seconds();
    state->timed_steps = 0;
}
```

The definition is always "completed steps since the last reset." The
step-1 report times one cycle and resets. The step-50 report therefore
averages steps 2 through 50, which is 49 completed cycles. Later
ordinary report windows contain fifty unless another excluded event
reset the clock between them.

## Schedule side effects after the update

Reporting needs no new model operation. Progress text needs a small
wrapper around the still-opaque `model_sample` call. Chapter 14 already
built `newline_id` and `print_text`; the wrapper arranges their buffers
and labels.

The relevant lines from the file's policy enum are:

```c
    SAMPLE_INTERVAL     = 250,     /* steps between generated samples */
    SAMPLE_LENGTH       = 200,     /* characters per training-time sample */
    CHECKPOINT_INTERVAL = 1000,    /* steps between saves */
```

One value schedules the hook, one sizes its output, and one schedules
checkpoint replacement. They are fixed command policy rather than
train flags.

The wrapper is exact:

```c
static void print_training_sample(Model *m, const Tokenizer *tk, Rng *rng, int step)
{
    int ids[1 + SAMPLE_LENGTH];

    ids[0] = newline_id(tk);
    model_sample(m, rng, ids, 1, 1 + SAMPLE_LENGTH, DEFAULT_TEMPERATURE);
    printf("---- sample at step %d ----\n", step);
    print_text(tk, ids + 1, SAMPLE_LENGTH);
    printf("---------------------------\n");
}
```

`SAMPLE_LENGTH` is 200, so the fixed local array has 201 integer slots:
one private newline seed and 200 result slots. `ids[0]` receives the
known newline id. The opaque handoff says that one id is known and 201
must exist when the call returns.

**Predict:** how many ids does `print_text` send to standard output, and
does it include the seed?

It begins at `ids + 1`, prints 200 ids, and skips slot zero. The two
`printf` calls put a step-numbered banner around that text. The
`DEFAULT_TEMPERATURE` value travels into the handoff, but
[Chapter 16](16-sampling.md#change-the-gaps-without-changing-their-order)
constructs what that scaling means and how the 200 choices are made.
The learner implements this wrapper in `main.c`; the Chapter 15 lab's
borrowed `model_sampling.c` supplies the called `model_sample` symbol.

Two small helpers own the remaining schedules:

```c
static void sample_training_if_due(TrainingResources *resources,
                                   TrainingObservationState *state, int step)
{
    if (step % SAMPLE_INTERVAL == 0) {
        print_training_sample(resources->model, resources->tokenizer,
                              resources->sample_rng, step);
        restart_training_timer(state);
    }
}

static void save_training_if_due(TrainingResources *resources,
                                 TrainingObservationState *state,
                                 const TrainOptions *options, int step)
{
    if (step % CHECKPOINT_INTERVAL == 0 || step == options->steps) {
        if (model_save(resources->model, resources->tokenizer,
                       options->out_path) != 0)
            die("cannot write checkpoint %s", options->out_path);
        restart_training_timer(state);
    }
}
```

`SAMPLE_INTERVAL` is 250. The call is an opaque progress-text hook in
this chapter. It receives the completed model and its separate-state
RNG, performs no backward pass or update, and has its time excluded.
[Chapter 16](16-sampling.md#a-largest-only-rule-throws-information-away)
constructs the draw, then
[builds the sliding context](16-sampling.md#one-choice-must-become-the-next-question)
inside that call. Its inference forwards may replace
the latest-forward record, but the current backward and update are
already complete. The next training forward writes the record that the
next backward will use.

`CHECKPOINT_INTERVAL` is 1,000. Chapter 2's
[logical OR](02-foundations.md#read-bytes-without-trusting-the-file)
operator `||` also saves after the final requested step even when that
step is not a multiple of 1,000.
Chapter 13's [version 1 promise](13-durable-checkpoints.md#state-the-version-1-promise-exactly)
stores configuration, tokenizer vocabulary, and parameter values. It
does not store AdamW moments, step number, or RNG states, so this is a
generation checkpoint rather than exact interrupted-training state.

Predict the due work for a 260-step run:

| Step | Loss report | Progress text | Checkpoint |
|---:|:---:|:---:|:---:|
| 1 | yes | no | no |
| 50, 100, 150, 200 | yes | no | no |
| 250 | yes | yes | no |
| 260 | no | no | yes, final |

At step 250, reporting runs before progress text because its `if` block
appears first. Both reset the timer after their own excluded work.

At step 1,000, all three conditions are true:

```text
1. print the loss and optional held-out grade
2. print progress text
3. save one checkpoint
```

The checkpoint branch executes once. The condition has two reasons to
enter the same block; it does not contain two save calls. Because every
branch follows `model_step`, the saved weights include update 1,000.
The `loss` printed immediately before them still describes the model
before that update.

## The complete route

All pieces now fit in one source-order picture:

```text
Chapter 14 checked setup
          |
          v
allocate batch buffers -> prepare four fixed validation batches
          |
          v
for step = 1 through requested steps
          |
          +-> draw training batch
          +-> clear gradients
          +-> forward with targets -> remember pre-update loss
          +-> backward immediately
          +-> checked update with one-based step
          |
          +-> report due?
          |      +-> fixed held-out forwards, no backward or update
          |
          +-> progress text due? -> Chapter 16 mechanism
          |
          +-> interval or final save due? -> Chapter 13 checkpoint
          |
          v
free validation and batch buffers -> return to Chapter 14 cleanup
```

The complete Chapter 15 control flow can now appear. Every operation
this chapter owns has been constructed; `model_sample` remains the
marked Chapter 16 handoff:

```c
static void train_loop(TrainingResources *resources,
                       const TrainOptions *options)
{
    TrainingStepState step_state = prepare_training_step_state(options);
    TrainingObservationState observation =
        prepare_training_observation(resources, options);

    for (int step = 1; step <= options->steps; step++) {
        float loss =
            run_training_step(resources, &step_state, options, step);

        observation.timed_steps++;
        report_training_if_due(resources, &observation, options, step, loss);
        sample_training_if_due(resources, &observation, step);
        save_training_if_due(resources, &observation, options, step);
    }
    free_training_observation(observation);
    free_training_step_state(step_state);
}
```

The signature carries two records. `resources` supplies the model,
tokenizer, datasets, and separate RNG handles. `options` supplies the
checked configuration, rate, count, seed, and output path.

The opening calls prepare step storage and observation storage. The
`for` body then joins the exact helpers already walked: one update, a
due report, a due progress sample, and a due save. The final two calls
release only storage owned by `train_loop`. The six command resources
remain valid until `run_train` calls `free_training_resources`.

## First light

The complete loop is now ready for a corpus run. Use a model small
enough that a mistake returns quickly:

```sh
./tiny-agenc train \
    --data data/cyberpunk.txt \
    --out first-light.bin \
    --steps 200 \
    --layers 1 \
    --heads 2 \
    --width 32 \
    --block 32 \
    --batch 4 \
    --lr 0.001 \
    --seed 1337
```

The cleaned NIGHT GRID corpus has `V = 80` byte tokens. Chapter 5's
[equal-bet calculation](05-forward-pass.md#cross-entropy-keeping-score)
gives the starting reference:

```text
target probability = 1/80
loss                = -log(1/80)
                    =  log(80)
                    =  about 4.382
```

Random initialization need not produce exactly equal bets, and each
report grades a different random batch. The first printed value should
therefore be near that scale, not equal to that decimal.

Before running the command, write down the three observations that
would convince you to continue:

```text
1. every reported loss is finite
2. later reports are lower overall, even if one point bounces upward
3. first-light.bin is written after the final update
```

The 200-step checkpoint is not expected to write polished text. Keep
`first-light.bin`; [Chapter 16](16-sampling.md#close-the-first-light-path)
will load it and test the generation path.
[Chapter 17](17-the-training-run.md#loss-needs-a-landmark) will build
stronger comparison references and audit the recorded 5,000-step runs.
This chapter needs the smaller claim that the real corpus can drive
sampled training loss downward overall and write the resulting
parameter values.

## Know what the Chapter 15 witness proves

The focused target combines several claims that a single falling number
cannot carry.

The fixed `\nabc` model witness checks finite updates, a loss drop of at
least half, checkpoint save/load, bit-exact parameter preservation, and
an identical loaded forward result. The stricter repository
memorization test requires a final loss below `0.10` and below one tenth
of its initial loss.

The CLI witness compares one-step and fifty-step checkpoints, so a loop
that only prints changing text while saving initialized values fails.
It also parses the deterministic tiny run's step reports and requires
the later reported loss to be lower.

Finally, byte-identical two-step checkpoints with and without validation
show that their saved parameter values match. The source audit supplies
the separate claim that validation never receives `batch_rng` and never
calls backward or update.

The target does not exercise the step-250 progress-text branch or the
step-1,000 interval save. It does not prove text quality, the Chapter 16
draw, timing equivalence across machines, complete validation coverage,
or exact training continuation after a checkpoint load.

## Build checkpoint: make it learn

### Build

Keep Chapter 14's parsing, preflight, tokenizer, datasets, model
construction, and cleanup. Remove the learner-only `no updates yet`
line and direct initialized-model save. Add reusable batch buffers, the
AdamW recipe, one-based repeated learning steps, failed-update handling,
fixed validation batches, isolated reporting, interval and final saves,
and the matching cleanup.

The learner target temporarily links the answer-key
`model_sampling.c` because Chapter 14's already-built sample command
needs its symbol. Implement the `print_training_sample` wrapper above,
but treat its `model_sample` call as an opaque handoff.
[Chapter 16](16-sampling.md#the-complete-generation-loop) replaces
that borrowed sampling module and teaches its mechanism.

### Verify

Run the learner checkpoint:

```sh
make -C labs check-15
```

Then inspect the answer key:

```sh
make -C labs WORK=../src check-15
```

After both pass, run the first-light command above and keep its
checkpoint for Chapter 16. On the learner track, replace
`./tiny-agenc train` in that command with
`./labs/build/tiny-agenc-before-sampling train`.

### Expected

The fixed batch remains finite and loses at least half its initial
grade. The one-step and fifty-step checkpoints differ, the later tiny
CLI report is lower than the first, and adding validation leaves the
two-step checkpoint byte-identical. First light reports finite losses,
trends downward overall, and saves after its final update. Chapter 16
performs the first-light load and generation check.

### Common failures

- Loss that never moves often means the learner kept Chapter 14's
  direct save, omitted `model_step`, or disconnected the final head.
- A sharp change followed by unstable loss can mean gradients were not
  cleared before the next target-bearing forward.
- An assertion in backward means another forward replaced the training
  record or the training forward omitted targets.
- Rejection at the first update can mean step numbering began at zero,
  the optimizer recipe is invalid, or a gradient became nonfinite.
- Different checkpoints with and without validation mean observation
  isolation failed somewhere; inspect batch draws and calls to backward
  or update first.
- A step-1,000 checkpoint that misses its newest update was saved before
  `model_step`.
- Timing that grows when validation or checkpointing runs means the
  clock was not reset after excluded work.

The loop is short because fourteen chapters earned that brevity. Its
parameter-changing path is now visible. The next chapter opens the
remaining choice mechanism and turns scores from the learned parameters
into text without changing them.

---

[Previous: The Command Line Is a Boundary](14-the-command-line.md) | [Contents](README.md) | [Next: Sampling](16-sampling.md)
