# Appendix B: Sources and Further Reading

Tiny AgenC is small, not parentless. Its pieces come from papers, books,
standards, and reference implementations whose ideas survived contact
with much larger machines.

Use this appendix as a map when one part of the model makes you want the
longer story. You do not need it before Chapter 1, and you do not need
to read it from front to back. Each entry says where Tiny teaches the
idea, where its current contract lives, what it borrowed, and where it
deliberately differs.

## A citation cannot answer every question

Suppose you want to know why an earlier position cannot read a later
position during attention. Four sources can answer four different
questions:

```text
I need the idea        -> read the chapter that first teaches it
I need Tiny's contract -> inspect its C and tests
I need the ancestry    -> read the original paper
I need another scale   -> read a bridge implementation or later paper
```

[Chapter 5 constructs causal
attention](05-forward-pass.md#let-one-position-consult-the-visible-past).
The loop in
[`attention_head_forward`](../src/ops.c)
settles what this program does today. `check_forward` in
[`tests/integration.c`](../tests/integration.c) checks one concrete
witness: changing its future query, key, and value row does not alter
its earlier output rows. *Attention Is All You Need* explains the larger
architecture from which the operation descends. FlashAttention later
asks how to schedule the same mathematical result with a different
memory plan.

Those answers are related, but they are not interchangeable.

**Predict:** if a future edit changes Tiny's loop from `t2 <= t` to
`t2 < time`, can the Transformer paper tell you whether the edited
program still hides the future?

No. The paper states the design. The current C and its tests state
whether this implementation preserves it. An outside source cannot
overrule code that has changed.

The completed source trace looks like this:

```text
question          How is future text hidden during attention?
first teaching    Chapter 5
current contract  attention_head_forward, with t2 <= t
test witness      tests/integration.c, check_forward
outside source    Transformer paper, Section 3.2.3
Tiny keeps        scaled scores, softmax, value mixing
Tiny changes      the paper's mask becomes a causal loop bound
stop when         I can explain why the visible count is t + 1
```

That record is a source trace. Start inside Tiny when you need
understanding or exact behavior. Read outward when you can name the
question the outside source must answer.

## The architecture route

### Attention Is All You Need

Ashish Vaswani and colleagues introduced the Transformer architecture
and scaled dot-product multi-head attention in
[Attention Is All You Need](https://arxiv.org/abs/1706.03762).

**Start inside Tiny.** Read [Chapter 1's complete
map](01-the-map.md#one-trip-through-the-machine), then Chapter 5's
[numeric attention
construction](05-forward-pass.md#let-one-position-consult-the-visible-past).
The chapters give every symbol in the paper a row, shape, and job before
the paper asks you to read its compact equations.

**Then read.** Begin with Figure 1 and Sections 3.2.1 and 3.2.2. Tiny
keeps query, key, and value projections; the head-size scale; softmax
weights; and the weighted mixture of value rows. Those parts correspond
to the projection inside
[`block_forward`](../src/model_forward.c), followed by
[`attention_forward`](../src/ops.c).

**Keep the boundary.** The original machine has an encoder stack that
reads an input sequence and a decoder stack that produces an output
sequence. Tiny is a decoder-only next-token model. It also uses
[pre-norm blocks](01-the-map.md#prepare-a-copy-preserve-the-highway),
learned position rows, and a causal loop bound. The paper's translation
results do not measure Tiny's training run.

Stop once you can point from each of the paper's `Q`, `K`, and `V`
symbols to one slice of Tiny's packed `qkv` row and explain why the
visible key count is `t + 1`.

### GPT-2

OpenAI's
[Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
describes GPT-2. The
[released `model.py`](https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/model.py)
makes several architectural details more direct than the report alone.

**Start inside Tiny.** Chapter 1 maps the decoder. Chapter 9 constructs
the [parameter
registry](09-parameters-and-the-blueprint.md#one-block-all-its-learned-tensors)
and [initialization
policy](09-parameters-and-the-blueprint.md#initialization-is-architecture).
Chapter 11 walks the [complete forward
route](11-wiring-the-model-forward.md#the-complete-source-route).

**Then read.** Start with Section 2.3 of the report. In the released
source, compare the two embedding tables, the normalization before each
block edit, the four-times-wider GELU path, the final normalization, and
the language-model head. The last matrix multiplication reads the input
token table in the other direction, as Tiny's tied head does.

Tiny combines the released source's `0.02` base scale with the report's
smaller residual-projection scale. In Tiny's
[`model_create_parameters`](../src/model_parameters.c), the residual
denominator is `sqrt(2 * layer_count)` because every block has two
residual edits.

**Keep the boundary.** Tiny uses the broad decoder skeleton, not GPT-2's
complete model or training recipe. Tiny builds its vocabulary from the
distinct bytes in one corpus. GPT-2 uses a much larger subword
tokenizer. Tiny has no separate linear bias arrays, uses much smaller
shapes, and runs handwritten CPU forward and backward loops. The
released GPT-2 source initializes position rows at `0.01`; Tiny uses
`0.02` for both token and position rows. GPT-2's parameter counts,
losses, samples, and task results are not Tiny evidence.

Stop when you can compare one GPT-2 block with
[`block_forward`](../src/model_forward.c) and account for the
architectural differences named above: tokenizer, bias arrays, shapes,
and handwritten CPU loops.

### Tied input and output rows

Ofir Press and Lior Wolf study reuse of the input token table at the
output in
[Using the Output Embedding to Improve Language
Models](https://arxiv.org/abs/1608.05859).

**Start inside Tiny.** Chapter 9 first builds [one table with two
jobs](09-parameters-and-the-blueprint.md#one-table-two-jobs). Chapter 12
then follows the [two returning gradient
paths](12-wiring-the-model-backward.md#one-tied-object-receives-two-returns).

**Then read.** The paper explains why an input table and an output table
can be one learned object. Tiny's forward pass looks up token rows near
the entrance and sends final rows through the same values near the
exit. Its backward pass accumulates both uses into one gradient buffer.

**Keep the boundary.** The paper studies larger language models and
reports their measurements. Tiny uses the structural choice and tests
its own shared-storage behavior. Weight tying does not make the two
gradient paths numerically identical; it gives them one destination.

Stop when you can explain why Tiny's parameter registry contains no
second `V x C` output table.

### llm.c

Andrej Karpathy's
[`llm.c`](https://github.com/karpathy/llm.c) is a bridge from a small
CPU model toward larger C and CUDA training code.

**Start inside Tiny.** Finish Chapter 18's distinction between
[transferable roles and Tiny-specific
choices](18-epilogue.md#carry-roles-not-filenames). Tiny keeps every
parameter, activation, gradient, and lifetime visible before a larger
implementation introduces accelerator-specific work.

**Then read.** At the pinned revision used for this map, begin with the
[CPU trainer](https://github.com/karpathy/llm.c/blob/f1e2ace651495b74ae22d45d1723443fd00ecd3a/train_gpt2.c).
Identify the same forward operations you know. Then compare its tests
with the
[CUDA trainer](https://github.com/karpathy/llm.c/blob/f1e2ace651495b74ae22d45d1723443fd00ecd3a/train_gpt2.cu)
and the operation-specific files under `dev/cuda`.

**Keep the boundary.** `llm.c` is not a Tiny dependency or an answer key
for this repository. Its data formats, supported models, hardware,
kernels, and current branch can change. File-specific links above are
pinned so the route described here does not move under the reader.

Stop after tracing one operation from Tiny's CPU loop through the CPU
trainer and into one accelerator implementation. Reading every
optimized kernel is a separate project.

## The operation route

### Layer normalization

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey Hinton introduced layer
normalization in
[Layer Normalization](https://arxiv.org/abs/1607.06450).

**Start inside Tiny.** Chapter 5 constructs the need to [control one
row's scale](05-forward-pass.md#keep-one-rows-scale-from-controlling-the-next-operation).
Chapter 6 builds its returning gradient, and Appendix A gives the
[slower symbolic
derivation](appendix-a-derivations.md#layernorm-backward-one-path-at-a-time).

**Then read.** Begin with the abstract and Section 3. The paper replaces
statistics collected across a batch with statistics collected within
one case. Tiny makes that choice concrete by taking the mean and
variance across the channels of each token row, then applying one
learned gain and bias per channel.

**Keep the boundary.** Tiny fixes epsilon at `1e-5`, records the mean and
reciprocal standard deviation for backward, and uses the same
calculation during training and generation. Those exact storage and
float choices come from [`layernorm_forward`](../src/ops.c), not from a
general promise made by the paper.

Stop when Equation 3 can be translated into the three loops in Tiny's
forward function without inventing a missing array.

### GELU

Dan Hendrycks and Kevin Gimpel proposed the activation in
[Gaussian Error Linear Units
(GELUs)](https://arxiv.org/abs/1606.08415).

**Start inside Tiny.** Chapter 5 builds the [need for a bend between the
wide and narrow
projections](05-forward-pass.md#put-a-bend-between-widen-and-narrow).
Chapter 6 differentiates the approximation. Appendix A unwinds its
[nested scalar
calculation](appendix-a-derivations.md#gelu-unwind-the-nested-scalar).

**Then read.** The paper defines the exact GELU as the input multiplied
by the accumulated area under a standard normal curve. It also gives
the tanh approximation used by GPT-2 and Tiny. Locate `0.044715` in the
paper, then in [`gelu_forward`](../src/ops.c). That constant identifies
the approximation rather than the exact normal-area calculation.

**Keep the boundary.** The paper's comparisons do not prove that GELU
beats another bend in Tiny. The repository proves only that its chosen
forward values, backward slopes, and whole-model connections agree with
their witnesses.

Stop once you can state whether a number in the paper belongs to the
exact definition, the tanh approximation, or an experiment.

### Adam

Diederik Kingma and Jimmy Ba introduced the adaptive update in
[Adam: A Method for Stochastic
Optimization](https://arxiv.org/abs/1412.6980).

**Start inside Tiny.** Chapter 8 constructs a fading gradient history,
a fading squared-gradient history, coordinate-wise scaling, and the
[zero-start
repair](08-adamw.md#repair-the-zero-start). Appendix A derives why the
[missing coefficient
mass](appendix-a-derivations.md#adams-cold-start-correction)
is `1 - beta^step`.

**Predict:** which source owns the two moment histories and their
zero-start correction, the Adam paper or the AdamW paper?

Adam owns them. Read Algorithm 1 after Chapter 8. The first history
smooths gradients, the second smooths squared gradients, and both
divide out the coefficient mass missing because their buffers began at
zero.

**Keep the boundary.** Tiny places epsilon after the corrected second
history's square root. It starts `step` at one, validates every recipe
field, and rejects a nonfinite or unsafe update before changing the
`Param`. Those are current [`param_adamw_step`](../src/param.c)
contracts, not consequences of the name Adam.

Stop once each line of the normal update loop can be paired with one
line of Algorithm 1 and every extra Tiny guard is marked as local
policy.

### AdamW

Ilya Loshchilov and Frank Hutter separated adaptive gradient scaling
from direct parameter shrinkage in
[Decoupled Weight Decay
Regularization](https://arxiv.org/abs/1711.05101).

**Start inside Tiny.** Chapter 8 first shows why adding shrinkage to the
gradient changes its meaning, then constructs [direct
decay](08-adamw.md#pull-selected-values-directly-toward-zero).

**Then read.** Section 2 and Algorithm 2 isolate the decay term from
Adam's moment calculation. Tiny preserves that separation. The moment
buffers see the loss gradient; the parameter adjustment adds a decay
direction afterward.

**Keep the boundary.** Tiny applies decay only when a `Param` has more
than one row and more than one column. It performs global clipping
before any parameter update, preflights a complete `Param`, and saves
parameter values without optimizer history. The AdamW paper does not
specify those repository policies.

Stop when you can point to the two addends inside Tiny's update and say
which one comes from Adam and which one makes the update AdamW.

### Gradient norm clipping

Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio propose norm clipping
as a response to exploding gradients in
[On the difficulty of training Recurrent Neural
Networks](https://arxiv.org/abs/1211.5063).

**Start inside Tiny.** Chapter 8 constructs [one limit around all
gradients](08-adamw.md#put-one-limit-around-all-gradients). The worked
`[3, 4]` example produces norm `5`, then scale `1/5`, preserving the
direction as `[0.6, 0.8]`.

**Then read.** Use the paper for the reason a norm threshold can prevent
one unusually large gradient from setting the step size. Tiny's
[`clip_gradient_norm`](../src/model.c) makes a particular choice:
combine every registered parameter gradient, use threshold `1`, and
clip before AdamW.

**Keep the boundary.** The paper studies recurrent networks. Tiny's
transformer and its threshold need their own tests and training
evidence. The integration witness checks the global norm, preserved
direction, nonfinite rejection, and the large-finite recovery path.

Stop when you can distinguish the general clipping mechanism from
Tiny's threshold, ordering, and overflow policy.

### FlashAttention

Tri Dao and colleagues describe a different execution plan in
[FlashAttention: Fast and Memory-Efficient Exact Attention with
IO-Awareness](https://arxiv.org/abs/2205.14135).

**Start inside Tiny.** Chapter 5 builds attention as explicit score
rows. Chapter 10 counts the resulting
[`B * H * T * T` storage](10-memory-planning.md#attention-owns-the-expensive-square).
Chapter 18 separates the [attention equation from its storage
plan](18-epilogue.md#attention-storage-can-change).

**Then read.** Begin with the abstract, Section 3, and Algorithm 1.
FlashAttention processes the score calculation in pieces so fewer
values travel between levels of GPU memory. It preserves exact
mathematical attention rather than replacing it with an approximation.

**Keep the boundary.** Tiny allocates full `B * H * T * T` score
backing storage for each layer and a matching gradient buffer. Only
each causal prefix is meaningful. Keeping the square buffers makes the
forward and backward loops inspectable. FlashAttention is not present
in this repository. A different grouping of floating-point operations
can also change last bits even when the mathematical result is the
same. The paper's hardware measurements are not Tiny measurements.

Stop when you can say which saved Tiny arrays a tiled implementation
would avoid and which query, key, value, and softmax relationships it
must preserve.

## Randomness without shrugging

### PCG

Melissa O'Neill's
[PCG paper](https://www.pcg-random.org/paper.html) presents the family
of permuted congruential generators behind `rng.c`. The official
[minimal C implementation](https://www.pcg-random.org/download.html#minimal-c-implementation)
shows the exact XSH-RR output permutation in compact form. That label
names the XOR-and-shift work followed by the rotation that Chapter 2
already built.

**Start inside Tiny.** Chapter 2 builds
[state, transition, and output
separately](02-foundations.md#build-a-generator-from-state), then walks
every line of [`pcg32_next`](../src/rng.c).

**Then read.** Compare the multiplier, the right shifts by `18`, `27`,
and `59`, and the final bit rotation. Tiny fixes one increment instead
of storing a selectable sequence value. The PCG portion of `Rng`
therefore contains one state word rather than the reference
implementation's state and sequence pair. The rest of `Rng` holds the
saved normal value that Chapter 2 built. `rng_new` advances once, adds
Tiny's one seed, then advances again; it is not the reference library's
two-argument seeding interface.

**Keep the boundary.** PCG supplies Tiny's 32-bit source values.
`rng_uniform` is Tiny's choice to retain the high 24 bits for a
`float`. Box-Muller and Lemire supply the later conversions. Replaying
a seed is useful experimental control; it does not make the generated
values secret or suitable for protecting anything.

The integration test compares known PCG-derived values, not only two
copies of Tiny's implementation. Stop when you can explain which parts
of `rng.c` come from PCG and which parts sit above it.

### Box-Muller

G. E. P. Box and Mervin E. Muller published
[A Note on the Generation of Random Normal
Deviates](https://doi.org/10.1214/aoms/1177706645) in 1958.

**Start inside Tiny.** Chapter 2 constructs the circle, radius, angle,
and bell-shaped result in [two flat draws become a
bell](02-foundations.md#turn-two-flat-draws-into-a-bell). It also states
the finite boundary: Tiny's 24-bit uniform grid limits the implemented
radius even though the ideal normal distribution is unbounded.

**Then read.** The paper is two pages. Match its pair of outputs with
Tiny's cosine result and saved sine result. Tiny uses `1 - u` inside the
logarithm so its finite uniform input never supplies zero there.

**Predict:** after one call computes the pair and returns the cosine
half, how many new uniform draws does the next `rng_gaussian` call
consume?

Zero. [`rng_gaussian`](../src/rng.c) returns the saved sine half and
clears the flag. That spare is an implementation choice around the
two-output construction.

**Keep the boundary.** The source paper describes ideal continuous
uniform inputs. Tiny feeds it one of finitely many `float` fractions,
uses `float` math-library functions, and can therefore only approximate
the ideal normal rule.

Stop when you can account for both output values, both uniform draws,
the saved-spare flag, and the finite maximum radius.

### Unbiased bounded integers

Daniel Lemire's
[Fast Random Integer Generation in an
Interval](https://arxiv.org/abs/1805.10941) develops the multiply-high
mapping and rejection step used by `rng_below`.

**Start inside Tiny.** Chapter 2 first shows why direct remainder can
[favor some
answers](02-foundations.md#pick-an-integer-without-favoring-one). Its
small 4-bit example makes every extra source value visible.

**Then read.** Match the paper's full-width multiplication with Tiny's
64-bit `product`. The high 32 bits select the bounded result. The low
32 bits decide whether this source draw lies in the uneven edge that
must be rejected.

**Keep the boundary.** Tiny accepts a positive `int` bound and draws
from its fixed PCG32 source. The paper's speed comparisons do not state
the speed of this program on the reader's compiler or processor. The
integration test instead checks deterministic known answers.

Stop when you can derive Tiny's `threshold` for a small range and
explain why every returned integer has the same number of accepted
32-bit sources.

## C as a design language

### C Interfaces and Implementations

David Hanson's
[C Interfaces and Implementations: Techniques for Creating Reusable
Software](https://www.informit.com/store/c-interfaces-and-implementations-techniques-for-creating-9780201498417)
supplies the design tradition behind Tiny's narrow public headers and
opaque stateful types.

**Start inside Tiny.** Chapter 1 maps [public, private, value, and
function
modules](01-the-map.md#the-map-becomes-c-modules). Chapters 2 and 4
construct allocation ownership and borrowed matrix views. Chapter 9
shows why the model's [private blueprint must be
exact](09-parameters-and-the-blueprint.md#why-the-private-blueprint-must-be-exact).

**Then read.** Begin with the book's chapters on interfaces,
implementations, assertions, and memory management. Compare `rng.h`
with `rng.c`: callers can create, draw from, and free an `Rng`, but
cannot name its fields. Repeat that comparison for `Dataset`,
`Tokenizer`, `Param`, and `Model`.

**Keep the boundary.** Tiny does not copy a general-purpose library
from Hanson. It applies the separation to one teaching program. `Mat`
is intentionally different: its small shape-and-address description is
public and copied by value, while another object owns the referenced
floats.

Stop after you can explain why exposing `struct Rng` would let a caller
break a promise that the current functions maintain.

### UNIX Network Programming

W. Richard Stevens, Bill Fenner, and Andrew Rudoff's
[UNIX Network Programming, Volume 1](https://unpbook.com/) uses
error-checking wrappers around fallible C and system interfaces. Its
published
[`Malloc` and `Calloc`
wrappers](https://github.com/unpbook/unpv13e/blob/31e089beb9fe2f48361dd395240d9bd173415738/lib/wrapunix.c)
make the connection concrete.

**Start inside Tiny.** Chapter 2 constructs [one place that decides how
to stop](02-foundations.md#one-place-decides-how-to-stop), then walks
`die`, `emalloc`, and `ecalloc` line by line.

**Then read.** Compare Stevens' capitalized wrappers with Tiny's
lowercase `e` prefix. Both call the fallible allocator once, check its
result once, and enforce one process-level failure policy. Callers then
receive usable storage or do not return.

**Keep the boundary.** Tiny does not make every failure fatal.
Allocation failure stops the process because the training run cannot
continue without the requested storage. File parsing, checkpoint
validation, and optimizer preflight return status so their callers can
report or recover at the correct boundary.

Stop when you can state which Tiny failures belong in `die` and which
must travel back to a caller.

### CRC-32

Chapter 13 constructs Tiny's reflected CRC-32 checksum from the
one-byte input `A`, then compares two separate two-byte inputs. The same
polynomial and bit update appear in the sample code in
[RFC 1952, Section
8](https://www.rfc-editor.org/rfc/rfc1952.html#section-8).

**Start inside Tiny.** Read [make changed bytes change the stored
check](13-durable-checkpoints.md#make-changed-bytes-change-the-stored-check),
then walk [`crc32_update`](../src/checkpoint.c).

**Then read.** Use the RFC to recognize the established CRC-32
arithmetic: initial all-one bits, byte input, reflected polynomial
`0xEDB88320`, and a final complement. Tiny applies it to its own
checkpoint prefix.

**Keep the boundary.** Tiny does not implement the gzip file format.
Its `TAGC` header, tokenizer bytes, parameter order, bounds, and atomic
replacement policy are repository contracts. A CRC mismatch detects
many accidental byte changes. A match cannot prove that the bytes are
unchanged, make a hostile file trustworthy, or repair a bad version.

Stop when you can separate the borrowed checksum arithmetic from
Tiny's checkpoint layout and validation gates.

### Standards settle contracts, not teaching order

The
[OpenMP specifications](https://www.openmp.org/specifications/) define
the compiler directives and runtime calls used in `ops.c` and `main.c`.
The
[POSIX.1-2017 specification](https://pubs.opengroup.org/onlinepubs/9699919799/)
defines interfaces such as `clock_gettime`, `fseeko`, `fsync`,
`openat`, and `renameat`. The Linux manual pages define the xattr and
`getrandom` calls used at the checkpoint boundary. The libacl manual
defines the access-ACL objects copied between open file descriptors.

These documents answer exact interface questions. They do not begin
with Tiny's problem or arithmetic floor. Read Chapter 5's [first
parallel
loop](05-forward-pass.md#make-every-output-from-one-input-row) before
the OpenMP specification. Read Chapter 2's clock and Chapter 13's
[atomic replacement
path](13-durable-checkpoints.md#write-beside-the-file-that-must-survive)
before POSIX.

The source remains the final witness for which part of a standard Tiny
uses. OpenMP defines its directives; the surrounding loop determines
whether independent iterations make parallel execution safe. POSIX
defines what `fsync` and `renameat` do; `model_save_durable` determines
their order. Tiny synchronizes the completed temporary inode, renames
it over the destination, and then synchronizes the parent directory.
Chapter 13 separates a failure before rename from a failure of that
last directory synchronization because only the second occurs after
the new file is visible.

Stop when you can separate the interface guarantee supplied by a
standard from the call order and narrower guarantee supplied by Tiny's
C.

## A citation is not evidence for Tiny

An outside source supports ancestry, a comparison, or a possible
extension. It does not supply measurements for this implementation.

The GELU paper can explain the chosen curve. It cannot show that
replacing GELU improves or harms NIGHT GRID loss. The FlashAttention
paper can show an exact tiled algorithm on its hardware. It cannot
provide Tiny's milliseconds per step. GPT-2 can identify an
architectural relative. Its samples cannot stand in for Tiny's
committed samples.

The direction of authority is:

```text
idea and ancestry        paper, book, or standard
Tiny's current behavior  src/ plus its tests
Tiny's measured result   EVIDENCE.md or committed logs
```

When these disagree, first check whether they answer different
questions. If prose and `src/` describe the same current behavior and
still disagree, the book must be corrected. If `src/` looks wrong, the
book's fidelity rule says to stop and flag it rather than rewrite the
implementation during a book pass.

## Read outward by goal

For architecture, read Chapters 1 and 5, then the Transformer paper,
then GPT-2. Move to `llm.c` only when you can identify the same block
roles in Tiny. Stop after tracing one complete block at the larger
scale.

For derivatives, read Chapter 6, then Appendix A. Use the layernorm and
GELU papers to learn where the forward operations came from, not as a
replacement for the backward derivations.

For optimization, read Chapter 8 and Appendix A before Adam, AdamW, and
the clipping paper. Stop when you can mark each part of Tiny's update as
borrowed mechanism or local policy.

For randomness, read Chapter 2 before PCG, Box-Muller, and Lemire. Stop
when one 32-bit PCG output can be followed through exactly one of
Tiny's three conversion paths.

For memory and performance, read Chapters 10 and 18 before `llm.c` and
FlashAttention. Stop when an optimization can be described as a change
to storage or scheduling without claiming that the model equation
changed.

For C interfaces, read Chapters 2, 4, and 9 before Hanson and Stevens.
Use the standards only after a concrete function leaves an interface
question unanswered.

Building one model small enough to hold makes the next source legible.

## Build checkpoint: trace one idea outward

### Build

Choose attention, layernorm, AdamW, or randomness. Write one source
trace with these eight fields:

```text
question
chapter that first teaches the idea
current C symbol
test witness
outside source and focused section
one choice Tiny keeps
one choice Tiny changes
stop when
```

Do not summarize the whole paper. The trace is complete when it answers
one named question and has a stopping point.

### Verify

If you add the trace to this book, run the book and graph checks:

```sh
make check-book
python3 scripts/check-graph.py
```

The trace itself needs a manual check. Inspect the named C symbol and
its test witness. Confirm that each outside claim in your trace is
labeled as ancestry, comparison, or extension rather than Tiny's
current behavior.

### Expected

The chapter that first teaches the idea does not require the outside
source. The C symbol settles the present implementation. The test
objects to the failure named in the trace. The outside reading begins
at a focused section, and the Tiny comparison contains at least one
kept choice and one changed choice.

### Common failures

- **Starting with the paper.** Dense notation then looks like a
  prerequisite even though the book already built its pieces.
- **Treating ancestry as the current contract.** Tiny is not an exact
  reproduction of the Transformer, GPT-2, or another repository.
- **Giving AdamW credit for Adam.** Moment histories and zero-start
  correction come from Adam; decoupled decay makes the update AdamW.
- **Treating tied uses as equal gradients.** The two paths share one
  destination, not one numeric contribution.
- **Calling equal mathematics equal bits.** A changed execution order
  can round differently.
- **Borrowing benchmark numbers.** Outside measurements do not become
  Tiny evidence.
- **Reading a moving line link.** Pin a revision before making a
  file-specific claim about another repository.
- **Using PCG to protect secrets.** Tiny selected it for reproducible
  experiments, not for an adversary.
- **Collecting sources without a question.** A citation pile gives the
  reader no entry point or stopping rule.

---

[Previous: Appendix A](appendix-a-derivations.md) | [Contents](README.md) | [Next: Appendix C](appendix-c-capstone-labs.md)
