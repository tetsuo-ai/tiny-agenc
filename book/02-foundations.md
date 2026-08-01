# Chapter 2: Foundations

Your Chapter 0 run reached its first training score only because three
lower-level jobs had already worked. The program found enough memory
for all 815,360 parameters, filled 813,056 from a repeatable random
stream, set the other 2,304 to fixed ones or zeroes, and chose random
windows from the corpus. If allocation had failed, there would have
been no model. If those random choices could not be replayed, a broken
run could disappear when you tried to inspect it.

Two small modules perform those jobs: `util` and `rng`. `util` decides
how the program allocates, reports failure, reads files, and measures
time. `rng` turns one seed into repeatable choices. Neither module
knows anything about the training data or model. Chapter 3 can use
both without taking on their internal machinery.

## One place decides how to stop

Memory does not appear because a program declares a pointer. The
program must ask C's allocator for a block of bytes. That request can
fail.

The training path allocates its long-lived buffers before entering the
step loop in Chapter 15. If one of those allocations fails, there is no
useful recovery for this command-line program.

Suppose four different model-building functions each ask for memory.
If every caller repeats the request and forgets the failure check once,
the program may use "no address" as though it pointed to storage. The
crash then happens far from the real failure.

Tiny AgenC makes a different choice. A command-line training process
cannot usefully continue without its model buffers, so allocation
failure ends the process immediately. One function prints the fatal
message for every caller.

The public [`util.h`](../src/util.h) header already declares the
functions callers may use. Start with the shared fatal function:

```c
void die(const char *format, ...)
{
    va_list arguments;

    va_start(arguments, format);
    fprintf(stderr, "tiny-agenc: ");
    vfprintf(stderr, format, arguments);
    fputc('\n', stderr);
    va_end(arguments);
    exit(EXIT_FAILURE);
}
```

This is the complete [`die`](../src/util.c) implementation. Its first
argument is a format string such as `"cannot open %s"`. In
`const char *format`, the star makes `format` a pointer to characters,
and `const` promises that this function will not change those
characters through that pointer. The `...` says that more arguments
may follow. C calls a function with that kind of argument list
*variadic*. A `va_list` is the cursor used to read those extra
arguments.

`va_start` positions the cursor after `format`. `fprintf` writes the
fixed program prefix to `stderr`, the stream reserved for diagnostics.
`vfprintf` uses the cursor to fill the format string. `fputc` finishes
the line, and `va_end` closes the variadic traversal. Finally,
`exit(EXIT_FAILURE)` ends the process with a failure status. The
declaration in [`util.h`](../src/util.h) begins with `_Noreturn`, a
promise that control never comes back from this function.

The `va_list`, `va_start`, and `va_end` names come from `stdarg.h`.
The complete implementation preamble appears after each of its jobs
has been constructed.

The public name is earned by its behavior: `die` reports why the
program cannot continue, then stops it.

The source file physically defines `emalloc` and `ecalloc` before
`die`. The earlier declaration in `util.h` makes those calls legal.
The book starts with `die` because the wrappers depend on its failure
policy.

Note what is not here: no logging framework, no cleanup registry, and
no second error policy. This process cannot train without its required
storage, so the fatal path stays small.

## A block of bytes and one owner

Now put the allocation policy behind one function:

```c
void *emalloc(size_t size)
{
    void *block = malloc(size);

    if (block == NULL)
        die("out of memory allocating %zu bytes", size);
    return block;
}
```

This is the complete [`emalloc`](../src/util.c). Read it from top to
bottom.

`size_t` is the unsigned integer type C uses for byte counts and object
sizes. Chapter 1 used it for a parameter count; here it reaches the
memory allocator. `malloc(size)` asks for exactly `size` bytes and
returns their address. Its return type is `void *`, a pointer that has
not yet been assigned an element type.

If the request fails, `malloc` returns `NULL`, the special pointer value
meaning "no address." That value is called the **null pointer**. The
wrapper checks it before any caller can use it. `%zu` is the `printf`
placeholder for a `size_t`. On success, the function returns the
address.

This is W. Richard Stevens' error-checking wrapper pattern: make the
request, check the result in one place, and keep that repeated policy
out of every caller. The `e` prefix means *error-checked*. Callers can
write:

```c
int *values = emalloc(4 * sizeof *values);
```

`sizeof *values` asks how many bytes one `int` occupies on this
platform. Although the expression contains `*values`, `sizeof` does
not follow the pointer or read memory here. Its operand is not
evaluated; C inspects the type `int`. Writing the expression from the
pointed-to object keeps it correct if the pointer's type changes later.
Multiplying by four asks for enough bytes for four integers. `emalloc`
returns an untyped address; C converts it to `int *` when assigning it
to `values`.

That creates a responsibility:

```text
values
  │
  ▼
┌───────── allocated block: four int objects ─────────┐
│ values[0]   values[1]   values[2]   values[3]       │
└─────────────────────────────────────────────────────┘
  caller may use this block until caller calls free(values)
```

The caller is the **owner** of the returned block. Ownership means it
must eventually pass the address to `free` exactly once. The block's
**lifetime** begins when allocation succeeds and ends at that `free`.
Reading it after its lifetime ends, or freeing it twice, is invalid.
[Chapter 4](04-poor-mans-tensors.md) will reuse this rule when a small
shape description borrows storage owned somewhere else.

`malloc` does not initialize the bytes. Sometimes the caller needs
zeroes instead. Repeating a zeroing loop would create another place to
make a size mistake, so `util` provides the matching wrapper:

```c
void *ecalloc(size_t count, size_t size)
{
    void *block = calloc(count, size);

    if (block == NULL)
        die("out of memory allocating %zu x %zu bytes", count, size);
    return block;
}
```

This is the complete [`ecalloc`](../src/util.c). `calloc` receives an
element count and the byte size of one element, allocates their product,
and clears every byte. The same failure policy applies. In the
Chapter 2 witness,

```c
int *zeroed = ecalloc(ALLOCATION_ELEMENT_COUNT, sizeof *zeroed);
```

The file-level `ALLOCATION_ELEMENT_COUNT` is `4`, so this must produce
four integer objects whose stored bytes are all zero.

There are two edges to keep straight. A caller must check that its own
size arithmetic cannot overflow before calling `emalloc`; the wrapper
checks allocation failure, not a product already calculated
incorrectly. Also, C permits `malloc(0)` and `calloc(0, n)` to return
`NULL` even though no memory is needed. Tiny AgenC's callers request at
least one object whenever an empty logical value is legal.

The source note for Stevens' wrapper style is in
[Appendix B](appendix-b-sources.md#unix-network-programming).

## Read bytes without trusting the file

Allocation failure has one policy. File failure cannot.

A training-corpus path that cannot be opened and a corpus whose byte
count exceeds the caller's safety ceiling are different failures. The
bounded reader returns a status so the training command can report the
right one before allocating storage for the corpus.

First construct the successful result. The Chapter 2 witness writes
three bytes:

```text
index        0      1      2
byte        'A'   0x00   'Z'
```

The zero in the middle is data. A C string function would treat it as
the end of the text, so the reader must return an explicit length. It
also appends one extra zero after the file:

```text
index        0      1      2      3
byte        'A'   0x00   'Z'   0x00
file data  <───────────────>
sentinel                         ^
reported length = 3
```

The sentinel lets a file with no embedded zero act as a C string. An
embedded zero still ends C string functions early, so the explicit
length remains authoritative for arbitrary bytes. The allocation must
therefore hold `file length + 1` bytes.

The bounded reader has three possible outcomes. C's `enum` gives those
outcomes names backed by integer values:

```c
typedef enum {
    FILE_SLURP_OK,
    FILE_SLURP_IO_ERROR,
    FILE_SLURP_TOO_LARGE,
} FileSlurpStatus;
```

The three enum names receive integer values zero, one, and two in
order. The `typedef` line makes `FileSlurpStatus` a type name, so a
declaration can say `FileSlurpStatus status` without the extra `enum`
prefix. Returning one of the three named values prevents callers from
guessing whether `NULL` meant a missing file or a rejected size. C
calls a type name introduced with `typedef` a **type alias**.

Start with the stage that measures and rewinds the stream:

```c
static FileSlurpStatus measure_file(FILE *stream, size_t maximum,
                                    size_t *length)
{
    if (fseeko(stream, 0, SEEK_END) != 0)
        return FILE_SLURP_IO_ERROR;

    off_t end = ftello(stream);

    if (end < 0 || fseeko(stream, 0, SEEK_SET) != 0)
        return FILE_SLURP_IO_ERROR;
    if ((uintmax_t)end > (uintmax_t)maximum
        || (uintmax_t)end >= (uintmax_t)SIZE_MAX)
        return FILE_SLURP_TOO_LARGE;

    *length = (size_t)end;
    return FILE_SLURP_OK;
}
```

`FILE *` is the standard library's handle for an open stream. `fseeko`
moves to the end, and `ftello` reports that position as an `off_t`, the
system's file-offset type. A negative offset signals failure.
`SEEK_SET` rewinds to byte zero so the later read starts at the
beginning.

The next condition answers two different questions before allocating:
does the file exceed the caller's ceiling, and can the program add the
sentinel byte without overflowing `size_t`?

`uintmax_t` is an unsigned integer type wide enough to hold any standard
unsigned integer. Converting both sides before comparison avoids
silently narrowing the file length. `SIZE_MAX` is the greatest value a
`size_t` can hold. Rejecting `end >= SIZE_MAX` makes `(size_t)end + 1`
safe.

Before revealing the result, apply those checks to the three-byte
fixture. With `maximum = 2`, which status should return? The file length
is three, so it returns `FILE_SLURP_TOO_LARGE` without allocating or
reading the body. With `maximum = 3`, the exact ceiling is accepted and
four bytes are allocated.

Predict the result for an empty file with `maximum = 0`. Its length
equals the ceiling, so the reader allocates one byte, writes the
sentinel at index zero, reports length zero, and succeeds.

The measured length can become stale before the read. A file could
shrink, grow, or report a size that does not match the bytes it
produces. The next stage requires exactly the measured bytes followed
immediately by a clean end of file:

```c
static int read_exact_file(FILE *stream, char *contents, size_t length)
{
    if (fread(contents, 1, length, stream) != length)
        return -1;
    if (fgetc(stream) != EOF || ferror(stream))
        return -1;
    return 0;
}
```

`fread` requests `length` objects of one byte each. If the return value
is smaller, the stream ended or failed before supplying the promised
bytes.
After those bytes, `fgetc` tries to read one more. An extra byte means
the file grew beyond the measured length. `EOF` can also report an I/O
failure, so `ferror` distinguishes a clean end from an error.

The public function now coordinates those two stages:

```c
FileSlurpStatus file_slurp_bounded(const char *path, size_t maximum,
                                   char **text, size_t *size)
{
    if (text == NULL || size == NULL)
        return FILE_SLURP_IO_ERROR;
    *text = NULL;
    *size = 0;
    if (path == NULL)
        return FILE_SLURP_IO_ERROR;

    FILE *stream = fopen(path, "rb");

    if (stream == NULL)
        return FILE_SLURP_IO_ERROR;

    size_t length;
    FileSlurpStatus status = measure_file(stream, maximum, &length);

    if (status != FILE_SLURP_OK) {
        fclose(stream);
        return status;
    }

    char *contents = emalloc(length + 1);

    if (read_exact_file(stream, contents, length) != 0) {
        free(contents);
        fclose(stream);
        return FILE_SLURP_IO_ERROR;
    }
    if (fclose(stream) != 0) {
        free(contents);
        return FILE_SLURP_IO_ERROR;
    }

    contents[length] = '\0';
    *text = contents;
    *size = length;
    return FILE_SLURP_OK;
}
```

`char **text` is the address of the caller's `char *` output. One star
reaches the caller's pointer; assigning `*text = NULL` changes that
pointer. `size_t *size` works the same way for the length. The `||`
operator means logical OR: C checks the left condition first, then the
right one only if needed. The function rejects a missing output
address. It then initializes both outputs before checking the path or
opening the file. A caller that receives an error will not mistake
stale values for a partial result.

`fopen(path, "rb")` opens the path for reading in binary mode. Every
path that has opened the stream later calls `fclose`. A read failure
releases the owned block before closing. Even after a complete read,
the close itself must succeed. Only then does the function append the
sentinel and publish the finished buffer and length.

The unbounded convenience function is now small:

```c
char *file_slurp(const char *path, size_t *size)
{
    char *text;

    if (file_slurp_bounded(path, SIZE_MAX - 1, &text, size)
        != FILE_SLURP_OK)
        return NULL;
    return text;
}
```

`SIZE_MAX - 1` reserves room for the sentinel. `&text` passes the
address of the local pointer so the bounded function can fill it.
Failure becomes `NULL`; success returns an owned buffer the caller must
free.

## Fixed integers and a clock that does not turn back

Saved-model files need to move exact 32-bit fields through a file.
`int` is not required to be 32 bits on every C platform, so the
interface uses `int32_t`, a signed integer type that is exactly 32 bits
when provided:

```c
int write_i32(FILE *stream, int32_t value)
{
    return fwrite(&value, sizeof value, 1, stream) == 1 ? 0 : -1;
}

int read_i32(FILE *stream, int32_t *value)
{
    return fread(value, sizeof *value, 1, stream) == 1 ? 0 : -1;
}
```

These are the complete [`write_i32`](../src/util.c) and
[`read_i32`](../src/util.c). `&value` gives `fwrite` the address of the
object. On Tiny AgenC's GNU/Linux reference environment, 32 bits
occupy four 8-bit bytes. `sizeof value` obtains that size from the
object, and the final `1` requests one complete object. The conditional
expression `condition ? 0 : -1` returns zero only when exactly one
object moved. `read_i32` mirrors the same contract and reports `-1` for
a short read.

The Chapter 2 witness writes `INT32_MIN` followed by `INT32_MAX`,
rewinds, and reads both back. Before the third read, predict its return
value. No complete integer remains, so it must return `-1`.

These helpers preserve the platform's native `int32_t` byte order.
Byte order is the choice of whether an integer's least-significant or
most-significant byte appears first in memory. The functions copy
those in-memory bytes unchanged. On common machines with eight-bit
bytes, two orders for the value one look like this:

```text
least-significant byte first:  01 00 00 00
most-significant byte first:   00 00 00 01
```

A machine with the same order reads the copied bytes back as one. A
machine with the opposite order gives those bytes a different value.
This machine-dependent choice is **native byte order**. Tiny AgenC's
GNU/Linux reference environment uses eight-bit bytes. Chapter 13
therefore builds a same-platform saved-model format.

Training speed needs a different utility. A wall clock can jump
backward when the system corrects its civil time. If the first reading
were 12:00:01 and the corrected second reading 12:00:00, subtraction
would claim that a step took negative time. A **monotonic clock** counts
forward from an unspecified origin and is intended for elapsed
durations:

```c
double time_seconds(void)
{
    struct timespec now;

    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec / NANOSECONDS_PER_SECOND;
}
```

This is the complete [`time_seconds`](../src/util.c).
`struct timespec` holds whole seconds in `tv_sec` and the remaining
nanoseconds in `tv_nsec`. The `.` operator in `now.tv_sec` selects a
named field directly from the struct value `now`; this is **struct
member access**. `now.tv_nsec` selects the other field.
`CLOCK_MONOTONIC` selects the nondecreasing clock, and `&now` passes
the address that `clock_gettime` fills. Two readings may be equal, but
a later one must not be smaller. The casts convert both integers to
`double`, C's wider floating-point type, before division. That width
preserves the fractional seconds better than `float`.

Before evaluating the return line, predict the result for 12 seconds
and 250,000,000 nanoseconds. The fractional part is
`250,000,000 / 1,000,000,000 = 0.25`, so the result is `12.25`. The
absolute origin is unspecified and has no calendar meaning;
differences between readings are elapsed seconds.

The source assumes `CLOCK_MONOTONIC` works on Tiny AgenC's supported
platforms and does not check `clock_gettime`'s return value. A port to
a platform without that clock would need to add a failure policy
before using `now`.

The functions have now earned every line of the implementation
preamble:

```c
#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L   /* clock_gettime, fseeko */

#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#include "util.h"

static const double NANOSECONDS_PER_SECOND = 1e9;
```

This is the complete file preamble from [`util.c`](../src/util.c).
The two `#define` lines must appear before the headers. A macro is a
name the C preprocessor replaces before compilation. These two are
feature requests: the first asks for 64-bit file offsets, and the
second asks the system headers to expose the 2008 POSIX interfaces
used for `fseeko` and `clock_gettime`. POSIX is the common interface
family used by Tiny AgenC's GNU/Linux reference environment.

The angle-bracket headers declare variadic arguments, fixed-width
integers, allocation and exit, and the clock. The quoted header
supplies Tiny AgenC's public utility declarations and the standard I/O
declarations used above.

`static` keeps `NANOSECONDS_PER_SECOND` private to this source file.
`const` prevents the code from changing it. `1e9` is C's spelling for
one times ten to the ninth, or 1,000,000,000.

## rng: randomness you can subpoena

Most parameter values start from random draws, and training eats random
batches. Chapter 0 used both. The word *random* describes the choices,
but debugging needs a stronger contract than "different every time." C's
implementation-defined, process-global `rand()` would give every
consumer one hidden sequence and weaken control over repeatability.

Imagine one global sequence of bounded draws:

```text
7, 4, 4, 2, 9, 4, 7, 8, ...
```

In a toy job with one corpus window per batch, the first two values
choose the first two batches and the next batch starts from the third.
If a progress report consumes one value between them, training instead
receives the fourth. Merely printing generated text has changed what
the model learns. The Chapter 0 configuration has 32 rows per batch,
so its batch handle consumes 32 bounded draws instead of one.

Tiny AgenC gives each job its own state. The training command uses:

```text
user seed     --> temporary initialization Rng --> parameter values
user seed     --> batch Rng                  --> training windows
user seed + 1 --> preview Rng                --> progress characters
user seed + 2 --> temporary checking Rng     --> second-corpus windows
```

The initialization and batch handles start from the same number, but
they are different objects. Advancing one does not move the other.
Model creation frees the initialization handle after filling the
tables. The checking handle exists only when a second corpus path was
supplied, and it is freed after preparing fixed windows. The command
that only generates text has its own map:

```text
user seed --> text Rng --> generated characters
```

Each state advances through an irregular-looking but completely
determined sequence. The same starting state replays the same
sequence. That construction is **pseudorandomness**: the outputs serve
as random choices, while an explicit algorithm and starting value make
them repeatable. Tiny AgenC does not use C's process-global `rand()`.

Training batches and generated-text previews use separate RNG streams,
so printing a preview never changes which corpus window trains next.

## Build a generator from state

Start with a toy state holding one number from 0 through 15. Advance it
with:

```text
new state = (5 * old state + 1) mod 16
```

From state 3, predict the first five new states before revealing the
line:

```text
3 -> 0 -> 1 -> 6 -> 15 -> 12 -> ...
```

For the fourth transition, `(5 * 6 + 1) mod 16 = 31 mod 16 = 15`.
The multiplication and addition move the state; `mod 16` wraps it into
the fixed range. Such a multiply-add-wrap step is called a **linear
congruential generator**.

Returning the state itself exposes a shorter pattern. Write the same
states with four bits and inspect only the lowest two:

```text
state          3    0    1    6   15   12   13    2
four bits    0011 0000 0001 0110 1111 1100 1101 0010
low two       11   00   01   10   11   00   01   10
```

The whole state has not repeated, but its low two bits repeat after
four steps. The real 64-bit constants have the same raw low-bit
problem:

```text
multiplier mod 4 = 1
increment  mod 4 = 3

new low bits = (old low bits + 3) mod 4
00 -> 11 -> 10 -> 01 -> 00
```

A caller should not receive those state bits unchanged. Build a
separate output path that folds distant bits together, then rotates
the result. The two bit operations are visible on four and eight bits:

```text
1100 XOR 1010 = 0110

rotate 10110001 right by three:
right piece   00010110
wrapped piece 00100000
joined result 00110110
```

XOR writes one where its two input bits differ. Rotation moves the
three bits that fall off the right back to the left instead of losing
them. The state still advances by multiply, add, and wrap; only the
returned bits take this extra path.

That state-plus-output construction is the published **PCG32**
generator: 64 bits of state are folded and rotated into a 32-bit
result. Its fixed state constants are:

```c
static const uint64_t PCG32_MULTIPLIER = 6364136223846793005ULL;
static const uint64_t PCG32_INCREMENT  = 1442695040888963407ULL;
```

This exact excerpt comes from [`rng.c`](../src/rng.c). `uint64_t` is an
unsigned integer with exactly 64 bits. `ULL` marks each literal as an
unsigned `long long`, a type wide enough for these values on Tiny
AgenC's supported platforms. They define the stream; they are not
model settings.

The output permutation also needs three shift counts and one output
width. This shortened exact excerpt is the opening of the source enum.
Its final uniform-conversion entry appears when that job is built:

```c
enum {
    PCG32_FOLD_SHIFT     = 18,   /* xor the high half onto the low */
    PCG32_OUT_SHIFT      = 27,   /* keep the best 32 of 64 bits */
    PCG32_ROTATION_SHIFT = 59,   /* top five bits pick the rotation */
    PCG32_OUTPUT_BITS    = 32,
```

The fold first moves old-state bits right by 18 before XOR. A second
shift by 27 selects 32 result bits. Shifting the old state by 59 leaves
its top five bits, which can hold a number from zero through 31. The
last name records the 32-bit output width. These names expose the
published layout instead of scattering bare counts through the
function.

`Rng` is the opaque handle pattern from Chapter 1. Its private
structure has a 64-bit `state` field, but callers cannot reach it. Here
`rng->state` means "follow the pointer named `rng`, then access its
`state` field." Here is the complete [`pcg32_next`](../src/rng.c):

```c
static uint32_t pcg32_next(Rng *rng)
{
    uint64_t old = rng->state;

    rng->state = old * PCG32_MULTIPLIER + PCG32_INCREMENT;

    uint32_t folded   = (uint32_t)(((old >> PCG32_FOLD_SHIFT) ^ old) >> PCG32_OUT_SHIFT);
    uint32_t rotation = (uint32_t)(old >> PCG32_ROTATION_SHIFT);

    /* rotate the 32 output bits right by `rotation` */
    return (folded >> rotation)
         | (folded << ((PCG32_OUTPUT_BITS - rotation) & (PCG32_OUTPUT_BITS - 1)));
}
```

Here `static` applies the same file-private rule to a function. Code
outside `rng.c` cannot call `pcg32_next` by that name.

`uint32_t` is the matching unsigned type with exactly 32 bits.
Unsigned 64-bit arithmetic wraps modulo `2^64`, so the state update
implements the wrap without an explicit `%`.

The output uses `old`, the state from before the update. `>>` shifts
bits right. `^` combines two bit strings with XOR: a result bit is one
when the input bits differ. Before rotation, output bit zero comes
from old-state bit 27 XOR old-state bit 45, not the short-cycling raw
bit zero. The cast to `uint32_t` keeps 32 bits. The top five old-state
bits choose a rotation from 0 through 31.

In the final expression, `|` joins the shifted pieces as in the
eight-bit example. Between integers, `&` means bitwise AND; this is
different from unary `&text`, which took an address in the file
reader. The mask `& 31` keeps the second shift between 0 and 31. When
`rotation` is zero, it changes the would-be shift by 32 into a shift
by zero, avoiding an invalid C operation.

The constants and permutation come from Melissa O'Neill's PCG
definition. [Appendix B](appendix-b-sources.md#pcg) points to the
paper and its statistical evidence. The small examples here establish
the raw low-bit problem and walk the output mechanism; they do not
stand in for testing the entire `2^64`-state stream.

## Enter the sequence through a seed

A deterministic generator still needs a precisely defined place to
start in its fixed cycle. If two programs advance the same PCG step but
initialize its state differently, they produce different sequences.

Tiny AgenC follows a four-step entry procedure:

```text
state = 0
advance once
state = state + user's starting number
advance once again
```

Apply the procedure to the four-bit multiply-add-wrap rule and starting
number three:

```text
state = 0
first advance:       (5*0 + 1) mod 16 = 1
add starting number: 1 + 3            = 4
second advance:      (5*4 + 1) mod 16 = 5
```

The first caller-visible draw begins from the prepared state five. The
two advances incorporate the user's number before a caller receives
output. Two programs using the same PCG step and starting number but
different entry procedures would produce different streams.

The starting value that selects this repeatable stream is the
**seed**. Before reading on, predict what two separate
`rng_new(1337)` handles produce. They advance independently but begin
from identical state, so corresponding draws match. A handle seeded
with 1338 begins another stream.

The exact constructor also initializes storage for a paired draw that
has not been built yet. Its source walk waits until that storage has a
job.

## Turn 32 bits into a fraction

The raw PCG result lies from 0 through `2^32 - 1`. Character selection
and the later bell conversion need a fraction starting at zero but
never reaching one. A first attempt might convert all 32 bits to
`float` and divide by `2^32`. That asks a `float` to distinguish more
bits than it carries. On the supported IEEE 754 representation,
converting `4,294,967,295` to `float` rounds it up to
`4,294,967,296`. Dividing by `2^32` would then produce `1.0`, outside
the required half-open interval.

An ordinary IEEE 754 `float` has 24 significant binary bits: 23 stored
fraction bits plus one leading bit supplied by the representation.
Use an eight-bit draw and a three-bit result to construct the fix:

```text
eight-bit draw        10110110
keep the high 3 bits  101 = 5
scale                  5 / 8 = 0.625
```

Every input from `10100000` through `10111111` has those same high
three bits. All 32 map to `5/8`. Across all 256 inputs, each of the
eight fractions `0/8` through `7/8` receives exactly 32 inputs. Zero
is included and one is not.

Tiny AgenC applies the same construction at the real sizes: discard
the low eight of 32 bits, keep the high 24, and divide by `2^24`.
The result has this exact finite shape:

```text
possible values  k / 2^24, for k = 0 through 2^24 - 1
spacing          2^-24 = 0.000000059604644775390625
largest value    1 - 2^-24 = 0.999999940395355224609375
```

Each of the `2^24` grid points has exactly 256 of the original 32-bit
inputs behind it. This evenly represented finite grid earns the name
**uniform random draw**. Equal runs of grid points have equal chances;
an arbitrary interval boundary can differ by one grid point.

The source adds the uniform count to the four PCG counts already
walked:

```c
enum {
    PCG32_FOLD_SHIFT     = 18,   /* xor the high half onto the low */
    PCG32_OUT_SHIFT      = 27,   /* keep the best 32 of 64 bits */
    PCG32_ROTATION_SHIFT = 59,   /* top five bits pick the rotation */
    PCG32_OUTPUT_BITS    = 32,
    UNIFORM_BITS         = 24,   /* a float holds 24 significant bits */
};
```

This is the complete enum from [`rng.c`](../src/rng.c). An `enum` can
also give compile-time integer names to fixed constants rather than
status values. The first four names repeat the output layout above.
The new `UNIFORM_BITS` name records the 24 bits a `float` can carry.
None is learned or tuned.

Now the source follows the constructed operation:

```c
float rng_uniform(Rng *rng)
{
    uint32_t bits = pcg32_next(rng) >> (PCG32_OUTPUT_BITS - UNIFORM_BITS);

    return (float)bits / (float)(1u << UNIFORM_BITS);
}
```

This is the complete [`rng_uniform`](../src/rng.c).
`PCG32_OUTPUT_BITS - UNIFORM_BITS` is `32 - 24 = 8`, so the first line
discards the low eight bits. `1u` is the unsigned integer one;
shifting it left 24 places produces `2^24 = 16,777,216`.

For seed 42, the first retained integer is 12,776,827:

Rounded to nine decimal places:

```text
12,776,827 / 16,777,216 = 0.761558235
```

What should a second handle created with seed 42 return on its first
call? The same rounded `0.761558235`. On the next call, both handles
return `0.418087244` when rounded to nine decimal places. The lab
compares the exact float bits, not two copies of the same possibly
wrong implementation.

## Measure a set's center and spread

Weight initialization needs a different draw rule: many values close
to zero and progressively fewer far away. Before building it, we need
two arithmetic tools for describing where a group of numbers sits.

Take the three numbers:

```text
1, 2, 3
```

Before calculating, which one sits at the center? Add them and divide
by the count to check the prediction:

```text
(1 + 2 + 3) / 3 = 2
```

This add-then-divide center is the **mean**.

Now ask how far the values sit from that center:

```text
value             1    2    3
value - mean     -1    0    1
```

Predict what happens if those three differences are added. They cancel
to zero even though the values are not all at the center. Square each
difference so distance in either direction counts positively:

```text
squared difference    1    0    1
mean squared difference = (1 + 0 + 1) / 3 = 2/3
```

That mean squared distance is the **variance**. Some statistics
formulas divide by `count - 1` when a small observed group is being
used to estimate a larger unseen group. Tiny AgenC is measuring the
complete row it has, so it divides by the count. Chapter 5 will apply
the same population-variance rule when it normalizes one complete row.
An empty group has no mean because there is no count to divide by. A
one-value group has variance zero because its only value equals its
mean.

Variance is measured in squared units. Taking its square root returns
to the original units:

```text
square root of 2/3 = about 0.816
```

That result is the **standard deviation**, the spread measure denoted
by `sigma` or `σ` in Chapter 1's parameter table. A short run of random
draws will not have exactly the intended mean and spread; those values
describe what repeated draws approach, not a promise about every small
sample.

Spreads need special bookkeeping when random contributions are added.
Let each of two choices be `-1` or `+1`, with each possibility equally
likely. The second choice does not react to the first, so all four
pairs have equal chances:

```text
A    B    A + B
-1  -1     -2
-1  +1      0
+1  -1      0
+1  +1      2
```

Before calculating, predict the mean of the four sums. Their total is
zero, so the mean is zero. Their variance and spread are:

```text
variance = ((-2)^2 + 0^2 + 0^2 + 2^2) / 4
         = (4 + 0 + 0 + 4) / 4
         = 2

standard deviation = sqrt(2)
```

Each input alone had variance one and standard deviation one. The
variances added, `1 + 1 = 2`; the standard deviations did not. The
mixed products cancel because the four sign pairs balance:

```text
(-1)(-1) + (-1)(+1) + (+1)(-1) + (+1)(+1) = 0
```

When learning one choice does not change the chances for the other,
the choices are **independent**. The same cancellation works beyond
the two-sign example. Center each choice by subtracting its own mean.
The average centered value is zero. Expand one squared sum:

```text
(centered A + centered B)^2
    = (centered A)^2
      + 2*(centered A)*(centered B)
      + (centered B)^2
```

Independence pairs every centered A chance with every centered B
chance. The average mixed product is therefore
`(average centered A)*(average centered B) = 0*0 = 0`. Only the two
variance terms remain. Repeating that bookkeeping for `N` independent
contributions of spread `s` adds `s*s` to the variance `N` times:

```text
combined variance = N*s*s
combined spread   = sqrt(N*s*s) = sqrt(N)*s
```

This is the square-root combination promised in Chapter 1. Chapter 9
will apply it to the `2L` layer outputs that can be added along the
model's main path.

## Growth and its undo

To turn flat uniform draws into a bell shape, the source needs one more
small piece of math.

Start at one and apply one 100-percent increase. The result is two.
Now split that same stated increase into two 50-percent updates. Before
reading the result, predict whether compounding makes it equal to two:

```text
one update:   (1 + 1)^1       = 2
two updates:  (1 + 1/2)^2     = 1.5 * 1.5 = 2.25
four updates: (1 + 1/4)^4     = 1.25^4 = 2.44140625
```

The smaller updates act on the growth produced by earlier ones, so the
result rises. Splitting the interval into more and more updates makes
the values approach:

```text
2.7182818...
```

That limiting growth factor is named **e**. It gives a scale where
equal additions to an input cause equal multiplications of the output.
Raise `e` to several whole-number inputs first.

Before reading the zero row, predict its result. Any nonzero number to
power zero is one.

| `x` | `e^x` rounded |
|---:|---:|
| -2 | 0.1353 |
| -1 | 0.3679 |
| 0 | 1.0000 |
| 1 | 2.7183 |

Negative inputs produce reciprocals, so `e^-1 = 1 / e^1`.
Multiplying two outputs adds their inputs:

```text
e^a * e^b = e^(a + b)

e^-1 * e^1 = 0.3679 * 2.7183 = about 1
e^(-1 + 1) = e^0                 = 1
```

Fractional inputs preserve that rule. Two half-input factors must make
the one-input factor:

```text
e^(1/2) * e^(1/2) = e^1 = e
e^(1/2) = sqrt(e) = about 1.6487
e^(-1/2) = 1 / sqrt(e) = about 0.6065
```

Ever finer fractions extend the operation between whole-number inputs.
The real-input operation constructed by those rows is the
**exponential**, written `exp(x) = e^x`. The math library supplies it.

That add-then-multiply behavior will turn a two-coordinate rule into a
rule based only on distance.

Suppose the output is known and the exponent is missing. Reverse each
of the three relationships:

```text
exp(0) = 1       so log(1) = 0
exp(1) = e       so log(e) = 1
exp(-1) = 0.3679 so log(0.3679) = about -1
```

The inverse operation constructed by those rows is the **natural
logarithm**, written `log`.

The multiplication rule gives one matching rule for `log`. If
`p = exp(a)` and `q = exp(b)`, then `p*q = exp(a+b)`. Reversing that
statement gives:

```text
log(p*q) = log(p) + log(q)
log(2^24) = 24*log(2)
log(2^-24) = -24*log(2)
```

Because the mathematical `exp` never produces zero or a negative
value, `log(x)` accepts only `x > 0`. In C, the float versions are
`expf` and `logf`. Chapter 5 will reuse both operations when it turns
raw scores into bets and grades them; their first need is the
bell-shaped generator here.

Those mathematical functions have no finite endpoint, but their C
versions return `float`. A sufficiently large positive input can
overflow `expf`, and a sufficiently negative one can round its result
to zero. Chapter 5 will keep its inputs in a safe range before calling
it.

## Turn two flat draws into a bell

If a uniform draw `u` lies in `[0, 1)`, then `2u - 1` lies in `[-1, 1)`.
That changes the endpoints but not the shape: every equal-width
interval still receives the same share of draws, and values beyond one
can never occur. It cannot initialize weights under a bell-shaped rule.

Build the desired shape before trying another conversion. Give a tiny
equal-width interval around `x` this relative weight:

```text
w(x) = exp(-x*x/2)
```

The square makes `+x` and `-x` match. The negative sign makes the
weight fall with distance from zero:

| `x` | `w(x)` rounded |
|---:|---:|
| 0 | 1.0000 |
| 1 or -1 | 0.6065 |
| 2 or -2 | 0.1353 |

These weights compare equal-width intervals; they are not percentages.
Here *weight* means a chance score, not a learned model parameter.
Multiplying every row by the same constant would leave their ratios
unchanged.

We need two output numbers. If the horizontal and vertical choices do
not alter each other's chances, a tiny equal-area patch around the pair
has relative weight `w(x)*w(y)`:

```text
                    y = 0    y = 1    y = 2
x = 0               1.0000   0.6065   0.1353
x = 1               0.6065   0.3679   0.0821
x = 2               0.1353   0.0821   0.0183
```

For fixed `x = 1`, every entry is the `x = 0` entry multiplied by
`0.6065`. That common factor cancels when comparing the vertical
choices. Learning `x` therefore does not change their relative
chances. This is the independence constructed in the spread example.

The exponential multiplication rule now does useful work:

```text
w(x) * w(y)
    = exp(-x*x/2) * exp(-y*y/2)
    = exp(-(x*x + y*y)/2)
```

The expression `x*x + y*y` measures squared distance in two
perpendicular directions. Build that rule with non-negative horizontal
length three and vertical length four. Put one resulting right triangle
in each corner of a square of side `3 + 4`, turning each copy by one
square corner from the copy before it. Their equally long slanted sides
enclose a smaller four-sided shape. Each turn also turns the next
slanted side by one square corner, so the inner shape has four equal
sides and four square corners. It is a square. Two copies of one
triangle form a `3` by `4` rectangle of area 12, so one triangle has
half that area, or 6:

```text
outer area      = 7*7 = 49
one triangle    = 3*4/2 = 6
four triangles  = 4*6 = 24
inner area      = 49 - 24 = 25
inner side      = sqrt(25) = 5
```

Call that straight-line distance from `(0, 0)` the radius `r`. For a
coordinate `x`, its horizontal length ignores whether `x` points left
or right, but squaring either sign gives `x*x`. The vertical length
works the same way for `y`. Call those non-negative lengths `a` and
`b`. The area bookkeeping gives:

```text
r*r = (a + b)^2 - 4*(a*b/2)
    = a*a + 2*a*b + b*b - 2*a*b
    = a*a + b*b
    = x*x + y*y
```

The pair's weight therefore depends only on radius. Every direction at
one radius should have the same chance. Construct the direction rule
first on a circle of radius one. Before reading the half-turn row,
predict its horizontal and vertical coordinates:

| Turn | Horizontal | Vertical |
|---:|---:|---:|
| 0 | 1 | 0 |
| 1/4 | 0 | 1 |
| 1/2 | -1 | 0 |
| 3/4 | 0 | -1 |

An **angle** records that direction. On the radius-one circle, the
horizontal coordinate is named the **cosine** of the angle and the
vertical coordinate is named its **sine**. At any radius:

```text
x = r*cos(angle)
y = r*sin(angle)
```

C measures angles in **radians**. The circle constant `pi`, about
3.14159, is circumference divided by diameter. A radius-one circle has
circumference `2*pi`; radians use that arc length as the angle number,
so a full turn is `2*pi`.

One uniform draw can now choose a direction without favoring one:

```text
angle = 2*pi*u2
```

Equal intervals of `u2` cover equal arcs of the circle. The remaining
problem is choosing the radius.

A wide outer ring covers more area than a narrow inner ring. Giving
both rings the same chance would not give each patch of the plane the
same bell weight. Change variables from `r` to:

```text
s = r*r/2
```

The needed circle-area rule also comes from `pi`. Cut a disk into many
thin wedges and alternate their pointed ends. They approach a
rectangle whose base is half the circumference and whose height is the
radius:

```text
base   = (2*pi*r)/2 = pi*r
height = r
area   = base*height = pi*r*r
```

Two equal-width bands of `s` now make rings with equal areas:

```text
s from 0.0 to 0.5: r*r from 0 to 1, area = pi
s from 0.5 to 1.0: r*r from 1 to 2, area = 2*pi - pi = pi
```

In general, a band from `s` to `s + d` has area:

```text
pi * 2*(s + d) - pi * 2*s = 2*pi*d
```

Equal bands of `s` cover equal areas. Within a narrow band, the bell
weight changes only a little, so its total relative weight is the
common area times about `exp(-s)`. That common area cancels when
comparing bands. As the width `d` narrows toward zero, the
approximation becomes exact. To find the share in the tail beyond a
chosen band, take `d > 0` and call `exp(-d)` by the shorter name `q`.
That makes `0 < q < 1`, so each term is smaller than the one before
it. Call the full series `A`. Multiplying it by `q` removes its first
term:

```text
A    = 1 + q + q*q + q*q*q + ...
q*A  =     q + q*q + q*q*q + ...
A    = 1 + q*A
A*(1 - q) = 1
A    = 1/(1 - q)

weight of all bands           = A
weight of bands starting at k = q^k * A

tail share = (q^k * A) / A
           = q^k
           = exp(-k*d)
```

As the bands narrow, `k*d` becomes the chosen value `s`. The ideal
tail share beyond `s` is therefore `exp(-s)`.

Let `v = 1 - u1`. For an ideal uniform draw, the share of values at or
below any number `a` is `a`. Because `exp(-s)` falls as `s` grows,
values with `v <= exp(-s)` must produce radii at or beyond `s`.
Matching the two tail shares requires:

```text
v = 1 - u1 = exp(-s)
```

Reverse the exponential, then recover the radius:

```text
log(1 - u1) = -s
s = -log(1 - u1)
r*r = 2*s
r = sqrt(-2*log(1 - u1))
```

The square root chooses the non-negative radius. Combining radius and
angle gives the pair:

```text
x = r*cos(angle)
y = r*sin(angle)
```

Check the promised spread without calculus. Start with four steps per
unit of `s`. At each step, make a fresh independent choice: stop with
chance `1/4`; otherwise continue. If `m` is the average number of steps
remaining, one step always passes and three quarters of runs continue:

```text
m = 1 + (3/4)*m
m - (3/4)*m = 1
(1/4)*m = 1
m = 4

average s = 4 steps * 1/4 unit per step = 1
```

After `k` steps, the continuing share is `(3/4)^k`. Now replace four
with any `n > 1`. The same average calculation becomes:

```text
m = 1 + ((n - 1)/n)*m
m/n = 1
m = n

average s = n steps * 1/n unit per step = 1
```

At a step boundary `s = k/n`, the count `n*s` is the integer `k`.
The continuing share is `(1 - 1/n)^(n*s)`. For one unit, rewrite that
factor:

```text
(1 - 1/n)^n
    = 1 / (1 + 1/(n - 1))^n
    = 1 / (1 + 1/(n - 1))^(n - 1)
      * 1 / (1 + 1/(n - 1))
```

As the steps narrow, the first denominator approaches `e` by the
compounding construction, and the second factor approaches one. The
continuing share for one unit therefore approaches `1/e = exp(-1)`.
For `s` units,
`((1 - 1/n)^n)^s` therefore approaches
`exp(-1)^s = exp(-s)`, the tail rule above.

In the limit, the stopping construction and the inverse-uniform
construction have the same share beyond every value of `s`. They
describe the same ideal rule. Since every finite stopping construction
has average `s = 1`, the limiting selected `s` used for the radius does
too.

Since `r*r = 2*s`, the average squared radius is two. Opposite angles
balance, so the average `x` and `y` are each zero. Rotation treats the
two coordinates alike, so each receives half of
`x*x + y*y = r*r`:

```text
average x*x = 1
average y*y = 1
```

With mean zero, each variance is one, and each standard deviation is
one. The product table already showed that the two coordinates are
independent.

Use one ideal pair that can be checked without a calculator:

```text
u1 = 1 - exp(-2)    so 1 - u1 = exp(-2)
u2 = 1/2

radius = sqrt(-2*log(exp(-2))) = sqrt(4) = 2
angle  = 2*pi*(1/2) = pi, a half turn
```

Before revealing the coordinates, predict the direction: a half turn
points left, so `x` must be negative and `y` must be zero.

```text
x = 2*cos(pi) = -2
y = 2*sin(pi) =  0
```

Work one pair from the committed seed-1337 stream. The second draw is
between one quarter and one half of a full turn, so the direction
points up and left. Predict the signs of `x` and `y`, then check the
last two rows. These rows round each source `float` to nine decimal
places:

```text
u1       = 0.581423998
u2       = 0.350974739
1 - u1   = 0.418576002
log(...) = -0.870896816
radius   = sqrt(1.741793633) = 1.319770336
angle    = 2*pi*u2           = 2.205239296

x = radius*cos(angle) = -0.782266140
y = radius*sin(angle) =  1.062945604
```

This uniform-pair conversion is the **Box-Muller construction**. Its
bell-shaped output rule is called the **normal distribution** or
**Gaussian distribution**. When its mean is zero and standard
deviation is one, it is the standard normal.

Before reading the implementation edge, predict whether a grid with
only `2^24` fractions can preserve the ideal rule's lack of an
endpoint. It cannot: the grid has a closest value to one, so
`1 - u1` has a smallest positive value.

The ideal construction assumes a uniform number can approach one
without a closest value. Its radius has no fixed endpoint. Tiny
AgenC's uniform generator has only `2^24` possible fractions, so its
largest `u1` is `1 - 2^-24`. Its smallest possible `1 - u1` and
largest radius are:

```text
smallest 1 - u1 = 2^-24
largest radius  = sqrt(-2*log(2^-24))
                = sqrt(48*log(2))
                = about 5.7681
```

The C function is therefore a finite 24-bit approximation to the
ideal standard normal. Neither returned coordinate can have magnitude
greater than about `5.7681`. Under the ideal rule, the share of pairs
beyond that radius is `2^-24`, or one in `16,777,216`. The finite grid
has no point available for that remaining tail.

The constructor can now earn the private fields it initialized. The
source stores the full-turn constant and its hidden state like this:

```c
static const float TWO_PI = 6.28318530717958647692f;
static const float BOX_MULLER_RADIUS_FACTOR = -2.0f;

struct Rng {
    uint64_t state;
    int      has_spare_gaussian;   /* Box-Muller yields two draws; bank one */
    float    spare_gaussian;
};
```

This complete excerpt appears earlier in [`rng.c`](../src/rng.c), but
the book waits until each field has a job. `state` drives PCG32.
`BOX_MULLER_RADIUS_FACTOR` names the `-2` in the radius equation. The
integer flag says whether `spare_gaussian` currently holds the second
coordinate from a pair. When the flag is zero, that float's stored
bits do not matter and must not be read. The `f` suffix on each
constant makes its literal a `float`.

Here is the complete constructor whose four-step entry procedure was
built earlier:

```c
Rng *rng_new(unsigned long long seed)
{
    Rng *rng = emalloc(sizeof *rng);

    rng->state = 0;
    rng->has_spare_gaussian = 0;
    pcg32_next(rng);
    rng->state += seed;
    pcg32_next(rng);
    return rng;
}
```

`emalloc` creates one owned `Rng`. The next two lines establish a zero
state and an empty spare. Each ignored `pcg32_next` result still
advances the state. Between those advances, `+= seed` adds the caller's
starting value. Finally, the function returns the owned handle. Two
calls with the same seed perform the same state changes in separate
objects.

The source computes the pair and saves half:

```c
float rng_gaussian(Rng *rng)
{
    if (rng->has_spare_gaussian) {
        rng->has_spare_gaussian = 0;
        return rng->spare_gaussian;
    }

    /* Box-Muller: two uniform draws become two independent gaussians.
     * 1 - u keeps the logarithm's argument in (0, 1], never zero. */
    float radius =
        sqrtf(BOX_MULLER_RADIUS_FACTOR
              * logf(1.0f - rng_uniform(rng)));
    float angle  = TWO_PI * rng_uniform(rng);

    rng->spare_gaussian     = radius * sinf(angle);
    rng->has_spare_gaussian = 1;
    return radius * cosf(angle);
}
```

This is the complete [`rng_gaussian`](../src/rng.c). The opening branch
returns the saved coordinate and clears its flag. Otherwise, the first
uniform draw creates the radius and the second creates the angle.
`1.0f - u` matters because `u` may equal zero: using `logf(u)` would
then ask for `log(0)`. Since `u` never reaches one, `1-u` stays in
`(0, 1]`, the logarithm's valid domain.

The vertical coordinate from `sinf` is stored, its flag becomes true,
and the horizontal coordinate from `cosf` returns. Before revealing
the next state change, predict how many new uniform draws the next
`rng_gaussian` call consumes. It consumes none; it returns the spare.
The third Gaussian call consumes the next pair of uniforms.

The `f` suffix on `sqrtf`, `logf`, `sinf`, and `cosf` selects the
version that accepts and returns `float`. The unsuffixed C functions
work with the wider `double` type.

The source comment names the ideal outputs as independent Gaussians.
The finite grid and pseudorandom PCG stream approximate that
mathematical rule. These functions also use finite-precision `float`
math. With the same executable and floating-point environment, the
stream replays. Another compiler, processor, or math library may differ
in its last bits, so the lab permits a small tolerance for committed
Gaussian answers.

## Pick an integer without favoring one

Corpus batching does not need a fraction or a bell. It needs one integer
from `0` through `bound - 1`, with no position favored.

The tempting conversion is `draw % bound`. Use a toy source with eight
equally likely draws, 0 through 7, and `bound = 3`. Before reading the
counts, predict which remainders receive an extra source value:

```text
remainder 0: source draws 0, 3, 6   (three ways)
remainder 1: source draws 1, 4, 7   (three ways)
remainder 2: source draws 2, 5      (two ways)
```

Eight does not divide evenly by three. Remainders zero and one receive
one extra source value. The same mismatch exists between `2^32` PCG
outputs and most bounds.

Daniel Lemire's mapping multiplies the source draw by the range, uses
the high word as the answer, and inspects the low word for the uneven
edge. A four-bit version makes the operation visible. With source draws
0 through 15 and range three, multiply by three. Splitting a product
into a high part and a low part means:

```text
high = product div 16
low  = product mod 16
```

Check the four boundary draws:

| Draw | Product | High | Low | Action |
|---:|---:|---:|---:|:---|
| 0 | 0 | 0 | 0 | reject |
| 1 | 3 | 0 | 3 | keep |
| 6 | 18 | 1 | 2 | keep |
| 11 | 33 | 2 | 1 | keep |

Sixteen leaves remainder one when divided by three. Reject source draw
zero, whose product low part is below that threshold. The remaining
15 draws give five ways to produce each answer: draws 1 through 5
produce zero, 6 through 10 produce one, and 11 through 15 produce two.

The toy threshold can be computed without writing `16` in a
fixed-width four-bit value:

```text
-3 wraps to 16 - 3 = 13
13 mod 3 = 1
```

For a 32-bit range, the same unsigned calculation gives
`2^32 mod range`, the number of source values on the uneven edge.
For range ten, that threshold is six, so low words zero through five
are rejected. If the range divides `2^32`, the threshold is zero and
no low word is rejected.

A zero bound would describe the empty interval `[0, 0)` and later make
remainder by zero invalid. That can only be an internal caller error;
there is no useful bounded draw to return. C's `assert(condition)`
checks such a programmer contract in assertion-enabled builds. If the
condition is false, it prints the failed condition and source location,
then stops the process. Defining `NDEBUG` when compiling removes
assertions, so they are not a substitute for checking recoverable user
input.

Rejection must draw once, then repeat only when the low part falls in
the uneven edge. C's `do { ... } while (condition);` loop matches that
order: it executes the body before its first condition check. The
semicolon after `while (condition)` is required syntax.

The real code performs that construction with 32-bit words:

```c
int rng_below(Rng *rng, int bound)
{
    assert(bound > 0);

    /*
     * Lemire's method: multiply a 32-bit draw by the range and keep the
     * high word.  Unless the range divides 2^32, multiply-high alone gives
     * some results one extra preimage.  Reject low-word values below the
     * threshold to make every result equiprobable.
     */
    uint32_t range = (uint32_t)bound;
    uint32_t threshold = (uint32_t)(-range) % range;
    uint32_t draw;
    uint32_t low;
    uint64_t product;

    do {
        draw    = pcg32_next(rng);
        product = (uint64_t)draw * (uint64_t)range;
        low     = (uint32_t)product;
    } while (low < threshold);

    return (int)(product >> PCG32_OUTPUT_BITS);
}
```

This is the complete [`rng_below`](../src/rng.c). Converting `bound` to
`uint32_t` makes the following arithmetic unsigned. In unsigned 32-bit
arithmetic, `-range` wraps around `2^32`; taking its remainder by
`range` computes the unwanted edge size `2^32 mod range`.

The declared `draw`, `low`, and `product` variables receive values
inside the loop. Casting both factors to `uint64_t` keeps the full
64-bit product. Casting that product back to `uint32_t` keeps its low
word for the rejection test. Once the low word reaches the threshold,
shifting right by 32 returns the high word, an integer from zero
through `bound - 1`.

With seed 42 and bound 10, the committed first results are:

```text
7, 4, 4, 2, 9, 4, 7, 8
```

Before checking the lab, predict whether a returned value can equal
10. It cannot; the interval is written `[0, 10)`, including zero and
excluding ten. With `bound = 1`, the only valid answer is zero;
`2^32 mod 1` is zero, so no draw needs rejection.

Every need in the module's include block has now appeared:

```c
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "rng.h"
#include "util.h"
```

This is the complete include block from [`rng.c`](../src/rng.c).
`assert.h` declares `assert`; `math.h` declares the circle, square-root,
and logarithm functions; `stdint.h` supplies fixed-width integers; and
`stdlib.h` declares `free`. The public `rng.h` declares the opaque
handle and its functions. `util.h` declares the error-checked allocator
used by `rng_new`.

The complete module has one final ownership operation:

```c
void rng_free(Rng *rng)
{
    free(rng);
}
```

`rng_new` creates and returns the owned object. `rng_free` ends its
lifetime. No caller may draw from that handle afterward.

## The foundation contract

The two modules now have a finite public contract:

| Function | Success | Failure or edge |
|---|---|---|
| `emalloc(n)` | owned `n`-byte block | prints and exits on `NULL` |
| `ecalloc(k,n)` | owned zeroed block | prints and exits on `NULL` |
| `die(...)` | never returns | always exits with failure |
| `file_slurp` | owned bytes, length, sentinel | `NULL` |
| `file_slurp_bounded` | same, under exact ceiling | I/O or too-large status |
| `write_i32` / `read_i32` | returns `0` | returns `-1` |
| `time_seconds` | monotonic seconds | assumes the fixed clock succeeds |
| `rng_new(seed)` | owned repeatable stream | caller later frees it |
| `rng_uniform` | one of `2^24` fractions in `[0,1)` | consumes one PCG draw |
| `rng_gaussian` | finite standard-normal approximation | capped near `5.7681`; consumes zero or two uniform draws |
| `rng_below(n)` | uniform integer in `[0,n)` | requires `n > 0` |

The RNG layers fit in one picture:

```text
seed
  |
  v
PCG32 state --> 32 output bits
                    |
          +-----------+------------+
          |           |            |
          v           v            v
       uniform    Box-Muller   multiply-high
        [0,1)   Gaussian approx  [0,bound)
```

Chapter 0's first training score depended on the Gaussian branch for
most initial parameter values and the bounded branch for corpus
windows. Separate handles keep a progress sample from changing the
batch stream. Chapter 3 now has the two tools it needs:
`file_slurp_bounded` to bring corpus bytes into memory, and `rng_below`
to choose training windows.

## Build checkpoint: repeat the future

The Chapter 0 workspace already contains the public
[`util.h`](../src/util.h) and [`rng.h`](../src/rng.h) interfaces. Your
job is to supply their implementations without copying the answer key.

**Build.** Implement `die`, `emalloc`, and `ecalloc` in
`labs/work/util.c`. Add the feature macros, headers, and private clock
constant shown in the chapter. Then implement `file_slurp`,
`file_slurp_bounded`, `write_i32`, `read_i32`, and
`time_seconds`. The bounded reader must initialize valid output
pointers before fallible work, reject an excessive length before
allocating or reading the body, accept its exact ceiling, preserve
binary zero bytes, reject bytes beyond the measured end, require a
successful close before publication, and append one sentinel byte.

In `labs/work/rng.c`, add the required headers, fixed PCG constants,
private `Rng` definition, and circle constant. Implement the PCG32 state
step and output permutation, the reference seeding sequence,
`rng_uniform`, `rng_gaussian` with its banked spare, `rng_below` with
rejection, and `rng_free`. Keep `Rng` opaque. Do not substitute
`rand()`.

**Verify.**

```sh
make -C labs check-02
# answer key: make check-foundations
```

**Expected.** On GNU/Linux, the witness reports all 330 foundation
checks passed.
File slurping preserves `A`, an embedded zero byte, and `Z`, reports
length three, and appends a sentinel. A ceiling of two rejects that
file; a ceiling of three accepts it. An empty file succeeds under a
zero-byte ceiling. Missing files and oversized files produce different
statuses, with outputs reset on both failures. The `/proc/self/cmdline`
witness reports a measured zero length but produces bytes, so the
reader rejects it rather than publishing a partial result.

`INT32_MIN` and `INT32_MAX` round-trip, a third read fails, allocated
zeroed storage contains zeroes, and successive clock reads do not move
backward. Seed 42 matches the committed uniform and bounded sequences.
Two seed-1337 generators replay Gaussian values, seed 1338 changes the
stream, and every bounded result lies in `[0, bound)`.

**Common failures.**

- A crash after allocation failure means a `NULL` pointer escaped the
  wrapper instead of reaching `die`.
- Garbage in `ecalloc` storage means the implementation called
  `malloc` without clearing the bytes.
- Losing bytes after an embedded zero means file length was inferred
  with a C string function instead of the explicit stream length.
- Correct file bytes but a missing final sentinel means the allocation
  omitted `+1` or the final zero assignment.
- Stale output values after a bounded-read failure mean `*text` and
  `*size` were initialized too late.
- Rejecting the exact ceiling usually means `>=` replaced `>`.
  Accepting `maximum + 1` means the comparison or ceiling arithmetic
  is wrong.
- Even correct statuses are insufficient if an oversized file is
  allocated or read before rejection; the size test belongs before
  both operations.
- A successful third integer read means `read_i32` did not require one
  complete `int32_t`.
- Whole-number elapsed times point to integer division. A later clock
  reading below an earlier one points to a civil-time clock.
- Matching the first uniform but not later draws often means the PCG
  output used the new state where the published permutation uses
  `old`.
- Getting the first Gaussian right but the second wrong often means
  Box-Muller did not bank and return its spare in the source order.
- A logarithm domain error means zero reached `logf`; the source uses
  `1.0f - u` so its argument stays in `(0, 1]`.
- Rarely favored bounded results mean `% bound` replaced the
  multiply-high rejection method.
- Negative or out-of-range bounded results usually come from signed
  arithmetic or a 32-bit product.
- Results that change when another consumer draws mean RNG state is
  global instead of owned by separate handles.

---

[Previous: The Map](01-the-map.md) | [Contents](README.md) | [Next: Data](03-data.md)
