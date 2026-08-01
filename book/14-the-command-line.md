# Chapter 14: The Command Line Is a Boundary

The model API uses assertions for impossible call sequences. A command
line cannot.

Consider this request:

```sh
./tiny-agenc train --data tiny.txt --steps 12oops
```

The model must eventually receive an integer step count. The terminal
has supplied six characters. If the program keeps the numeric prefix
and silently calls the value `12`, the run succeeds at doing something
the person did not ask for.

People mistype numbers. Files disappear. A validation corpus contains a
byte the training tokenizer has never seen. A checkpoint belongs to an
older format. These are ordinary input failures, and the terminal should
explain them without turning them into a C assertion puzzle.

The model functions built so far are allowed to require valid shapes,
live objects, and legal call order. The terminal is not allowed to make
those promises. Something must inspect the outside values before they
reach that core:

```text
untrusted outside values
    command words, number strings, paths, environment
                         |
                         v
              +---------------------+
              | inspect and convert |
              | reject bad requests |
              | route output streams|
              +---------------------+
                         |
                         v
admitted values at the next owned boundary
    typed fields, checked setup objects, deliberately routed output
```

This chapter builds the boundary between a careful mathematical core and
an unreliable outside world. By the time the boundary calls the model,
strings such as `12oops` are gone. What remains is either a checked
integer `12` from the complete string `12`, or a diagnostic and a
[failure status](02-foundations.md#one-place-decides-how-to-stop).
Paths and prompt text do not become trustworthy strings; they move to
the defensive file, checkpoint, and tokenizer operations that own their
next checks.

## Give two jobs separate command grammars

Suppose every option lived in one flat command:

```text
tiny-agenc --data corpus.txt --steps 5000 --model trained.bin \
           --prompt RAZR:
```

Chapter 16 will construct
[sampling](16-sampling.md#drawing-from-the-distribution), which chooses
generated ids from the model's probability sheet, and
[temperature](16-sampling.md#change-the-gaps-without-changing-their-order), the
number that controls how concentrated those choices are. In this
chapter both remain opaque command values.

Some values describe a new training run. Others describe generation
from an existing checkpoint. The program would have to infer which job
was intended from whichever flags happened to appear. A missing
`--data` could turn a malformed training request into a sampling
request. A stray `--model` could do the reverse.

Put one required word before the options instead:

```text
tiny-agenc train  ...
tiny-agenc sample ...
```

Now the first word after the program selects one command grammar.
`train` accepts a corpus, the
[capacity dimensions](01-the-map.md#capacity-and-the-active-call)
grouped in `ModelConfig`, and the
[learning rate](08-adamw.md#give-the-gradient-a-distance-control) that
scales parameter motion.
`sample` accepts a checkpoint, prompt, output length, and temperature.
That first selection is **command dispatch**.

Two top-level information paths need no model or corpus:

```text
tiny-agenc --help
tiny-agenc --version
```

No implicit default subcommand exists. Guessing whether a filename was
meant for training or sampling makes errors less legible, not more
friendly. Sampling likewise requires an explicit `--model FILE`.
Silently guessing a default checkpoint makes a typo look like a missing
file in an unrelated place.

Training does have one safe output default: `trained.bin`. It is not the
committed `tiny-agenc.bin` showcase, so an experimental run cannot
replace that file by omission. The longer commands in this book still
spell out `--out` so the destination is visible before a run starts.

**Predict:** should `tiny-agenc train --help` print help, or should it be
rejected?

Tiny AgenC has one help surface, exactly `tiny-agenc --help`. The
subcommand form is rejected as an invalid training option. One exact
form is easier to document and test than separate help rules inside
each parser.

## See the words that C receives

The shell removes its own quoting and starts the program with a count
and an ordered array of strings. For this command:

```sh
./tiny-agenc train --data tiny.txt --steps 12
```

C receives:

| Index | String |
|---:|:---|
| `0` | `./tiny-agenc` |
| `1` | `train` |
| `2` | `--data` |
| `3` | `tiny.txt` |
| `4` | `--steps` |
| `5` | `12` |

The count is six. C conventionally names it `argc`, short for
*argument count*. The array is `argv`, short for *argument vector*.
Here *vector* means an ordered C array, not the mathematical vector from
the model. Each element is a `char *`, an address of the first character
in one zero-terminated string. `char **argv` points to the first of
those string addresses.

The top level owns the first decision. Once it sees `train`, the
training parser should see this smaller view:

| Training-parser index | String |
|---:|:---|
| `0` | `train` |
| `1` | `--data` |
| `2` | `tiny.txt` |
| `3` | `--steps` |
| `4` | `12` |

`argv + 1` advances the pointer by one string-address slot, from the
program name to `train`. `argc - 1` changes the matching count from six
to five. This is the same
[pointer-arithmetic rule](03-data.md#draw-every-legal-start) used for
arrays elsewhere: advancing a typed pointer moves by one object of its
pointed-to type.

The complete source signature is `int main(int argc, char **argv)`.
Making that entry point perform setup and inspect every argument would
hide its two jobs in a list of conditions. Its complete body instead
names those jobs:

```c
int main(int argc, char **argv)
{
    configure_runtime();
    return dispatch_command_line(argc, argv);
}
```

`configure_runtime` owns the process-wide output and thread choices
constructed later in this chapter. `dispatch_command_line` owns every
decision based on `argc` and `argv`. The return passes its success or
failure result back to the process.

Help and version are valid only when they are the sole argument after
the program name. One helper records that repeated condition:

```c
static int is_standalone_argument(int argc, char **argv,
                                  const char *expected)
{
    return argc == TOP_LEVEL_COMMAND_ARGUMENTS
        && strcmp(argv[COMMAND_ARGUMENT_INDEX], expected) == 0;
}
```

`TOP_LEVEL_COMMAND_ARGUMENTS` is two: the program name plus one command.
`COMMAND_ARGUMENT_INDEX` is one: the slot holding that command. Naming
the two roles keeps the indexing policy out of the helper. Its
`expected` parameter lets help and version share the rule without
sharing their output. `strcmp(a, b) == 0` means the two strings have
exactly the same bytes, including case.

The top-level dispatcher can now read in source order as the public
command-line grammar:

```c
static int dispatch_command_line(int argc, char **argv)
{
    if (is_standalone_argument(argc, argv, "--help")) {
        print_usage(stdout);
        return EXIT_SUCCESS;
    }
    if (is_standalone_argument(argc, argv, "--version")) {
        printf("tiny-agenc %s\n", TINY_AGENC_VERSION);
        return EXIT_SUCCESS;
    }
    if (argc < TOP_LEVEL_COMMAND_ARGUMENTS)
        usage_error(NULL);
    return dispatch_subcommand(argc, argv);
}
```

Each completed information request returns immediately. Therefore the
remaining path is neither a valid help nor a valid version request. No
argument reaches `usage_error`; every other request has a word that can
be dispatched. The exact-count helper means `tiny-agenc --help extra`
cannot discard the extra word and report success.

The remaining helper gives the shifted view an explicit name before
choosing its parser:

```c
static int dispatch_subcommand(int argc, char **argv)
{
    const char *subcommand      = argv[COMMAND_ARGUMENT_INDEX];
    int         subcommand_argc = argc - COMMAND_ARGUMENT_INDEX;
    char      **subcommand_argv = argv + COMMAND_ARGUMENT_INDEX;

    if (strcmp(subcommand, "train") == 0)
        return parse_train(subcommand_argc, subcommand_argv);
    if (strcmp(subcommand, "sample") == 0)
        return parse_sample(subcommand_argc, subcommand_argv);
    usage_error("unknown command '%s'", subcommand);
}
```

No argument means `argc < 2`; `usage_error(NULL)` prints the full usage
without inventing a more specific reason. A recognized command receives
the shifted count and pointer built above. Every other word reaches the
final error. `Train` is not `train`.
`usage_error` never returns, so C does not need a return statement after
that last call.

Chapter 2 used
[`EXIT_FAILURE`](02-foundations.md#one-place-decides-how-to-stop) when
`die` could not return. The dispatcher adds the matching `EXIT_SUCCESS`
when an information request completes. The process's **exit status**
gives the calling shell or script one small result: success means the
requested command completed; failure means it did not.

## Let each parser fill one record

The word `train` selects a grammar, but the remaining strings still
arrive in arbitrary order:

```text
--steps 20 --data tiny.txt
--data tiny.txt --steps 20
```

Both should describe the same request. Calling the model as each flag is
encountered would make order matter and could start work before a later
flag fails. Instead, begin with one record of defaults, replace fields
as options arrive, then validate the complete record.

The training record starts as:

| Field | Initial value |
|:---|---:|
| data path | required, not yet present |
| validation path | absent |
| output path | `trained.bin` |
| steps | `5000` |
| layers `L` | `4` |
| heads `H` | `4` |
| width `C` | `128` |
| block length `T` | `128` |
| batch size `B` | `32` |
| learning rate | `0.001` |
| seed | `1337` |

Generation starts with an absent required model path, an empty prompt,
length `400`, temperature `0.8`, and seed `1337`.

Each accepted flag has one field and one local limit:

| Command flag | Accepted value |
|:---|:---|
| `train --data`, `--val-data`, `--out` | path string |
| `train --steps` | integer `1` through `1,073,741,824` |
| `train --layers` | integer `1` through `1,024` |
| `train --heads` | integer `1` through `256` |
| `train --width` | integer `1` through `16,384` |
| `train --block` | integer `1` through `65,536` |
| `train --batch` | integer `1` through `1,048,576` |
| `train --lr` | finite positive number at most `1,000,000` |
| `train --seed` | unsigned decimal whole number |
| `sample --model`, `--prompt` | string |
| `sample --length` | integer `1` through `16,777,216` |
| `sample --temperature` | finite positive number at most `1,000,000` |
| `sample --seed` | unsigned decimal whole number |

A local range does not prove the fields work together. Batch one million
and block two each pass their individual parsers, but their product of
two million tokens cannot fit one model pass. Width seven and two heads
are both inside their local ranges, but seven channels cannot be divided
evenly between two heads. Those combined checks wait until the record is
full.

The option table tells the GNU `getopt_long` parser which names require
following values. This is the complete training table:

```c
static const struct option TRAIN_FLAGS[] = {
    { "data",   required_argument, NULL, 'd' },
    { "val-data", required_argument, NULL, 'v' },
    { "out",    required_argument, NULL, 'o' },
    { "steps",  required_argument, NULL, 's' },
    { "layers", required_argument, NULL, 'l' },
    { "heads",  required_argument, NULL, 'h' },
    { "width",  required_argument, NULL, 'w' },
    { "block",  required_argument, NULL, 'k' },
    { "batch",  required_argument, NULL, 'b' },
    { "lr",     required_argument, NULL, 'r' },
    { "seed",   required_argument, NULL, 'x' },
    { NULL, 0, NULL, 0 },
};
```

Each row initializes four fields in order. The first is the public name
without its leading `--`. The second says whether a following value is
required. The third can point at an integer for the library to fill;
Tiny AgenC supplies `NULL`, so no such destination exists. With that
field null, `getopt_long` returns the fourth field. The private letters
therefore become compact results for the later selection. The all-zero
row marks the end of the table. These letters are parser labels; Tiny
AgenC does not advertise `-s` or any other short option.

Begin with the loop in [`parse_train`](../src/main.c):

```c
int letter;

opterr = 0;
while ((letter = getopt_long(argc, argv, "", TRAIN_FLAGS, NULL)) != -1)
    parse_train_option(&options, letter, argc, argv);
```

The loop only discovers options and sends each result to a small
dispatcher. Its first three cases handle strings:

```c
switch (letter) {
case 'd':
    options->data_path = optarg;
    return;
case 'v':
    options->validation_path = optarg;
    return;
case 'o':
    options->out_path = optarg;
    return;
```

These are shortened excerpts: the first omits the function wrapper and
the second stops before the numeric cases.

`opterr = 0` stops the library from printing a diagnostic in its own
voice. Each successful parse puts the option value in `optarg`. The
empty string after `argv` says that no short-option spellings are
accepted. `TRAIN_FLAGS` supplies the long-option table. The final
`NULL` says that the caller does not need the table index of the matched
long option.

`switch (letter)` selects one labeled branch. A line such as
`case 'd':` runs when `letter` contains that character. Each assignment
is followed by `return`, which leaves the dispatcher before execution
can fall into the following case. C calls this multiway selection a
**switch statement**. These first three branches assign path strings
directly.

**Predict:** after the loop sees `--out first.bin --out second.bin`,
which path remains in `options.out_path`?

The later switch assignment wins:

```text
--out first.bin --out second.bin
                    ^^^^^^^^^^ stored value
```

This is a consequence of the loop, not a separate merge rule.
GNU `getopt_long` also accepts an unambiguous abbreviation such as
`--dat` for `--data`. Tiny AgenC documents and tests the complete names,
so scripts should use those. The accepted abbreviation is a GNU parser
edge, not a promised portable spelling.

The remaining branches need actual numbers, not borrowed prefixes.
Build those converters before returning to the loop.

## Numbers must consume the whole string

The C library can find a numeric prefix. That is not yet a valid command
value.

Trace `12oops` through `strtol`:

```text
input bytes       1  2  o  o  p  s  \0
                  \--/
converted value    12
end pointer            ^
first unconsumed byte  'o'
```

The numeric result alone looks useful. The end pointer reveals that four
characters remain. The address returned through `end` is the
**conversion end pointer**.

**Predict:** where should the end pointer land for the valid input
`"12"`?

It lands on the terminating zero byte. That makes `*end == '\0'`.
An empty string also leaves the end pointer at its first byte, so it
needs its own rejection check. Finally, the converted number must fit
the range belonging to this flag.

Here is the complete [`parse_int`](../src/main.c):

```c
static int parse_int(const char *text, int min, int max, const char *what)
{
    char *end;
    long  value = strtol(text, &end, DECIMAL_RADIX);

    if (*text == '\0' || *end != '\0' || value < min || value > max)
        die("%s wants an integer in [%d, %d], not '%s'", what, min, max, text);
    return (int)value;
}
```

`strtol` means *string to long*. `DECIMAL_RADIX` is ten, selecting
decimal notation without leaving an unexplained literal at the call.
The function writes the address of the first unconsumed
character into `end`. The `if` joins the three refusals: empty text,
leftover text, or a value outside the caller's range. Only then does the
cast narrow the `long` to `int`.

The configured maxima all fit in `int`, and therefore in the supported
platform's `long`. If `strtol` overflows, it returns a `long` endpoint
that is outside these much narrower accepted ranges. This parser does
not need a separate `errno` check to reject that result.

`strtol` permits leading whitespace and a leading plus sign. The whole
string rule rejects trailing whitespace:

| Text | Result for `--steps` |
|:---|:---|
| `"12"` | accept `12` |
| `"+12"` | accept `12` |
| `" 12"` | accept `12` |
| `"12 "` | reject leftover space |
| `"12oops"` | reject leftover `o` |
| `"0"` | reject below minimum |

The shell normally removes unquoted separating whitespace, so leading
or trailing spaces reach this function only when the person quotes
them.

This construction is **whole-string numeric parsing**: conversion
succeeds only when every supplied character belongs to the accepted
number and the resulting value satisfies the flag's policy.

## Floats and seeds need different grammars

Reusing the integer grammar for every number would hide two different
problems. Learning rate and temperature may contain decimal fractions.
A seed must never turn a minus sign into a large unsigned value.

For `--lr 0.001`, `strtof` converts a floating-point prefix and supplies
the same kind of end pointer:

```text
text             "0.001"
converted value   0.001f
end               points to '\0'
policy            finite, greater than 0, at most 1,000,000
```

Chapter 5 constructed
[nonfinite floating-point values](05-forward-pass.md#turn-arbitrary-scores-into-usable-shares).
Chapter 8 then opened
[the source's exponent-bit test](08-adamw.md#check-the-recipe-before-changing-history).
The tempting CLI reuse is the usual `isfinite(value)`. Tiny AgenC is
built with `-ffast-math`, which permits optimizer assumptions that make
ordinary nonfinite tests unreliable. In the four-byte IEEE 754 `float`
representation used by the supported target, infinity and NaN share one
signature: every exponent bit is one.

The source copies the bits into a 32-bit unsigned integer and tests that
signature. Here are the complete helpers and parser:

```c
static const float MAX_FLAG_VALUE = 1e6f;
static const uint32_t FLOAT_EXPONENT_BITS = 0x7F800000u;

static int is_finite_number(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof bits);
    return (bits & FLOAT_EXPONENT_BITS) != FLOAT_EXPONENT_BITS;
}

static float parse_positive(const char *text, const char *what)
{
    char *end;
    float value = strtof(text, &end);

    if (*text == '\0' || *end != '\0' || !is_finite_number(value)
        || value <= 0.0f || value > MAX_FLAG_VALUE)
        die("%s wants a positive number at most %g, not '%s'",
            what, (double)MAX_FLAG_VALUE, text);
    return value;
}
```

Chapter 3's
[byte copy](03-data.md#draw-every-legal-start) established that
`memcpy(destination, source, byte_count)` moves raw bytes. Here it moves
the float's four stored bytes into `bits` without asking C to read the
float through an incompatible pointer type. `bits &
FLOAT_EXPONENT_BITS` keeps only the exponent positions. If they all
match the mask, the value is rejected. The comparisons then reject zero,
negative finite values, and values above the human-scale ceiling.

This code deliberately assumes the supported four-byte IEEE
representation; it does not contain a compile-time check for another
`float` layout. With fast-math, an extremely tiny positive value may
compare equal to zero and is rejected. Such a rate would behave as zero
for the run anyway. `strtof` also accepts scientific notation such as
`1e-3` and C hexadecimal floating notation such as `0x1p-10`, where
`p` supplies a power of two. Like `strtol`, it accepts leading
whitespace and a leading plus sign, while the end-pointer check rejects
trailing whitespace. Full consumption, finiteness, and range are the
actual public checks.

The seed has a different worked failure. The unsigned library converter
can accept a sign:

```text
text              "-1"
naive strtoull     a very large unsigned value
wanted CLI result  reject
```

Require the first byte to be a decimal digit, require the end pointer to
reach `'\0'`, and check the library's overflow signal:

```c
static unsigned long long parse_seed(const char *text)
{
    char *end;

    /* Digits only: strtoull would happily negate '-5' into a huge
     * value and silently clamp overflow, both lies about the seed. */
    errno = 0;

    unsigned long long value = strtoull(text, &end, DECIMAL_RADIX);

    if (!isdigit((unsigned char)text[0]) || *end != '\0' || errno == ERANGE)
        die("--seed wants a whole number, not '%s'", text);
    return value;
}
```

Some C library calls record an error in an indicator named `errno`.
The caller sets it to zero first; `strtoull` sets it to `ERANGE` when
the decimal value does not fit. `isdigit` receives an `unsigned char`
conversion because passing a negative plain `char` value to that
character-classification function would be invalid.

On the supported 64-bit target, compare these adjacent decimal strings:

```text
18446744073709551615
18446744073709551616
```

**Predict:** does the second string pass merely because every character
is a digit and the end pointer reaches `'\0'`?

No. The first string is the largest `unsigned long long`. `strtoull`
cannot represent the second value, sets `errno` to `ERANGE`, and the
final condition rejects it.

That reset, library call, and comparison construct an **`errno` range
signal**. Unlike a return value, the signal is meaningful only after the
caller clears the old value and the documented library call sets a new
one.

The resulting seed grammar accepts `0` and decimal digits through the
largest `unsigned long long`. It rejects a plus sign, minus sign,
leading whitespace, trailing text, and overflow. The three converters
are separate because the three command values mean different things.

Now every branch in the complete option dispatcher has a constructed
meaning:

```c
static void parse_train_option(TrainOptions *options, int letter,
                               int argc, char **argv)
{
    switch (letter) {
    case 'd':
        options->data_path = optarg;
        return;
    case 'v':
        options->validation_path = optarg;
        return;
    case 'o':
        options->out_path = optarg;
        return;
    case 's':
        options->steps = parse_int(optarg, MINIMUM_OPTION_VALUE,
                                   MAX_STEP_COUNT, "--steps");
        return;
    case 'l':
        options->cfg.layer_count =
            parse_int(optarg, MINIMUM_OPTION_VALUE,
                      MODEL_MAX_LAYER_COUNT, "--layers");
        return;
    case 'h':
        options->cfg.head_count =
            parse_int(optarg, MINIMUM_OPTION_VALUE,
                      MODEL_MAX_HEAD_COUNT, "--heads");
        return;
    case 'w':
        options->cfg.d_model =
            parse_int(optarg, MINIMUM_OPTION_VALUE,
                      MODEL_MAX_D_MODEL, "--width");
        return;
    case 'k':
        options->cfg.block_size =
            parse_int(optarg, MINIMUM_OPTION_VALUE,
                      MODEL_MAX_BLOCK_SIZE, "--block");
        return;
    case 'b':
        options->cfg.batch_size =
            parse_int(optarg, MINIMUM_OPTION_VALUE,
                      MODEL_MAX_TOKENS_PER_PASS, "--batch");
        return;
    case 'r':
        options->learning_rate = parse_positive(optarg, "--lr");
        return;
    case 'x':
        options->seed = parse_seed(optarg);
        return;
    default:
        invalid_option("train", argc, argv);
    }
}
```

This is the complete [`parse_train_option`](../src/main.c). The numeric
cases select the grammar belonging to each flag. Every recognized case
returns immediately, so the control flow does not build a rightward
nest. The `default` branch catches every unlisted parser result.
`invalid_option` never returns.

After the loop, the coordinator validates the filled record and hands
it to training:

```c
validate_train_options(&options, argc, argv);
return run_train(options);
```

`getopt_long` returns `-1` when no options remain. `optind` then names
the first unconsumed word. `validate_train_options` rejects that extra
word rather than guessing what it means, and checks the required data
path. The checks happen before `run_train`, so setup never sees a null
training path.

After constructing the record this way, the mechanism has earned its
name: **command-line option parsing** turns option strings into one
typed command record before execution begins.

## Diagnostics belong to the program

Allowing `getopt_long` to print one error and Tiny AgenC to print another
would produce competing voices:

```text
program: unrecognized option '--oops'
tiny-agenc: train: invalid option '--oops'
tiny-agenc: usage:
...
```

Setting `opterr = 0` suppresses the library line. The parser then routes
an unknown option or missing option value through `invalid_option`, and
that helper routes it through `usage_error`.

Here is the complete error-usage path:

```c
_Noreturn static void usage_error(const char *format, ...)
{
    if (format != NULL) {
        va_list arguments;

        va_start(arguments, format);
        fputs("tiny-agenc: ", stderr);
        vfprintf(stderr, format, arguments);
        fputc('\n', stderr);
        va_end(arguments);
    }
    print_usage(stderr);
    exit(EXIT_FAILURE);
}

_Noreturn static void invalid_option(const char *command, int argc, char **argv)
{
    const char *option =
        optind > 0 && optind <= argc ? argv[optind - 1] : "(unknown)";

    usage_error("%s: invalid option '%s'", command, option);
}
```

Chapter 2 built the shared
[`die`](02-foundations.md#one-place-decides-how-to-stop) function for
fatal runtime failures. `usage_error` follows the same variadic C
pattern, but adds the full usage because the request's shape was wrong.
Its `format == NULL` case prints usage alone. That usage still begins
with `tiny-agenc: usage:`.

`optind` normally points past the offending option when
`invalid_option` runs, so `argv[optind - 1]` recovers its spelling. The
range guards prevent an invalid array access if the library reports an
unexpected index.

The exact contract is narrower than "every possible failure is
prefixed." Diagnostics deliberately emitted by `usage_error` and
`die` begin with `tiny-agenc:`. An assertion failure, a C library
message, an OpenMP runtime message, or termination by a signal need not.
The command boundary handles expected bad requests; it cannot rewrite
every message produced below or outside it.

Early parse and setup failures happen before training narration begins,
so they write diagnostics to standard error and leave standard output
empty. A later failure can occur after step lines have already reached
standard output. For example, a final checkpoint write can fail after
training has printed its architecture and loss. Failure status remains
the reliable machine-readable result.

## A path string is not a file identity

The default output makes omission safe. A supplied output can still be
dangerous:

```sh
./tiny-agenc train --data corpus.txt --out corpus.txt
```

A string comparison catches that exact spelling. It misses two names
for one existing filesystem object:

```text
corpus.txt        ----+
                       +--> device 8, inode 42017
corpus-link.txt   ----+
```

A hard link has a different path string but the same device and inode
numbers. A symbolic link that resolves to the same target has the same
identity at this lookup. The current atomic checkpoint writer would
rename over the alias name, repurposing that directory entry rather than
rewriting the original object through it. Even so, an output name that
currently denotes an input object is not a distinct destination. The
CLI adopts the conservative rule that all three roles must name
different filesystem objects before training begins.

The hard-link name and resolved symbolic-link name are **filesystem link
aliases**: distinct path text that identifies the same existing object
for this check.

The first attempt therefore remains useful as a fast check, then `stat`
asks the filesystem what each existing name resolves to. Chapter 13
used the same
[`stat` metadata query](13-durable-checkpoints.md#write-beside-the-file-that-must-survive)
to preserve permission bits:

```c
static int paths_name_same_file(const char *first, const char *second)
{
    if (strcmp(first, second) == 0)
        return 1;

    struct stat first_status;
    struct stat second_status;

    if (stat(first, &first_status) != 0 || stat(second, &second_status) != 0)
        return 0;
    return first_status.st_dev == second_status.st_dev
        && first_status.st_ino == second_status.st_ino;
}
```

`struct stat` is a record filled by the operating system. `st_dev`
identifies the containing filesystem device and `st_ino` identifies an
object within it. Equal pairs mean that these two lookups named the same
existing object. The code has now constructed **filesystem object
identity**, which is stronger than equal path text.

Training applies the check to all three dangerous pairs:

```text
--data      versus --out
--val-data  versus --out
--data      versus --val-data
```

The last pair protects the no-peeking boundary built in
[Chapter 3](03-data.md#keep-complete-scenes-out-of-training): one file
cannot serve as both the examples that change parameters and the
held-out examples used to inspect them.

There are limits to this preflight. If either `stat` lookup fails, the
helper reports no known identity match. That permits a new output path,
which does not exist yet. Another process could change a path after the
check and before later file operations. The check prevents ordinary
aliases and mistakes; it is not a lock or an adversarial filesystem
security boundary.

## Read data before allocating the model

Passing the option grammar is not enough. This valid-looking request is
still impossible:

```sh
./tiny-agenc train --data four-bytes.txt --block 128
```

A training window of length 128 needs 129 consecutive token ids: 128
inputs and the next-token answer after each position. Four bytes cannot
supply one. If `model_new` allocated a large model first, the command
would discover this corpus failure after spending the model's memory.

The setup therefore narrows the request in stages:

```text
checked TrainOptions
        |
        v
reject known path aliases
        |
        v
bounded training bytes -> training tokenizer -> training Dataset
        |                         |
        |                         +-> newline and T+1 checks
        v
optional bounded validation bytes
        |
        v
same tokenizer -> validation Dataset
                        |
                        +-> known-byte and T+1 checks
        v
fill V -> validate combined geometry -> calculate model memory
        |
        v
model_new -> separate RNG objects -> Chapter 15 training loop
```

Chapter 2 built
[`file_slurp_bounded`](02-foundations.md#read-bytes-without-trusting-the-file).
The caller supplies a ceiling, and the reader checks the file length
before allocating or reading its body. `dataset_max_text_bytes()` is the
smallest of three limits:

```text
256 MiB policy ceiling
SIZE_MAX / sizeof(int)
INT_MAX
```

On the supported 64-bit target, the result is exactly `268,435,456`
source bytes. The representability limits ensure that a later
one-`int`-per-source-byte allocation and the model's integer interfaces
remain expressible.

The beginning of [`run_train`](../src/main.c) is divided by task.
`run_train` rejects path aliases first. The next helper owns the
bounded training read:

```c
static char *read_training_text(const TrainOptions *options, size_t *length)
{
    char *text;
    FileSlurpStatus status =
        file_slurp_bounded(options->data_path, dataset_max_text_bytes(),
                           &text, length);

    if (status == FILE_SLURP_TOO_LARGE)
        die("corpus %s exceeds the platform limit of %zu bytes",
            options->data_path, dataset_max_text_bytes());
    if (status != FILE_SLURP_OK)
        die("cannot read corpus %s (try `make corpus` or `make data`)",
            options->data_path);
    return text;
}
```

One caller turns those bytes into the tokenizer and training dataset:

```c
static Dataset *load_training_data(const TrainOptions *options,
                                   Tokenizer **tokenizer)
{
    size_t length;
    char  *text = read_training_text(options, &length);

    *tokenizer = tokenizer_new(text, length);
    (void)newline_id(*tokenizer);

    Dataset *dataset = dataset_new(*tokenizer, text, length);

    free(text);
    if (dataset == NULL)
        die("corpus %s cannot fit in a token buffer on this platform",
            options->data_path);
    if (dataset_token_count(dataset)
        < (size_t)options->cfg.block_size + NEXT_TOKEN_OFFSET)
        die("corpus %s is smaller than one training window",
            options->data_path);
    return dataset;
}
```

The path check precedes file reads. The bounded reader distinguishes a
too-large file from another read failure. The training bytes construct
the one tokenizer that owns the run's alphabet.

`newline_id` tries to encode `"\n"` and fails if the alphabet has no
newline. Even a one-step command performs this check because the full
training command promises progress sampling later in a longer run; that
sampling starts from a newline. `(void)` says the returned id is
deliberately discarded here. The call is a preflight.

`load_training_data` returns the dataset and writes the tokenizer
pointer through its second argument. `dataset_new` allocates room for
as many integer ids as source bytes, encodes the text, and owns that
token buffer. Only then can the raw source byte allocation be freed.
The final comparison constructs the `T + 1` requirement from
[Chapter 3's shifted windows](03-data.md#one-extra-token-supplies-every-answer).

Validation must reuse the training alphabet. Work through this pair:

```text
training text     "ab\nab\n"     6 source bytes, 6 known ids
validation text   "abX\n"        4 source bytes
```

The training tokenizer knows `a`, `b`, and newline. Chapter 3 specified
that encoding
[skips unknown bytes](03-data.md#encode-without-inventing-a-token).
The validation `X` therefore disappears:

```text
validation source count     4
validation encoded count    3
                              ^
                        mismatch exposes X
```

**Predict:** with block length four, does the training text contain a
complete window?

Yes. It has six ids and needs five. The validation text violates two
requirements: its count mismatch exposes the unknown byte, and its
three known ids are also too short. Source order reports the unknown
byte first and exits, so the later window check does not run for this
request.

The corresponding helper is:

```c
static Dataset *load_validation_data(const TrainOptions *options,
                                     const Tokenizer *tokenizer)
{
    if (options->validation_path == NULL)
        return NULL;

    size_t length;
    char  *text    = read_validation_text(options, &length);
    Dataset *dataset = dataset_new(tokenizer, text, length);

    free(text);
    if (dataset == NULL)
        die("validation corpus %s cannot fit in a token buffer on this platform",
            options->validation_path);
    if (dataset_token_count(dataset) != length)
        die("validation corpus %s contains bytes absent from training data",
            options->validation_path);
    if (dataset_token_count(dataset)
        < (size_t)options->cfg.block_size + NEXT_TOKEN_OFFSET)
        die("validation corpus %s is smaller than one training window",
            options->validation_path);
    return dataset;
}
```

`read_validation_text` performs the same bounded read in its own small
helper. Comparing encoded count with source length works because this
project gives every accepted byte exactly one id. Building a second
tokenizer would hide the unknown `X` by adding it to a different
alphabet, and the validation score would no longer measure the same
model vocabulary.

Only after both datasets survive does setup fill the vocabulary size,
check combined dimensions, and ask for a memory report:

```c
static void validate_training_model(TrainOptions *options,
                                    const Tokenizer *tokenizer)
{
    options->cfg.vocab_size = tokenizer_vocab_size(tokenizer);
    if (!model_config_valid(options->cfg))
        die("impossible model configuration: --width must be divisible by --heads "
            "and --batch x --block must stay within %d tokens",
            MODEL_MAX_TOKENS_PER_PASS);

    ModelMemory memory;

    if (!model_memory_requirements(options->cfg, &memory))
        die("model memory requirements overflow this platform");
    if (memory.total_bytes > MODEL_MAX_CHECKPOINT_RESIDENT_BYTES)
        die("model needs %.1f MiB of buffers; the CLI limit is %.0f MiB",
            (double)memory.total_bytes / BYTES_PER_MEBIBYTE,
            (double)MODEL_MAX_CHECKPOINT_RESIDENT_BYTES
                / BYTES_PER_MEBIBYTE);
}
```

One coordinator preserves the required order:

```c
static TrainingResources prepare_training_resources(TrainOptions *options)
{
    TrainingResources resources = { 0 };

    resources.training_data =
        load_training_data(options, &resources.tokenizer);
    resources.validation_data =
        load_validation_data(options, resources.tokenizer);
    validate_training_model(options, resources.tokenizer);
    resources.model      = model_new(options->cfg, options->seed);
    resources.batch_rng  = rng_new(options->seed);
    resources.sample_rng = rng_new(options->seed + SAMPLE_SEED_OFFSET);
    return resources;
}
```

The geometry check includes `C % H == 0` and
`B * T <= 1,048,576`. The memory calculation from
[Chapter 10](10-memory-planning.md#the-public-memory-report-is-an-independent-calculation)
must be representable and at most one GiB before `model_new` allocates
the model.

That one-GiB limit applies to the model report: parameter values,
gradients, AdamW moments, activation values, activation gradients, and
cached token ids. It is not a whole-process resident-memory limit.
Training and validation each have an independent 256 MiB source
ceiling, each token buffer can approach one GiB, and both datasets may
coexist with the model. Small objects, raw bytes that temporarily
overlap token construction, and allocator overhead also sit outside the
model report.

Setup prevents bad external input from reaching `model_new`; it does
not promise that every bad corpus fails before a large data allocation.
For example, validation unknown-byte detection happens after allocating
and encoding its complete token buffer. The source calls the corpus cap
a platform limit even when the controlling value is the 256 MiB policy
ceiling.

This staged narrowing is the command's **setup preflight**: inspect
recoverable conditions that can be checked before allocating the model
or entering the long-running operation. It reduces late failures. It
cannot prove that future allocation or output I/O will succeed.

`TrainingResources` keeps the six owners needed by the command in one
small record: tokenizer, two datasets, model, and two RNG handles. The
coordinator does not perform their work. Its calls make the setup order
visible.

The final three constructors separate randomness by job. Model
initialization and batch selection receive the same numeric seed but
different `Rng` objects. Progress sampling receives `seed + 1`.
Validation batches, created later inside the training loop, receive
`seed + 2`. Chapter 2 showed that
[separate seeded handles](02-foundations.md#enter-the-sequence-through-a-seed)
advance separately, even when equal seeds give them identical starting
state. A progress sample therefore cannot move the batch handle or
consume its next draw. Chapter 15 will use and verify that state
isolation.

## Sampling crosses the boundary in the other direction

Training turns external text into checked model inputs. Sampling begins
with a checkpoint already protected by Chapter 13's
[defensive loader](13-durable-checkpoints.md#validate-before-publishing),
then sends generated text back outside.

The setup center is short:

```c
Tokenizer *tk;
Model     *m = model_load(&tk, options.model_path);

if (m == NULL)
    die("cannot load checkpoint %s (train one first?)", options.model_path);
print_architecture(stderr, m);

size_t prompt_length = strlen(options.prompt);
int *ids = emalloc((SAMPLE_SEED_TOKENS + prompt_length
                    + (size_t)options.length) * sizeof *ids);

/* A newline before the prompt puts the model at start-of-line, the
 * state every line of its corpus began from. */
ids[0] = newline_id(tk);

int known = SAMPLE_SEED_TOKENS
          + (int)tokenizer_encode(tk, ids + SAMPLE_SEED_TOKENS,
                                  options.prompt, prompt_length);
int  total = known + options.length;
Rng *rng   = rng_new(options.seed);

model_sample(m, rng, ids, known, total, options.temperature);
print_text(tk, ids + SAMPLE_SEED_TOKENS,
           total - SAMPLE_SEED_TOKENS);
```

This shortened excerpt from [`run_sample`](../src/main.c) shows setup
and the sampling handoff. The function wrapper, cleanup, and success
return are omitted.

`model_load` exposes one failure result, `NULL`, for missing, damaged,
incompatible, oversized, and old-format checkpoint files. The CLI can
report the path but cannot truthfully diagnose which internal check
failed. Its suggestion to train one first is therefore a hint, not a
classification.

`SAMPLE_SEED_TOKENS` is one, the hidden newline prefix length. The
allocation reserves that seed-newline slot, at most one slot per prompt
byte, and the requested generated length. Unknown prompt bytes
are skipped under
[Chapter 3's tokenizer contract](03-data.md#encode-without-inventing-a-token);
the allocation remains large enough. A separate newline places the
model at the start-of-line state, but `print_text` omits that seed from
the result.

The checked command record supplies a positive length and temperature.
The loaded checkpoint supplies a valid model and matching tokenizer.
Only then does `model_sample` take over. Chapter 16 constructs what
happens inside that call. Chapter 14 owns the values that enter it and
the stream to which its decoded result leaves.

Work through the handoff with a checkpoint whose tokenizer knows
newline, `a`, and `b`. Let the prompt be the three bytes `"aXb"`, where
`X` is unknown, and request two generated ids.

**Predict:** how many integer slots are reserved, how many prompt ids
survive encoding, and how many ids does `model_sample` receive in
total?

The allocation reserves the worst case:

```text
seed newline + prompt bytes + requested tail = 1 + 3 + 2 = 6 slots
```

Encoding keeps `a` and `b` but skips `X`, so two prompt ids survive.
The known prefix and final total are therefore:

```text
known = 1 seed + 2 prompt ids = 3
total = 3 known + 2 generated = 5
```

`model_sample` fills `ids[3]` and `ids[4]`. `print_text` begins at
`ids + 1` and prints `total - 1 = 4` ids: `a`, `b`, and the two new
ids. Neither the hidden seed newline nor the unknown `X` reaches
standard output.

This loading, prompt preparation, and checked handoff form the
**sample-command setup**. It owns the outside values around
`model_sample`, not the choice mechanism inside it.

## Keep narration away from generated text

One output stream is not enough for sampling:

```text
architecture: 815360 parameters ...
RAZR: neon rain ...
```

If both lines go to the same place, saving a continuation also saves a
diagnostic banner inside the generated text. C gives every process two
pre-opened output streams with different jobs. Tiny AgenC uses them as:

| Situation | Stream |
|:---|:---|
| successful top-level help and version | standard output |
| training narration and progress | standard output |
| generated sample text | standard output |
| command diagnostics and usage errors | standard error |
| sampling architecture banner | standard error |

These are conventionally named `stdout` and `stderr`. Assigning each
public output to one of them creates the CLI's **stream routing**
contract.

Shell redirection can now save only the generated text:

**Predict:** with the routing table above, does the architecture banner
enter the file in this command?

```sh
./tiny-agenc sample --model trained.bin > continuation.txt
```

No. `>` is output redirection, not a pipe. It connects standard output
to the file while the architecture banner remains on standard error,
normally the terminal. A pipe uses `|` to connect standard output to
another process:

```sh
./tiny-agenc sample --model trained.bin | wc -c
```

`wc -c` counts the bytes it receives. The banner does not become part
of that count.

Training has a different problem. When standard output is redirected to
a file, a C library may hold many lines in memory before writing them.
A person following the file would see long pauses. The runtime setup
groups the line-buffering request with the thread choice built next:

```c
static void configure_runtime(void)
{
    /* Training narrates as it goes; line-buffering keeps the story
     * streaming even when stdout is a file being tailed. */
    setvbuf(stdout, NULL, _IOLBF, 0);
    choose_thread_count();
}
```

`setvbuf` changes a stream's buffering policy; `_IOLBF` requests that a
completed line be sent onward. `stdout` selects the stream. `NULL` and
the final zero supply no caller-owned buffer storage, leaving that
storage to the C library. The call occurs before any output, including
help and version. Its return value is ignored, so this is a best-effort
**line-buffering request**, not a guarantee.

Most `printf`, `putchar`, and `setvbuf` results are also unchecked. A
full output device can therefore be detected late. If the process at
the other end of a pipe closes it, the operating system may end Tiny
AgenC while it writes. Stream routing says which kind of text is sent
where during ordinary operation. It does not promise successful delivery
after the receiving file or process breaks.

## OpenMP is optional, not invisible

Chapter 5 introduced the
[OpenMP loop directives](05-forward-pass.md#make-every-output-from-one-input-row).
Choosing a runtime thread count still belongs at the command boundary
because the environment may already contain an operator's choice.

For example, Chapter 3's
[command environment assignment](03-data.md#where-the-raw-text-came-from)
can choose four threads for one run:

```sh
OMP_NUM_THREADS=4 ./tiny-agenc train --data corpus.txt
```

The new process receives a table of environment names and values.
`getenv("OMP_NUM_THREADS")` returns a non-null pointer when that name is
present and `NULL` when it is absent. It does not decide whether the
value is a valid OpenMP setting. This presence check is an
**environment-variable lookup**. Without it, the program's fallback
could silently replace the operator's explicit choice.

Using every reported hardware thread was slower on the recorded
32-core, 64-thread development machine. For the default model after
enabling fast-math, the committed sweep measured `43.3 ms/step` with
32 threads and `225.7 ms/step` with 64. These are measurements from one
machine and workload, not universal speed ratios. The full commands and
environment are in
[`dev-measurements.md`](logs/dev-measurements.md#same-sweep-after-adding--ffast-math-to-cflags).

The source turns that observation into a modest default:

```c
static void choose_thread_count(void)
{
#ifdef _OPENMP
    int half =
        omp_get_num_procs() / LOGICAL_THREADS_PER_PHYSICAL_CORE;

    if (getenv("OMP_NUM_THREADS") == NULL && half > 0)
        omp_set_num_threads(half);
#endif
}
```

**Predict:** if OpenMP reports 64 processors and `OMP_NUM_THREADS` is
absent, how many threads does this function request?

This is the complete [`choose_thread_count`](../src/main.c).
`omp_get_num_procs()` asks the OpenMP runtime for its available processor
count; integer division by two gives 32. If the result is positive and
`OMP_NUM_THREADS` is absent, Tiny AgenC requests that many threads.

Presence matters, not validity. Even an empty `OMP_NUM_THREADS`
environment value prevents the program from replacing the operator's
choice; the OpenMP runtime then interprets that value. If only one
processor is reported, `half` is zero and the program leaves the runtime
default alone. `configure_runtime` calls this function before command
dispatch, so the OpenMP runtime may be consulted even for an
information command. The operator-respecting fallback is Tiny AgenC's
**runtime thread default**.

Chapter 4's
[conditional header region](04-poor-mans-tensors.md#the-complete-header)
used `#ifndef` and `#endif`. Here `#ifdef _OPENMP` includes the body only
when the compiler defines `_OPENMP`; otherwise the preprocessor removes
it. The default Make target enables OpenMP because training is the
showcase. A compiler without it can build:

```sh
make OPENMP=0
```

The default optimization is `-O3 -ffast-math` without tuning for the
build host. A measured local build can opt in:

```sh
make NATIVE=1
```

That adds `-march=native` and may produce an executable that assumes
instructions from the build CPU. It is useful for a controlled local
measurement, not as the default artifact to hand to another machine.
`OPENMP` and `NATIVE` are build-time policies. `OMP_NUM_THREADS` and
`configure_runtime` govern runtime policy.

All parts are now in place. The **command-line boundary** narrows
untrusted words and paths into the values accepted by the next
defensive owner, refuses problems it can detect before long-running
work, routes successful output and diagnostics separately, and returns
an exit status. It does not make paths immutable, make arbitrary text
known to the tokenizer, or guarantee that future allocation and I/O
will succeed.

## Stop at an honest learner-track seam

The completed `src/main.c` already contains the Chapter 15 training
loop. The building workspace does not: `labs/start.sh` creates
`labs/work/main.c` with one TODO, and Chapter 14 must not ask the reader
to invent learning before it is taught.

The focused check needs a checkpoint so it can exercise the load and
sample boundary. It does not compare pre-save and post-save parameters
or require loss to fall. The learner can therefore stop after setup
with this explicit provisional behavior:

```text
Chapter 14 learner-only ending

checked tokenizer, datasets, configuration, and memory report
                         |
                         v
                 construct Model
                         |
                         v
       print architecture and dataset counts
       print "tiny-agenc: boundary check; no updates yet"
                         |
                         v
       save the initialized model directly to --out
       print "tiny-agenc: initialized checkpoint saved to PATH"
                         |
                         v
       free Model, Datasets, and Tokenizer; return success
```

This is behavior to implement in the learner workspace, not an excerpt
from the finished source. Use the `model_save` failure policy already
built in Chapter 13. Do not create batch or progress-sample RNGs, and do
not call `model_forward`, `model_backward`, or `model_step` in this
temporary ending. The words *no updates yet* keep the result honest:
the saved checkpoint contains initialized parameters, not learned ones.

Chapter 15 will keep the parsing and setup, replace the two temporary
status lines and direct save with RNG construction and the real training
loop, then preserve the same final cleanup. The Chapter 14 scaffold
exists only so the boundary can be tested at the chapter where it is
owned.

## Know what the boundary witness proves

The Chapter 14 lab compiles the learner's `main.c` and every completed
runtime module. Sampling mathematics belongs to Chapter 16, so this
checkpoint temporarily links the answer-key `src/model_sampling.c`.
That borrowed function lets the lab inspect the command boundary without
pretending the learner has already implemented sampling.

The witness exercises two successful information requests, nineteen
rejection paths, one valid train-shaped request with `--steps 1`, and one
valid sampling command. Among those cases it checks exact version
output, public help paths, dispatch, required flags, extra arguments,
malformed integer, float, and seed values, the source-size ceiling,
impossible geometry, model-memory preflight, direct and hard-link path
collisions, an unknown validation byte, a missing checkpoint, and the
sample stream split.

Against the learner workspace, that valid request reaches the
provisional save above. Against `WORK=../src`, the finished answer key
does execute one real update. The script only requires a loadable
checkpoint and the promised stream text in either case. It proves that
a valid request crossed the boundary and completed; it does not use the
checkpoint contents to claim that learning occurred. Chapter 15 adds
the witnesses for changed parameters and falling loss.

This focused lab does not test every accepted range endpoint, quoted
whitespace, hexadecimal floats, duplicate options, GNU option
abbreviations, thread selection, line-buffer timing, late failures, or
the path-check race. It does not prove learning quality or the sampling
calculation. The repository's broader `make check-cli` later adds more
path aliases, validation independence, deterministic sampling, unknown
prompt bytes, and malformed-checkpoint cases.

## Build checkpoint: make bad input boring

### Build

Implement top-level help and version, command dispatch, flag parsers,
required `sample --model`, whole-string number checks, output
separation, path collision checks, and the training and sampling setup
paths in `main.c`. Disable `getopt`'s own diagnostics so expected
failures pass through the program's error path. End the learner's
training setup with the explicit no-update scaffold above. The finished
answer key instead contains the Chapter 15 loop.

### Verify

Build and run the learner checkpoint:

```sh
make -C labs check-14
```

Probe the most tempting partial integer by hand:

```sh
labs/build/tiny-agenc-before-sampling train \
    --data labs/tiny-corpus.txt --steps 12oops
```

Then inspect the complete focused witness:

```sh
make -C labs WORK=../src check-14
```

### Expected

The focused answer-key run ends with:

```text
check-14: information, dispatch, validation, and stream contracts passed
```

Help and version succeed on standard output. The learner's valid
train-shaped request says that no updates ran and saves the initialized
model as default `trained.bin`; the answer key performs its one update.
One valid sampling command writes only text to standard output while its
banner goes to standard error. The exercised early failures leave
standard output empty, write a `tiny-agenc:`-prefixed diagnostic to
standard error, and return failure.

### Common failures

- **`--steps 12oops` is accepted:** the conversion end pointer was
  ignored.
- **`sample` silently searches for a checkpoint:** `--model` was not
  made explicit and required.
- **Training can replace its corpus:** the exact output and input names
  were not compared before checkpoint replacement.
- **An unknown option begins with another program's wording:**
  `getopt_long` printed its own diagnostic before Tiny AgenC could apply
  the CLI contract.
- **Validation accepts `X` absent from training:** it built a second
  tokenizer or failed to compare encoded count with source length.
- **Piped or redirected samples begin with an architecture line:** the
  banner went to standard output.
- **A typo aborts in an assertion:** recoverable external input reached
  the model before the command preflight.
- **The lab cannot link `model_sample`:** the Chapter 14 target did not
  borrow the answer-key sampling file at this temporary track seam.

The boundary now refuses nonsense cleanly. Chapter 15 gives the valid
training path something worth doing.

---

[Previous: Durable Checkpoints](13-durable-checkpoints.md) | [Contents](README.md) | [Next: The Training Loop](15-the-training-loop.md)
