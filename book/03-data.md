# Chapter 3: Data

In Chapter 0, the model began with broken-looking text. After training,
it produced speaker names, header fragments, and pieces of dialogue.
The training loop did not acquire those shapes from C code. It saw them
repeated in the text.

Consider two possible training files. The first contains unrelated
paragraphs about hundreds of subjects. The second repeats a four-line
scene shape with the same small cast:

```text
header
speaker: dialogue
speaker: dialogue
speaker: dialogue
```

**Predict:** which file gives a small model a pattern it can encounter
again within 128 characters?

The second one does. Repetition does not guarantee good writing, but it
gives the model several chances to encounter the same local structure.
We chose the text around that constraint instead of treating any large
file as interchangeable training material.

That deliberately selected body of training text is a **corpus**. A
character-level model with 815K parameters cannot learn the internet.
It can learn a *voice* if the corpus has enough structure to grab onto.
This chapter builds that structure, turns its bytes into integers, and
then turns those integers into questions with known next-byte answers.

## NIGHT GRID: a corpus built to be watched

We wanted the training demo to have an arc: word-like text, recurring
slang, speaker tags, and whole scenes should become increasingly
recognizable as training progresses. That dictates the corpus design.
It needs rigid local patterns, a small cast of repeated names, a
recurring lexicon, and one unmistakable large-scale format. So:
cyberpunk transmission logs.

```text
=== TRANSMISSION 1017 // SECTOR 3: BLACKOUT DISTRICT ===
RAZR: Found a trace, but it's gone cold like this district in blackout.
GHOST: The Grid's like a maze. No easy exits when you're marked.
DOC: We need to stay one step ahead of the chrome.
```

That block is `data/cyberpunk.txt`, lines 1153 through 1156,
verbatim.

The fixed parts do different jobs. Every scene starts with
`=== TRANSMISSION`, so those characters appear in the same order many
times. The four-digit number and sector vary inside a fixed frame.
Eight handles recur:

```text
RAZR  GHOST  VYPR  NYX  JINX  DOC  WIRES  MOTH
```

The dialogue reuses a small lexicon such as `choom`, `ice`, `mesh`,
`jack in`, `stay frosty`, `chrome`, `corpo`, `netrun`, `blackout`,
`cred-chips`, and `the Grid`. A 128-byte input window cannot hold a
long story. It can hold a header, a speaker tag, and nearby dialogue.
NIGHT GRID puts its learnable repetitions at that scale.

Look again at the sample and predict what should be easiest to copy:
the exact sentence about the district, the `GHOST:` label, or the
general idea of a city-wide conspiracy. `GHOST:` has the strongest
local evidence. It is short and occurs in the same role across many
scenes. The conspiracy is spread across words and scenes, so this tiny
model receives much weaker evidence for it.

This is the first data-design rule: decide what visible behavior you
want to watch, then put repeated evidence for that behavior inside the
model's input span.

## Where the raw text came from

Writing more than a megabyte of dialogue by hand would hide the parts
this book is trying to show. [`gen-corpus.sh`](../scripts/gen-corpus.sh)
asked a locally running Ollama model, `dolphin3`, for three scenes per
request. It rotated through sixteen scenario prompts, including a data
heist, a chase, and a mole inside the crew. Those strings are prompts,
not random seeds. The generation was not exactly reproducible.

Each request included a newly sampled example header. The prompt said
that the example showed syntax and must not be copied. That varying
example was intended to discourage one demonstration header from
dominating later responses. It is not a guarantee about what the model
returned.

The generator appends usable responses until the raw file reaches its
target. An interrupted run can continue from the bytes already
written. A response must contain both a valid header and a valid
dialogue line. Three unusable responses in a row stop the script
instead of polling a broken server forever.

There are two different byte counts to keep separate. The historical
generation log ended after 780 calls and reported 1,200,020 bytes. That
exact byte stream was not retained. The committed
`data/cyberpunk.raw.txt` is the canonical input and is exactly
1,200,000 bytes. Every reproducible data claim begins at that file.

The default target is also 1,200,000 bytes. The canonical raw file has
already reached it, so this command currently makes zero Ollama calls:

```sh
make generate-corpus
```

Extending the raw artifact is a deliberate data change. For example,
this asks Ollama to append until the file reaches 1,300,000 bytes:

```sh
TARGET_BYTES=1300000 make generate-corpus
```

In the shell, putting `NAME=value` before one command gives that
environment variable to that command only. It does not change later
commands. Here `make` receives the new target and passes it to the
generator. This prefix is a **command environment assignment**.

That command needs the configured Ollama model and mutates the tracked
raw input. It does not recreate the historical dialogue. The ordinary
book path is different:

```sh
make corpus
```

`make corpus` reads the canonical raw file, contacts no network
service, and derives the clean corpus deterministically.

## Cleaning as a grammar, not a mop

Raw model output contains useful scenes mixed with chatter, markdown,
misspelled headers, trailing spaces, and occasional Unicode
punctuation. These source lines come from two places in the canonical
raw file and have been assembled into one small grammar test.
`[space]` replaces one trailing space so the otherwise invisible byte
can be seen:

```text
Here are the scenes formatted correctly with the given requirements and slang terms incorporated.
=== TRANSMISION 0498 // SECTOR 11: ICE BAY WAREHOUSE ===
DOC: Any signs of the mole?[space]
=== TRANSMISSION 0521 // SECTOR 3: CHROME SPARRING GROUND ===
JINX: Someone's been dipping into our creds.
NYX: Could be a mole. Stay sharp out there.
```

The first line is commentary, not NIGHT GRID. The first header
misspells `TRANSMISSION`. `[space]` marks the trailing space after the
`DOC` line's punctuation. That line also belongs under the malformed
header.

**Predict:** which header and dialogue lines should enter the training
text?

Only the `0521` header and its two valid dialogue lines survive. A
patch-based cleaner might remove the commentary, repair the misspelling,
and trim the trailing space. Each new failure would need another patch.
That approach can accidentally admit a different malformed shape.

The safer construction starts from an empty output and admits only two
line forms. An accepted header has four digits, a sector from 1 through
12, and an uppercase location. An accepted dialogue line starts with
one of the eight handles, has a non-space first content character, and
ends in sentence punctuation with an optional straight quote. A header
is not emitted until a matching dialogue line proves that the scene is
complete.

The source spells those forms with these exact pattern definitions:

```sh
MAX_REPEATS_PER_LINE=3   # guard against excessive exact-line repetition

HEADER_PATTERN="^=== TRANSMISSION [0-9]{4} // SECTOR ([1-9]|1[0-2]): [A-Z0-9][A-Z0-9 '-]* ===$"
DIALOGUE_PATTERN="^(RAZR|GHOST|VYPR|NYX|JINX|DOC|WIRES|MOTH): [^ ].*[.!?][\"']?$"
```

These three lines differ from `NAME=value command`: no command follows
the value on the same line. Each value remains available to later
lines in this shell script. This is a **shell variable assignment**.
When the later command contains `"$HEADER_PATTERN"`, the shell replaces
that name with the stored pattern; the double quotes keep its spaces
inside one argument.

In this pattern language, `^` anchors the beginning of a line and `$`
anchors the end. `[0-9]{4}` requires four digits. The parenthesized
sector choice accepts one digit from 1 through 9 or `1` followed by a
digit from 0 through 2. The location must begin with an uppercase
letter or digit. Its `*` then permits zero or more uppercase letters,
digits, spaces, apostrophes, or hyphens.

The dialogue pattern begins with a choice among the eight handles,
followed by a literal colon and space. `[^ ]` requires the first
dialogue character not to be a space. `.*` permits the remaining
characters. `[.!?]` requires final sentence punctuation, and
`[\"']?` permits zero or one straight closing quote. The separate
constant caps an accepted exact dialogue line at three copies.
The backslash before `"` belongs to the shell's double-quoted string:
it puts a literal double quote into the variable. AWK receives the
final piece as `["']?`.

Because the cleaner starts from empty output and keeps only lines on
its allowlist, it is a **whitelist**. Everything else disappears. The
accepted forms plus the saved scene state make a tiny
**corpus grammar**.

Here is the grammar pass from
[`clean-corpus.sh`](../scripts/clean-corpus.sh). The excerpt is
shortened to the AWK program:

```sh
apply_grammar() {
    awk -v header="$HEADER_PATTERN" \
        -v dialogue="$DIALOGUE_PATTERN" \
        -v max_repeats="$MAX_REPEATS_PER_LINE" '
        $0 ~ header { pending = $0; scene_open = 0; next }
        /^===/      { pending = ""; scene_open = 0; next }
        $0 ~ dialogue && (pending != "" || scene_open) {
            if (seen[$0] >= max_repeats)
                next
            seen[$0]++
            if (pending != "") {
                print ""
                print pending
                pending = ""
                scene_open = 1
            }
            print
            next
        }
    '
}
```

`apply_grammar() { ... }` defines a shell function. The single quote
after the last `-v` begins the AWK program, and the matching quote near
the end closes it. Those quotes pass `$0` to AWK literally instead of
letting the shell replace it. The final `}` closes the shell function.

AWK's three `-v` options copy the header, dialogue, and repeat values
into AWK variables before input begins. A trailing `\` continues one
shell command on the next source line. For each input line, AWK tries
the displayed pattern-action rules from top to bottom until a `next`
moves to the following line. Before the first line, an unset scalar
such as `pending` or `scene_open` behaves as an empty string in a
string test and as zero in a numeric test.

Inside AWK, `~` asks whether a line matches a pattern. `&&` requires
both surrounding conditions; `||` accepts either one; `!=` tests that
two values differ.

AWK places the current input line in `$0`. The first rule recognizes a
valid header. It stores that line in `pending`, closes any previous
scene, and uses `next` to move to the following input line. Nothing is
printed yet. A header with no valid dialogue will therefore vanish.

The second rule catches every other line beginning with `===`. Such a
line looks like an attempted header but failed the accepted pattern. It
clears `pending` and closes the scene. That reset is why the `DOC` line
in the worked example cannot attach itself to an older valid header.

The third rule accepts dialogue only when a valid header is pending or
a valid scene is already open. `seen[$0]` counts matching dialogue
candidates inside retained scenes across the complete file. When the
count has already reached three, `next` discards another copy. The cap
is global, not one cap per scene. An AWK array entry that has not been
used yet has numeric value zero. `seen[$0]++` uses that current count
and then increases it.

For the first retained dialogue line, `pending != ""` is true. The
program prints a blank separator, prints the saved header, clears it,
and marks the scene open. The final bare `print` emits the current
dialogue line. Later valid dialogue enters through the same rule but
does not print the header again.

Before that grammar runs, Unicode is transliterated and all remaining
non-ASCII bytes are removed:

```sh
to_ascii() {
    iconv -f UTF-8 -t ASCII//TRANSLIT -c | LC_ALL=C tr -cd '\n -~'
}
```

`iconv` turns forms such as curly quotes into their closest ASCII
spelling when possible. `-f UTF-8` names the input encoding,
`-t ASCII//TRANSLIT` requests transliterated ASCII output, and `-c`
omits a byte sequence that cannot be converted. The pipe sends that
output into `tr`. `LC_ALL=C` uses the same one-command environment
assignment, this time for `tr`; it makes byte ranges use the C locale.
`-c` complements the stated set and `-d` deletes that complement. The
remaining set is newline plus printable ASCII from space through `~`.
This keeps typographic variants from becoming extra byte values.

The script writes into a temporary directory beside the destination.
It rejects identical raw and clean paths, refuses to install output
with no complete scene, preserves existing file permissions, and moves
the completed temporary file into place. A failed pass does not leave
a half-written corpus at the destination.

Starting from the canonical raw artifact, `make corpus` produces these
recorded results:

```text
bytes                  1,089,394
complete transmissions     2,011
distinct byte values           80
```

Every one of the eight handles is represented. A retained scene has at
least one retained dialogue line; the cleaner does not require the 20
through 35 lines requested from the generator. The clean file is
ignored by Git because the canonical raw input and deterministic
cleaner reproduce it.

## Give each byte a compact number

The clean file is still text. The model stores its inputs in arrays,
and an array position needs an integer. We could use each byte's ASCII
value directly. For the lab text `"zaba\n"`, those values are:

```text
byte       newline   a    b    z
ASCII           10  97   98  122
```

That would reserve positions 0 through 122 while using only four of
them. The model needs a compact row for each value that can occur.

Assigning numbers in first-seen order looks compact. Try it on two
texts with the same distinct bytes:

```text
text 1: z a b a newline    first ids: z=0 a=1 b=2 newline=3
text 2: a b a z newline    first ids: a=0 b=1 z=2 newline=3
```

The text order changed the meaning of every row except the last one.
Saving a model built with the first mapping and loading the second
mapping would make row 0 mean `a` instead of `z`.

There are 256 possible byte values. Scan those values from 0 through
255 and assign the next compact number whenever a value appeared.

**Predict:** scan the four byte values above from smallest to largest.
Which compact id does each receive, and why does newline receive id 0?

The result for `"zaba\n"` is:

```text
compact id       0   1   2   3
byte         newline   a   b   z
byte value       10  97  98  122
```

Its byte value, 10, is smaller than the byte values for `a`, `b`, and
`z`. Sorting by byte value makes both example texts produce the same
mapping. The mapping remains stable when the set of present bytes is
the same. Adding or removing a smaller byte can still shift later ids.

Now the pieces have names. In Tiny AgenC, one byte occurrence in the
input sequence is a **token**. More generally, a token is whichever
piece of text receives one id. `"zaba\n"` contains five token
occurrences:

```text
z  a  b  a  newline
```

The two `a` occurrences are two input tokens, but they have the same
value. The set of distinct token values is the **vocabulary**, so this
five-token text has a vocabulary of four values. A token occurrence's
compact integer is its **id**. Both `a` occurrences use id 1.
Converting the text sequence into ids is **tokenization**. The object
that owns both the byte-to-id and id-to-byte mappings is a
**tokenizer**.

For NIGHT GRID, one token is one byte. The cleaner makes the text
ASCII, so one displayed character also occupies one byte. That
coincidence does not hold for general UTF-8 text, where one displayed
character can occupy several bytes.

A byte tokenizer spends one input position on every byte.

**Predict:** how many positions does `"the the"` occupy?

```text
byte tokens:  t  h  e  space  t  h  e       count 7
```

Suppose a larger tokenizer's vocabulary also contains the common piece
`th`.

**Predict:** if each `t, h` pair becomes one `th`, how many positions
remain?

```text
larger tokens:  th  e  space  th  e          count 5
```

A token that covers a recurring part of a word is a **subword token**.
The shorter sequence leaves more of a fixed input span for later text,
but the vocabulary needs another row and the tokenizer needs rules for
choosing pieces. Tiny AgenC keeps one byte per token so every choice
remains visible.

## tokenizer: the alphabet, sorted

The compact mapping must answer two questions quickly:

```text
encoding: byte -> id
decoding: id   -> byte
```

Scanning the vocabulary for every input byte would repeat work. Two
tables make either answer one array lookup. Here is the private
representation and construction from
[`tokenizer.c`](../src/tokenizer.c). The excerpt is shortened to the
constants, tables, and constructors; the private routine map between
the record and the first helper is omitted:

```c
enum {
    BYTE_VALUES  = 256,
    TOKEN_ABSENT = -1,
};

struct Tokenizer {
    int  vocab_size;
    char id_to_byte[BYTE_VALUES];
    int  byte_to_id[BYTE_VALUES];
};

/* Walking byte values in ascending order sorts the vocabulary. */
static Tokenizer *from_seen_bytes(const int seen[BYTE_VALUES])
{
    Tokenizer *tk = emalloc(sizeof *tk);

    tk->vocab_size = 0;
    for (int byte = 0; byte < BYTE_VALUES; byte++) {
        tk->byte_to_id[byte] = TOKEN_ABSENT;
        if (!seen[byte])
            continue;
        tk->byte_to_id[byte] = tk->vocab_size;
        tk->id_to_byte[tk->vocab_size] = (char)byte;
        tk->vocab_size++;
    }
    return tk;
}

Tokenizer *tokenizer_new(const char *text, size_t length)
{
    int seen[BYTE_VALUES] = { 0 };

    for (size_t i = 0; i < length; i++)
        seen[(unsigned char)text[i]] = 1;
    return from_seen_bytes(seen);
}
```

`BYTE_VALUES` sizes both fixed arrays for every possible byte.
`TOKEN_ABSENT` is `-1`, which cannot be confused with a valid id
starting at 0. `vocab_size` records how much of `id_to_byte` contains
meaningful entries. Every position of `byte_to_id` will be initialized.

The spelling `const int seen[BYTE_VALUES]` looks as though a
256-element array travels into the helper. In a C function parameter,
brackets are adjusted to a pointer: the helper actually receives
`const int *seen`. The bracket count neither allocates storage nor
checks the caller's array length. The caller here supplies the real
256-element `seen` array, and `const` prevents the helper from changing
its entries. This bracket spelling is an **array parameter**.
At the call `from_seen_bytes(seen)`, the array expression `seen`
becomes a pointer to its first element, `&seen[0]`. That separate
caller-side step is an **array-to-pointer conversion**.

The helper is `static`, the Chapter 2 rule that keeps a function private
to this C file. `emalloc(sizeof *tk)` allocates enough bytes for the
structure and stops the program if allocation fails. The expression
uses the pointed-to object's size, so it stays correct if the type of
`tk` changes.

Inside `tokenizer_new`, `{ 0 }` supplies zero for the first array
element. C fills every remaining element with zero too, so the whole
`seen` array starts clear. The loop then marks each byte that occurs in
the input.

The cast in `(unsigned char)text[i]` matters. A plain C `char` may be
negative for a byte above 127. Negative array indexes would access
memory before `seen`. Converting to `unsigned char` produces an index
from 0 through 255.

`from_seen_bytes` starts with no vocabulary entries. For each possible
byte it first writes `TOKEN_ABSENT`. If the byte was not seen,
`continue` skips the rest of this loop iteration. A seen byte receives
the current compact id in `byte_to_id`; the reverse table receives that
byte at the same id. Casting back to `char` stores the original byte
pattern for NIGHT GRID because its ASCII values all fit in plain
`char`. For a byte above plain `char`'s positive range, C leaves the
conversion result implementation-defined. The supported GNU/Linux
compiler preserves the eight-bit pattern, but that is a platform
boundary rather than a portable C promise. Incrementing `vocab_size`
prepares the next compact id.

The test `!seen[byte]` uses C's `!` operator. For an unseen byte,
`seen[byte]` is 0 and `!0` is 1, so the `if` body runs. For a seen byte,
`seen[byte]` is 1 and `!1` is 0, so the body does not run. Turning zero
into true and a nonzero value into false this way is **logical NOT**.

For `"zaba\n"`, the loop reaches byte 10 first and fills both halves:

```text
byte_to_id[10] = 0       id_to_byte[0] = newline
byte_to_id[97] = 1       id_to_byte[1] = a
byte_to_id[98] = 2       id_to_byte[2] = b
byte_to_id[122] = 3      id_to_byte[3] = z
```

The tables are inverse mappings only on vocabulary bytes and valid ids.
For example, `byte_to_id['?']` is `-1` here. There is no valid id to
send back through `id_to_byte`.

`tokenizer_vocab_size` returns the final count. The caller owns the
returned tokenizer and eventually passes it to `tokenizer_free`.

## Encode without inventing a token

Take the tokenizer built from `"zaba\n"` and encode `"az?b\n"`.

**Predict:** which ids are written, and what happens to `?`?

The known bytes map as follows:

```text
input byte      a   z   ?   b   newline
mapped id       1   3  none 2      0
written slot    0   1   -   2      3
```

The result is `[1, 3, 2, 0]`. The unknown `?` is skipped, so later ids
move left and the returned count is four. Decoding those four ids
produces `"azb\n"`, not the original five-byte string.

One caller-side boundary is not visible in an `int *`. Suppose `ids`
points at space for only two integers. The writes for `a` and `z` fit
in slots 0 and 1. The write for `b` tries slot 2, outside that array.
C does not stop the write. From that point, the C language promises no
result. The write may damage another object, or it may appear to work
and fail later. Code that leaves the language's guarantees this way
has **undefined behavior**.

An address does not carry the number of writable elements behind it.
That number is the **buffer capacity**. The caller must give
`tokenizer_encode` capacity for at least `length` integers, the
worst-case result when every input byte is known. Current callers
reserve that many slots. The function returns the smaller count after
unknown bytes have been skipped; it cannot discover the capacity from
`ids`.

Here are the complete encoding and decoding functions:

```c
size_t tokenizer_encode(const Tokenizer *tk, int *ids, const char *text, size_t length)
{
    size_t written = 0;

    for (size_t i = 0; i < length; i++) {
        int id = tk->byte_to_id[(unsigned char)text[i]];

        if (id == TOKEN_ABSENT)
            continue;
        ids[written++] = id;
    }
    return written;
}

char tokenizer_decode(const Tokenizer *tk, int id)
{
    assert(id >= 0 && id < tk->vocab_size);
    return tk->id_to_byte[id];
}
```

`const Tokenizer *` means this function may read the tokenizer but may
not change it through `tk`. `written` begins at zero. The loop visits
each input position, casts its byte to a safe table index, and reads the
corresponding id. `continue` leaves `written` unchanged for an absent
byte. In `ids[written++] = id`, C uses the old value as the array index
and then increments it. The returned `written` count is the number of
known bytes, which is the compaction seen in the table.

`tokenizer_decode` accepts only an id from this vocabulary. Its
`assert` is the Chapter 2 programmer contract: a bad id is a bug in the
caller, not input to recover from. A valid id selects the corresponding
byte.

Training text cannot contain an unknown byte when its tokenizer was
built from that same text. A user's later prompt can, so prompt
encoding deliberately skips unknown bytes. Held-aside text uses the
training tokenizer too, but the command-line path rejects it if any
byte would be skipped. Chapter 14 walks that diagnostic.

## Keep the alphabet beside the learned rows

Chapter 1 showed that the model has rows associated with token ids.
Suppose the saved row at id 0 was learned for newline. If a loader
changed id 0 to `a`, every number in that row would keep its bits while
acquiring the wrong meaning.

**Predict:** is it enough to save the four bytes from our example in
the order `a`, `b`, `z`, newline?

No. The bytes are all present, but their positions define their ids.
The tokenizer writes the vocabulary in exact id order and the reader
accepts only the canonical ascending order.

Turning an in-memory object into an ordered byte representation and
reconstructing it later is **serialization**. This first instance saves
the tokenizer as a count followed by the byte at each id.

For the `"zaba\n"` tokenizer, the writer performs these two writes:

```text
32-bit count value:             4
vocabulary bytes in id order:  10  97  98  122
meaning:                       newline  a  b  z
```

The reader gets 4 first, then reads exactly four vocabulary bytes. It
checks `10 < 97 < 98 < 122`, marks those four byte values as present,
and rebuilds both tables in ascending order.

**Predict:** after that reconstruction, what are
`byte_to_id['z']` and `id_to_byte[0]`?

They are 3 and newline. The ordered fields have made a complete numeric
round trip without changing any id.

Here is the complete writer:

```c
int tokenizer_write(const Tokenizer *tk, FILE *stream)
{
    if (write_i32(stream, tk->vocab_size) != 0)
        return -1;
    if (fwrite(tk->id_to_byte, 1, (size_t)tk->vocab_size, stream) != (size_t)tk->vocab_size)
        return -1;
    return 0;
}
```

It first stores the vocabulary count as a 32-bit integer. It then calls
`fwrite` with the address of the id-to-byte table, an item size of one
byte, and exactly `vocab_size` items. Either failed write returns `-1`.

The reader separates four jobs. First it reads and checks the count:

```c
static int read_vocabulary_size(FILE *stream, int32_t *vocab_size)
{
    if (read_i32(stream, vocab_size) != 0)
        return -1;
    if (*vocab_size < 1 || *vocab_size > BYTE_VALUES)
        return -1;
    return 0;
}
```

Then it reads that many bytes:

```c
static int read_vocabulary_bytes(FILE *stream, char bytes[BYTE_VALUES],
                                 int32_t vocab_size)
{
    return fread(bytes, 1, (size_t)vocab_size, stream)
               == (size_t)vocab_size
        ? 0 : -1;
}
```

The next helper checks the serialized id order:

```c
static int vocabulary_is_canonical(const char bytes[BYTE_VALUES],
                                   int32_t vocab_size)
{
    for (int32_t i = 1; i < vocab_size; i++) {
        /* Weight rows use this serialized id order.  The writer emits
         * ascending bytes, so accepting any other order would silently
         * attach the loaded weights to different characters. */
        if ((unsigned char)bytes[i - 1] >= (unsigned char)bytes[i])
            return 0;
    }
    return 1;
}
```

For each adjacent pair, the unsigned casts compare byte values from 0
through 255. `>=` rejects both descending order and duplicate
neighbors. Starting the loop at id 1 means both compared indexes exist.

Only a canonical array reaches reconstruction:

```c
static Tokenizer *from_canonical_vocabulary(const char bytes[BYTE_VALUES],
                                            int32_t vocab_size)
{
    int seen[BYTE_VALUES] = { 0 };

    for (int32_t i = 0; i < vocab_size; i++)
        seen[(unsigned char)bytes[i]] = 1;
    return from_seen_bytes(seen);
}
```

Each byte is marked in `seen`, and `from_seen_bytes` rebuilds both
lookup tables. The strict adjacent comparison has already proved that
the bytes are unique. No second duplicate check is needed after
construction.

The public reader is now the ordered list of those jobs:

```c

Tokenizer *tokenizer_read(FILE *stream)
{
    int32_t vocab_size;

    if (read_vocabulary_size(stream, &vocab_size) != 0)
        return NULL;

    char bytes[BYTE_VALUES];

    if (read_vocabulary_bytes(stream, bytes, vocab_size) != 0)
        return NULL;
    if (!vocabulary_is_canonical(bytes, vocab_size))
        return NULL;
    return from_canonical_vocabulary(bytes, vocab_size);
}
```

Both public signatures use the `FILE *` stream from Chapter 2. The
reader rejects a missing count, a count below 1 or above 256, a short
byte read, and any noncanonical order. Every rejection returns `NULL`
instead of handing the caller a partly valid mapping. If allocation
inside `from_seen_bytes` fails, Chapter 2's `emalloc` contract
terminates the process instead.

There is one deliberate boundary. `tokenizer_new("", 0)` can construct
an empty runtime tokenizer, but `tokenizer_read` rejects a serialized
count of zero. Training always needs at least one byte. A saved-model
**checkpoint**, the file built in
[Chapter 13](13-durable-checkpoints.md), also carries a nonempty
tokenizer. The application therefore never promises an empty-tokenizer
round trip.

The 32-bit count uses Chapter 2's same-platform integer format.
[Chapter 13](13-durable-checkpoints.md) will place this tokenizer
payload beside the model parameters and add the complete saved-model
boundary.

## Keep complete scenes out of training

If every clean scene can change the model, checking those same scenes
later measures only text that was eligible to shape it. We also want a
second group whose scenes never participate in those updates.

Consider a deliberately weak predictor that stores each question and
its answer without learning a reusable pattern. For a question absent
from its table, it returns `?`, which counts as wrong:

```text
record   question   answer
A        RA         Z
B        GH         O
C        VY         P
D        NY         X
```

**Predict:** if it stores all four records and is checked on those same
four, how many answers does it get right?

It gets 4 out of 4. That result makes the predictor look perfect even
though it can answer only questions already in its table. If record
`D` never enters the table, its answer is `?`; checking `D` gives 0 out
of 1 and exposes that limit. A real model stores patterns in numbers
rather than a literal lookup table, but reusing its update-eligible
records creates the same blind spot.

A byte cut is a tempting first attempt. Put the first 90 percent of the
file in one group and the last 10 percent in another. The cut can land
between a header and its dialogue. A line-by-line shuffle is worse: one
scene's header might enter the first group while its replies enter the
second. In either case, the two groups no longer contain complete
NIGHT GRID records.

The cleaner already separates complete scenes with blank lines. Treat
each scene as one record, choose record membership, and then write the
whole record to one side.

Work through four records named `A`, `B`, `C`, and `D`. Their numeric
indexes begin in source order:

```text
order = [0, 1, 2, 3]
record =  A  B  C  D
```

Chapter 2 built `rng_below`. For seed 1337, the three bounded draws
needed by this four-record shuffle are 2, 1, and 0:

```text
i=3, swap positions 3 and 2: [0, 1, 3, 2]
i=2, swap positions 2 and 1: [0, 3, 1, 2]
i=1, swap positions 1 and 0: [3, 0, 1, 2]
```

The default 10 percent calculation keeps at least one record on each
side. With four records it selects one for the held-aside group.

**Predict:** which record is selected when the first shuffled index is
used?

Index 3 comes first, so record `D` is held aside. The selection order is
`D, A, B, C`, but the output files do not use that shuffled order. The
final membership read in original order is:

```text
A: may update the model
B: may update the model
C: may update the model
D: held aside
```

We have now built a deterministic partition of whole records. The
update-eligible side is the **training split**. The held-aside side is
the **validation split**. In this example, the first three records form
the training split and `D` forms the validation split.

Splitting one scene between the two sides would put evidence from the
same record into both. Information meant for the held-aside check would
then enter the training data. That failure is **data leakage**.
Building the alphabet from both files would be another case: the
held-aside byte set would influence the training setup. Tiny AgenC
blocks both cases by keeping records whole and building one tokenizer
from the training split.

This boundary begins after cleaning. The fixed grammar runs over the
complete canonical raw file, and its global three-copy count is applied
before record membership is chosen. A held-aside record can therefore
affect which repeated line copies survive the global cleaning pass. The
validation split is isolated from tokenizer construction, vocabulary
selection, and parameter updates; it is not isolated from all earlier
text processing. An experiment requiring that stronger boundary would
have to choose records before any data-dependent cleaning. Tiny AgenC
treats its canonical clean corpus as the input to the split and states
that boundary directly.

Here is the selection core from
[`split-order.c`](../scripts/split-order.c). The excerpt omits argument
parsing and allocation:

```c
for (int i = 0; i < record_count; i++)
    order[i] = i;
```

The omitted allocation creates `order` with `malloc` and the
`validation` flags with `calloc`. Chapter 2 showed that `calloc` starts
every byte at zero, so no record is marked yet. The displayed loop
fills `order` with the original indexes. The source then validates the
seed argument and creates `rng`; Chapter 14 will walk command-line
arguments.

The remainder of the selection core is exact:

```c
for (int i = record_count - 1; i > 0; i--) {
    int other = rng_below(rng, i + 1);
    int held = order[i];

    order[i] = order[other];
    order[other] = held;
}
rng_free(rng);

for (int i = 0; i < validation_count; i++)
    validation[order[i]] = 1;
for (int i = 0; i < record_count; i++)
    puts(validation[i] ? "validation" : "train");
```

The backward loop performs the three swaps from the worked example. At
each position `i`, `rng_below(rng, i + 1)` chooses one of the
still-eligible positions from 0 through `i`. The temporary `held`
preserves one value while the two array entries trade places. The
generator is freed after its last draw.

After the shuffle, the next loop marks the first
`validation_count` indexes. The final loop visits indexes from 0
upward, so each destination keeps the original relative order of its
records. `condition ? first : second` chooses one of the two strings,
and `puts` writes that string followed by a newline. The shuffled array
chooses membership; it does not become the training order.

[`split-corpus.sh`](../scripts/split-corpus.sh) performs the surrounding
file work. It reads blank-line-delimited records, counts at least two,
and computes:

```text
train count = records * (100 - validation percent) / 100
validation count = records - train count
```

These are integer calculations, so division discards any remainder.
The percentage must be an integer from 1 through 50. The default is 10,
and the default seed is 1337. The script clamps the counts so at least
one record reaches each side. Source, training, and held-aside paths
must all differ. Both outputs share a directory so temporary files,
backups, and the final two-file installation can be managed together.

Running the same split with the same seed replays its membership.
Changing the seed normally changes which records are held aside. If a
validation byte is absent from the training vocabulary, the later
training command rejects the validation file instead of silently
shortening it.

The canonical clean corpus contains 2,011 records. At the 90/10
default:

```text
2,011 * 90                    = 180,990
180,990 / 100                 =   1,809 remainder 90
training records              =   1,809
validation records = 2,011 - 1,809 = 202
```

The committed evidence records 982,693 training bytes and 106,701
validation bytes. Their sum is the 1,089,394-byte clean corpus. The
split holds out complete records, not every repeated string. Because
the source has recurring labels and permits up to three copies of one
dialogue line, the same phrase can still occur on both sides.

Chapter 15 will use fixed validation batches during training, and
[Chapter 17](17-the-training-run.md#read-the-held-out-run) will
interpret the resulting measurements. Here the important contract is
precise: after canonical global cleaning, choose whole held-aside
scenes before tokenization or parameter updates.

## Store one owned sequence of ids

Tokenizing the same million-byte file before every batch would repeat
the same conversion. The program instead allocates one integer array,
encodes the text into it once, remembers the number of ids written, and
reuses that sequence.

For the four-byte text `"zaba"` and its sorted mapping, that owned state
would look like this:

```text
source bytes:  z  a  b  a
owned ids:     2  0  1  0
token count:   4
```

An object that owns this encoded corpus sequence is a **Dataset**.

**Predict:** after construction, can the caller free the source text
without losing these four ids?

Yes. Construction copies the encoded values into a new allocation. The
Dataset keeps no pointer to the source text or tokenizer and does not
own either one. It owns only its allocated token array and count.

Before allocating, the program needs a bound. One source byte can
produce at most one integer id, so `length` source bytes need at most:

```text
length * sizeof(int) bytes
```

Three limits apply. Project policy accepts at most 256 MiB of source
text. `SIZE_MAX / sizeof(int)` is the largest source length whose byte
product fits in `size_t`. `INT_MAX` is C's largest representable
`int`; limiting the source length to it keeps later start positions and
random bounds representable as `int`. The effective limit is the
smallest of those three.

Here are the complete storage-related definitions from
[`dataset.c`](../src/dataset.c). The excerpt omits includes and moves
`dataset_free` beside the other storage operations; the displayed
definitions themselves are complete:

```c
struct Dataset {
    int   *tokens;
    size_t token_count;
};

static const size_t DATASET_POLICY_MAX_BYTES =
    (size_t)256 * (size_t)1024 * (size_t)1024;

size_t dataset_max_text_bytes(void)
{
    size_t allocation_limit = SIZE_MAX / sizeof(int);
    size_t limit = DATASET_POLICY_MAX_BYTES;

    if (limit > allocation_limit)
        limit = allocation_limit;
    if (limit > (size_t)INT_MAX)
        limit = (size_t)INT_MAX;
    return limit;
}

Dataset *dataset_new(const Tokenizer *tk, const char *text, size_t length)
{
    if (length > dataset_max_text_bytes())
        return NULL;

    Dataset *ds = emalloc(sizeof *ds);
    size_t token_bytes = length * sizeof *ds->tokens;

    /* malloc(0) may return NULL even though the empty Dataset is valid. */
    ds->tokens = emalloc(token_bytes == 0 ? sizeof *ds->tokens : token_bytes);
    ds->token_count = tokenizer_encode(tk, ds->tokens, text, length);
    return ds;
}

size_t dataset_token_count(const Dataset *ds)
{
    return ds->token_count;
}

void dataset_free(Dataset *ds)
{
    free(ds->tokens);
    free(ds);
}
```

The two fields are private. `tokens` points to a separate allocation;
`token_count` says how many entries contain encoded ids. The policy
constant spells 256 MiB as `256 * 1024 * 1024` using `size_t`
arithmetic.

`dataset_max_text_bytes` starts at the policy limit and lowers it when
allocation arithmetic or later `int` indexing requires a smaller
value. Because `dataset_new` checks `length` first, the multiplication
cannot wrap.

After allocating the structure, `dataset_new` calculates the maximum
token-buffer byte count. C permits `malloc(0)` to return either a
pointer or `NULL`. Tiny AgenC's `emalloc` treats `NULL` as failure, but
an empty Dataset is valid. The conditional therefore allocates one
unused `int` when `token_bytes` is zero.

`tokenizer_encode` fills the allocation and returns the actual number
of ids. This can be less than `length` when the text has unknown bytes.
Training text has none because it created the tokenizer. The
command-line validation path compares the returned count with the file
length and rejects any unknown validation byte.

`dataset_token_count` exposes the count without exposing the private
array. `dataset_free` releases the token allocation first and then the
structure. It does not free the tokenizer; the caller owns that
separate object.

An empty Dataset is valid at this module boundary. The training command
first looks up the required newline id because generated samples begin
at a natural line boundary. An empty file or a file with no newline
therefore fails there; next-token batching itself does not require a
newline. A newline-bearing file that is still too short later fails the
input-window check. Chapter 14 walks that command path.

On the reference platform, `sizeof(int)` is four.

**Predict:** how much token storage could a source at the full 256 MiB
policy ceiling reserve?

Each source byte reserves one four-byte integer, so
`256 MiB * 4 = 1,024 MiB = 1 GiB`. That number is not a C-wide promise.
The formula and the two platform limits determine the bound elsewhere.

## One extra token supplies every answer

Next-character prediction needs a run of input tokens and the same run
moved one position forward. Use `SP` to make the invisible space byte
visible:

```text
source:   R  A  Z  R  :  SP  S  t  a  y
inputs:   R  A  Z  R  :  SP  S  t  a
targets:  A  Z  R  :  SP  S  t  a  y
```

Each target is the source byte immediately after the input above it.
The final target, `y`, is not present in the input row. It comes from
the one extra source position at the right.

A width-three numeric case uses `"abcd"`. Sorting those bytes gives ids
`a=0`, `b=1`, `c=2`, and `d=3`. Choose an input width of three:

```text
source ids:  0  1  2  3
inputs:      0  1  2
targets:     1  2  3
```

**Predict:** could the same three input positions produce all three
targets if the source stopped after id 2?

No. The last answer needs id 3. An input width `T` therefore requires a
source run of `T + 1` tokens. Pairing each input id with the immediately
following answer creates a **next-token pair**.

The relation `target[t] == input[t + 1]` is visible for the first
`T - 1` columns. At the final column, `input[T]` lies outside the
`T`-wide input row. It exists in the source run and is copied only into
the targets.

Think of the legal source runs as a deck of flashcards. One run is one
flashcard: the front contains `T` inputs and the back contains their
`T` next-token answers, making `T` next-token pairs. Drawing a card
does not remove it from the deck.

## Draw every legal start

A long token sequence contains many overlapping cards. Lay out ten
token occurrences, number their source positions from 0 through 9, and
keep `T = 3`:

```text
position:   0 1 2 3 4 5 6 7 8 9
source id:  4 1 1 7 2 4 6 3 0 5
```

A start at 0 consumes source positions 0 through 3. A start at 6
consumes 6 through 9. A start at 7 would need position 10 for its final
target, which is outside the sequence.

**Predict:** what is the last legal start, and how many legal starts
are there?

The last start is 6. The inclusive set `0, 1, 2, 3, 4, 5, 6` has seven
members. With `N` tokens and input width `T`:

```text
last legal start = N - T - 1
number of starts = last legal start + 1 = N - T
```

For the Chapter 2 generator with seed 42, the first two draws below 7
are 5 and 2.

**Predict:** using the position and source-id rows above, which three
input ids and target ids does each start produce?

They produce:

```text
start 5: inputs [4, 6, 3]    targets [6, 3, 0]
start 2: inputs [1, 7, 2]    targets [7, 2, 4]
```

Drawing start 5 does not change the seven-start list.

**Predict:** can the next row draw start 5 again?

Yes. Every row chooses from the full list. Returning a chosen window
before the next draw is **sampling with replacement**.

Call the integer returned by `rng_below` `start` while reading the code.
The implementation must point at that chosen token occurrence without
copying all earlier values. Suppose `ds->tokens` points at source
position 0, `start` is 5, and one `int` occupies four bytes.

**Predict:** which source position and stored id does `ds->tokens + 5`
select, and how many raw bytes away is it?

It points at the token occurrence in position 5, whose id is 4. C
scales the move by the pointed-to type, so five four-byte integers move
20 bytes. Adding an object count to a typed pointer this way is
**pointer arithmetic**.

C defines that addition only within one array allocation or at the
position immediately after it. That one-past position may be formed,
but it may not be read. The legal-start calculation protects this
boundary. For `N = 10` and `T = 3`, the last run reads input source
positions 6 through 8 and target source positions 7 through 9. Their
input ids are `[6, 3, 0]`, and their target ids are `[3, 0, 5]`.
Position 10 is one past the array and is never copied.

The row now needs three consecutive integers in separate destination
storage.

**Predict:** on the same four-byte platform, how many bytes reproduce
that row?

Three integers need 12 bytes.
`memcpy(destination, source, byte_count)` copies that many bytes.
Using a byte count to reproduce the integer row is the chapter's
**byte copy**.

The two destination pointers have the same capacity rule as `ids`.
With `B = 2` and `T = 3`, each must provide six integer slots. A
five-slot destination would place the last value of row 1 outside its
array. `dataset_batch` cannot inspect either capacity. The application
validates the [Chapter 1 `B*T`
ceiling](01-the-map.md#build-checkpoint-specify-the-machine) before
allocating `B*T` slots for each array. It allocates `inputs` and
`targets` as two distinct, non-overlapping buffers; this function
relies on that caller contract.

The file names the one-position relationship first:

```c
enum { NEXT_TOKEN_OFFSET = 1 };
```

Here is the complete batch function:

```c
void dataset_batch(const Dataset *ds, Rng *rng, int *inputs, int *targets,
                   int batch_size, int block_size)
{
    /* A run starting at s uses tokens s .. s+block_size as input and
     * target, so the final answer determines the last legal start. */
    assert(ds->token_count
           >= (size_t)block_size + NEXT_TOKEN_OFFSET);

    int last_start =
        (int)(ds->token_count - (size_t)block_size - NEXT_TOKEN_OFFSET);

    for (int row = 0; row < batch_size; row++) {
        const int *run =
            ds->tokens + rng_below(rng, last_start + NEXT_TOKEN_OFFSET);

        memcpy(inputs + row * block_size, run, (size_t)block_size * sizeof *inputs);
        memcpy(targets + row * block_size, run + NEXT_TOKEN_OFFSET,
               (size_t)block_size * sizeof *targets);
    }
}
```

The signature receives the owned sequence, the Chapter 2 generator,
two destination arrays, the number of rows, and the width of each row.
The caller has already validated a positive block size. Given that
precondition, the `assert` enforces the `T + 1` requirement. Converting
`block_size` to `size_t` makes the comparison use the same unsigned
size type as `token_count`.

`NEXT_TOKEN_OFFSET` names the one-position shift from each input to its
answer. `last_start` implements the arithmetic from the ten-token
example. The earlier `INT_MAX` ceiling makes the cast back to `int`
representable. Passing `last_start + NEXT_TOKEN_OFFSET` to `rng_below`
includes both endpoint starts. At the last legal start,
`start + block_size` is
`token_count - 1`, the final real array element rather than the
one-past position.

The expression `ds->tokens + start` applies the pointer arithmetic from
the worked start. Adding one to an `int *` advances by one complete
`int`, not one raw byte. If the draw is 5, `run` points at the token
occurrence in source position 5, whose stored id is 4.
`run + NEXT_TOKEN_OFFSET` points
at position 6, whose id is 6. The same rule makes
`inputs + row * block_size` point at the first integer slot for that
flattened row.

`memcpy(destination, source, byte_count)` copies raw bytes between
valid, non-overlapping storage. The two destination buffers and
`ds->tokens` occupy separate storage, so each copy meets that rule. The
first call copies `block_size` integers from `run` into the selected
input row. Multiplying by `sizeof *inputs` converts that integer count
into a byte count. The second call copies the same number of integers
beginning at `run + NEXT_TOKEN_OFFSET`, which creates the shifted target
row. This fixed-size operation is the byte copy constructed above.

Nothing marks start 5 as used. The next row draws from all seven starts
again and could also choose 5. This is the replacement behavior from
the prediction. Each legal start is selected uniformly by `rng_below`,
and duplicate windows are valid.

The uniformly chosen objects are start positions, not scenes. A long
scene supplies more starts than a short one. The encoded files also
retain blank separators, so a window may cross from the end of one
scene through the separator into the next scene. These are deliberate
properties of the source sequence used by the implementation.

Chapter 1 named the flattened batch dimensions `B` and `T`. With the
default `B = 32` and `T = 128`, one call supplies:

```text
32 * 128 = 4,096 next-token answers
```

That is a count, not a claim about how quickly training must improve.
[Chapter 17](17-the-training-run.md#read-the-full-corpus-run) will show
the recorded training curve.

## The complete data boundary

The clean corpus supports two explicit paths:

```text
canonical raw -> ASCII and grammar -> clean corpus

full-corpus path:
clean corpus -> build tokenizer -> Dataset -> sampled batches

held-out path:
clean corpus -> whole-scene split -> training bytes
                                  -> validation bytes

training bytes -> build tokenizer -> training Dataset -> sampled batches
                        |
validation bytes -------+----------> validation Dataset -> fixed checks
```

Chapter 0 used the full-corpus path for the shortest first run. The
recorded held-out run uses the second path. There, the split happens
before the application tokenizer is built. Training bytes define the
one alphabet, and both Dataset objects use it. Training batches draw
legal starts with replacement. Validation later uses fixed starts so
measurements at different steps ask the same questions.

Trace five places where bad input stops:

```text
oversized command file    bounded reader rejects before reading its body
oversized Dataset length  dataset_new returns NULL before multiplication
validation unknown byte   command rejects the shortened encoding
fewer than T + 1 tokens   command rejects before dataset_batch
bad saved alphabet        tokenizer_read returns NULL
```

Those boundaries keep file input, token meanings, allocation sizes,
and shifted windows separate. There is no hidden text-processing
library between a corpus byte and the integer copied into a batch.

## Build checkpoint: deal the flashcards

**Build.** Implement the sorted byte vocabulary, encode and decode, and
the canonical `tokenizer_write`/`tokenizer_read` round trip. The reader
must reject invalid counts, short bodies, duplicates, and descending
unsigned byte values. Store one encoded sequence in `Dataset`,
including the platform-aware source limit and the valid empty
representation. Implement shifted windows with replacement. Do not
bind a Dataset to one block size.

The supplied `make validation-data` path performs the whole-scene split;
inspect its reported counts rather than reimplementing the script in
the lab. Preserve this invariant for Chapter 14: command-line training
will build one tokenizer from training bytes and reuse it for
validation bytes.

**Verify.**

```sh
make corpus
make validation-data
make -C labs check-03
# answer key: make check-data
make check-evidence
```

**Expected.** Corpus cleaning reports 1,089,394 bytes, 80 distinct byte
values, and 2,011 transmission headers. Splitting reports 1,809
training transmissions in 982,693 bytes and 202 validation
transmissions in 106,701 bytes. The lab and answer key each report that
all data checks passed. `make check-evidence` finishes after confirming
the canonical hashes, derived split, bundled checkpoint, and recorded
replay evidence.

Encoding and then decoding vocabulary bytes preserves them.
Serialization preserves the exact id-to-byte mapping, accepts
canonical values through byte 255, and rejects short, duplicate, or
reordered mappings. `"abcd"` at block size three produces inputs
`[0,1,2]` and targets `[1,2,3]`. Empty input constructs safely, while a
source beyond the effective limit is rejected before multiplication.

**Common failures.**

- First-seen ids change when the text order changes. Scan all 256 byte
  values in ascending order.
- A negative table index means plain `char` was used where an
  `unsigned char` index was required.
- Separate tokenizers for training and validation give one byte two
  possible meanings. Build the mapping from training bytes once.
- Silently accepting a shortened validation encoding hides an unknown
  byte. Reject it at the command boundary.
- A reordered serialized vocabulary attaches saved rows to different
  bytes. Require strict ascending byte order.
- A last-window crash usually comes from forgetting that the random
  bound is `last_start + 1`.
- A shifted final target needs `T + 1` source tokens, not `T`.
- Wrong later rows usually mean the destination pointer omitted
  `row * block_size`.
- Passing an integer count instead of a byte count to `memcpy` omits
  the `sizeof` multiplication.
- An empty allocation failure means construction depended on one
  possible result of `malloc(0)`.
- A huge source becoming a small allocation means the limit check
  happened after the token-buffer multiplication.

---

[Previous: Foundations](02-foundations.md) | [Contents](README.md) |
[Next: Poor Man's Tensors](04-poor-mans-tensors.md)
