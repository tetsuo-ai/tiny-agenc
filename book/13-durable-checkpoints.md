# Chapter 13: Durable Checkpoints

Chapter 0 trained a model, wrote `trained.bin`, started a new process,
and generated text. The second process never saw the training run. It
had only the file.

Now make that handoff explicit. At the end of training, one learned
table might contain:

```text
token-table values
row 0              [ 0.10 ]
row 1              [ 0.70 ]
```

Writing the two floats is not enough. A later process cannot answer:

```text
How many rows and columns did this table have?
Which character does row 0 mean?
What comes after this table?
How many transformer blocks must be rebuilt?
Did the write finish without changing a byte?
```

The old process owned all those answers in memory. Process exit erased
them.

The new process needs one self-contained handoff:

```text
a fixed mark telling the reader what kind of file this is
model dimensions
id-to-character alphabet
learned values in one agreed order
a check value recalculated from the preceding bytes
```

That complete saved-model handoff is a **checkpoint**. Tiny AgenC's
checkpoint is sufficient to reconstruct the model for generation. It
does not attempt to preserve every fact from the training process.

A checkpoint is a promise made by a previous process:

> These bytes describe this model, with this vocabulary, in this order.

Loading turns disk bytes into allocation sizes and floating-point
parameters. Hope is not a parser.

## Work one tiny file by hand

Begin with a small file that gives both learned tables two rows while
still exercising every field:

```text
V=2   T=2   C=1   H=1   L=1   B=1
alphabet: A B
```

The six dimensions were built in
[Chapter 1](01-the-map.md#the-shapes-now-have-somewhere-to-live). They
tell the loader how many token rows, positions, channels, heads, layers,
and batch sequences to construct.

Count the learned floats from the shapes already built in
[Chapter 9](09-parameters-and-the-blueprint.md#one-block-all-its-learned-tensors):

```text
token and position tables       (V + T)C
one transformer block           12C*C + 4C
final gain and bias              2C

P = (V + T)C + L(12C*C + 4C) + 2C
  = (2 + 2)*1 + 1*(12*1*1 + 4*1) + 2*1
  = 4 + 16 + 2
  = 22 floats
```

The fixed opening needs eight 32-bit integers: an identifying integer,
an integer selecting the layout, and the six dimensions. The tokenizer
then needs one 32-bit count followed by its two alphabet bytes. On Tiny
AgenC's supported platform, each integer and float occupies four bytes.

**Predict:** after adding a four-byte check value at the end, how many
bytes does the complete file occupy?

```text
8 opening integers       8*4 = 32 bytes
tokenizer count          1*4 =  4 bytes
alphabet bytes             2 =  2 bytes
parameter values         22*4 = 88 bytes
final check value        1*4 =  4 bytes
                              ----------
                                 130 bytes
```

Here is the same count as an offset map:

```text
0       4       8              32      36   38       126    130
+-------+-------+---------------+-------+----+---------+-------+
| TAGC  | ver 1 | V T C H L B   | tk=2  | AB | 22 f32  | check |
+-------+-------+---------------+-------+----+---------+-------+
```

Offsets mark byte boundaries. Byte 38 begins the first float; byte 126
begins the final four-byte check value; byte 130 is one past the file.

**Predict:** should an otherwise valid file with a 131st byte load?

No. A reader that accepts unexplained trailing bytes cannot tell whether
it understands the whole file or only an old prefix.

The first four bytes are an identifying mark. Such a fixed mark at the
start of a file is called a **magic value**. The source writes the
32-bit value `0x43474154`. On the reference little-endian machine,
[Chapter 2's native byte
order](02-foundations.md#fixed-integers-and-a-clock-that-does-not-turn-back)
places it on disk
as:

```text
54 41 47 43
 T  A  G  C
```

Suppose a future writer inserts another field while an old reader still
expects parameter bytes at the old offset. Matching magic alone would
let the old reader assign new bytes the wrong meaning. The layout needs
its own integer selector.

The next integer is `1`. It is the **format version**. A changed layout
must use a new version rather than letting an old reader guess what new
bytes mean. The fixed opening through `B` is the file's **header**. The
alphabet and learned values are its **payload**.

## The TAGC version 1 layout

The resulting layout is TAGC version 1:

| Offset | Bytes | Content |
|---:|---:|---|
| 0 | 4 | native `int32_t` magic `0x43474154` |
| 4 | 4 | native `int32_t` version `1` |
| 8 | 4 | vocabulary size `V` |
| 12 | 4 | block capacity `T` |
| 16 | 4 | model width `C` |
| 20 | 4 | head count `H` |
| 24 | 4 | layer count `L` |
| 28 | 4 | maximum batch `B` |
| 32 | 4 | tokenizer count, required to equal `V` |
| 36 | `V` | strictly increasing vocabulary bytes |
| `36+V` | `P*sizeof(float)` | parameter values |
| last 4 | 4 | check value for every preceding byte |

This is deliberately a same-platform format. It uses the machine's
native `int32_t` and `float` representations. It is not a universal
model-exchange format between byte orders or floating-point formats.
The old project format has a different magic value and is rejected.

## Order gives unnamed values their meaning

Parameter order is serialized meaning.

The file does not write a name before each learned array. Consider two
same-shaped tables:

```text
token table       [ 0.10  0.70 ]
position table    [ 0.20 -0.30 ]
```

The writer emits:

```text
0.10  0.70  0.20 -0.30
```

**Predict:** if the reader gives the first two values to the position
table, can either the scalar count or file length detect the mistake?

No. Both tables still receive two values:

```text
loaded position table    [ 0.10  0.70 ]   wrong meaning
loaded token table       [ 0.20 -0.30 ]   wrong meaning
```

The file can have the right length and an intact check value while
rebuilding the wrong model.

Chapter 9 constructed one numbered
[parameter registry](09-parameters-and-the-blueprint.md#one-assignment-creates-both-views).
The writer and reader both walk that registry:

```text
token table
position table
for layer 0, then layer 1, and so on:
    norm1 gain
    norm1 bias
    packed query-key-value weights
    attention projection weights
    norm2 gain
    norm2 bias
    MLP up weights
    MLP down weights
final gain
final bias
```

The output head reuses the token table, as
[Chapter 9](09-parameters-and-the-blueprint.md#one-table-two-jobs)
showed. It contributes no second payload.

The fixed registry sequence is therefore more than convenient
iteration. It is the **canonical parameter order**: order on disk
supplies the names that the raw values do not carry. Changing that
order without changing the format version can produce a clean read of
a wrong model.

## Make changed bytes change the stored check

Exact length detects missing and trailing bytes. It does not detect a
changed byte in the middle. A damaged float can even remain finite:

```text
1.0f bytes        00 00 80 3f
one changed bit   01 00 80 3f
```

A first attempt could add all byte values and store the total.

**Predict:** will that total distinguish `01 02` from `00 03`?

```text
bytes             ordinary byte sum
01 02                       3
00 03                       3
```

No. The bytes differ, but their sum does not. Reordering bytes defeats
the same scheme.

We need a running 32-bit state in which the position of every incoming
bit matters. Chapter 2 already built
[fixed-width integers, shifts, XOR, and unsigned
wraparound](02-foundations.md#build-a-generator-from-state). Reuse
those tools here.

Start the state with all 32 bits set. XOR one input byte into its low
bits. Then repeat eight times, once for each bit in that byte:

```text
shift the state right by one bit
if the discarded low bit was 1, XOR in 0xEDB88320
```

For the one-byte input `A`, whose byte value is `0x41`, the source's
states are:

```text
initial state                         FFFFFFFF
after XOR with 41                     FFFFFFBE
after bit 0                           7FFFFFDF
after bit 1                           D2477CCF
after bit 2                           849B3D47
after bit 3                           AFF51D83
after bit 4                           BA420DE1
after bit 5                           B09985D0
after bit 6                           584CC2E8
after bit 7                           2C266174
after complement                      D3D99E8B
```

Every printed state can be reproduced with the shift and XOR rule.
The final complement flips every bit.

A stored value recalculated from the earlier bytes for the purpose of
detecting accidental change is a **checksum**. This particular
shift-and-XOR checksum is reflected **CRC32**, using polynomial
`0xEDB88320`, initial state `0xFFFFFFFF`, and a final complement.

It separates the failed byte-sum pair:

```text
bytes             byte sum        CRC32
01 02                    3         B6CC4292
00 03                    3         D8D04345
```

There are only 2 to the 32nd power possible CRC32 results, so different
long inputs can share one result. A person who edits the file can
calculate a new CRC. It detects the tested bit corruption and many
accidental changes; it does not prove who made the file or that the
contents describe a sensible model.

Evidence that data came from a trusted creator is **authentication**.
TAGC version 1 provides no authentication.

Keep the validation questions separate:

```text
CRC matches        bytes pass this accidental-change detector
format matches     this reader knows the intended byte layout
resources fit      the requested model stays inside policy
payload is valid   alphabet, count, and floats satisfy their rules
authentication     not provided by TAGC version 1
```

A valid recomputed CRC does not excuse a bad version, hostile
dimensions, malformed alphabet, or nonfinite float.

## Walk the checksum code

The production
[`crc32_update`](../src/checkpoint.c) is the rule above written
directly in C:

```c
static uint32_t crc32_update(uint32_t crc, const unsigned char *bytes,
                             size_t count)
{
    for (size_t i = 0; i < count; i++) {
        crc ^= bytes[i];
        for (int bit = 0; bit < 8; bit++)
            crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
    return crc;
}
```

The outer loop visits each byte in order. `crc ^= bytes[i]` mixes that
byte into the low eight bits. The inner loop advances one bit.

The expression `crc & 1u` extracts the current low bit. If it is zero,
`0u - 0u` is zero. If it is one, unsigned wraparound makes
`0u - 1u` all one bits. The `&` therefore selects either zero or the
polynomial. One source line performs the conditional XOR without an
`if`.

The file pass reads bounded chunks rather than allocating a copy of the
whole file:

```c
static int crc32_prefix(FILE *stream, off_t length, uint32_t *result)
{
    unsigned char buffer[8192];
    uint32_t      crc = 0xFFFFFFFFu;

    if (fseeko(stream, 0, SEEK_SET) != 0)
        return -1;
    while (length > 0) {
        size_t wanted =
            length < (off_t)sizeof buffer ? (size_t)length : sizeof buffer;
        size_t received = fread(buffer, 1, wanted, stream);

        if (received != wanted)
            return -1;
        crc = crc32_update(crc, buffer, received);
        length -= (off_t)received;
    }
    *result = ~crc;
    return 0;
}
```

`off_t` is Chapter 2's file-offset type. `fseeko` first returns to byte
zero. Each loop asks for the lesser of the remaining length and the
8,192-byte local buffer. A short read is failure. The update continues
from the preceding chunk's state, so chunk boundaries do not change the
answer. Only after exactly `length` bytes does `~crc` complement every
bit and publish the result.

On save, those bytes must reach the stream before the function rereads
them:

```c
static int append_checksum(FILE *stream)
{
    if (fflush(stream) != 0)
        return -1;

    off_t payload_end = ftello(stream);
    uint32_t checksum;

    if (payload_end < 0 || crc32_prefix(stream, payload_end, &checksum) != 0
        || fseeko(stream, payload_end, SEEK_SET) != 0
        || write_i32(stream, (int32_t)checksum) != 0
        || fflush(stream) != 0)
        return -1;
    return fsync(fileno(stream));
}
```

`FILE *` output is buffered in the C library. `fflush` pushes that
buffer toward the operating system before the same stream is reread.
`ftello` records the first byte after the payload. The prefix pass
calculates through that point, `fseeko` returns there, and `write_i32`
appends the checksum. The second `fflush` pushes the new checksum too.

`fileno` recovers the operating system's integer handle from the C
stream. The handle is called a **file descriptor**. `fsync` asks the
operating system to synchronize the file contents and metadata needed
to retrieve them. The later save section will separate that file
synchronization from directory persistence.

The loader refuses an enormous file before paying to checksum it. The
source-exact gate is:

```c
static int checksum_matches(FILE *stream)
{
    if (fseeko(stream, 0, SEEK_END) != 0)
        return 0;

    off_t end = ftello(stream);

    if (end < (off_t)(3 * sizeof(int32_t))
        || end > (off_t)MODEL_MAX_CHECKPOINT_FILE_BYTES)
        return 0;

    off_t   checksum_at = end - (off_t)sizeof(int32_t);
    int32_t stored;
    uint32_t computed;

    if (fseeko(stream, checksum_at, SEEK_SET) != 0
        || read_i32(stream, &stored) != 0
        || crc32_prefix(stream, checksum_at, &computed) != 0
        || (uint32_t)stored != computed)
        return 0;
    return fseeko(stream, 0, SEEK_SET) == 0;
}
```

The first seek and `ftello` find the exact file length. The preliminary
minimum guarantees only two opening words and the final checksum. Later
parsing still requires the full header. The maximum blocks a padded
multi-gigabyte file before the CRC scan.

`checksum_at` is four bytes before the end. The loader reads the stored
word there, calculates the prefix ending immediately before it, and
compares the bit patterns as `uint32_t`. Success rewinds the stream so
ordinary parsing can start at byte zero.

Checksum validation and parsing are two passes over the same open file.
Tiny AgenC's own writer replaces a completed file by rename, so it does
not modify that open file in place. An unrelated writer that changes
the same underlying open file between the two passes could make the
parsed bytes differ from the checked bytes. TAGC version 1 does not
defend against that concurrent in-place mutation.

## Count bytes and memory separately

A small file can request a huge machine.

The checksum pass has an exact maximum:

```text
MODEL_MAX_CHECKPOINT_FILE_BYTES = 268,439,552 bytes
```

Chapter 9's `Param` object owns four equally sized buffers: values,
gradients, first moments, and second moments. The file stores only the
values. A model inside the one-GiB resident ceiling therefore cannot
have more than one quarter GiB of stored parameter values. The source
adds 4,096 bytes for the header and other file data:

```text
1 GiB / 4 + 4096
= 1,073,741,824 / 4 + 4096
= 268,439,552 bytes
```

That is a cheap outer envelope, not the final model-size calculation.
Activations can dominate memory without appearing in the file.

Use the hostile but geometrically valid header from the Chapter 13
witness:

```text
V=1   T=1024   C=1   H=1   L=1   B=1024
```

Its parameter payload is small:

```text
P = (1 + 1024)*1 + 1*(12*1*1 + 4*1) + 2*1
  = 1025 + 16 + 2
  = 1043 floats

file bytes = 40 + V + 4P
           = 40 + 1 + 4*1043
           = 4213 bytes
```

The 40 fixed bytes are the nine opening integers plus the final
checksum.

Now calculate one attention-score region:

```text
score floats = B*H*T*T
             = 1024*1*1024*1024
             = 1,073,741,824 floats

score bytes  = 1,073,741,824*4
             = 4,294,967,296 bytes
             = 4 GiB
```

**Predict:** should a complete, checksummed 4,213-byte file be allowed
to call `model_new` for this configuration?

No. The header's file request is small, but one runtime region is
already four times the complete checkpoint resident ceiling.

The source reuses Chapter 10's allocation-free
[model memory
report](10-memory-planning.md#the-public-memory-report-is-an-independent-calculation):

```c
static int checkpoint_config_supported(ModelConfig cfg)
{
    ModelMemory memory;
    size_t parameter_floats = model_parameter_float_count(cfg);
    size_t fixed_bytes =
        (CHECKPOINT_HEADER_I32S + CHECKPOINT_TOKENIZER_SIZE_I32S
         + CHECKPOINT_CHECKSUM_I32S) * sizeof(int32_t);

    if (!model_memory_requirements(cfg, &memory)
        || memory.total_bytes > MODEL_MAX_CHECKPOINT_RESIDENT_BYTES
        || parameter_floats > SIZE_MAX / sizeof(float))
        return 0;

    size_t parameter_bytes = parameter_floats * sizeof(float);

    return fixed_bytes <= MODEL_MAX_CHECKPOINT_FILE_BYTES
        && (size_t)cfg.vocab_size
           <= MODEL_MAX_CHECKPOINT_FILE_BYTES - fixed_bytes
        && parameter_bytes
           <= MODEL_MAX_CHECKPOINT_FILE_BYTES - fixed_bytes
            - (size_t)cfg.vocab_size;
}
```

The report first validates the dimensions and performs Chapter 10's
checked calculations for parameter buffers, arenas, gradients, and
cached ids. Anything above one GiB is rejected before construction.
The next check proves the float-byte multiplication fits `size_t`.

Only then does the function calculate parameter bytes. The final
subtractions are ordered so no unsigned expression can wrap. Together
they prove that fixed fields, alphabet, and parameter payload fit the
file ceiling.

These allocation-free file and resident checks are the **checkpoint
resource preflight**. They reject the 4,213-byte hostile request before
`model_new`.

Overflow is failure, not a smaller allocation.

Save and load both call this predicate. A locally constructed model can
be legal under the public geometry rules yet deliberately unsupported
by this checkpoint policy.

## Keep the alphabet attached to its rows

Suppose the learned token rows are:

```text
id 0 value       [ 0.10 ]
id 1 value       [ 0.70 ]
```

With alphabet `A B`, row 0 means `A`. With alphabet `B A`, the same row
means `B`.

**Predict:** can the float payload reveal which alphabet was used?

No. The alphabet has to cross the process boundary with the floats.
Chapter 3's
[canonical tokenizer
serialization](03-data.md#keep-the-alphabet-beside-the-learned-rows)
writes its count followed by bytes in strictly increasing unsigned
order. Its reader rejects duplicates, descending bytes, and counts
outside 1 through 256. The loader also requires the tokenizer
count to equal model dimension `V`.

Canonical order prevents a file from presenting the same alphabet in a
different id order. There remains one caller precondition that bytes
cannot prove. `model_save` can check that the supplied tokenizer has
`V` entries; it cannot infer whether this is the tokenizer that gave
the model's token table its meanings. Supplying a different
same-sized alphabet is a caller error.

## Reject nonfinite learned values

CRC answers whether bytes match their stored checksum. An editor can
recompute that checksum after inserting infinity or `NaN`. Those values
must fail a separate payload rule.

Chapter 8 used the float's exponent bits at the
[checkpoint value
boundary](08-adamw.md#save-values-not-the-training-history).
For the supported 32-bit representation:

```text
value       bit pattern      exponent bits       accepted
1.0f        3F800000         3F800000             yes
infinity    7F800000         7F800000             no
```

An all-one exponent identifies infinity or `NaN`. `-ffast-math` is the
project's compiler option that permits transformations assuming
ordinary finite inputs. Arithmetic-based finiteness checks can
therefore be optimized under assumptions about what they are supposed
to test. The bit test remains reliable under that option.

The checkpoint helpers in
[`param.c`](../src/param.c) use the rule on both sides. These are
complete, source-exact functions:

```c
int param_write(const Param *p, FILE *stream)
{
    size_t count = mat_size(p->values);

    if (!values_are_finite(p))
        return -1;
    return fwrite(p->values.vals, sizeof *p->values.vals, count, stream) == count ? 0 : -1;
}

int param_read(Param *p, FILE *stream)
{
    size_t count = mat_size(p->values);

    if (fread(p->values.vals, sizeof *p->values.vals, count, stream) != count)
        return -1;
    return values_are_finite(p) ? 0 : -1;
}
```

`mat_size` supplies the exact scalar count. Saving checks every value
before `fwrite`; loading requires an exact `fread` count and then checks
every received value. A short read and a nonfinite value both return
`-1`.

Only `p->values` is written. The gradient and two AdamW moment buffers
belong to the in-memory `Param` but not to TAGC version 1.

Finite values are the minimum. They do not prove that the model is
trained or useful.

## Write beside the file that must survive

Consider replacing an existing good `trained.bin` by opening that path
directly for writing.

**Predict:** if the new write fails halfway, which complete version
remains at the destination?

```text
old trained.bin       complete model A
open with "wb"        model A is truncated immediately
write model B         half complete
write fails           neither A nor B remains
```

Neither. Opening with `"wb"` truncated A before B was ready.

The destination must not change until every byte of B is ready.

Construct a neighboring name:

```text
trained.bin                 visible complete model A
trained.bin.tmp.4kP9sQ      separate temporary B under construction
```

If B fails, remove the temporary file and leave A alone. If B succeeds,
replace the destination name in one filesystem operation:

```text
trained.bin=A
      |
write neighboring temporary B
      |
flush C buffer -> synchronize file -> close
      |
rename temporary over destination
      |
trained.bin=B
```

Another process opening `trained.bin` on the stated local-filesystem
boundary sees complete A or complete B. It does not see B halfway
through. Building beside the destination and then renaming is
**atomic file replacement**.

The first half of
[`model_save`](../src/checkpoint.c) creates and owns that temporary
file. This shortened source excerpt begins after the function opening:

```c
    static const char TEMP_SUFFIX[] = ".tmp.XXXXXX";
    size_t path_length = strlen(path);

    if (!checkpoint_config_supported(m->cfg)
        || tokenizer_vocab_size(tk) != m->cfg.vocab_size
        || path_length > SIZE_MAX - sizeof TEMP_SUFFIX)
        return -1;

    char *temp_path = emalloc(path_length + sizeof TEMP_SUFFIX);

    memcpy(temp_path, path, path_length);
    memcpy(temp_path + path_length, TEMP_SUFFIX, sizeof TEMP_SUFFIX);

    int descriptor = mkstemp(temp_path);

    if (descriptor < 0) {
        free(temp_path);
        return -1;
    }
```

Policy and tokenizer size are checked before creating anything.
The path-length comparison makes the allocation addition safe.
`sizeof TEMP_SUFFIX` includes the terminating zero byte, so the two
`memcpy` calls produce a complete C string.

`mkstemp` requires the final six `X` characters. It replaces them with
a unique suffix, creates the file exclusively, opens it, and returns
its file descriptor. Because the template begins with the destination
path, the temporary file is in the same directory and therefore the
same filesystem as the destination.

The source next handles existing access bits. This is the next
source-exact fragment:

```c
    struct stat existing;

    if (stat(path, &existing) == 0
        && fchmod(descriptor, existing.st_mode & 0777) != 0) {
        close(descriptor);
        remove(temp_path);
        free(temp_path);
        return -1;
    }
```

`stat` fills a `struct stat` with file metadata. `st_mode` contains
file-type and permission bits. The leading zero makes `0777` an
**octal integer literal** in C. Its nine one bits select the owner's,
group's, and others' read, write, and execute permissions. `fchmod`
applies only those selected bits to the new descriptor.

If metadata lookup succeeds but changing the permissions fails, the
descriptor is closed, the temporary name is removed, and its allocated
path is freed. If `stat` itself fails, this implementation continues
with `mkstemp`'s permissions. It does not distinguish a missing
destination from other metadata errors, and it does not copy ownership,
access-control lists, or other metadata.

The descriptor is an operating-system handle, not a buffered C stream.
The next source-exact fragment supplies the stream operations used
throughout the book:

```c
    FILE *stream = fdopen(descriptor, "w+b");

    if (stream == NULL) {
        close(descriptor);
        remove(temp_path);
        free(temp_path);
        return -1;
    }
```

`fdopen` wraps the existing descriptor in a `FILE *` open for both
writing and reading in binary mode. On success, closing the stream will
also close its descriptor. On failure, no stream took ownership, so the
code closes the descriptor directly.

The final shortened source excerpt contains the write and commit half:

```c
    int failed = write_i32(stream, (int32_t)CHECKPOINT_MAGIC) != 0
              || write_i32(stream, CHECKPOINT_VERSION) != 0
              || write_i32(stream, m->cfg.vocab_size) != 0
              || write_i32(stream, m->cfg.block_size) != 0
              || write_i32(stream, m->cfg.d_model) != 0
              || write_i32(stream, m->cfg.head_count) != 0
              || write_i32(stream, m->cfg.layer_count) != 0
              || write_i32(stream, m->cfg.batch_size) != 0
              || tokenizer_write(tk, stream) != 0;

    for (int i = 0; !failed && i < m->param_count; i++)
        failed = param_write(m->params[i], stream) != 0;
    if (!failed)
        failed = append_checksum(stream) != 0;
    if (fclose(stream) != 0)
        failed = 1;
    if (!failed && rename(temp_path, path) != 0)
        failed = 1;
    if (failed)
        remove(temp_path);
    free(temp_path);
    return failed ? -1 : 0;
```

C's `||` evaluates left to right and stops at the first true operand.
The header and tokenizer sequence therefore stops on its first failed
write. The registry loop does the same for parameters.

`append_checksum` flushes, rereads, appends, flushes, and synchronizes
the completed temporary file. `fclose` reports a final buffered close
failure. Only a fully successful file reaches `rename`.

`rename` replaces the destination's directory entry atomically on the
local same-filesystem boundary. A handled failure removes the temporary
name, although `remove` itself is best effort here: its return value is
ignored, so a failed cleanup can leave a `.tmp.*` file. A process crash
can leave one too. If there was no old destination, a failed first save
leaves no checkpoint.

File contents and the directory's name-to-file mapping are separate
persistent state:

```text
temporary B contents synchronized        yes
rename trained.bin from A to B            completed while running
directory record for that rename          not synchronized here
```

Suppose power fails after the second line. The source has asked for B's
bytes to reach storage, but it has not asked the parent directory's
updated mapping to do the same. Recovery behavior is filesystem
dependent; this code cannot promise whether the recovered name selects
the old or new file.

**Durability** is the promise that acknowledged state survives a crash
or power loss. Tiny AgenC synchronizes the completed file and provides
atomic runtime replacement. Without parent-directory synchronization,
it does not claim a universal power-loss durability guarantee.

## Validate before publishing

The loader must reject bad bytes without publishing half-built objects.
Its gates run in source order:

```text
file size and CRC
        |
magic, version, geometry
        |
resident-memory and file policy
        |
canonical tokenizer and matching V
        |
exact remaining payload length
        |
construct candidate model
        |
registry count and every finite parameter
        |
final checksum field and end of file
        |
publish Model and Tokenizer
```

**Predict:** if the last parameter is nonfinite after a candidate
tokenizer has been built, should the caller receive that tokenizer?

No. A late failure must free every candidate and leave the output
pointer `NULL`.

Start with [`read_config`](../src/checkpoint.c):

```c
static int read_config(FILE *stream, ModelConfig *cfg)
{
    int32_t magic;
    int32_t version;

    if (read_i32(stream, &magic) != 0
        || (uint32_t)magic != CHECKPOINT_MAGIC
        || read_i32(stream, &version) != 0
        || version != CHECKPOINT_VERSION)
        return -1;
    if (read_dimension(stream, &cfg->vocab_size) != 0
        || read_dimension(stream, &cfg->block_size) != 0
        || read_dimension(stream, &cfg->d_model) != 0
        || read_dimension(stream, &cfg->head_count) != 0
        || read_dimension(stream, &cfg->layer_count) != 0
        || read_dimension(stream, &cfg->batch_size) != 0)
        return -1;
    return model_config_valid(*cfg) ? 0 : -1;
}
```

Each `read_i32` requires one complete native integer. Magic and version
must match exactly. `read_dimension` converts each `int32_t` into the C
`int` field used by `ModelConfig`; on the supported targets those types
represent the required values. The final call applies Chapter 1's
positive bounds, row ceiling, and equal head division before any memory
calculation.

After the tokenizer has been read, the stream is exactly where parameter
values should begin. The reader compares remaining file bytes with the
formula:

```c
static int payload_size_matches(FILE *stream, ModelConfig cfg)
{
    off_t here = ftello(stream);

    if (here < 0 || fseeko(stream, 0, SEEK_END) != 0)
        return 0;

    off_t end = ftello(stream);
    size_t parameter_floats = model_parameter_float_count(cfg);

    if (end < here || end - here < (off_t)sizeof(int32_t)
        || parameter_floats > SIZE_MAX / sizeof(float)
        || fseeko(stream, here, SEEK_SET) != 0)
        return 0;
    return (uintmax_t)(end - here - (off_t)sizeof(int32_t))
        == (uintmax_t)parameter_floats * sizeof(float);
}
```

`here` remembers the first parameter byte. The function seeks to the
end and proves there is room for the final checksum. It then returns to
`here`. Casts to `uintmax_t`, the widest unsigned integer type available
for this comparison, let the offset difference and `size_t` product
meet without narrowing either. Equality rejects both truncation and
trailing data.

Replay the tiny file at this gate:

```text
here                              38
end                              130
bytes after tokenizer and check  130 - 38 - 4 = 88
expected parameter bytes          22 * 4 = 88
```

The two 88-byte results match. A 131st byte would make the left side 89
and fail.

Now walk
[`model_load`](../src/checkpoint.c). This is the complete production
function:

```c
Model *model_load(Tokenizer **tk, const char *path)
{
    if (tk == NULL)
        return NULL;
    *tk = NULL;

    FILE *stream = fopen(path, "rb");
    ModelConfig cfg;
    if (stream == NULL)
        return NULL;
    if (!checksum_matches(stream)
        || read_config(stream, &cfg) != 0
        || !checkpoint_config_supported(cfg)) {
        fclose(stream);
        return NULL;
    }

    Tokenizer *loaded_tk = tokenizer_read(stream);

    if (loaded_tk == NULL
        || tokenizer_vocab_size(loaded_tk) != cfg.vocab_size) {
        if (loaded_tk != NULL)
            tokenizer_free(loaded_tk);
        fclose(stream);
        return NULL;
    }

    if (!payload_size_matches(stream, cfg)) {
        tokenizer_free(loaded_tk);
        fclose(stream);
        return NULL;
    }

    Model *m = model_new(cfg, 0);

    if (model_parameter_count(m) != model_parameter_float_count(cfg)) {
        model_free(m);
        tokenizer_free(loaded_tk);
        fclose(stream);
        return NULL;
    }

    for (int i = 0; i < m->param_count; i++) {
        if (param_read(m->params[i], stream) != 0) {
            model_free(m);
            tokenizer_free(loaded_tk);
            fclose(stream);
            return NULL;
        }
    }

    int32_t checksum;

    if (read_i32(stream, &checksum) != 0 || fgetc(stream) != EOF) {
        model_free(m);
        tokenizer_free(loaded_tk);
        fclose(stream);
        return NULL;
    }

    fclose(stream);
    *tk = loaded_tk;
    return m;
}
```

A missing output pointer cannot receive ownership, so it fails first.
`*tk = NULL` establishes the failure result before the file is opened.
Every later rejection either owns nothing or frees its candidate
objects before returning. Files are input. They do not earn assertions.

The first combined gate checks the complete-file envelope and CRC,
parses the header, and applies the resource policy. The tokenizer is
then reconstructed by Chapter 3's reader and compared with `V`.
Exact payload length is proved before model allocation.

`model_new(cfg, 0)` needs a seed because ordinary construction
initializes parameters. Every saved value will overwrite those initial
values, so the chosen zero seed has no surviving effect on the loaded
parameters. Construction also creates zeroed parameter-gradient and
moment buffers plus the planned arenas. The activation-gradient arena
still needs Chapter 12's explicit clear before backward.

The one-GiB preflight bounds the allocation request; it cannot promise
that the machine currently has that memory available. Allocation
failure still follows the project's fatal allocator policy rather than
returning `NULL`.

The next comparison is an internal cross-check. The scalar formula
used for the file must equal the count in the registry actually created
for this model. Each registry entry then reads its exact number of
finite values.

The stored checksum word is consumed after the parameters. It was
already compared during the first pass. `fgetc` asks for one more byte;
`EOF` means no byte remains. In C, the same `EOF` result can also
represent a read error. The source does not call `feof` and `ferror` to
separate those cases, and it ignores a final successful-path `fclose`
error. Its reliable public failures are the open, read, format, and
resource-policy failures checked explicitly above.

Only after all gates does ownership cross the API boundary:
`*tk = loaded_tk`, and the model pointer is returned. The caller never
receives a partial tokenizer or model.

## State the version 1 promise exactly

The loaded model can generate because the file preserves the
dimensions, alphabet, and learned values. The caller supplies a new RNG
for [Chapter 16's weighted next-character
draw](16-sampling.md#drawing-from-the-distribution). The file cannot
resume the exact next training update.

Use Chapter 8's first AdamW moment. Suppose training had reached:

```text
old first moment    0.20
next gradient       0.10
beta1               0.90
```

Continuing the same process gives:

```text
0.90*0.20 + (1 - 0.90)*0.10
= 0.18 + 0.01
= 0.19
```

**Predict:** after loading TAGC version 1, what first moment results
when the newly constructed moment begins at zero?

```text
0.90*0.00 + (1 - 0.90)*0.10
= 0.00 + 0.01
= 0.01
```

It is `0.01`, not the continued run's `0.19`.

The weights and next gradient can match while the next update differs.
The missing state changes the training trajectory.

The process boundary is:

| State | TAGC v1 treatment | Consequence |
|---|---|---|
| six model dimensions | saved | model geometry is reconstructed |
| sorted alphabet | saved | token ids regain their byte meanings |
| parameter values | saved | generation uses the learned model |
| parameter gradients | omitted, newly zero | a fresh cycle can fill them |
| activation gradients | omitted, allocated | clear them before backward |
| activations and cached ids | omitted, allocated | a new forward fills them |
| AdamW moments and step | omitted, reset | exact update continuation fails |
| optimizer settings | omitted | caller must choose them again |
| training and Chapter 16 draw RNG state | omitted | later draws do not continue |
| corpus, split, and data position | omitted | batch sequence is not resumed |

The committed evidence records enough configuration and provenance to
replay the documented run from its beginning. Replay from the beginning
is not interrupted-run continuation. A format that promises the latter
would need a new version and more fields.

## Know what the witnesses prove

The focused [`labs/check13.c`](../labs/check13.c) witness makes four
checkpoint claims:

```text
saved parameter-object shapes and every parameter value bit round-trip
one changed finite payload bit is rejected by CRC32
the legacy magic is rejected even after its CRC is recomputed
the complete 4,213-byte resource fixture is rejected before construction
```

Its 18 checks include fixture creation, seeking, writing, and closing.
It does not compare every configuration field or tokenizer byte. Its
successful-save check does not observe a concurrent reader or simulate
a crash.

The broader integration witness also checks configuration,
every vocabulary id, loaded forward loss, a seeded generation replay,
unknown versions, invalid dimensions, malformed tokenizer order,
nonfinite values, truncation, trailing data, an oversized file, and
preservation of the old destination after a handled nonfinite save.

Neither witness proves CRC collision resistance, authentication,
permission propagation, concurrent in-place mutation, process-crash
recovery, power-loss recovery, or parent-directory durability. Those
limits are part of the promise, not footnotes outside it.

Both witnesses build corrupt fixtures with a copy of the production CRC
loop. Neither pins the named CRC variant with an independent
known-answer test. The worked values in this chapter were checked
independently; the executables themselves do not make that comparison.

## Build checkpoint: make the handoff complete

**Build.** Implement `checkpoint.c`. Begin with the format constants
and the source-exact CRC32 update. Add the bounded prefix pass, checksum
append, and checksum-first load gate. Reuse Chapter 3's tokenizer
serializer, Chapter 9's canonical parameter registry, and Chapter 10's
checked memory report.

Implement save as adjacent temporary creation, optional existing
read/write/execute bit propagation, header and payload writes, checksum
append, file synchronization, close, and rename. Implement load as the
ordered gates above. Keep candidate tokenizer and model ownership
inside the loader until every check passes. Do not serialize gradients,
optimizer moments, arenas, or cached forward state.

**Verify.**

```sh
make -C labs check-13
# answer key
make -C labs WORK=../src check-13
```

**Expected.**

```text
check-13: all 18 checkpoint checks passed
```

The focused result means the exact claims in the preceding witness
section, not every property of atomic replacement or persistence.

**Common failures.**

- **The tiny example has the wrong size:** count eight header integers,
  one tokenizer-size integer, `V` alphabet bytes, `P` floats, and one
  checksum integer. For the worked model the total is 130 bytes.
- **A changed payload loads:** confirm that CRC covers every byte before
  the stored checksum and that the loader compares it before parsing.
- **A recomputed bad version loads:** CRC integrity does not replace
  magic and version validation.
- **A small file asks for a giant arena:** apply the Chapter 10 memory
  report before `model_new`, not only the parameter-file count.
- **Weights load into the wrong objects:** writer and reader must walk
  the same Chapter 9 registry order. The tied head has no second copy.
- **A failed save damages the old destination:** never open the
  destination for the new payload. Write and synchronize beside it,
  close, then rename.
- **A failed load leaves a tokenizer pointer:** set `*tk = NULL` at
  entry and publish ownership only after the final gate.
- **Loaded training immediately diverges:** version 1 restores learned
  values for generation, not optimizer state or RNG position.

The model can now survive its process. Chapter 14 makes the process
report failures to a person without exposing these internal ownership
paths.

---

[Previous: Wiring the Model Backward](12-wiring-the-model-backward.md) | [Contents](README.md) | [Next: The Command Line Is a Boundary](14-the-command-line.md)
