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

    if (payload_end < 0
        || crc32_prefix(stream, payload_end, &checksum) != 0
        || fseeko(stream, payload_end, SEEK_SET) != 0
        || write_i32(stream, (int32_t)checksum) != 0)
        return -1;
    return fflush(stream);
}
```

`FILE *` output is buffered in the C library. `fflush` pushes that
buffer toward the operating system before the same stream is reread.
`ftello` records the first byte after the payload. The prefix pass
calculates through that point, `fseeko` returns there, and `write_i32`
appends the checksum. The second `fflush` pushes the new checksum too.

That last flush is not a durability promise. A later save stage
recovers the operating system's integer handle with `fileno`. The
handle is called a **file descriptor**. That stage synchronizes the
complete temporary file only after its metadata is also ready.

The old load path checked the file, rewound it, then parsed it. An
unrelated process could change bytes through the same inode between
those two passes:

```text
checksum pass sees       A B C
other process writes         D
parser sees              A B D
```

The parser must consume the bytes that were checked. The loader now
reads the bounded file once into owned memory. It calculates the CRC
over that memory, then opens a read stream over the same allocation.
No later filesystem write can change those owned bytes.

The checksum gate over that allocation is:

```c
static int snapshot_checksum_matches(const char *snapshot, size_t size)
{
    if (size < 3 * sizeof(int32_t))
        return 0;

    size_t checksum_at = size - sizeof(int32_t);
    int32_t stored;
    uint32_t computed =
        ~crc32_update(0xFFFFFFFFu,
                      (const unsigned char *)snapshot, checksum_at);

    memcpy(&stored, snapshot + checksum_at, sizeof stored);
    return computed == (uint32_t)stored;
}
```

The size check proves room for magic, version, and the final checksum.
The final four bytes begin at `checksum_at`. `crc32_update` visits every
earlier byte in the allocation, and `memcpy` reads the stored native
integer without assuming that its address meets an integer alignment.
The comparison is between the resulting 32-bit patterns.

The file-sized allocation plus the stream over it is an **immutable
byte snapshot** for this load. Immutable here means that the loader
never changes it and later changes to the path cannot reach it. It does
not mean the original file has become read-only.

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

The source first calculates both demands without allocating. This
source-exact helper uses Chapter 10's
[model memory
report](10-memory-planning.md#the-public-memory-report-is-an-independent-calculation):

```c
static int checkpoint_layout(ModelConfig cfg, CheckpointLayout *layout)
{
    CheckpointLayout candidate = {0};
    size_t fixed_bytes = checkpoint_fixed_bytes();

    candidate.parameter_floats = model_parameter_float_count(cfg);
    if (!model_memory_requirements(cfg, &candidate.memory)
        || candidate.parameter_floats > SIZE_MAX / sizeof(float))
        return 0;

    candidate.parameter_bytes =
        candidate.parameter_floats * sizeof(float);
    if (fixed_bytes > MODEL_MAX_CHECKPOINT_FILE_BYTES
        || (size_t)cfg.vocab_size
           > MODEL_MAX_CHECKPOINT_FILE_BYTES - fixed_bytes
        || candidate.parameter_bytes
           > MODEL_MAX_CHECKPOINT_FILE_BYTES - fixed_bytes
            - (size_t)cfg.vocab_size)
        return 0;

    candidate.file_bytes =
        fixed_bytes + (size_t)cfg.vocab_size + candidate.parameter_bytes;
    if (candidate.memory.total_bytes
        > MODEL_MAX_CHECKPOINT_RESIDENT_BYTES)
        return 0;
    *layout = candidate;
    return 1;
}
```

`candidate` keeps a failed calculation from publishing a partial
layout. The parameter count and Chapter 10 report are both derived from
`cfg`. The division check proves that converting the float count to
bytes cannot wrap.

The ordered subtractions prove that fixed fields, alphabet, and
parameter values fit the file ceiling. Only after those checks does the
code add the three pieces into `file_bytes`. The separate resident
check rejects the 4,213-byte hostile request because its model report
exceeds one GiB. The final assignment publishes a complete layout.

Loading now owns a file-sized snapshot at the same time as the model.
Checking each owner against one GiB separately would allow their sum to
cross the policy:

```text
snapshot allocation       300 MiB
model-owned memory        800 MiB
separate checks              pass
combined resident use    1,100 MiB
combined check               fail
```

Chapter 2's bounded reader adds one byte for its terminating zero, even
though the binary parser does not consume that byte. The source-exact
combined check counts it:

```c
static int snapshot_fits_resident_limit(
    const LoadCandidate *candidate, const CheckpointLayout *layout)
{
    if (candidate->snapshot_size == SIZE_MAX)
        return 0;

    size_t allocated_snapshot = candidate->snapshot_size + 1;

    return allocated_snapshot <= MODEL_MAX_CHECKPOINT_RESIDENT_BYTES
        && layout->memory.total_bytes
           <= MODEL_MAX_CHECKPOINT_RESIDENT_BYTES - allocated_snapshot;
}
```

The first check makes `snapshot_size + 1` safe. The final comparison is
written as subtraction, so adding the two allocations cannot wrap.
That exact sum is the load-side **checkpoint resource preflight**:

```text
(file bytes + terminating byte) + model-owned bytes <= 1 GiB
```

Here `model-owned bytes` means the four payload families in Chapter
10's report: parameter buffers, value arena, gradient arena, and cached
ids. The bound is exact for those families and the snapshot allocation.
Small structures, allocator metadata, and the construction RNG remain
outside it, as Chapter 10 records.

Overflow is failure, not a smaller allocation. Save checks the model
and file layout. Load additionally counts the immutable snapshot before
`model_new`. A locally constructed model can be legal under the public
geometry rules yet deliberately unsupported by this checkpoint policy.

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

Construct a neighboring file:

```text
trained.bin                 visible complete model A
.tiny-agenc-7b2c...         separate temporary B under construction
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

That promise covers readers while the save runs. A process with the
same filesystem authority can also rename entries in the destination
directory or change the existing file's metadata. The caller must
exclude both kinds of concurrent change during the save.

There is another name race to close. Suppose the save checks
`runs/trained.bin`, then an unrelated operation replaces `runs` before
the temporary file is created. Looking up the path again could inspect
one directory and write into another.

Split the path once into a parent and one final name. Open the parent
directory, keep its descriptor, and perform later lookups with
`fstatat`, `openat`, `renameat`, and `unlinkat` relative to that same
descriptor:

```text
"runs/trained.bin"
        |
        +-> open "runs" once -> directory descriptor 5
        |
        +-> use name "trained.bin" relative to descriptor 5
```

The directory descriptor, rather than a repeated text path lookup, is
the stable reference for the transaction. This is an **anchored parent
directory**.

Empty final names and the special names `.` and `..` do not identify a
checkpoint file, so path splitting rejects them. `fstatat` asks what
the final name itself selects without following an indirect entry:

```text
name is absent                  allow creation
name selects a regular file     allow replacement
name selects anything else      reject
lookup fails for another reason reject
```

Only the operating system's not-found result means absent. Permission
and I/O failures are not mistaken for permission to create.

When a target exists, `openat` uses `O_NOFOLLOW`, then `fstat` proves
that the opened descriptor still has the device and inode numbers
recorded by `fstatat`. A changed object, indirect entry, directory,
device, pipe, or socket fails before a temporary file is committed. An
absent destination remains allowed.

The temporary name does not borrow user-controlled destination text.
`getrandom` supplies 16 bytes, and `format_temp_name` writes their 32
hexadecimal digits after `.tiny-agenc-`. `openat` uses
`O_CREAT | O_EXCL`, so creation succeeds only if that exact name is
absent. A collision draws a fresh name; any other failure stops.

Requested mode `0600` is an upper bound during creation: the process
mask can remove bits, and a default directory ACL can affect the
resulting access record. It requests no access beyond owner read and
write. The later `fchmod` makes the final mode bits exactly `0600`.

Before the first descriptor becomes a C stream, `fcntl` with
`F_DUPFD_CLOEXEC` makes a second descriptor for the same temporary
inode. The first descriptor carries the payload and closes after file
sync. The duplicate stays open through commit or cleanup and stays out
of an executed child process.

`fstat` records the duplicate's device and inode and proves it is
regular. Keeping that descriptor open prevents the inode number from
being released and reused while the name is checked. Commit and cleanup
act on the temporary name only when a fresh `fstatat` finds that same
identity.

Linux provides no rename-or-unlink operation conditioned on an inode
number. A same-authority writer could still replace the random name
after the check and before the name operation. The 128 random bits keep
the name out of ordinary collisions; they do not widen the concurrency
contract above.

### Preserve the old file's access rules

Replacing a directory entry selects a new inode. Even when the payload
is correct, creating that inode with mode `0600` can discard rules
attached to the old one.

Start with the familiar permission bits:

```text
owner     read write execute
group     read write execute
other     read write execute
special   set-user-ID set-group-ID sticky
```

The first nine use mask `0777`. Including the three special bits gives
mask `07777`. Tiny AgenC records all twelve, plus the numeric owner and
group IDs.

Some files grant named users or groups access that those twelve bits
cannot express. Linux stores that rule as a separate access record.
The record is a POSIX **access-control list**, or **ACL**. Tiny AgenC
reads and writes it through libacl's `acl_get_fd` and `acl_set_fd`
instead of treating its binary encoding as an ordinary attribute.

A file can also carry named byte strings such as `user.review`,
`security.capability`, or `security.selinux`. The value may contain
zero bytes or may have length zero. These records are **extended
attributes**, shortened to **xattrs**.

The save manages every xattr reported by `flistxattr` except
`system.posix_acl_access`, which libacl owns. It captures names and
exact byte counts, applies them after ownership, removes unmatched
attributes from the temporary inode, then reads everything back.

Some attributes describe an inode's content or use a different ACL
model. Copying them to new bytes would make a false promise. The
source-exact rejection list is:

```c
static int xattr_is_unsupported(const char *name)
{
    return strcmp(name, "system.posix_acl_default") == 0
        || strcmp(name, "system.nfs4_acl") == 0
        || strcmp(name, "system.nfs4acl") == 0
        || strcmp(name, "trusted.nfs4_acl") == 0
        || strcmp(name, "security.ima") == 0
        || strcmp(name, "security.evm") == 0;
}
```

The first name belongs on directories, not a regular checkpoint. The
next three represent NFSv4 ACL data that libacl's POSIX access-ACL
comparison does not preserve. IMA and EVM values can authenticate or
protect the old inode's bytes and metadata. Encountering any listed
name stops before replacement.

Ordinary captured xattrs include Linux capability and SELinux labels
when the filesystem reports them. Applying ownership or security
metadata can require privilege. Failure is not downgraded into silent
metadata loss.

This source-exact helper applies the captured metadata:

```c
static int apply_metadata(int descriptor,
                          const CheckpointMetadata *metadata)
{
    if (fchown(descriptor, metadata->uid, metadata->gid) != 0
        || acl_set_fd(descriptor, metadata->access_acl) != 0
        || fchmod(descriptor, metadata->mode) != 0
        || apply_saved_xattrs(descriptor, metadata) != 0
        || remove_unmatched_xattrs(descriptor, metadata) != 0)
        return -1;
    return metadata_matches(descriptor, metadata) ? 0 : -1;
}
```

The order matters. `fchown` can clear set-user-ID and set-group-ID bits,
so ownership is applied first. `acl_set_fd` can adjust permission bits,
so `fchmod` restores the exact `07777` mask after it. Then xattrs are
set and any unrecorded xattr on the new inode is removed.

`metadata_matches` rereads UID, GID, all twelve mode bits, the access
ACL, the complete managed xattr name set, every xattr length, and every
value byte. A mismatch returns failure while the old destination is
still in place.

### Synchronize bytes, then the name

The payload, metadata, and directory entry become ready at different
times. Work through this order:

```text
probe parent-directory synchronization before creating a temporary
write header, vocabulary, parameters, checksum
apply and verify old metadata when replacing a file
synchronize the temporary file
close the temporary file
recheck the old destination's identity and metadata
rename the temporary file
synchronize and close the parent directory
```

The first directory sync proves that this filesystem accepts the
operation before the save creates or replaces anything. It does not
confirm a future rename, so the final directory sync remains required.

If writing or metadata work fails, the old name still selects A.
Synchronizing the temporary before close asks the filesystem to persist
B's bytes and the metadata needed to retrieve them. The source-exact
stage is:

```c
static int prepare_temporary_checkpoint(
    SaveTransaction *transaction, const Model *m, const Tokenizer *tk)
{
    if (write_checkpoint(transaction->stream, m, tk) != 0)
        return -1;

    int descriptor = fileno(transaction->stream);

    if (descriptor < 0)
        return -1;
    if ((transaction->path.target_exists
         && apply_metadata(descriptor, &transaction->metadata) != 0)
        || (!transaction->path.target_exists
            && fchmod(descriptor, 0600) != 0))
        return -1;
    if (fsync(descriptor) != 0)
        return -1;
    return transaction_close_stream(transaction);
}
```

`write_checkpoint` performs the TAGC v1 writes and final C-stream
flush. `fileno` retrieves the descriptor. Existing metadata is applied
and verified before `fsync`. For a new destination, `fchmod` forces
mode `0600` after creation so the process mask or an inherited default
access rule cannot leave different mode bits. A sync or close failure
prevents rename.

The original file is checked once more just before commit. For an
existing destination, rename replaces only the same regular inode whose
metadata was captured. For an initially absent destination,
`RENAME_NOREPLACE` refuses to overwrite a file that appeared during the
save. The source-exact commit is:

```c
static int commit_temporary_checkpoint(SaveTransaction *transaction)
{
    int renamed;

    if (close_original_destination(transaction) != 0
        || !path_matches_temporary_file(transaction,
                                        transaction->temp_name))
        return -1;
    if (transaction->path.target_exists) {
        renamed =
            renameat(transaction->path.directory, transaction->temp_name,
                     transaction->path.directory,
                     transaction->path.name);
    } else {
        renamed =
            renameat2(transaction->path.directory, transaction->temp_name,
                      transaction->path.directory,
                      transaction->path.name, RENAME_NOREPLACE);
    }
    if (renamed != 0)
        return -1;
    transaction->renamed = 1;
    transaction->temp_name[0] = '\0';
    return 0;
}
```

`close_original_destination` first compares the open old descriptor's
metadata and current destination identity before closing it.
`path_matches_temporary_file` then proves the temporary name still
selects the transaction's recorded inode immediately before rename.
Both rename calls use the already opened directory for source and
destination. After a successful rename, clearing `temp_name` prevents
cleanup from unlinking a name that no longer exists.

The rename has changed visible state, but file contents and the
directory's name-to-file mapping are separate persistent state:

```text
temporary B contents synchronized        yes
rename trained.bin from A to B            completed while running
directory record for that rename          not synchronized yet
```

The save now calls `fsync` on the open parent directory after rename,
then closes it. That final sync is what confirms the new mapping.

**Durability** is the promise that acknowledged state survives a crash
or power loss on the documented GNU/Linux local-filesystem boundary.
Tiny AgenC claims it only after temporary-file sync, close, rename, and
parent-directory sync and close all succeed.

### Report what happened after rename

A two-way success/failure return loses information at one sharp point.
Suppose rename succeeds, then directory sync fails:

```text
trained.bin now names complete B      yes
crash recovery confirmed              no
old A can be restored safely          no
```

Returning an ordinary failure sounds as though A remains. Returning
ordinary success claims durability that was not confirmed. Rolling back
would require another rename whose own durability could fail.

The caller needs three outcomes:

```text
failure before rename      destination not replaced
rename and directory sync  replacement durable
rename, directory failure  replacement visible, durability unconfirmed
```

The public names for those constructed states are
`MODEL_SAVE_NOT_COMMITTED`, `MODEL_SAVE_DURABLE`, and
`MODEL_SAVE_COMMITTED_DURABILITY_UNCONFIRMED`.

This source-exact function keeps the commit point visible:

```c
ModelSaveResult model_save_durable(const Model *m, const Tokenizer *tk,
                                   const char *path)
{
    CheckpointLayout layout;

    if (m == NULL || tk == NULL || path == NULL
        || !checkpoint_layout(m->cfg, &layout)
        || model_parameter_count(m) != layout.parameter_floats
        || tokenizer_vocab_size(tk) != m->cfg.vocab_size)
        return MODEL_SAVE_NOT_COMMITTED;

    SaveTransaction transaction;
    ModelSaveResult result = MODEL_SAVE_NOT_COMMITTED;

    transaction_init(&transaction);
    if (prepare_transaction(&transaction, path) != 0
        || prepare_temporary_checkpoint(&transaction, m, tk) != 0
        || commit_temporary_checkpoint(&transaction) != 0)
        goto done;

    result = confirm_directory_update(&transaction);

done:
    transaction_cleanup(&transaction);
    return result;
}
```

Invalid pointers and unsupported geometry fail before transaction
setup. The registry scalar count must match the layout count, and the
tokenizer size must match `V`. `result` begins in the only correct
pre-rename state. Preparation opens the path and temporary file,
writing makes the candidate complete, and commit performs rename. Only
after commit can `confirm_directory_update` choose between durable and
visible-but-unconfirmed.

`transaction_cleanup` closes owned descriptors and streams, frees
captured metadata, and unlinks only this transaction's still-named
temporary file. After rename the temporary name has been cleared, so
cleanup never tries to undo the published checkpoint.

Existing callers can still use `model_save`. Its complete wrapper is:

```c
int model_save(const Model *m, const Tokenizer *tk, const char *path)
{
    return model_save_durable(m, tk, path) == MODEL_SAVE_DURABLE
        ? 0 : -1;
}
```

It returns zero only for confirmed durability. Its nonzero result does
not prove that replacement was avoided, because it also represents the
visible-but-unconfirmed state. A caller that must distinguish the
post-commit state uses `model_save_durable`.

## Validate before publishing

The loader must reject bad bytes without publishing half-built objects.
Its gates run in source order:

```text
bounded immutable snapshot and CRC
        |
magic, version, geometry
        |
exact TAGC v1 size and combined resident policy
        |
canonical tokenizer and matching V
        |
construct candidate model
        |
registry count and every finite parameter
        |
checksum field, clean end of snapshot, successful close
        |
publish Model and Tokenizer
```

**Predict:** if the last parameter is nonfinite after a candidate
tokenizer has been built, should the caller receive that tokenizer?

No. A late failure must free every candidate and leave the output
pointer `NULL`.

Start by making the owned snapshot. This complete function is
source-exact:

```c
static int load_snapshot(LoadCandidate *candidate, const char *path)
{
    if (path == NULL
        || file_slurp_bounded(path, MODEL_MAX_CHECKPOINT_FILE_BYTES,
                              &candidate->snapshot,
                              &candidate->snapshot_size)
           != FILE_SLURP_OK
        || !snapshot_checksum_matches(candidate->snapshot,
                                      candidate->snapshot_size))
        return -1;

    candidate->stream =
        fmemopen(candidate->snapshot, candidate->snapshot_size, "rb");
    return candidate->stream == NULL ? -1 : 0;
}
```

A null path fails before I/O. Chapter 2's bounded reader publishes a
complete allocation only when the file fits the outer ceiling and ends
cleanly. CRC is then calculated over that allocation. `fmemopen`
creates the ordinary `FILE *` parser interface over the same bytes.

No header field has influenced an allocation yet. If the checksum
fails, candidate cleanup frees the snapshot without constructing a
tokenizer or model.

`read_config` still owns the opening fields:

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

The next complete helper is source-exact:

```c
static int load_header_and_tokenizer(
    LoadCandidate *candidate, CheckpointLayout *layout)
{
    ModelConfig cfg;

    if (read_config(candidate->stream, &cfg) != 0
        || !checkpoint_layout(cfg, layout)
        || layout->file_bytes != candidate->snapshot_size
        || !snapshot_fits_resident_limit(candidate, layout))
        return -1;

    candidate->tokenizer = tokenizer_read(candidate->stream);
    if (candidate->tokenizer == NULL
        || tokenizer_vocab_size(candidate->tokenizer) != cfg.vocab_size)
        return -1;

    candidate->model = model_new(cfg, 0);
    return model_parameter_count(candidate->model)
        == layout->parameter_floats ? 0 : -1;
}
```

Header parsing comes first. `checkpoint_layout` derives the one valid
TAGC v1 size from those dimensions. Equality with `snapshot_size`
rejects truncation and trailing bytes before a tokenizer or model is
allocated. The combined resident check then proves room for both the
still-owned snapshot and the requested model.

Replay the tiny file at this equality:

```text
fixed fields                       40 bytes
alphabet                            2 bytes
parameter values                   88 bytes
layout file_bytes                 130 bytes
snapshot_size                     130 bytes
```

The two 130-byte results match. A 131st byte fails.

Only then does `tokenizer_read` reconstruct the canonical alphabet and
prove its count equals `V`. `model_new(cfg, 0)` constructs the candidate
model. It needs a seed because ordinary construction initializes
parameters. Every saved value will overwrite those initial values, so
the zero seed has no surviving effect on loaded parameters.

The registry count is compared with the scalar count used to derive the
file size. Loading then walks every registry entry through Chapter 8's
finite-value gate.

After the parameters, one checksum word must remain. It was already
compared against the snapshot prefix. The parser still has to consume
that field, prove that no byte follows it, distinguish clean EOF from a
stream error, and observe close failure. The complete source-exact
helper does all four:

```c
static int consume_checkpoint_trailer(LoadCandidate *candidate)
{
    int32_t checksum;

    if (read_i32(candidate->stream, &checksum) != 0
        || fgetc(candidate->stream) != EOF
        || ferror(candidate->stream))
        return -1;

    FILE *stream = candidate->stream;

    candidate->stream = NULL;
    return fclose(stream);
}
```

The stream pointer moves into a local before the candidate field is
cleared. That prevents later cleanup from closing it twice. The return
value of `fclose` is the stage result.

Now walk
[`model_load`](../src/checkpoint.c). This is the complete production
function:

```c
Model *model_load(Tokenizer **tk, const char *path)
{
    if (tk == NULL)
        return NULL;
    *tk = NULL;

    LoadCandidate candidate;
    CheckpointLayout layout;
    Model *model = NULL;

    load_candidate_init(&candidate);
    if (load_snapshot(&candidate, path) != 0
        || load_header_and_tokenizer(&candidate, &layout) != 0
        || load_checkpoint_parameters(&candidate) != 0
        || consume_checkpoint_trailer(&candidate) != 0)
        goto done;
    model = publish_load_candidate(&candidate, tk);

done:
    load_candidate_free(&candidate);
    return model;
}
```

A missing output pointer cannot receive ownership, so it fails first.
`*tk = NULL` establishes the failure result before reading the path.
The candidate begins with no owners, and the four named stages run in
order. C's `||` stops at the first failure.

Only the success path calls `publish_load_candidate`. That helper moves
the model and tokenizer out of the candidate and clears its pointers.
The shared cleanup then closes and frees everything still owned,
including the snapshot. The caller never receives a partial tokenizer
or model.

The combined preflight bounds the requested allocation; it cannot
promise that the machine currently has that memory available.
Allocation failure still follows the project's fatal allocator policy
rather than returning `NULL`. Construction also creates zeroed
parameter-gradient and moment buffers plus the planned arenas. The
activation-gradient arena still needs Chapter 12's explicit clear
before backward.

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

The focused [`labs/check13.c`](../labs/check13.c) witness makes six
checkpoint claims:

```text
a save reports confirmed durability and every parameter bit round-trips
an indirect non-regular destination is rejected before commit
invalid save input reports that no replacement was committed
one changed finite payload bit is rejected by CRC32
the legacy magic is rejected even after its CRC is recomputed
the complete 4,213-byte resource fixture is rejected before construction
```

Its 21 checks include fixture creation, seeking, writing, and closing.
It does not compare every configuration field or tokenizer byte. Its
successful-save check does not simulate a crash.

The broader integration witness also checks configuration,
every vocabulary id, loaded forward loss, a seeded generation replay,
unknown versions, invalid dimensions, malformed tokenizer order,
nonfinite values, truncation, trailing data, an oversized file, and
preservation of the old destination after a handled nonfinite save.

The dedicated fault executable makes 108 checks. A restrictive process
mask removes every requested creation bit, then the explicit `fchmod`
restores mode `0600` and the resulting checkpoint loads. Replacement
preserves all mode bits, UID, GID, a named access ACL, an empty xattr,
and a binary xattr. Indirect entries, FIFOs, and directories are
rejected.

Its linker-wrapped pre-commit failures cover directory-sync probing,
payload flush, file sync and close, owner, mode, ACL, xattr, metadata
reread, and rename. Each reports `MODEL_SAVE_NOT_COMMITTED` and leaves
the old destination bytes. Final directory-sync and directory-close
failures occur after rename, leave a loadable new checkpoint, and
report `MODEL_SAVE_COMMITTED_DURABILITY_UNCONFIRMED`.

A temporary-name swap after stream close proves that the retained
identity descriptor prevents commit and that cleanup leaves the foreign
replacement alone. A registry-count mismatch also fails before commit.
On load, one fixture changes the source path after the bounded snapshot
and still parses the checked bytes. A forced `fmemopen` failure
publishes neither model nor tokenizer.

One final fixture freezes the writer result from before this refactor.
It uses `V=T=C=H=L=B=1`, fills its 20 parameter scalars with a fixed
sequence, and requires a 121-byte file:

```text
fixed fields       40 bytes
alphabet            1 byte
20 parameter f32s  80 bytes
                  ---------
                   121 bytes
```

On the reference little-endian IEEE binary32 representation, a 64-bit
FNV-1a fingerprint of all 121 bytes must equal
`0xdf6959f32448a235`, the value recorded from commit `af73d2d`. This is
a regression witness that the new transaction machinery preserved the
pre-refactor TAGC v1 writer result on that platform.

The witnesses do not cause an actual process crash or power loss. They
do not prove CRC collision resistance, authentication, behavior on an
undocumented filesystem, or preservation of privileged security
attributes unavailable to the test user. They do not support a writer
with the same authority concurrently changing the destination or its
directory. The golden writer check is skipped on another native integer
order or float representation. Its fingerprint can collide, does not
make the native format portable, and is not authentication. Those
limits are part of the promise, not footnotes outside it.

The focused and broader integration witnesses build corrupt fixtures
with a copy of the production CRC loop. The golden writer fingerprint
also covers the stored CRC word, but no executable isolates the named
CRC variant with an independent standard known-answer input. The worked
CRC values in this chapter were checked independently.

## Build checkpoint: make the handoff complete

**Build.** Implement `checkpoint.c`. Begin with the format constants
and the source-exact CRC32 update. Add the bounded prefix pass, checksum
append, and immutable checksum-first load snapshot. Reuse Chapter 3's
tokenizer serializer, Chapter 9's canonical parameter registry, and
Chapter 10's checked memory report. Count the snapshot and model
together under the resident ceiling.

Implement save around one opened parent directory. Accept an absent or
regular destination, create an exclusive random neighbor, and preserve
UID, GID, `07777` mode bits, the libacl access ACL, and managed xattrs
when replacing. Reject unsupported attribute models before rename.
Write TAGC v1, apply and verify metadata, synchronize and close the
temporary file, recheck the destination, rename, then synchronize and
close the parent directory.

Return the three commit states without attempting post-rename rollback.
Keep candidate tokenizer and model ownership inside the loader until
every check passes. Do not serialize gradients, optimizer moments,
arenas, or cached forward state.

**Verify.**

```sh
make -C labs check-13
# answer key
make -C labs WORK=../src check-13
```

**Expected.**

```text
check-13: all 21 checkpoint checks passed
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
  report before `model_new`, and count its bounded snapshot in the same
  resident ceiling.
- **Weights load into the wrong objects:** writer and reader must walk
  the same Chapter 9 registry order. The tied head has no second copy.
- **A failed save damages the old destination:** never open the
  destination for the new payload. Work relative to one opened parent,
  write and synchronize beside it, close, then rename.
- **Replacement drops access rules:** capture and verify UID, GID, all
  `07777` mode bits, the libacl access ACL, and complete managed xattrs
  before commit.
- **A directory-sync failure says nothing was written:** rename already
  published the new file. Return
  `MODEL_SAVE_COMMITTED_DURABILITY_UNCONFIRMED`.
- **A failed load leaves a tokenizer pointer:** set `*tk = NULL` at
  entry and publish ownership only after the final gate.
- **Loaded training immediately diverges:** version 1 restores learned
  values for generation, not optimizer state or RNG position.

The model can now survive its process. Chapter 14 makes the process
report failures to a person without exposing these internal ownership
paths.

---

[Previous: Wiring the Model Backward](12-wiring-the-model-backward.md) | [Contents](README.md) | [Next: The Command Line Is a Boundary](14-the-command-line.md)
