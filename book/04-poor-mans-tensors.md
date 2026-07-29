# Chapter 4: Poor Man's Tensors

Chapter 0 trained a model and printed a falling
[loss](05-forward-pass.md#cross-entropy-keeping-score), the numeric
grade Chapter 5 will construct. Chapter 1 drew the grids of numbers
that carried one batch through that model. Chapter 3 produced the
integer token rows that enter those grids. The next chapter will
perform the arithmetic, but the C functions need a way to receive a
grid before they can work on it.

A pointer alone cannot do that job.

The title reuses Chapter 1's
[`tensor`](01-the-map.md#rows-columns-and-channels), one name
for a shaped collection of numbers. Tiny AgenC's implementation in
this chapter handles the two-dimensional float case.

## A pointer forgets the grid

Put six floats in one array:

```c
float storage6[6] = {
    10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f,
};
```

A `float *` can point at the first `10.0f`. The
[Chapter 1 pointer](01-the-map.md#an-address-instead-of-exposed-fields)
records an
address, not an arrangement. These two grids begin at the same address
and contain the same six values:

```text
2 rows by 3 columns          3 rows by 2 columns

10  20  30                  10  20
40  50  60                  30  40
                            50  60
```

**Predict:** if a function receives only `storage6`, can it decide
whether the value `30` belongs to row 0 or row 1?

No. In the left grid, `30` is at row 0, column 2. In the right grid,
it is at row 1, column 0. The address does not distinguish those
readings.

One hypothetical interface could pass three separate arguments:

```c
operate(storage6, 2, 3);
```

That convention leaves three pieces which every caller must keep
together and in the right order. A returned address would lose the
shape again. We need one small value containing:

```text
address of the first float
number of rows
number of columns
```

Now the six floats have one unambiguous description:

```text
address ──► storage6[0]
rows         2
columns      3
```

The description is not another array of six floats. It is three facts
about the existing array.

## Put the shape beside the address

C's `struct`, introduced in
[Chapter 1](01-the-map.md#an-address-instead-of-exposed-fields),
groups named
fields into one value. Here is the complete description from
[`mat.h`](../src/mat.h):

```c
typedef struct {
    float *vals;
    int    rows;
    int    cols;
} Mat;
```

Read it from the inside out. `vals` stores the address of the first
float. `rows` and `cols` store the shape, always rows first. The
`typedef` applies
[Chapter 2's type-alias
rule](02-foundations.md#read-bytes-without-trusting-the-file):
it gives this otherwise unnamed struct the type name `Mat`. No name
appears between `struct` and `{`, so `struct Mat` would not be another
valid spelling.

The six-float description now has a name. A `Mat` is a float pointer
paired with a two-dimensional shape. It is the small public
[value type](01-the-map.md#when-copying-the-description-is-useful)
previewed in
Chapter 1.

The four functions in `mat.h` are defined in the header itself.
Repeating an ordinary externally visible function definition in every
C file would create competing definitions. Giving each C file its own
private helper applies [Chapter 2's `static` rule](02-foundations.md)
and avoids that conflict. The `inline` keyword suggests that replacing
a call with the helper body may be useful. The compiler decides
whether to do that, and it may make the same optimization without the
keyword. C spells the combination `static inline`; `static` is the
part that gives each C file its own helper.

The first helper fills the three fields:

```c
static inline Mat mat_make(float *vals, int rows, int cols)
{
    Mat m = { vals, rows, cols };

    return m;
}
```

The signature receives one float pointer and two integers. Its result
type is `Mat`, so the function returns a complete three-field value.
Inside, the brace list initializes fields in declaration order:
`vals` goes into `m.vals`, `rows` into `m.rows`, and `cols` into
`m.cols`. This is a positional struct initializer.

Call it on the six floats:

```c
Mat grid = mat_make(storage6, 2, 3);
```

**Predict:** what are `grid.vals`, `grid.rows`, and `grid.cols` after
the call?

This call reuses Chapter 3's
[array-to-pointer conversion](03-data.md#give-each-byte-a-compact-number):
the array expression `storage6` supplies a pointer to its first
element, `&storage6[0]`.

The three fields are `&storage6[0]`, `2`, and `3`. The `.` operator
reuses [Chapter 2's struct member access](02-foundations.md) to select
a field from a struct value. Passing and returning `Mat` copies those
three fields. Copying `grid.vals` copies an address; it does not copy
the six floats behind that address.

`mat_make` performs no allocation and no validation. It preserves the
three values the caller supplies, including
[Chapter 2's null pointer](02-foundations.md) or an invalid dimension.
That narrow behavior is useful later when Chapter 10 first describes
shapes without touching their storage. It also means callers must
establish the shape contract before making a usable grid.

## Turn two coordinates into one offset

Choose the `2 × 3` reading of the six floats:

```text
             column
             0   1   2
row 0       [ 10  20  30 ]
row 1       [ 40  50  60 ]

flat index   0   1   2    3   4   5
```

Row 0 begins at flat index 0. Three values belong to that row, so row
1 begins at flat index 3.

**Predict:** where would row 2 begin if a third three-column row were
present?

It would begin at index 6. Each row advances by three floats:

```text
row 0 start = 0 * 3 = 0
row 1 start = 1 * 3 = 3
row 2 start = 2 * 3 = 6
```

To reach one cell, start at its row and then move by its column:

```text
flat index = row * column count + column
```

For row 1, column 2:

```text
1 * 3 + 2 = 5
storage6[5] = 60
```

Storing every column of row 0, then every column of row 1, is
**row-major storage**. The rows occupy consecutive parts of one array;
there are no separately allocated row arrays between them.

The distance between adjacent row starts is the **row stride**. In this
row-major layout, the stride is 3, exactly the column count. No separate
stride field is needed.

The source packages the row-start calculation as `mat_row`:

```c
static inline float *mat_row(Mat m, int row)
{
    assert(row >= 0 && row < m.rows);
    return m.vals + (size_t)row * (size_t)m.cols;
}
```

The function receives a copy of the `Mat` description and the desired
row. The `assert` applies
[Chapter 2's programmer-contract
check](02-foundations.md#pick-an-integer-without-favoring-one).
Both conditions must hold: the row cannot be negative, and it must be
smaller than the row count. A three-row matrix has rows 0, 1, and 2;
row 3 is already one past the end.

The return expression first converts `row` and `m.cols` to `size_t`,
then multiplies in that unsigned object-size type. Adding the result to
a `float *` uses the
[pointer arithmetic from Chapter 3](03-data.md#draw-every-legal-start):
adding 3 advances by three complete `float` objects, not three bytes.
The result points at the first float of the requested row.

The returned pointer can then use an ordinary array index for the
column:

```c
float value = mat_row(grid, 1)[2];
```

First `mat_row` reaches flat index 3. Then `[2]` advances two more
floats to index 5, whose value is `60.0f`.

`mat_row` checks the row only. It does not know whether `grid.vals`
still points at live storage, whether the storage really contains six
floats, or whether the later column index is below `grid.cols`. The
caller must keep those facts true.

## A flattened batch uses the same row rule

Chapter 1
[stacked a batch's sequences](01-the-map.md)
into one taller matrix with `R = B*T` rows. `Mat` does not need a
special batch field. The row count already carries that flattened
shape.

Use `B = 2` sequences, `T = 3` positions per sequence, and `C = 2`
channels per position. The matrix has `2 * 3 = 6` rows and 2 columns:

```text
matrix row   sequence   position   flat float offset
0            0          0          0
1            0          1          2
2            0          2          4
3            1          0          6
4            1          1          8
5            1          2         10
```

**Predict:** where does `mat_row(batch, 4)` point?

The matrix has two columns, so row 4 starts at `4 * 2 = 8`. The row is
sequence 1, position 1. Flattening changed the row labels, not the
row-major calculation. The answer-key Chapter 4 check includes this
exact offset.

## Count the visible floats

Some operations work row by row. Others apply the same calculation to
every visible float and need one total count.

**Predict:** how many floats are visible through shapes `2 × 3` and
`3 × 4`?

They expose `2 * 3 = 6` and `3 * 4 = 12` floats. The source calculation
is:

```c
static inline size_t mat_size(Mat m)
{
    return (size_t)m.rows * (size_t)m.cols;
}
```

Chapter 1 built
[`size_t`](01-the-map.md#build-checkpoint-specify-the-machine) as C's
unsigned type for object sizes. Each conversion happens before the
multiplication, so C does not first compute `rows * cols` in `int`.
The result counts floats, not bytes. A later allocation or copy must
still use [Chapter 2's `sizeof`
rule](02-foundations.md#a-block-of-bytes-and-one-owner) to multiply
that count by `sizeof(float)` or `sizeof *m.vals`.

These conversions solve one narrow problem. They do not make arbitrary
dimensions safe. A negative `int` becomes a large unsigned value, and
an enormous positive product can exceed what `size_t` represents.
`mat_size` does not detect either case. Tiny AgenC validates model
dimensions. [Chapter 10's checked memory
arithmetic](10-memory-planning.md#check-arithmetic-before-performing-it)
rejects a sum or product before `size_t` could wrap and before the
descriptions reach allocation code. The local precondition remains:
`rows` and `cols` are valid nonnegative dimensions whose product fits.

The pointer does not participate in this calculation. A shape-only
description made with `mat_make(NULL, 2, 3)` still has size 6. Calling
`mat_row` on that description would attempt pointer arithmetic without
backing storage and is not allowed.

## Expose fewer rows without copying

Use a fresh array so this worked example is independent of the
six-float grid:

```c
float storage12[12] = {
     0.0f, 1.0f,  2.0f,  3.0f,
     4.0f, 5.0f,  6.0f,  7.0f,
     8.0f, 9.0f, 10.0f, 11.0f,
};
```

Its twelve initializer values fill indexes 0 through 11 in order.
Describe them as three four-column rows:

```text
             column
             0  1  2  3
row 0       [ 0  1  2  3 ]
row 1       [ 4  5  6  7 ]
row 2       [ 8  9 10 11 ]
```

The full description is:

```c
Mat full = mat_make(storage12, 3, 4);
```

Now imagine that one call needs only the first two rows. Copying the
first eight floats would require new storage, a copy loop, and a new
owner responsible for releasing the copy. None of those jobs changes
the desired arithmetic. The existing first eight floats are already
in the right order.

Make a second three-field description:

```text
description   first address   rows   columns   visible floats
full          storage12       3      4         12
prefix        storage12       2      4          8
```

Both descriptions begin at the same address. Only the visible row
count changes.

**Predict:** if `prefix` writes row 1, column 2, which flat storage
index changes?

The row starts at `1 * 4 = 4`, and the column adds 2, so index 6
changes. `full` and `prefix` both reach that same float:

```c
mat_row(prefix, 1)[2] = 99.0f;
```

After the write, the owned array is:

```text
0  1  2  3   4  5  99  7   8  9  10  11
```

Nothing was copied. A small description that borrows existing storage
this way is a **matrix view**. It is non-owning because the description
does not decide when the storage's lifetime ends. This particular
form, which keeps the address and columns while selecting a leading
row count no greater than the parent, is a **leading-row view**.

The source helper is:

```c
static inline Mat mat_first_rows(Mat m, int rows)
{
    assert(rows >= 0 && rows <= m.rows);
    return mat_make(m.vals, rows, m.cols);
}
```

The assertion accepts row counts from zero through the parent's full
row count. The upper comparison uses `<=` because asking for all rows
is valid. The return line preserves `m.vals` and `m.cols`, substitutes
the requested row count, and returns the new description by value.
The original `m` was itself copied into the function, so its row count
does not change.

**Predict:** what fields will the two boundary calls return?

```text
mat_first_rows(full, 0)
mat_first_rows(full, 3)
```

The first returns 0 rows, 4 columns, and the same address. The
zero-row view has size zero, and no row is legal to pass to `mat_row`.
The second returns 3 rows, 4 columns, and the same address. It
describes the same visible shape as its parent but remains a separate
three-field value.

The storage still has one owner:

```text
storage12 owns: [0 1 2 3  4 5 99 7  8 9 10 11]
                 ▲
                 ├── full.vals
                 └── prefix.vals
```

Here the local array variable `storage12` owns the floats until its
enclosing function returns. The local variables `full` and `prefix`
hold only descriptions. If the function returns, both pointers become
invalid. Calling `free` on this local array would itself be invalid
because no allocation produced it.

The same borrowing rule applies when storage does come from Chapter
2's [allocator](02-foundations.md#a-block-of-bytes-and-one-owner).
Only that allocation's owner calls `free`, exactly once. A view never
extends either kind of lifetime, and a view helper never frees
`view.vals`.

## Changing width needs a new description

Reducing the row count preserves the row stride. That is correct for
the prefix above because both descriptions have four columns.
Changing the logical width is a different operation.

Start with another array and describe its sixteen float slots at
maximum capacity as `4 × 4`:

```c
float storage16[16];
Mat maximum = mat_make(storage16, 4, 4);
```

This example reads no stored value, so the uninitialized floats do not
affect it. Only their addresses and the two shapes matter:

```text
maximum 4 x 4

row 0 starts at  0
row 1 starts at  4
row 2 starts at  8
row 3 starts at 12
```

A smaller call needs a packed `2 × 2` grid in the first four slots:

```text
active 2 x 2

row 0 starts at 0
row 1 starts at 2
```

**Predict:** does `mat_first_rows(maximum, 2)` describe that active
grid?

No. It produces `2 × 4`, exposes eight floats, and starts row 1 at
offset 4. It shortened the height while preserving the four-float
stride.

The required description supplies both current dimensions:

```c
Mat active = mat_make(maximum.vals, 2, 2);
```

Now `mat_size(active)` is 4, and `mat_row(active, 1)` advances by two
floats. The values did not move. Giving the same storage a new logical
shape without moving its values is a **reshape view**.

The caller must prove that the new shape fits the backing
[capacity](03-data.md#encode-without-inventing-a-token). `Mat` stores
no capacity field and cannot prove that `2 * 2` slots exist. It also
cannot prove that the first four stored values were packed for this
reading.

Chapter 10 applies the same distinction to a packed collection of
square work tables whose current width follows the current text length.
Most of that chapter's descriptions need a leading-row view. These
tables need a shorter prefix with a new width, so their description
uses `mat_make`.

## What assertions can protect

The two assertions catch bad requests relative to the dimensions
already recorded in a description. For a `3 × 4` `full`, predict each
result before reading the answers:

```text
mat_row(full, -1)
mat_row(full, 3)
mat_first_rows(full, -1)
mat_first_rows(full, 4)
```

The two `mat_row` calls fail their row-bound assertion. Rows below zero
and rows equal to the row count are invalid. The two
`mat_first_rows` calls fail their view-bound assertion. A negative
visible count and a count above the parent's rows are invalid.

An assertion is an internal programmer-contract check, not input
validation. If `full.rows` falsely says 30, `mat_row(full, 20)` accepts
that lie even when only twelve floats exist. If `full.vals` points to
storage whose lifetime ended, the assertion cannot detect that either.
The helper checks the fact it owns: the requested row against the
recorded row count.

C removes `assert` checks when a build defines the `NDEBUG`
[macro from Chapter 2](02-foundations.md). Tiny AgenC's normal
optimized and lab flags do not define it, so these checks remain active
in the project's supported commands. Correctness still cannot depend
on an assertion repairing an invalid shape.

The complete caller contract is:

```text
rows and columns are nonnegative
rows * columns fits in size_t
vals is live before any element access
backing storage has at least rows * columns floats
every column access is in 0 .. cols - 1
```

`mat_make` records whatever it receives. A description intended for
element access must satisfy all five facts. `mat_size` needs the first
two. `mat_row` checks the requested row, then relies on all the storage
facts. `mat_first_rows` checks only that its new row count is a prefix.

## What a Mat never owns or remembers

A `Mat` never allocates, frees, or copies its float payload. It copies
three descriptor fields whenever it is assigned, passed, or returned
by value. The distinction matters:

```text
Mat copy                 float payload
--------                 -------------
pointer copied ────────► same storage
rows copied
columns copied
```

Changing `copy.rows` changes that description only. Writing through
`copy.vals` changes shared storage and is visible through every other
description of the same floats.

The type is deliberately narrow. Every element is a `float`. Every
shape has two dimensions. The row stride is always the column count.
There is no capacity field, no owner field, and no separate read-only
form. The `float *` in every `Mat` permits a write. When a function
must only read a matrix, that promise belongs to the function's
contract; this type does not record it.

A `Mat` also records no operation history. After a function writes a
value, the description cannot say which inputs or calculation produced
it. Our network is not arbitrary. It is one fixed pipeline, so later
chapters write the reverse calculations directly in the source:
[Chapter 6](06-backprop-by-hand.md) constructs the arithmetic,
[Chapter 7](07-trust-but-verify.md) builds an independent checker,
and [Chapter 12](12-wiring-the-model-backward.md) wires the reverse
path.
That checker prints its exact current check count when run; this
chapter does not freeze an unsupported count in prose.

The public functions in [`ops.c`](../src/ops.c) pass their
matrix-shaped operands as `Mat` values. Private one-dimensional
helpers still receive raw pointers. Passing the descriptions by value
keeps shape beside the matrix operands without copying their payloads.

## The complete header

The helper definitions live in a header, and several other headers can
reach `mat.h` through different include paths. Without a guard, one C
file could see the `Mat` definition twice and reject the redefinition.

The preprocessor can remember a marker after the first inclusion:

```text
marker absent  -> define marker and read the header body
marker present -> skip the header body
```

The `#ifndef`, `#define`, and final `#endif` lines implementing that
choice are a **header guard**. Chapter 2 introduced the
[preprocessor's replacement
step](02-foundations.md).
Here the marker controls whether a region is included at all.

You have now walked every non-comment line that implements the type and
its helpers. Here is the complete [`mat.h`](../src/mat.h) in source
order:

```c
/*
 * mat.h -- a matrix is a pointer and a shape.
 *
 * Mat does not own memory; it is a view, three fields passed by value,
 * the C equivalent of a slice.  Whoever allocates the floats decides
 * their lifetime; Mat just gives the math a shape to work with.
 * Storage is row-major and contiguous: row r of an R x C matrix is the
 * C consecutive floats starting at vals[r * C].
 */
#ifndef TINY_AGENC_MAT_H
#define TINY_AGENC_MAT_H

#include <assert.h>
#include <stddef.h>

typedef struct {
    float *vals;
    int    rows;
    int    cols;
} Mat;

static inline Mat mat_make(float *vals, int rows, int cols)
{
    Mat m = { vals, rows, cols };

    return m;
}

static inline float *mat_row(Mat m, int row)
{
    assert(row >= 0 && row < m.rows);
    return m.vals + (size_t)row * (size_t)m.cols;
}

static inline size_t mat_size(Mat m)
{
    return (size_t)m.rows * (size_t)m.cols;
}

/* A shorter view of the same storage: batches smaller than the maximum
 * use the leading rows of a full-size buffer. */
static inline Mat mat_first_rows(Mat m, int rows)
{
    assert(rows >= 0 && rows <= m.rows);
    return mat_make(m.vals, rows, m.cols);
}

#endif
```

The opening comment uses `slice` as an analogy from other programming
languages. The needed fact is in the same comment: a `Mat` borrows its
float storage while carrying two shape dimensions. This book calls
that description a matrix view.

`<assert.h>` declares `assert`. `<stddef.h>` declares `size_t`. The
guard prevents repeated definitions within one C file, while
`static inline` lets the four complete helper bodies remain in the
header. There is no `mat.c`: together, the type and helpers are all of
`mat.h`'s implementation.

## Build checkpoint: walk the rows

**Build.** Implement `mat_make`, `mat_size`, `mat_row`, and
`mat_first_rows` in `labs/work/mat.h`. Preserve the pointer and both
dimensions, calculate visible element counts in `size_t`, assert the
two row bounds, and return leading-row descriptions that share their
parent's storage. Do not add allocation or ownership to `Mat`.
Chapter 8 builds storage for learned values; Chapter 10 builds storage
for the model's temporary grids.

Before running the check, predict these facts:

```text
3 x 4 visible size                 ?
row 2 start offset                 ?
2 x 4 prefix visible size          ?
prefix row 1, column 2 offset      ?
mat_row(full, -1)                  pass or assert?
mat_row(full, 3)                   pass or assert?
mat_first_rows(full, -1)           pass or assert?
mat_first_rows(full, 4)            pass or assert?
```

**Verify.**

```sh
make -C labs check-04
# answer key: make check-mat
```

**Expected.** The learner check reports that all matrix-view checks
passed. It verifies pointer and shape preservation, row-major offsets,
visible sizes, shared writes, and four invalid-bound assertion cases.
The answer-key target adds a known column value and the Chapter 1
`B*T` flattened-row ordering. It does not repeat the learner check's
assertion-death cases.

**Common failures.**

- Using `row * m.rows` makes non-square matrices walk to the wrong
  offset.
- Treating a pointer offset as bytes instead of `float` objects lands
  inside the wrong value.
- Multiplying in `int` and converting afterward allows signed overflow
  before `size_t` receives the result.
- Allocating inside `mat_first_rows` creates a copy and a second owner.
- Changing `cols` in `mat_first_rows` changes the row stride instead of
  exposing a row prefix.
- Claiming more elements than the backing capacity makes a valid-looking
  description unsafe to access.
- Checking `row <= m.rows` admits the one-past row.
- Freeing `m.vals` from a helper ends storage still borrowed by other
  descriptions.

---

[Previous: Data](03-data.md) | [Contents](README.md) | [Next: The Forward Pass](05-forward-pass.md)
