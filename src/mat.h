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

static inline Mat mat_make(float *vals, int rows, int cols);
static inline float *mat_row(Mat m, int row);
static inline size_t mat_size(Mat m);
static inline Mat mat_first_rows(Mat m, int rows);

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
