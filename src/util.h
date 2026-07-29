/*
 * util.h -- the ground floor: error-checked allocation, fatal errors,
 * whole-file reads, binary integer I/O, and a wall clock.
 *
 * Allocation failure is the one error nothing in a training run can
 * recover from, so the `e' wrappers (for "error-checked") die instead
 * of returning NULL and spare every caller a test that could never
 * usefully fail.  Everything else reports failure through its return
 * value and lets the caller decide.
 */
#ifndef TINY_AGENC_UTIL_H
#define TINY_AGENC_UTIL_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef enum {
    FILE_SLURP_OK,
    FILE_SLURP_IO_ERROR,
    FILE_SLURP_TOO_LARGE,
} FileSlurpStatus;

void          *emalloc(size_t size);
void          *ecalloc(size_t count, size_t size);
_Noreturn void die(const char *format, ...);

/* Read a whole file into a fresh NUL-terminated buffer the caller
 * frees; NULL on failure.  The bounded form checks the file length
 * before allocating or reading and initializes `text` to NULL and
 * `size` to zero on failure.  Binary i32 I/O returns 0 on success. */
char  *file_slurp(const char *path, size_t *size);
FileSlurpStatus file_slurp_bounded(const char *path, size_t maximum,
                                   char **text, size_t *size);
int    write_i32(FILE *stream, int32_t value);
int    read_i32(FILE *stream, int32_t *value);

double time_seconds(void);   /* monotonic, for timing training steps */

#endif
