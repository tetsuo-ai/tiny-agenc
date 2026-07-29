#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L   /* clock_gettime, fseeko */

#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#include "util.h"

static const double NANOSECONDS_PER_SECOND = 1e9;

void *emalloc(size_t size)
{
    void *block = malloc(size);

    if (block == NULL)
        die("out of memory allocating %zu bytes", size);
    return block;
}

void *ecalloc(size_t count, size_t size)
{
    void *block = calloc(count, size);

    if (block == NULL)
        die("out of memory allocating %zu x %zu bytes", count, size);
    return block;
}

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

FileSlurpStatus file_slurp_bounded(const char *path, size_t maximum,
                                   char **text, size_t *size)
{
    if (text == NULL || size == NULL)
        return FILE_SLURP_IO_ERROR;
    *text = NULL;
    *size = 0;

    FILE *stream = fopen(path, "rb");

    if (stream == NULL)
        return FILE_SLURP_IO_ERROR;
    if (fseeko(stream, 0, SEEK_END) != 0) {
        fclose(stream);
        return FILE_SLURP_IO_ERROR;
    }

    off_t end = ftello(stream);

    if (end < 0 || fseeko(stream, 0, SEEK_SET) != 0) {
        fclose(stream);
        return FILE_SLURP_IO_ERROR;
    }
    if ((uintmax_t)end > (uintmax_t)maximum
        || (uintmax_t)end >= (uintmax_t)SIZE_MAX) {
        fclose(stream);
        return FILE_SLURP_TOO_LARGE;
    }

    char *contents = emalloc((size_t)end + 1);

    if (fread(contents, 1, (size_t)end, stream) != (size_t)end) {
        free(contents);
        fclose(stream);
        return FILE_SLURP_IO_ERROR;
    }
    fclose(stream);
    contents[(size_t)end] = '\0';
    *text = contents;
    *size = (size_t)end;
    return FILE_SLURP_OK;
}

char *file_slurp(const char *path, size_t *size)
{
    char *text;

    if (file_slurp_bounded(path, SIZE_MAX - 1, &text, size)
        != FILE_SLURP_OK)
        return NULL;
    return text;
}

int write_i32(FILE *stream, int32_t value)
{
    return fwrite(&value, sizeof value, 1, stream) == 1 ? 0 : -1;
}

int read_i32(FILE *stream, int32_t *value)
{
    return fread(value, sizeof *value, 1, stream) == 1 ? 0 : -1;
}

double time_seconds(void)
{
    struct timespec now;

    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec / NANOSECONDS_PER_SECOND;
}
