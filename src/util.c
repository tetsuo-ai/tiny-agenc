#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L   /* clock_gettime, fseeko */

#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#include "util.h"

static const double NANOSECONDS_PER_SECOND = 1e9;

static FileSlurpStatus measure_file(FILE *stream, size_t maximum,
                                    size_t *length);
static int read_exact_file(FILE *stream, char *contents, size_t length);

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

static int read_exact_file(FILE *stream, char *contents, size_t length)
{
    if (fread(contents, 1, length, stream) != length)
        return -1;
    if (fgetc(stream) != EOF || ferror(stream))
        return -1;
    return 0;
}

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
