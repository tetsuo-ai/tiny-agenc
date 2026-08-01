/*
 * Chapter 2 witness: utilities and deterministic randomness.
 *
 * This file deliberately knows nothing about tokenizers or models. A
 * foundations implementation should not need future modules to link.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "rng.h"
#include "util.h"

static int checks;
static int failures;

enum {
    TEMPORARY_PATH_CAPACITY = 128,
    ABSENT_FILE_LIMIT = 10,
    ALLOCATION_ELEMENT_COUNT = 4,
    KNOWN_RNG_SEED = 42,
    RNG_BOUND = 10,
    GAUSSIAN_REPLAY_SEED = 1337,
    DIFFERENT_GAUSSIAN_SEED = 1338,
    GAUSSIAN_DRAW_COUNT = 16,
    RANGE_TEST_SEED = 99,
    RANGE_DRAW_COUNT = 256,
    RANGE_LIMIT = 17,
};

static void expect(int condition, const char *message);
static uint32_t float_bits(float value);
static int write_slurp_fixture(char *path, const unsigned char *bytes,
                               size_t count);
static void check_existing_file_slurp(
    const char *path, const unsigned char *bytes, size_t count);
static void check_empty_file_slurp(const char *path);
static void check_absent_file_slurp(const char *path);
static void check_changing_file_slurp(void);
static void check_binary_io(void);
static void check_utilities(void);
static void check_allocations(void);
static void check_uniform_rng(void);
static void check_gaussian_rng(void);
static void check_bounded_rng_range(void);
int main(void);

static void expect(int condition, const char *message)
{
    checks++;
    if (condition)
        return;
    fprintf(stderr, "check-02: %s\n", message);
    failures++;
}

static uint32_t float_bits(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof bits);
    return bits;
}

static int write_slurp_fixture(char *path, const unsigned char *bytes,
                               size_t count)
{
    int written = snprintf(path, TEMPORARY_PATH_CAPACITY,
                           "/tmp/tiny-agenc-check02-%ld.bin",
                           (long)getpid());

    expect(written > 0 && written < TEMPORARY_PATH_CAPACITY,
           "the file-slurp fixture path fits");
    if (written <= 0 || written >= TEMPORARY_PATH_CAPACITY)
        return -1;

    FILE *fixture = fopen(path, "wb");

    expect(fixture != NULL, "the file-slurp fixture opens");
    if (fixture == NULL)
        return -1;

    int wrote = fwrite(bytes, 1, count, fixture) == count;
    int closed = fclose(fixture) == 0;

    expect(wrote, "the file-slurp fixture is written");
    expect(closed, "the file-slurp fixture closes");
    return wrote && closed ? 0 : -1;
}

static void check_existing_file_slurp(
    const char *path, const unsigned char *bytes, size_t count)
{
    size_t size = 0;
    char *slurped = file_slurp(path, &size);

    expect(slurped != NULL, "file_slurp reads an existing file");
    if (slurped != NULL) {
        expect(size == count, "file_slurp reports the binary byte count");
        expect(memcmp(slurped, bytes, count) == 0,
               "file_slurp preserves embedded zero bytes");
        expect(slurped[size] == '\0',
               "file_slurp appends a sentinel zero byte");
    }
    free(slurped);

    char sentinel;
    char *bounded_text = &sentinel;
    size_t bounded_size = SIZE_MAX;

    expect(file_slurp_bounded(path, count - 1,
                              &bounded_text, &bounded_size)
               == FILE_SLURP_TOO_LARGE,
           "bounded slurp rejects a file before an oversized read");
    expect(bounded_text == NULL && bounded_size == 0,
           "bounded slurp clears outputs after size rejection");
    expect(file_slurp_bounded(path, count, &bounded_text, &bounded_size)
               == FILE_SLURP_OK,
           "bounded slurp accepts its exact byte ceiling");
    expect(bounded_text != NULL && bounded_size == count
               && memcmp(bounded_text, bytes, count) == 0,
           "bounded slurp preserves accepted bytes");
    free(bounded_text);
}

static void check_empty_file_slurp(const char *path)
{
    FILE *fixture = fopen(path, "wb");

    expect(fixture != NULL, "the empty-file fixture opens");
    if (fixture == NULL)
        return;
    expect(fclose(fixture) == 0, "the empty-file fixture closes");

    char sentinel;
    char *bounded_text = &sentinel;
    size_t bounded_size = SIZE_MAX;

    expect(file_slurp_bounded(path, 0, &bounded_text,
                              &bounded_size) == FILE_SLURP_OK,
           "bounded slurp accepts an empty exact-ceiling file");
    expect(bounded_text != NULL && bounded_size == 0
               && bounded_text[0] == '\0',
           "empty slurp publishes its sentinel-only buffer");
    free(bounded_text);

    bounded_size = SIZE_MAX;
    expect(file_slurp_bounded(path, 0, NULL, &bounded_size)
               == FILE_SLURP_IO_ERROR
               && bounded_size == SIZE_MAX,
           "bounded slurp rejects a missing output address");

    bounded_text = &sentinel;
    bounded_size = SIZE_MAX;
    expect(file_slurp_bounded(NULL, 0, &bounded_text, &bounded_size)
               == FILE_SLURP_IO_ERROR
               && bounded_text == NULL && bounded_size == 0,
           "bounded slurp rejects a missing path and clears outputs");
}

static void check_absent_file_slurp(const char *path)
{
    size_t absent_size = SIZE_MAX;

    expect(file_slurp(path, &absent_size) == NULL,
           "file_slurp reports a missing file without exiting");

    char sentinel;
    char *absent_text = &sentinel;

    expect(file_slurp_bounded(path, ABSENT_FILE_LIMIT,
                              &absent_text, &absent_size)
               == FILE_SLURP_IO_ERROR
               && absent_text == NULL && absent_size == 0,
           "bounded slurp distinguishes I/O failure and clears outputs");
}

static void check_changing_file_slurp(void)
{
    static const char PATH[] = "/proc/self/cmdline";
    FILE *changing = fopen(PATH, "rb");

    if (changing == NULL)
        return;

    int measured_zero = fseek(changing, 0, SEEK_END) == 0
                     && ftell(changing) == 0
                     && fseek(changing, 0, SEEK_SET) == 0;
    int has_bytes = measured_zero && fgetc(changing) != EOF;

    fclose(changing);
    if (!has_bytes)
        return;

    char sentinel;
    char *changing_text = &sentinel;
    size_t changing_size = SIZE_MAX;

    expect(file_slurp_bounded(PATH, 0,
                              &changing_text, &changing_size)
               == FILE_SLURP_IO_ERROR,
           "bounded slurp rejects bytes beyond the measured end");
    expect(changing_text == NULL && changing_size == 0,
           "changed-file rejection leaves no partial output");
}

static void check_binary_io(void)
{
    FILE *binary = tmpfile();

    expect(binary != NULL, "the binary-I/O fixture opens");
    if (binary == NULL)
        return;

    expect(write_i32(binary, INT32_MIN) == 0
               && write_i32(binary, INT32_MAX) == 0,
           "binary i32 writes report success");
    rewind(binary);

    int32_t first = 0;
    int32_t second = 0;

    expect(read_i32(binary, &first) == 0 && first == INT32_MIN
               && read_i32(binary, &second) == 0 && second == INT32_MAX,
           "binary i32 values round-trip exactly");
    expect(read_i32(binary, &first) != 0,
           "binary i32 reads reject a short stream");
    fclose(binary);
}

static void check_utilities(void)
{
    static const unsigned char BYTES[] = { 'A', 0, 'Z' };
    char path[TEMPORARY_PATH_CAPACITY];

    if (write_slurp_fixture(path, BYTES, sizeof BYTES) == 0) {
        check_existing_file_slurp(path, BYTES, sizeof BYTES);
        check_empty_file_slurp(path);
    }
    remove(path);
    check_absent_file_slurp(path);
    check_changing_file_slurp();
    check_binary_io();

    double first_time = time_seconds();
    double second_time = time_seconds();

    expect(first_time > 0.0 && second_time >= first_time,
           "the monotonic clock never moves backward");
}

static void check_allocations(void)
{
    int *plain = emalloc(ALLOCATION_ELEMENT_COUNT * sizeof *plain);
    int *zeroed = ecalloc(ALLOCATION_ELEMENT_COUNT, sizeof *zeroed);

    expect(plain != NULL, "emalloc returns storage");
    expect(zeroed != NULL, "ecalloc returns storage");
    for (int i = 0; i < ALLOCATION_ELEMENT_COUNT; i++)
        expect(zeroed[i] == 0, "ecalloc clears every byte");
    free(plain);
    free(zeroed);
}

static void check_uniform_rng(void)
{
    static const uint32_t UNIFORM_BITS[] = {
        0x3F42F57Bu, 0x3ED60F88u, 0x3EE56F64u, 0x3E8842A6u,
        0x3F75AF5Eu, 0x3ED17D6Cu, 0x3F4BC731u, 0x3F55EFC7u,
    };
    static const int BELOW_TEN[] = { 7, 4, 4, 2, 9, 4, 7, 8 };
    Rng *uniform = rng_new(KNOWN_RNG_SEED);

    for (size_t i = 0; i < sizeof UNIFORM_BITS / sizeof UNIFORM_BITS[0]; i++)
        expect(float_bits(rng_uniform(uniform)) == UNIFORM_BITS[i],
               "uniform draw matches the PCG32 known answer");
    rng_free(uniform);

    Rng *bounded = rng_new(KNOWN_RNG_SEED);

    for (size_t i = 0; i < sizeof BELOW_TEN / sizeof BELOW_TEN[0]; i++)
        expect(rng_below(bounded, RNG_BOUND) == BELOW_TEN[i],
               "bounded draw matches the known answer");
    rng_free(bounded);
}

static void check_gaussian_rng(void)
{
    static const float GAUSSIAN_ANSWERS[] = {
        -0.782266140f, 1.062945604f, 0.196520776f, 0.952153027f,
        -0.718591213f, -1.716835856f, 0.370696634f, 0.374907821f,
    };
    static const float TOLERANCE = 1e-6f;
    Rng *first = rng_new(GAUSSIAN_REPLAY_SEED);
    Rng *second = rng_new(GAUSSIAN_REPLAY_SEED);
    Rng *different = rng_new(DIFFERENT_GAUSSIAN_SEED);
    int changed = 0;

    for (int i = 0; i < GAUSSIAN_DRAW_COUNT; i++) {
        float first_value = rng_gaussian(first);
        float second_value = rng_gaussian(second);
        float different_value = rng_gaussian(different);

        expect(float_bits(first_value) == float_bits(second_value),
               "equal seeds replay Gaussian draws");
        if (i < (int)(sizeof GAUSSIAN_ANSWERS
                      / sizeof GAUSSIAN_ANSWERS[0]))
            expect(fabsf(first_value - GAUSSIAN_ANSWERS[i]) <= TOLERANCE,
                   "Gaussian draw matches the known answer");
        changed |= float_bits(first_value) != float_bits(different_value);
    }
    expect(changed, "a different seed changes the stream");
    rng_free(first);
    rng_free(second);
    rng_free(different);
}

static void check_bounded_rng_range(void)
{
    Rng *range = rng_new(RANGE_TEST_SEED);

    for (int i = 0; i < RANGE_DRAW_COUNT; i++) {
        int draw = rng_below(range, RANGE_LIMIT);

        expect(draw >= 0 && draw < RANGE_LIMIT,
               "bounded draw stays inside its range");
    }
    rng_free(range);
}

int main(void)
{
    check_utilities();
    check_allocations();
    check_uniform_rng();
    check_gaussian_rng();
    check_bounded_rng_range();

    if (failures != 0) {
        fprintf(stderr, "check-02: %d of %d checks failed\n", failures, checks);
        return EXIT_FAILURE;
    }
    printf("check-02: all %d foundation checks passed\n", checks);
    return EXIT_SUCCESS;
}
