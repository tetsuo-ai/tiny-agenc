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

static void check_utilities(void)
{
    char path[128];
    int written = snprintf(path, sizeof path,
                           "/tmp/tiny-agenc-check02-%ld.bin",
                           (long)getpid());

    expect(written > 0 && (size_t)written < sizeof path,
           "the file-slurp fixture path fits");

    FILE *fixture = fopen(path, "wb");
    static const unsigned char BYTES[] = { 'A', 0, 'Z' };

    expect(fixture != NULL, "the file-slurp fixture opens");
    if (fixture != NULL) {
        expect(fwrite(BYTES, 1, sizeof BYTES, fixture) == sizeof BYTES,
               "the file-slurp fixture is written");
        expect(fclose(fixture) == 0, "the file-slurp fixture closes");

        size_t size = 0;
        char *slurped = file_slurp(path, &size);

        expect(slurped != NULL, "file_slurp reads an existing file");
        if (slurped != NULL) {
            expect(size == sizeof BYTES,
                   "file_slurp reports the binary byte count");
            expect(memcmp(slurped, BYTES, sizeof BYTES) == 0,
                   "file_slurp preserves embedded zero bytes");
            expect(slurped[size] == '\0',
                   "file_slurp appends a sentinel zero byte");
        }
        free(slurped);

        char *bounded_text = (char *)1;
        size_t bounded_size = 99;

        expect(file_slurp_bounded(path, sizeof BYTES - 1,
                                  &bounded_text, &bounded_size)
                   == FILE_SLURP_TOO_LARGE,
               "bounded slurp rejects a file before an oversized read");
        expect(bounded_text == NULL && bounded_size == 0,
               "bounded slurp clears outputs after size rejection");
        expect(file_slurp_bounded(path, sizeof BYTES,
                                  &bounded_text, &bounded_size)
                   == FILE_SLURP_OK,
               "bounded slurp accepts its exact byte ceiling");
        expect(bounded_text != NULL && bounded_size == sizeof BYTES
               && memcmp(bounded_text, BYTES, sizeof BYTES) == 0,
               "bounded slurp preserves accepted bytes");
        free(bounded_text);
    }
    remove(path);

    size_t absent_size = 99;

    expect(file_slurp(path, &absent_size) == NULL,
           "file_slurp reports a missing file without exiting");

    char *absent_text = (char *)1;

    expect(file_slurp_bounded(path, 10, &absent_text, &absent_size)
               == FILE_SLURP_IO_ERROR
           && absent_text == NULL && absent_size == 0,
           "bounded slurp distinguishes I/O failure and clears outputs");

    FILE *binary = tmpfile();

    expect(binary != NULL, "the binary-I/O fixture opens");
    if (binary != NULL) {
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

    double first_time = time_seconds();
    double second_time = time_seconds();

    expect(first_time > 0.0 && second_time >= first_time,
           "the monotonic clock never moves backward");
}

int main(void)
{
    check_utilities();

    int *plain = emalloc(4 * sizeof *plain);
    int *zeroed = ecalloc(4, sizeof *zeroed);

    expect(plain != NULL, "emalloc returns storage");
    expect(zeroed != NULL, "ecalloc returns storage");
    for (int i = 0; i < 4; i++)
        expect(zeroed[i] == 0, "ecalloc clears every byte");
    free(plain);
    free(zeroed);

    static const uint32_t UNIFORM_BITS[] = {
        0x3F42F57Bu, 0x3ED60F88u, 0x3EE56F64u, 0x3E8842A6u,
        0x3F75AF5Eu, 0x3ED17D6Cu, 0x3F4BC731u, 0x3F55EFC7u,
    };
    static const int BELOW_TEN[] = { 7, 4, 4, 2, 9, 4, 7, 8 };

    Rng *uniform = rng_new(42);

    for (size_t i = 0; i < sizeof UNIFORM_BITS / sizeof UNIFORM_BITS[0]; i++)
        expect(float_bits(rng_uniform(uniform)) == UNIFORM_BITS[i],
               "uniform draw matches the PCG32 known answer");
    rng_free(uniform);

    Rng *bounded = rng_new(42);

    for (size_t i = 0; i < sizeof BELOW_TEN / sizeof BELOW_TEN[0]; i++)
        expect(rng_below(bounded, 10) == BELOW_TEN[i],
               "bounded draw matches the known answer");
    rng_free(bounded);

    Rng *first = rng_new(1337);
    Rng *second = rng_new(1337);
    Rng *different = rng_new(1338);
    static const float GAUSSIAN_ANSWERS[] = {
        -0.782266140f, 1.062945604f, 0.196520776f, 0.952153027f,
        -0.718591213f, -1.716835856f, 0.370696634f, 0.374907821f,
    };
    int changed = 0;

    for (int i = 0; i < 16; i++) {
        float a_value = rng_gaussian(first);
        float b_value = rng_gaussian(second);
        float c_value = rng_gaussian(different);
        uint32_t a = float_bits(a_value);
        uint32_t b = float_bits(b_value);
        uint32_t c = float_bits(c_value);

        expect(a == b, "equal seeds replay Gaussian draws");
        if (i < (int)(sizeof GAUSSIAN_ANSWERS
                      / sizeof GAUSSIAN_ANSWERS[0]))
            expect(fabsf(a_value - GAUSSIAN_ANSWERS[i]) <= 1e-6f,
                   "Gaussian draw matches the known answer");
        changed |= a != c;
    }
    expect(changed, "a different seed changes the stream");
    rng_free(first);
    rng_free(second);
    rng_free(different);

    Rng *range = rng_new(99);

    for (int i = 0; i < 256; i++) {
        int draw = rng_below(range, 17);

        expect(draw >= 0 && draw < 17, "bounded draw stays inside its range");
    }
    rng_free(range);

    if (failures != 0) {
        fprintf(stderr, "check-02: %d of %d checks failed\n", failures, checks);
        return EXIT_FAILURE;
    }
    printf("check-02: all %d foundation checks passed\n", checks);
    return EXIT_SUCCESS;
}
