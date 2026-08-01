/*
 * Produce one deterministic train/validation assignment per record.
 * The caller remains responsible for parsing and writing whole records.
 */
#include <errno.h>
#include <ctype.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#include "rng.h"

enum {
    DECIMAL_RADIX = 10,
    MINIMUM_COUNT = 1,
    PROGRAM_ARGUMENTS = 4,
    RECORD_COUNT_ARGUMENT = 1,
    VALIDATION_COUNT_ARGUMENT = 2,
    SEED_ARGUMENT = 3,
};

static int parse_count(const char *text, const char *label);
static unsigned long long parse_seed(const char *text);
static void fill_identity_order(int *order, int count);
static void shuffle_order(int *order, int count, Rng *rng);
static int print_assignments(const int *order, int record_count,
                             int validation_count);
int main(int argc, char **argv);

static int parse_count(const char *text, const char *label)
{
    char *end;
    long value;

    errno = 0;
    value = strtol(text, &end, DECIMAL_RADIX);
    if (errno != 0 || *text == '\0' || *end != '\0'
        || value < MINIMUM_COUNT || value > INT_MAX) {
        fprintf(stderr, "split-order: invalid %s: %s\n", label, text);
        exit(EXIT_FAILURE);
    }
    return (int)value;
}

static unsigned long long parse_seed(const char *text)
{
    char *end;
    unsigned long long value;

    errno = 0;
    value = strtoull(text, &end, DECIMAL_RADIX);
    if (errno != 0 || !isdigit((unsigned char)text[0]) || *end != '\0') {
        fprintf(stderr, "split-order: invalid seed: %s\n", text);
        exit(EXIT_FAILURE);
    }
    return value;
}

static void fill_identity_order(int *order, int count)
{
    for (int i = 0; i < count; i++)
        order[i] = i;
}

static void shuffle_order(int *order, int count, Rng *rng)
{
    for (int i = count - 1; i > 0; i--) {
        int other = rng_below(rng, i + 1);
        int held = order[i];

        order[i] = order[other];
        order[other] = held;
    }
}

static int print_assignments(const int *order, int record_count,
                             int validation_count)
{
    unsigned char *validation =
        calloc((size_t)record_count, sizeof *validation);

    if (validation == NULL)
        return -1;

    for (int i = 0; i < validation_count; i++)
        validation[order[i]] = 1;
    for (int i = 0; i < record_count; i++)
        puts(validation[i] ? "validation" : "train");

    free(validation);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc != PROGRAM_ARGUMENTS) {
        fputs("usage: split-order RECORDS VALIDATION_RECORDS SEED\n",
              stderr);
        return EXIT_FAILURE;
    }

    int record_count =
        parse_count(argv[RECORD_COUNT_ARGUMENT], "record count");
    int validation_count =
        parse_count(argv[VALIDATION_COUNT_ARGUMENT], "validation count");

    if (validation_count >= record_count) {
        fputs("split-order: validation count must be smaller than records\n",
              stderr);
        return EXIT_FAILURE;
    }

    int *order = malloc((size_t)record_count * sizeof *order);

    if (order == NULL) {
        fputs("split-order: out of memory\n", stderr);
        return EXIT_FAILURE;
    }

    fill_identity_order(order, record_count);

    Rng *rng = rng_new(parse_seed(argv[SEED_ARGUMENT]));

    shuffle_order(order, record_count, rng);
    rng_free(rng);
    int printed = print_assignments(order, record_count, validation_count);

    free(order);
    if (printed != 0) {
        fputs("split-order: out of memory\n", stderr);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
