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

static int parse_count(const char *text, const char *label)
{
    char *end;
    long value;

    errno = 0;
    value = strtol(text, &end, 10);
    if (errno != 0 || *text == '\0' || *end != '\0'
        || value < 1 || value > INT_MAX) {
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
    value = strtoull(text, &end, 10);
    if (errno != 0 || !isdigit((unsigned char)text[0]) || *end != '\0') {
        fprintf(stderr, "split-order: invalid seed: %s\n", text);
        exit(EXIT_FAILURE);
    }
    return value;
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr,
                "usage: split-order RECORDS VALIDATION_RECORDS SEED\n");
        return EXIT_FAILURE;
    }

    int record_count = parse_count(argv[1], "record count");
    int validation_count = parse_count(argv[2], "validation count");

    if (validation_count >= record_count) {
        fprintf(stderr,
                "split-order: validation count must be smaller than records\n");
        return EXIT_FAILURE;
    }

    int *order = malloc((size_t)record_count * sizeof *order);
    unsigned char *validation =
        calloc((size_t)record_count, sizeof *validation);

    if (order == NULL || validation == NULL) {
        fprintf(stderr, "split-order: out of memory\n");
        free(order);
        free(validation);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < record_count; i++)
        order[i] = i;

    Rng *rng = rng_new(parse_seed(argv[3]));

    for (int i = record_count - 1; i > 0; i--) {
        int other = rng_below(rng, i + 1);
        int held = order[i];

        order[i] = order[other];
        order[other] = held;
    }
    rng_free(rng);

    for (int i = 0; i < validation_count; i++)
        validation[order[i]] = 1;
    for (int i = 0; i < record_count; i++)
        puts(validation[i] ? "validation" : "train");

    free(validation);
    free(order);
    return EXIT_SUCCESS;
}
