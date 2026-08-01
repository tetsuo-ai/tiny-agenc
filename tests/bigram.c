#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "evaluation.h"
#include "rng.h"
#include "util.h"

enum {
    BYTE_VALUES = 256,
    VALIDATION_BATCHES = TINY_AGENC_VALIDATION_BATCHES,
    VALIDATION_BATCH_SIZE = TINY_AGENC_DEFAULT_BATCH_SIZE,
    VALIDATION_BLOCK_SIZE = TINY_AGENC_DEFAULT_BLOCK_SIZE,
    PROGRAM_ARGUMENTS = 3,
    TRAIN_PATH_ARGUMENT = 1,
    VALIDATION_PATH_ARGUMENT = 2,
    NEXT_BYTE_OFFSET = 1,
};

static const unsigned long long VALIDATION_SEED =
    TINY_AGENC_VALIDATION_SEED;
static const double ADD_ONE_PSEUDOCOUNT = 1.0;

typedef struct {
    unsigned long long pair[BYTE_VALUES][BYTE_VALUES];
    unsigned long long previous[BYTE_VALUES];
    int                seen[BYTE_VALUES];
    int                vocab_size;
} BigramCounts;

typedef struct {
    double loss;
    size_t predictions;
} LossMeasurement;

static double transition_loss(
    const unsigned long long pair[BYTE_VALUES][BYTE_VALUES],
    const unsigned long long previous[BYTE_VALUES], int vocab_size,
    unsigned char a, unsigned char b);
static void count_training_bigrams(BigramCounts *counts, const char *train,
                                   size_t train_length);
static LossMeasurement measure_full_validation(
    const BigramCounts *counts, const char *validation,
    size_t validation_length);
static void measure_validation_window(const BigramCounts *counts,
                                      const char *validation,
                                      size_t start,
                                      LossMeasurement *measurement);
static LossMeasurement measure_fixed_windows(
    const BigramCounts *counts, const char *validation,
    size_t validation_length);
static void print_measurements(const BigramCounts *counts,
                               LossMeasurement full,
                               LossMeasurement fixed);
int main(int argc, char **argv);

static double transition_loss(
    const unsigned long long pair[BYTE_VALUES][BYTE_VALUES],
    const unsigned long long previous[BYTE_VALUES],
    int vocab_size, unsigned char a, unsigned char b)
{
    double probability =
        ((double)pair[a][b] + ADD_ONE_PSEUDOCOUNT)
        / ((double)previous[a] + (double)vocab_size);

    return -log(probability);
}

static void count_training_bigrams(BigramCounts *counts, const char *train,
                                   size_t train_length)
{
    for (size_t i = 0; i < train_length; i++) {
        unsigned char byte = (unsigned char)train[i];

        if (counts->seen[byte])
            continue;
        counts->seen[byte] = 1;
        counts->vocab_size++;
    }
    for (size_t i = 0; i + NEXT_BYTE_OFFSET < train_length; i++) {
        unsigned char a = (unsigned char)train[i];
        unsigned char b =
            (unsigned char)train[i + NEXT_BYTE_OFFSET];

        counts->pair[a][b]++;
        counts->previous[a]++;
    }
}

static LossMeasurement measure_full_validation(
    const BigramCounts *counts, const char *validation,
    size_t validation_length)
{
    double total_loss = 0.0;
    size_t predictions = 0;

    for (size_t i = 0; i + NEXT_BYTE_OFFSET < validation_length; i++) {
        unsigned char a = (unsigned char)validation[i];
        unsigned char b =
            (unsigned char)validation[i + NEXT_BYTE_OFFSET];

        if (!counts->seen[a] || !counts->seen[b])
            die("validation corpus contains a byte absent from training");

        total_loss += transition_loss(counts->pair, counts->previous,
                                      counts->vocab_size, a, b);
        predictions++;
    }

    if (predictions == 0)
        die("validation corpus has no next-byte predictions");
    LossMeasurement measurement = {
        .loss = total_loss / (double)predictions,
        .predictions = predictions,
    };

    return measurement;
}

static void measure_validation_window(const BigramCounts *counts,
                                      const char *validation,
                                      size_t start,
                                      LossMeasurement *measurement)
{
    for (int time = 0; time < VALIDATION_BLOCK_SIZE; time++) {
        size_t at = start + (size_t)time;
        unsigned char previous = (unsigned char)validation[at];
        unsigned char next =
            (unsigned char)validation[at + NEXT_BYTE_OFFSET];

        measurement->loss +=
            transition_loss(counts->pair, counts->previous,
                            counts->vocab_size, previous, next);
        measurement->predictions++;
    }
}

static LossMeasurement measure_fixed_windows(
    const BigramCounts *counts, const char *validation,
    size_t validation_length)
{
    if (validation_length <= VALIDATION_BLOCK_SIZE
        || validation_length > (size_t)INT_MAX)
        die("validation corpus cannot supply the default fixed windows");

    size_t last_start =
        validation_length - VALIDATION_BLOCK_SIZE - NEXT_BYTE_OFFSET;
    Rng *rng = rng_new(VALIDATION_SEED);
    LossMeasurement measurement = { 0 };

    for (int batch = 0; batch < VALIDATION_BATCHES; batch++) {
        for (int row = 0; row < VALIDATION_BATCH_SIZE; row++) {
            size_t start = (size_t)rng_below(
                rng, (int)(last_start + NEXT_BYTE_OFFSET));

            measure_validation_window(counts, validation, start,
                                      &measurement);
        }
    }
    rng_free(rng);
    measurement.loss /= (double)measurement.predictions;
    return measurement;
}

static void print_measurements(const BigramCounts *counts,
                               LossMeasurement full,
                               LossMeasurement fixed)
{
    printf("bigram: vocab %d | validation predictions %zu\n",
           counts->vocab_size, full.predictions);
    printf("bigram: uniform loss %.6f | full-file add-one loss %.6f | perplexity %.4f\n",
           log((double)counts->vocab_size), full.loss, exp(full.loss));
    printf("bigram: fixed-window seed %llu | %d x %d x %d = %zu predictions\n",
           VALIDATION_SEED, VALIDATION_BATCHES, VALIDATION_BATCH_SIZE,
           VALIDATION_BLOCK_SIZE, fixed.predictions);
    printf("bigram: fixed-window add-one loss %.6f | perplexity %.4f\n",
           fixed.loss, exp(fixed.loss));
}

int main(int argc, char **argv)
{
    if (argc != PROGRAM_ARGUMENTS) {
        fputs("usage: bigram TRAIN_TEXT VALIDATION_TEXT\n", stderr);
        return EXIT_FAILURE;
    }

    size_t train_length;
    size_t validation_length;
    char *train = file_slurp(argv[TRAIN_PATH_ARGUMENT], &train_length);
    char *validation =
        file_slurp(argv[VALIDATION_PATH_ARGUMENT], &validation_length);

    if (train == NULL || validation == NULL)
        die("bigram cannot read both corpora");

    BigramCounts counts = { 0 };

    count_training_bigrams(&counts, train, train_length);
    LossMeasurement full =
        measure_full_validation(&counts, validation, validation_length);
    LossMeasurement fixed =
        measure_fixed_windows(&counts, validation, validation_length);

    print_measurements(&counts, full, fixed);

    free(validation);
    free(train);
    return EXIT_SUCCESS;
}
