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
};

static const unsigned long long VALIDATION_SEED =
    TINY_AGENC_VALIDATION_SEED;

static double transition_loss(
    const unsigned long long pair[BYTE_VALUES][BYTE_VALUES],
    const unsigned long long previous[BYTE_VALUES],
    int vocab_size, unsigned char a, unsigned char b)
{
    double probability =
        ((double)pair[a][b] + 1.0)
        / ((double)previous[a] + (double)vocab_size);

    return -log(probability);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fputs("usage: bigram TRAIN_TEXT VALIDATION_TEXT\n", stderr);
        return EXIT_FAILURE;
    }

    size_t train_length;
    size_t validation_length;
    char  *train = file_slurp(argv[1], &train_length);
    char  *validation = file_slurp(argv[2], &validation_length);

    if (train == NULL || validation == NULL)
        die("bigram cannot read both corpora");

    unsigned long long pair[BYTE_VALUES][BYTE_VALUES] = { 0 };
    unsigned long long previous[BYTE_VALUES] = { 0 };
    int seen[BYTE_VALUES] = { 0 };
    int vocab_size = 0;

    for (size_t i = 0; i < train_length; i++) {
        unsigned char byte = (unsigned char)train[i];

        if (!seen[byte]) {
            seen[byte] = 1;
            vocab_size++;
        }
    }
    for (size_t i = 0; i + 1 < train_length; i++) {
        unsigned char a = (unsigned char)train[i];
        unsigned char b = (unsigned char)train[i + 1];

        pair[a][b]++;
        previous[a]++;
    }

    double total_loss = 0.0;
    size_t predictions = 0;

    for (size_t i = 0; i + 1 < validation_length; i++) {
        unsigned char a = (unsigned char)validation[i];
        unsigned char b = (unsigned char)validation[i + 1];

        if (!seen[a] || !seen[b])
            die("validation corpus contains a byte absent from training");

        total_loss += transition_loss(pair, previous, vocab_size, a, b);
        predictions++;
    }

    if (predictions == 0)
        die("validation corpus has no next-byte predictions");
    if (validation_length <= VALIDATION_BLOCK_SIZE
        || validation_length > (size_t)INT_MAX)
        die("validation corpus cannot supply the default fixed windows");

    double full_loss = total_loss / (double)predictions;
    size_t last_start = validation_length - VALIDATION_BLOCK_SIZE - 1;
    Rng *rng = rng_new(VALIDATION_SEED);
    double sampled_total = 0.0;
    size_t sampled_predictions = 0;

    for (int batch = 0; batch < VALIDATION_BATCHES; batch++) {
        for (int row = 0; row < VALIDATION_BATCH_SIZE; row++) {
            size_t start = (size_t)rng_below(
                rng, (int)(last_start + 1));

            for (int time = 0; time < VALIDATION_BLOCK_SIZE; time++) {
                unsigned char a = (unsigned char)validation[start + (size_t)time];
                unsigned char b =
                    (unsigned char)validation[start + (size_t)time + 1];

                sampled_total +=
                    transition_loss(pair, previous, vocab_size, a, b);
                sampled_predictions++;
            }
        }
    }
    rng_free(rng);

    double sampled_loss = sampled_total / (double)sampled_predictions;

    printf("bigram: vocab %d | validation predictions %zu\n",
           vocab_size, predictions);
    printf("bigram: uniform loss %.6f | full-file add-one loss %.6f | perplexity %.4f\n",
           log((double)vocab_size), full_loss, exp(full_loss));
    printf("bigram: fixed-window seed %llu | %d x %d x %d = %zu predictions\n",
           VALIDATION_SEED, VALIDATION_BATCHES, VALIDATION_BATCH_SIZE,
           VALIDATION_BLOCK_SIZE, sampled_predictions);
    printf("bigram: fixed-window add-one loss %.6f | perplexity %.4f\n",
           sampled_loss, exp(sampled_loss));

    free(validation);
    free(train);
    return EXIT_SUCCESS;
}
