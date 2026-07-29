/*
 * evaluation.h -- the shared fixed-window comparison contract.
 *
 * The CLI and the matched bigram baseline include this file so a change
 * to the evidence windows cannot silently update only one side.
 */
#ifndef TINY_AGENC_EVALUATION_H
#define TINY_AGENC_EVALUATION_H

enum {
    TINY_AGENC_DEFAULT_BATCH_SIZE = 32,
    TINY_AGENC_DEFAULT_BLOCK_SIZE = 128,
    TINY_AGENC_VALIDATION_BATCHES = 4,
};

#define TINY_AGENC_DEFAULT_SEED           1337ULL
#define TINY_AGENC_SAMPLE_SEED_OFFSET        1ULL
#define TINY_AGENC_VALIDATION_SEED_OFFSET    2ULL
#define TINY_AGENC_VALIDATION_SEED \
    (TINY_AGENC_DEFAULT_SEED + TINY_AGENC_VALIDATION_SEED_OFFSET)

#endif
