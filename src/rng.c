#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "rng.h"
#include "util.h"

/*
 * PCG32 is a 64-bit linear congruential step whose state is scrambled
 * down to 32 well-mixed output bits by a permutation.  The multiplier,
 * increment, and permutation shifts below are the algorithm's published
 * definition (O'Neill, pcg32_random_r), not tunable choices.
 */
static const uint64_t PCG32_MULTIPLIER = 6364136223846793005ULL;
static const uint64_t PCG32_INCREMENT  = 1442695040888963407ULL;

enum {
    PCG32_FOLD_SHIFT     = 18,   /* xor the high half onto the low */
    PCG32_OUT_SHIFT      = 27,   /* keep the best 32 of 64 bits */
    PCG32_ROTATION_SHIFT = 59,   /* top five bits pick the rotation */
    PCG32_OUTPUT_BITS    = 32,
    UNIFORM_BITS         = 24,   /* a float holds 24 significant bits */
};

static const float TWO_PI = 6.28318530717958647692f;
static const float BOX_MULLER_RADIUS_FACTOR = -2.0f;

struct Rng {
    uint64_t state;
    int      has_spare_gaussian;   /* Box-Muller yields two draws; bank one */
    float    spare_gaussian;
};

static uint32_t pcg32_next(Rng *rng);

static uint32_t pcg32_next(Rng *rng)
{
    uint64_t old = rng->state;

    rng->state = old * PCG32_MULTIPLIER + PCG32_INCREMENT;

    uint32_t folded   = (uint32_t)(((old >> PCG32_FOLD_SHIFT) ^ old) >> PCG32_OUT_SHIFT);
    uint32_t rotation = (uint32_t)(old >> PCG32_ROTATION_SHIFT);

    /* rotate the 32 output bits right by `rotation` */
    return (folded >> rotation)
         | (folded << ((PCG32_OUTPUT_BITS - rotation) & (PCG32_OUTPUT_BITS - 1)));
}

Rng *rng_new(unsigned long long seed)
{
    Rng *rng = emalloc(sizeof *rng);

    rng->state = 0;
    rng->has_spare_gaussian = 0;
    pcg32_next(rng);
    rng->state += seed;
    pcg32_next(rng);
    return rng;
}

float rng_uniform(Rng *rng)
{
    uint32_t bits = pcg32_next(rng) >> (PCG32_OUTPUT_BITS - UNIFORM_BITS);

    return (float)bits / (float)(1u << UNIFORM_BITS);
}

float rng_gaussian(Rng *rng)
{
    if (rng->has_spare_gaussian) {
        rng->has_spare_gaussian = 0;
        return rng->spare_gaussian;
    }

    /* Box-Muller: two uniform draws become two independent gaussians.
     * 1 - u keeps the logarithm's argument in (0, 1], never zero. */
    float radius =
        sqrtf(BOX_MULLER_RADIUS_FACTOR
              * logf(1.0f - rng_uniform(rng)));
    float angle  = TWO_PI * rng_uniform(rng);

    rng->spare_gaussian     = radius * sinf(angle);
    rng->has_spare_gaussian = 1;
    return radius * cosf(angle);
}

int rng_below(Rng *rng, int bound)
{
    assert(bound > 0);

    /*
     * Lemire's method: multiply a 32-bit draw by the range and keep the
     * high word.  Unless the range divides 2^32, multiply-high alone gives
     * some results one extra preimage.  Reject low-word values below the
     * threshold to make every result equiprobable.
     */
    uint32_t range = (uint32_t)bound;
    uint32_t threshold = (uint32_t)(-range) % range;
    uint32_t draw;
    uint32_t low;
    uint64_t product;

    do {
        draw    = pcg32_next(rng);
        product = (uint64_t)draw * (uint64_t)range;
        low     = (uint32_t)product;
    } while (low < threshold);

    return (int)(product >> PCG32_OUTPUT_BITS);
}

void rng_free(Rng *rng)
{
    free(rng);
}
