/*
 * rng.h -- deterministic pseudo-randomness.
 *
 * Repeatable pseudo-random draws make experiments debuggable.  With the
 * same executable and floating-point environment, the same seed produces
 * the same initial weights and batches.  This module wraps a PCG32
 * generator (O'Neill, 2014) behind an opaque handle: uniform draws for
 * choosing, gaussian draws for initializing.
 */
#ifndef TINY_AGENC_RNG_H
#define TINY_AGENC_RNG_H

typedef struct Rng Rng;

Rng   *rng_new(unsigned long long seed);
float  rng_uniform(Rng *rng);            /* uniform in [0, 1) */
float  rng_gaussian(Rng *rng);           /* normal: mean 0, stddev 1 */
int    rng_below(Rng *rng, int bound);   /* uniform in [0, bound); bound >= 1 */
void   rng_free(Rng *rng);

#endif
