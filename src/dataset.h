/*
 * dataset.h -- the corpus as a stream of training examples.
 *
 * A batch for next-character prediction is `batch_size` runs of
 * `block_size` consecutive token ids, plus the same runs shifted left
 * by one: the model reads inputs[t] and must predict
 * targets[t] == inputs[t + 1].  Runs start at positions drawn uniformly
 * at random, giving this demo a simple sampling-with-replacement stream.
 *
 * A Dataset is not bound to one block size; callers confirm the corpus
 * is long enough via dataset_token_count before batching.  Corpora are
 * capped deliberately so a source file cannot imply an unbounded token
 * allocation.
 */
#ifndef TINY_AGENC_DATASET_H
#define TINY_AGENC_DATASET_H

#include <stddef.h>

#include "rng.h"
#include "tokenizer.h"

typedef struct Dataset Dataset;

/* The source-file policy ceiling is 256 MiB, further bounded by the
 * largest one-int-per-byte token allocation representable on this
 * platform.  dataset_new returns NULL above that ceiling. */
size_t   dataset_max_text_bytes(void);
Dataset *dataset_new(const Tokenizer *tk, const char *text, size_t length);
size_t   dataset_token_count(const Dataset *ds);
void     dataset_batch(const Dataset *ds, Rng *rng, int *inputs, int *targets,
                       int batch_size, int block_size);
void     dataset_free(Dataset *ds);

#endif
