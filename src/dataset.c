#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "dataset.h"
#include "util.h"

struct Dataset {
    int   *tokens;
    size_t token_count;
};

static const size_t DATASET_POLICY_MAX_BYTES =
    (size_t)256 * (size_t)1024 * (size_t)1024;

size_t dataset_max_text_bytes(void)
{
    size_t allocation_limit = SIZE_MAX / sizeof(int);
    size_t limit = DATASET_POLICY_MAX_BYTES;

    if (limit > allocation_limit)
        limit = allocation_limit;
    if (limit > (size_t)INT_MAX)
        limit = (size_t)INT_MAX;
    return limit;
}

Dataset *dataset_new(const Tokenizer *tk, const char *text, size_t length)
{
    if (length > dataset_max_text_bytes())
        return NULL;

    Dataset *ds = emalloc(sizeof *ds);
    size_t token_bytes = length * sizeof *ds->tokens;

    /* malloc(0) may return NULL even though the empty Dataset is valid. */
    ds->tokens = emalloc(token_bytes == 0 ? sizeof *ds->tokens : token_bytes);
    ds->token_count = tokenizer_encode(tk, ds->tokens, text, length);
    return ds;
}

size_t dataset_token_count(const Dataset *ds)
{
    return ds->token_count;
}

void dataset_batch(const Dataset *ds, Rng *rng, int *inputs, int *targets,
                   int batch_size, int block_size)
{
    /* A run starting at s uses tokens s .. s+block_size as input and
     * target, so the last legal start is token_count - block_size - 1. */
    assert(ds->token_count >= (size_t)block_size + 1);

    int last_start = (int)(ds->token_count - (size_t)block_size - 1);

    for (int row = 0; row < batch_size; row++) {
        const int *run = ds->tokens + rng_below(rng, last_start + 1);

        memcpy(inputs + row * block_size, run, (size_t)block_size * sizeof *inputs);
        memcpy(targets + row * block_size, run + 1, (size_t)block_size * sizeof *targets);
    }
}

void dataset_free(Dataset *ds)
{
    free(ds->tokens);
    free(ds);
}
