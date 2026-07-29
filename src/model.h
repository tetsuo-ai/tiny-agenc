/*
 * model.h -- the autoregressive transformer itself.
 *
 * A Model owns its parameters, its optimizer state, and every
 * activation buffer a training step touches, all allocated up front in
 * model_new; no training step allocates memory.  The compute path uses
 * learned token and position embeddings, layer_count pre-norm
 * transformer blocks, a final layernorm, and a language-model head
 * tied to the token embedding table.
 *
 * One training step, in call order:
 *
 *   model_zero_gradients(m);
 *   loss = model_forward(m, tokens, targets, batch, time);
 *   model_backward(m);
 *   if (model_step(m, opt, step) != 0) ...;
 *
 * model_forward with NULL targets is inference: it fills the logits
 * and returns 0.  model_backward replays the immediately preceding
 * forward pass, which must have been the with-targets kind (asserted):
 * an intervening inference pass invalidates the cache.
 */
#ifndef TINY_AGENC_MODEL_H
#define TINY_AGENC_MODEL_H

#include <stddef.h>

#include "param.h"
#include "rng.h"
#include "tokenizer.h"

typedef struct Model Model;

typedef struct {
    int vocab_size;
    int block_size;    /* longest sequence, in tokens */
    int d_model;       /* width of the residual stream */
    int head_count;
    int layer_count;
    int batch_size;    /* most sequences one forward pass can carry */
} ModelConfig;

/* The parameters as a flat list: the view the gradient checker walks.
 * The array belongs to the model. */
typedef struct {
    Param **params;
    int     count;
} ModelParams;

/* Buffer requirements derived without allocating.  The total includes
 * parameter values, gradients, AdamW moments, activations, activation
 * gradients, and cached token ids. */
typedef struct {
    size_t parameter_bytes;
    size_t activation_bytes;
    size_t gradient_bytes;
    size_t token_bytes;
    size_t total_bytes;
} ModelMemory;

/* Hard dimension and token-count ceilings for the 64-bit targets this
 * program supports.  Checkpoint loading adds a one-GiB buffer ceiling
 * before model construction, so a valid but hostile header cannot
 * demand an effectively unbounded allocation. */
enum {
    MODEL_MAX_TOKENS_PER_PASS = 1 << 20,   /* batch_size * block_size */
    MODEL_MAX_BLOCK_SIZE      = 1 << 16,
    MODEL_MAX_D_MODEL         = 1 << 14,
    MODEL_MAX_HEAD_COUNT      = 1 << 8,
    MODEL_MAX_LAYER_COUNT     = 1 << 10,
    MODEL_MAX_VOCAB_SIZE      = 256,
};

#define MODEL_MAX_CHECKPOINT_RESIDENT_BYTES ((size_t)1 << 30)
#define MODEL_MAX_CHECKPOINT_FILE_BYTES \
    (MODEL_MAX_CHECKPOINT_RESIDENT_BYTES / 4 + (size_t)4096)

int model_config_valid(ModelConfig config);
int model_memory_requirements(ModelConfig config, ModelMemory *memory);

/* Construction does not return for an invalid or unrepresentable
 * configuration, or when allocation fails. */
Model      *model_new(ModelConfig config, unsigned long long seed);
ModelConfig model_config(const Model *m);
ModelParams model_params(const Model *m);
size_t      model_parameter_count(const Model *m);

float       model_forward(Model *m, const int *tokens, const int *targets,
                          int batch, int time);
void        model_zero_gradients(Model *m);
void        model_backward(Model *m);
/* 0 on success.  Invalid optimizer settings and non-finite gradients are
 * rejected before gradient scaling or parameter updates begin.  A model
 * whose parameter or moment storage was corrupted is not transactionally
 * rolled back across Params. */
int         model_step(Model *m, AdamW opt, int step);

/* ids arrives holding prompt_count seed tokens and leaves holding
 * total_count: the model continues the sequence one draw at a time. */
void        model_sample(Model *m, Rng *rng, int *ids, int prompt_count,
                         int total_count, float temperature);

/* A checkpoint is magic + version + config + vocabulary + weights,
 * self-contained for sampling later.  `tk` must be the tokenizer whose
 * ids index the model's token table; the writer can verify its size but
 * cannot infer that semantic identity.  Save returns 0 on success.
 * Load returns NULL for I/O, format, or resource-policy failures;
 * allocation failure follows the project's fatal allocation policy. */
int         model_save(const Model *m, const Tokenizer *tk, const char *path);
Model      *model_load(Tokenizer **tk, const char *path);

void        model_free(Model *m);

#endif
