#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "model_internal.h"

static const float GRADIENT_CLIP_NORM = 1.0f;

static int finite_float(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof bits);
    return (bits & 0x7F800000u) != 0x7F800000u;
}

static int checked_add(size_t left, size_t right, size_t *result)
{
    if (left > SIZE_MAX - right)
        return 0;
    *result = left + right;
    return 1;
}

static int checked_multiply(size_t left, size_t right, size_t *result)
{
    if (left != 0 && right > SIZE_MAX / left)
        return 0;
    *result = left * right;
    return 1;
}

static int checked_scaled_add(size_t *total, size_t count, size_t scale)
{
    size_t term;

    return checked_multiply(count, scale, &term)
        && checked_add(*total, term, total);
}

int model_config_valid(ModelConfig cfg)
{
    long long tokens = (long long)cfg.batch_size * cfg.block_size;

    return cfg.vocab_size  >= 1 && cfg.vocab_size  <= MODEL_MAX_VOCAB_SIZE
        && cfg.block_size  >= 1 && cfg.block_size  <= MODEL_MAX_BLOCK_SIZE
        && cfg.d_model     >= 1 && cfg.d_model     <= MODEL_MAX_D_MODEL
        && cfg.head_count  >= 1 && cfg.head_count  <= MODEL_MAX_HEAD_COUNT
        && cfg.layer_count >= 1 && cfg.layer_count <= MODEL_MAX_LAYER_COUNT
        && cfg.batch_size  >= 1 && tokens          <= MODEL_MAX_TOKENS_PER_PASS
        && cfg.d_model % cfg.head_count == 0;
}

static int parameter_floats(ModelConfig cfg, size_t *result)
{
    size_t width      = (size_t)cfg.d_model;
    size_t embeddings;
    size_t vector_floats;
    size_t square_floats;
    size_t total = 0;

    if (!checked_add((size_t)cfg.vocab_size, (size_t)cfg.block_size,
                     &embeddings)
        || !checked_multiply(embeddings, width, &embeddings)
        || !checked_multiply((size_t)cfg.layer_count, 4 * width,
                             &vector_floats)
        || !checked_add(vector_floats, 2 * width, &vector_floats)
        || !checked_multiply(width, width, &square_floats)
        || !checked_multiply(square_floats, 12, &square_floats)
        || !checked_multiply(square_floats, (size_t)cfg.layer_count,
                             &square_floats)
        || !checked_add(total, embeddings, &total)
        || !checked_add(total, vector_floats, &total)
        || !checked_add(total, square_floats, &total))
        return 0;
    *result = total;
    return 1;
}

size_t model_parameter_float_count(ModelConfig cfg)
{
    size_t result;

    return parameter_floats(cfg, &result) ? result : 0;
}

int model_memory_requirements(ModelConfig cfg, ModelMemory *memory)
{
    if (memory == NULL || !model_config_valid(cfg))
        return 0;

    size_t rows;
    size_t channels;
    size_t scores;
    size_t logits;
    size_t block_values = 0;
    size_t block_gradients = 0;
    size_t value_floats = 0;
    size_t gradient_floats = 0;
    size_t parameter_count;

    if (!checked_multiply((size_t)cfg.batch_size, (size_t)cfg.block_size,
                          &rows)
        || !checked_multiply(rows, (size_t)cfg.d_model, &channels)
        || !checked_multiply(rows, (size_t)cfg.head_count, &scores)
        || !checked_multiply(scores, (size_t)cfg.block_size, &scores)
        || !checked_multiply(rows, (size_t)cfg.vocab_size, &logits)
        || !parameter_floats(cfg, &parameter_count)
        || !checked_scaled_add(&block_values, channels, 18)
        || !checked_add(block_values, scores, &block_values)
        || !checked_scaled_add(&block_values, rows, 4)
        || !checked_scaled_add(&block_gradients, channels, 18)
        || !checked_add(block_gradients, scores, &block_gradients)
        || !checked_scaled_add(&value_floats, channels, 2)
        || !checked_scaled_add(&value_floats, block_values,
                               (size_t)cfg.layer_count)
        || !checked_scaled_add(&value_floats, rows, 2)
        || !checked_scaled_add(&value_floats, logits, 2)
        || !checked_scaled_add(&gradient_floats, channels, 2)
        || !checked_scaled_add(&gradient_floats, block_gradients,
                               (size_t)cfg.layer_count)
        || !checked_add(gradient_floats, logits, &gradient_floats)
        || !checked_multiply(parameter_count, 4 * sizeof(float),
                             &memory->parameter_bytes)
        || !checked_multiply(value_floats, sizeof(float),
                             &memory->activation_bytes)
        || !checked_multiply(gradient_floats, sizeof(float),
                             &memory->gradient_bytes)
        || !checked_multiply(rows, 2 * sizeof(int), &memory->token_bytes))
        return 0;

    memory->total_bytes = 0;
    return checked_add(memory->total_bytes, memory->parameter_bytes,
                       &memory->total_bytes)
        && checked_add(memory->total_bytes, memory->activation_bytes,
                       &memory->total_bytes)
        && checked_add(memory->total_bytes, memory->gradient_bytes,
                       &memory->total_bytes)
        && checked_add(memory->total_bytes, memory->token_bytes,
                       &memory->total_bytes);
}

ModelConfig model_config(const Model *m)
{
    return m->cfg;
}

ModelParams model_params(const Model *m)
{
    ModelParams view = { m->params, m->param_count };

    return view;
}

size_t model_parameter_count(const Model *m)
{
    size_t total = 0;

    for (int i = 0; i < m->param_count; i++)
        total += mat_size(param_values(m->params[i]));
    return total;
}

void model_zero_gradients(Model *m)
{
    for (int i = 0; i < m->param_count; i++)
        param_zero_gradient(m->params[i]);
    memset(m->gradient_arena, 0,
           m->gradient_floats * sizeof *m->gradient_arena);
}

static double exact_gradient_norm_squared(const Model *m)
{
    double total = 0.0;

    for (int i = 0; i < m->param_count; i++) {
        Mat gradient = param_gradient(m->params[i]);

        for (size_t at = 0; at < mat_size(gradient); at++)
            total += (double)gradient.vals[at] * (double)gradient.vals[at];
    }
    return total;
}

static void scale_gradients_exactly(Model *m, double factor)
{
    for (int i = 0; i < m->param_count; i++) {
        Mat gradient = param_gradient(m->params[i]);

        for (size_t at = 0; at < mat_size(gradient); at++)
            gradient.vals[at] =
                (float)((double)gradient.vals[at] * factor);
    }
}

static int clip_gradient_norm(Model *m)
{
    for (int i = 0; i < m->param_count; i++)
        if (!param_gradient_is_finite(m->params[i]))
            return -1;

    double norm_squared = 0.0;

    for (int i = 0; i < m->param_count; i++)
        norm_squared += (double)param_gradient_norm_squared(m->params[i]);

    /* Preserve the original float-rounded path for every ordinary
     * training step.  Only a norm that overflowed float takes the
     * double-precision recovery path. */
    float norm = (float)sqrt(norm_squared);

    if (finite_float(norm)) {
        if (norm <= GRADIENT_CLIP_NORM)
            return 0;
        for (int i = 0; i < m->param_count; i++)
            param_scale_gradient(m->params[i], GRADIENT_CLIP_NORM / norm);
        return 0;
    }

    double exact_norm = sqrt(exact_gradient_norm_squared(m));

    if (exact_norm <= GRADIENT_CLIP_NORM)
        return 0;
    scale_gradients_exactly(m, (double)GRADIENT_CLIP_NORM / exact_norm);
    return 0;
}

int model_step(Model *m, AdamW opt, int step)
{
    if (!param_adamw_recipe_valid(opt, step))
        return -1;
    if (clip_gradient_norm(m) != 0)
        return -1;
    for (int i = 0; i < m->param_count; i++) {
        if (param_adamw_step(m->params[i], opt, step) != 0)
            return -1;
    }
    return 0;
}

void model_free(Model *m)
{
    for (int i = 0; i < m->param_count; i++)
        param_free(m->params[i]);
    free(m->params);
    free(m->blocks);
    free(m->values_arena);
    free(m->gradient_arena);
    free(m->tokens);
    free(m->targets);
    free(m);
}
