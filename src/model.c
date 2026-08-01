#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "model_internal.h"

static const float GRADIENT_CLIP_NORM = 1.0f;
static const uint32_t FLOAT_EXPONENT_MASK = 0x7F800000u;

enum {
    NORM_PARAMETER_VECTORS_PER_BLOCK = 4,
    FINAL_NORM_PARAMETER_VECTORS     = 2,
    ATTENTION_OUTPUT_PARAMETER_WIDTHS = 1,
    MLP_PARAMETER_MATRICES            = 2,
    ATTENTION_PARAMETER_WIDTHS =
        QKV_STREAMS + ATTENTION_OUTPUT_PARAMETER_WIDTHS,
    MLP_PARAMETER_WIDTHS =
        MLP_PARAMETER_MATRICES * MODEL_MLP_WIDENING,
    MATRIX_WIDTHS_PER_BLOCK =
        ATTENTION_PARAMETER_WIDTHS + MLP_PARAMETER_WIDTHS,
    /* normed1, attended, projected, after_attention, normed2, down,
     * and after_mlp each occupy one model-width tensor. */
    BLOCK_MODEL_WIDTH_TENSORS = 7,
    BLOCK_WIDE_TENSORS        = 2,
    BLOCK_CHANNEL_WIDTHS =
        BLOCK_MODEL_WIDTH_TENSORS + QKV_STREAMS
        + BLOCK_WIDE_TENSORS * MODEL_MLP_WIDENING,
    BLOCK_STATISTIC_BUFFERS        = 4,
    MODEL_STREAM_BUFFERS           = 2,
    MODEL_STATISTIC_BUFFERS        = 2,
    MODEL_LOGIT_VALUE_BUFFERS      = 2,
    MODEL_LOGIT_GRADIENT_BUFFERS   = 1,
    PARAMETER_STORAGE_BUFFERS      = 4,
    TOKEN_CACHE_BUFFERS            = 2,
};

typedef struct {
    size_t rows;
    size_t channels;
    size_t scores;
    size_t logits;
} ShapeCounts;

typedef struct {
    size_t values;
    size_t gradients;
} BlockFloatCounts;

typedef struct {
    size_t parameters;
    size_t values;
    size_t gradients;
} ModelFloatCounts;

static int finite_float(float value);
static int checked_add(size_t left, size_t right, size_t *result);
static int checked_multiply(size_t left, size_t right, size_t *result);
static int checked_scaled_add(size_t *total, size_t count, size_t scale);
static int model_dimensions_in_range(ModelConfig cfg);
static int model_pass_geometry_valid(ModelConfig cfg);
static int embedding_parameter_floats(ModelConfig cfg, size_t *result);
static int vector_parameter_floats(ModelConfig cfg, size_t *result);
static int matrix_parameter_floats(ModelConfig cfg, size_t *result);
static int measure_parameter_floats(ModelConfig cfg, size_t *result);
static int measure_shape_counts(ModelConfig cfg, ShapeCounts *result);
static int measure_block_float_counts(ShapeCounts shape,
                                      BlockFloatCounts *result);
static int measure_model_float_counts(ModelConfig cfg, ShapeCounts shape,
                                      BlockFloatCounts block,
                                      ModelFloatCounts *result);
static int total_memory_bytes(ModelMemory *memory);
static int build_memory_report(ShapeCounts shape, ModelFloatCounts floats,
                               ModelMemory *result);
static int measure_model_memory(ModelConfig cfg, ModelMemory *result);
static double exact_gradient_norm_squared(const Model *m);
static void scale_gradients_exactly(Model *m, double factor);
static int clip_gradient_norm(Model *m);

static int finite_float(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof bits);
    return (bits & FLOAT_EXPONENT_MASK) != FLOAT_EXPONENT_MASK;
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

static int model_dimensions_in_range(ModelConfig cfg)
{
    return cfg.vocab_size  >= 1 && cfg.vocab_size  <= MODEL_MAX_VOCAB_SIZE
        && cfg.block_size  >= 1 && cfg.block_size  <= MODEL_MAX_BLOCK_SIZE
        && cfg.d_model     >= 1 && cfg.d_model     <= MODEL_MAX_D_MODEL
        && cfg.head_count  >= 1 && cfg.head_count  <= MODEL_MAX_HEAD_COUNT
        && cfg.layer_count >= 1 && cfg.layer_count <= MODEL_MAX_LAYER_COUNT;
}

static int model_pass_geometry_valid(ModelConfig cfg)
{
    long long tokens = (long long)cfg.batch_size * cfg.block_size;

    return cfg.batch_size >= 1
        && tokens <= MODEL_MAX_TOKENS_PER_PASS
        && cfg.d_model % cfg.head_count == 0;
}

int model_config_valid(ModelConfig cfg)
{
    return model_dimensions_in_range(cfg)
        && model_pass_geometry_valid(cfg);
}

static int embedding_parameter_floats(ModelConfig cfg, size_t *result)
{
    size_t table_rows;

    return checked_add((size_t)cfg.vocab_size, (size_t)cfg.block_size,
                       &table_rows)
        && checked_multiply(table_rows, (size_t)cfg.d_model, result);
}

static int vector_parameter_floats(ModelConfig cfg, size_t *result)
{
    size_t width      = (size_t)cfg.d_model;
    size_t block_vectors;

    return checked_multiply((size_t)cfg.layer_count,
                            NORM_PARAMETER_VECTORS_PER_BLOCK * width,
                            &block_vectors)
        && checked_add(block_vectors,
                       FINAL_NORM_PARAMETER_VECTORS * width, result);
}

static int matrix_parameter_floats(ModelConfig cfg, size_t *result)
{
    size_t square;

    return checked_multiply((size_t)cfg.d_model, (size_t)cfg.d_model,
                            &square)
        && checked_multiply(square, MATRIX_WIDTHS_PER_BLOCK, &square)
        && checked_multiply(square, (size_t)cfg.layer_count, result);
}

static int measure_parameter_floats(ModelConfig cfg, size_t *result)
{
    size_t embeddings;
    size_t vectors;
    size_t matrices;
    size_t total;

    if (!embedding_parameter_floats(cfg, &embeddings)
        || !vector_parameter_floats(cfg, &vectors)
        || !matrix_parameter_floats(cfg, &matrices)
        || !checked_add(embeddings, vectors, &total)
        || !checked_add(total, matrices, &total))
        return 0;
    *result = total;
    return 1;
}

size_t model_parameter_float_count(ModelConfig cfg)
{
    size_t result;

    return model_config_valid(cfg)
        && measure_parameter_floats(cfg, &result) ? result : 0;
}

static int measure_shape_counts(ModelConfig cfg, ShapeCounts *result)
{
    ShapeCounts counts;

    if (!checked_multiply((size_t)cfg.batch_size, (size_t)cfg.block_size,
                          &counts.rows)
        || !checked_multiply(counts.rows, (size_t)cfg.d_model,
                             &counts.channels)
        || !checked_multiply(counts.rows, (size_t)cfg.head_count,
                             &counts.scores)
        || !checked_multiply(counts.scores, (size_t)cfg.block_size,
                             &counts.scores)
        || !checked_multiply(counts.rows, (size_t)cfg.vocab_size,
                             &counts.logits))
        return 0;
    *result = counts;
    return 1;
}

static int measure_block_float_counts(ShapeCounts shape,
                                      BlockFloatCounts *result)
{
    BlockFloatCounts counts = {0};

    if (!checked_scaled_add(&counts.values, shape.channels,
                            BLOCK_CHANNEL_WIDTHS)
        || !checked_add(counts.values, shape.scores, &counts.values)
        || !checked_scaled_add(&counts.values, shape.rows,
                               BLOCK_STATISTIC_BUFFERS)
        || !checked_scaled_add(&counts.gradients, shape.channels,
                               BLOCK_CHANNEL_WIDTHS)
        || !checked_add(counts.gradients, shape.scores,
                        &counts.gradients))
        return 0;
    *result = counts;
    return 1;
}

static int measure_model_float_counts(ModelConfig cfg, ShapeCounts shape,
                                      BlockFloatCounts block,
                                      ModelFloatCounts *result)
{
    ModelFloatCounts counts = {0};

    if (!measure_parameter_floats(cfg, &counts.parameters)
        || !checked_scaled_add(&counts.values, shape.channels,
                               MODEL_STREAM_BUFFERS)
        || !checked_scaled_add(&counts.values, block.values,
                               (size_t)cfg.layer_count)
        || !checked_scaled_add(&counts.values, shape.rows,
                               MODEL_STATISTIC_BUFFERS)
        || !checked_scaled_add(&counts.values, shape.logits,
                               MODEL_LOGIT_VALUE_BUFFERS)
        || !checked_scaled_add(&counts.gradients, shape.channels,
                               MODEL_STREAM_BUFFERS)
        || !checked_scaled_add(&counts.gradients, block.gradients,
                               (size_t)cfg.layer_count)
        || !checked_scaled_add(&counts.gradients, shape.logits,
                               MODEL_LOGIT_GRADIENT_BUFFERS))
        return 0;
    *result = counts;
    return 1;
}

static int total_memory_bytes(ModelMemory *memory)
{
    size_t total;

    return checked_add(memory->parameter_bytes, memory->activation_bytes,
                       &total)
        && checked_add(total, memory->gradient_bytes, &total)
        && checked_add(total, memory->token_bytes, &memory->total_bytes);
}

static int build_memory_report(ShapeCounts shape, ModelFloatCounts floats,
                               ModelMemory *result)
{
    ModelMemory memory;

    if (!checked_multiply(floats.parameters,
                          PARAMETER_STORAGE_BUFFERS * sizeof(float),
                          &memory.parameter_bytes)
        || !checked_multiply(floats.values, sizeof(float),
                             &memory.activation_bytes)
        || !checked_multiply(floats.gradients, sizeof(float),
                             &memory.gradient_bytes)
        || !checked_multiply(shape.rows,
                             TOKEN_CACHE_BUFFERS * sizeof(int),
                             &memory.token_bytes)
        || !total_memory_bytes(&memory))
        return 0;
    *result = memory;
    return 1;
}

static int measure_model_memory(ModelConfig cfg, ModelMemory *result)
{
    ShapeCounts shape;
    BlockFloatCounts block;
    ModelFloatCounts floats;

    return measure_shape_counts(cfg, &shape)
        && measure_block_float_counts(shape, &block)
        && measure_model_float_counts(cfg, shape, block, &floats)
        && build_memory_report(shape, floats, result);
}

int model_memory_requirements(ModelConfig cfg, ModelMemory *memory)
{
    ModelMemory result;

    if (memory == NULL || !model_config_valid(cfg)
        || !measure_model_memory(cfg, &result))
        return 0;
    *memory = result;
    return 1;
}

ModelConfig model_config(const Model *m)
{
    return m->cfg;
}

ModelParams model_params(const Model *m)
{
    ModelParams view = {
        .params = m->params,
        .count = m->param_count,
    };

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
