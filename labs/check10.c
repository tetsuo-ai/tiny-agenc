/*
 * Chapter 10 witness: the learner's arena plan supports full-capacity
 * and shorter views when driven by known-good forward and backward code.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model_internal.h"
#include "param.h"

static int checks;
static int failures;
static const uint32_t FLOAT_EXPONENT_MASK = 0x7F800000u;
enum {
    PARAMETER_STORAGE_BUFFERS = 4,
    TOKEN_CACHE_BUFFERS = 2,
    NEXT_TOKEN_OFFSET = 1,
};

typedef struct {
    const float *base;
    size_t       next;
    size_t       capacity;
    const char  *name;
} ArenaWalk;

static void expect(int condition, const char *message);
static void expect_named(int condition, const char *name,
                         const char *property);
static int finite_float(float value);
static void walk_span(ArenaWalk *walk, const float *start, size_t count,
                      const char *name);
static void walk_mat(ArenaWalk *walk, Mat mat, int rows, int cols,
                     const char *name);
static void walk_block_tensors(ArenaWalk *walk, const BlockTensors *tensors,
                               ModelConfig config);
static void walk_value_arena(const Model *model, size_t capacity);
static void walk_gradient_arena(const Model *model, size_t capacity);
static void check_arena_layout(const Model *model, ModelMemory memory);
static void check_whole_gradient_arena_clears(Model *model);
static ModelConfig arena_test_config(void);
static void check_full_capacity_view(Model *model, ModelConfig config);
static void check_short_arena_view(Model *model);
static void check_finite_model_gradients(const Model *model);
int main(void);

static void expect(int condition, const char *message)
{
    checks++;
    if (condition)
        return;
    fprintf(stderr, "check-10: %s\n", message);
    failures++;
}

static void expect_named(int condition, const char *name,
                         const char *property)
{
    checks++;
    if (condition)
        return;
    fprintf(stderr, "check-10: %s %s\n", name, property);
    failures++;
}

static int finite_float(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof bits);
    return (bits & FLOAT_EXPONENT_MASK) != FLOAT_EXPONENT_MASK;
}

static void walk_span(ArenaWalk *walk, const float *start, size_t count,
                      const char *name)
{
    int fits = walk->next <= walk->capacity
            && count <= walk->capacity - walk->next;

    expect_named(fits, name, "fits inside its reported arena");
    if (!fits)
        return;
    expect_named(start == walk->base + walk->next, name,
                 "starts where the previous view ends");
    walk->next += count;
}

static void walk_mat(ArenaWalk *walk, Mat mat, int rows, int cols,
                     const char *name)
{
    expect_named(mat.rows == rows && mat.cols == cols, name,
                 "has the expected shape");
    walk_span(walk, mat.vals, mat_size(mat), name);
}

static void walk_block_tensors(ArenaWalk *walk, const BlockTensors *tensors,
                               ModelConfig config)
{
    int rows = config.batch_size * config.block_size;
    int wide = MODEL_MLP_WIDENING * config.d_model;

    walk_mat(walk, tensors->normed1, rows, config.d_model,
             "block normed1");
    walk_mat(walk, tensors->qkv, rows, QKV_STREAMS * config.d_model,
             "block qkv");
    walk_mat(walk, tensors->scores, rows * config.head_count,
             config.block_size, "block scores");
    walk_mat(walk, tensors->attended, rows, config.d_model,
             "block attended");
    walk_mat(walk, tensors->projected, rows, config.d_model,
             "block projected");
    walk_mat(walk, tensors->after_attention, rows, config.d_model,
             "block after-attention");
    walk_mat(walk, tensors->normed2, rows, config.d_model,
             "block normed2");
    walk_mat(walk, tensors->up, rows, wide, "block up");
    walk_mat(walk, tensors->activated, rows, wide, "block activated");
    walk_mat(walk, tensors->down, rows, config.d_model, "block down");
    walk_mat(walk, tensors->after_mlp, rows, config.d_model,
             "block after-MLP");
}

static void walk_value_arena(const Model *model, size_t capacity)
{
    ModelConfig config = model->cfg;
    int rows = config.batch_size * config.block_size;
    ArenaWalk walk = {
        .base = model->values_arena,
        .next = 0,
        .capacity = capacity,
        .name = "activation arena",
    };

    walk_mat(&walk, model->embedded, rows, config.d_model, "embedded");
    for (int layer = 0; layer < config.layer_count; layer++) {
        const Block *block = &model->blocks[layer];

        walk_block_tensors(&walk, &block->acts, config);
        walk_span(&walk, block->means1, (size_t)rows, "block means1");
        walk_span(&walk, block->rstds1, (size_t)rows, "block rstds1");
        walk_span(&walk, block->means2, (size_t)rows, "block means2");
        walk_span(&walk, block->rstds2, (size_t)rows, "block rstds2");
    }
    walk_mat(&walk, model->final_normed, rows, config.d_model,
             "final normed");
    walk_span(&walk, model->final_means, (size_t)rows, "final means");
    walk_span(&walk, model->final_rstds, (size_t)rows, "final rstds");
    walk_mat(&walk, model->logits, rows, config.vocab_size, "logits");
    walk_mat(&walk, model->probs, rows, config.vocab_size, "probabilities");
    expect_named(walk.next == walk.capacity, walk.name,
                 "ends at the reported arena boundary");
}

static void walk_gradient_arena(const Model *model, size_t capacity)
{
    ModelConfig config = model->cfg;
    int rows = config.batch_size * config.block_size;
    ArenaWalk walk = {
        .base = model->gradient_arena,
        .next = 0,
        .capacity = capacity,
        .name = "gradient arena",
    };

    walk_mat(&walk, model->d_embedded, rows, config.d_model,
             "d_embedded");
    for (int layer = 0; layer < config.layer_count; layer++)
        walk_block_tensors(&walk, &model->blocks[layer].grads, config);
    walk_mat(&walk, model->d_final_normed, rows, config.d_model,
             "d_final_normed");
    walk_mat(&walk, model->d_logits, rows, config.vocab_size, "d_logits");
    expect_named(walk.next == walk.capacity, walk.name,
                 "ends at the reported arena boundary");
}

static void check_arena_layout(const Model *model, ModelMemory memory)
{
    size_t value_floats = memory.activation_bytes / sizeof(float);
    size_t gradient_floats = memory.gradient_bytes / sizeof(float);
    size_t parameter_bytes =
        model_parameter_count(model)
        * PARAMETER_STORAGE_BUFFERS * sizeof(float);
    size_t token_bytes =
        (size_t)model->cfg.batch_size * (size_t)model->cfg.block_size
        * TOKEN_CACHE_BUFFERS * sizeof(int);

    expect(memory.activation_bytes % sizeof(float) == 0,
           "activation report contains whole floats");
    expect(memory.gradient_bytes % sizeof(float) == 0,
           "gradient report contains whole floats");
    expect(memory.parameter_bytes == parameter_bytes,
           "parameter report covers values, gradients, and two moments");
    expect(memory.token_bytes == token_bytes,
           "token report covers both full-capacity integer caches");
    expect(memory.total_bytes == memory.parameter_bytes
           + memory.activation_bytes + memory.gradient_bytes
           + memory.token_bytes,
           "total report is the exact sum of its storage families");
    expect(model->gradient_floats == gradient_floats,
           "gradient clearing count matches the reported arena size");

    walk_value_arena(model, value_floats);
    walk_gradient_arena(model, gradient_floats);
}

static void check_whole_gradient_arena_clears(Model *model)
{
    int all_zero = 1;

    for (size_t i = 0; i < model->gradient_floats; i++)
        model->gradient_arena[i] = 1.0f;
    model_zero_gradients(model);
    for (size_t i = 0; i < model->gradient_floats; i++)
        if (model->gradient_arena[i] != 0.0f)
            all_zero = 0;
    expect(all_zero, "zeroing reaches every reported gradient arena float");
}

static ModelConfig arena_test_config(void)
{
    ModelConfig config = {
        .vocab_size  = 5,
        .block_size  = 4,
        .d_model     = 8,
        .head_count  = 2,
        .layer_count = 2,
        .batch_size  = 2,
    };

    return config;
}

static void check_full_capacity_view(Model *model, ModelConfig config)
{
    int token_count = config.batch_size * config.block_size;
    int inputs[token_count];
    int targets[token_count];

    for (int i = 0; i < token_count; i++) {
        inputs[i] = i % config.vocab_size;
        targets[i] = (i + NEXT_TOKEN_OFFSET) % config.vocab_size;
    }

    model_zero_gradients(model);
    float loss = model_forward(model, inputs, targets,
                               config.batch_size, config.block_size);
    model_backward(model);
    expect(finite_float(loss) && loss > 0.0f,
           "full-capacity arena views complete forward and backward");
}

static void check_short_arena_view(Model *model)
{
    int inputs[] = { 0, 1, 2 };
    int targets[] = { 1, 2, 3 };
    int token_count = (int)(sizeof inputs / sizeof inputs[0]);

    model_zero_gradients(model);
    float loss = model_forward(model, inputs, targets, 1, token_count);
    model_backward(model);
    expect(finite_float(loss) && loss > 0.0f,
           "short arena views reshape attention without stale capacity");
}

static void check_finite_model_gradients(const Model *model)
{
    ModelParams params = model_params(model);

    for (int parameter = 0; parameter < params.count; parameter++) {
        Mat gradient = param_gradient(params.params[parameter]);

        for (size_t i = 0; i < mat_size(gradient); i++)
            expect(finite_float(gradient.vals[i]),
                   "arena-backed model gradients stay finite");
    }
}

int main(void)
{
    ModelConfig config = arena_test_config();
    ModelMemory memory;

    expect(model_memory_requirements(config, &memory),
           "arena sizes are computable without allocating");
    expect(memory.activation_bytes > memory.token_bytes
           && memory.gradient_bytes > memory.token_bytes,
           "activation and gradient arenas dominate token caches");

    Model *model = model_new(config, 17);

    check_arena_layout(model, memory);
    check_whole_gradient_arena_clears(model);
    check_full_capacity_view(model, config);
    check_short_arena_view(model);
    check_finite_model_gradients(model);

    model_free(model);
    if (failures != 0) {
        fprintf(stderr, "check-10: %d of %d checks failed\n",
                failures, checks);
        return EXIT_FAILURE;
    }
    printf("check-10: all %d arena checks passed\n", checks);
    return EXIT_SUCCESS;
}
