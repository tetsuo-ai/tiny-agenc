/*
 * Chapter 9 witness: the model blueprint constructs every parameter in
 * the promised order before activation memory or compute is involved.
 */
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model_internal.h"
#include "param.h"
#include "rng.h"

static int checks;
static int failures;

static void expect(int condition, const char *message)
{
    checks++;
    if (condition)
        return;
    fprintf(stderr, "check-09: %s\n", message);
    failures++;
}

static uint32_t float_bits(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof bits);
    return bits;
}

static Model *blueprint_model(ModelConfig config, unsigned long long seed)
{
    Model *model = calloc(1, sizeof *model);
    Rng   *rng   = rng_new(seed);

    if (model == NULL) {
        fputs("check-09: out of memory\n", stderr);
        exit(EXIT_FAILURE);
    }
    model->cfg = config;
    model->blocks = calloc((size_t)config.layer_count,
                           sizeof *model->blocks);
    if (model->blocks == NULL) {
        fputs("check-09: out of memory\n", stderr);
        exit(EXIT_FAILURE);
    }
    model_create_parameters(model, rng);
    rng_free(rng);
    return model;
}

static void expect_parameter(ModelParams params, int index, Param *named,
                             int rows, int cols, const char *name)
{
    checks++;
    if (params.params != NULL && index >= 0 && index < params.count
        && params.params[index] != NULL && params.params[index] == named) {
        Mat values = param_values(params.params[index]);

        if (values.rows == rows && values.cols == cols)
            return;
    }
    fprintf(stderr,
            "check-09: parameter %d must be %s with shape [%d, %d]\n",
            index, name, rows, cols);
    failures++;
}

static void expect_constant(Param *param, int rows, int cols, float wanted,
                            const char *message)
{
    int matches = param != NULL;

    if (matches) {
        Mat values = param_values(param);

        matches = values.rows == rows && values.cols == cols;
        for (size_t i = 0; matches && i < mat_size(values); i++)
            matches = float_bits(values.vals[i]) == float_bits(wanted);
    }
    expect(matches, message);
}

static void expect_gaussian(Param *param, int rows, int cols, float stddev,
                            Rng *expected_rng, const char *message)
{
    Mat values = { 0 };
    int matches = param != NULL;

    if (matches) {
        values = param_values(param);
        matches = values.rows == rows && values.cols == cols;
    }

    size_t count = (size_t)rows * (size_t)cols;

    for (size_t i = 0; i < count; i++) {
        float wanted = stddev * rng_gaussian(expected_rng);

        if (matches && float_bits(values.vals[i]) != float_bits(wanted))
            matches = 0;
    }
    expect(matches, message);
}

static int parameter_values_equal(const Model *left, const Model *right)
{
    ModelParams a = model_params(left);
    ModelParams b = model_params(right);

    if (a.count != b.count || a.params == NULL || b.params == NULL)
        return 0;
    for (int p = 0; p < a.count; p++) {
        Mat av = param_values(a.params[p]);
        Mat bv = param_values(b.params[p]);

        if (av.rows != bv.rows || av.cols != bv.cols
            || memcmp(av.vals, bv.vals,
                      mat_size(av) * sizeof *av.vals) != 0)
            return 0;
    }
    return 1;
}

static void check_geometry(void)
{
    ModelConfig showcase = {
        .vocab_size  = 80,
        .block_size  = 128,
        .d_model     = 128,
        .head_count  = 4,
        .layer_count = 4,
        .batch_size  = 1,
    };

    expect(model_config_valid(showcase), "the showcase configuration is valid");

    ModelConfig invalid = showcase;
    invalid.d_model = 127;
    expect(!model_config_valid(invalid), "width must divide evenly among heads");

    invalid = showcase;
    invalid.vocab_size = 0;
    expect(!model_config_valid(invalid), "vocabulary size must be positive");

    Model *model = blueprint_model(showcase, 1337);

    expect(model_parameter_count(model) == 815360,
           "the assembled showcase model has 815360 parameters");
    expect(model_params(model).count == 36,
           "four blocks produce the expected parameter tensor count");
    model_free(model);
}

static void check_parameter_blueprint(void)
{
    ModelConfig config = {
        .vocab_size  = 5,
        .block_size  = 3,
        .d_model     = 4,
        .head_count  = 1,
        .layer_count = 2,
        .batch_size  = 1,
    };
    static const unsigned long long SEED = 123;
    static const float INIT_STDDEV = 0.02f;
    Model *model = blueprint_model(config, SEED);
    ModelParams params = model_params(model);
    int expected_count =
        MODEL_TENSORS_ELSEWHERE
        + MODEL_TENSORS_PER_BLOCK * config.layer_count;

    expect(params.count == expected_count,
           "two blocks expose exactly twenty parameter tensors");
    if (params.params != NULL && params.count == expected_count) {
        int at = 0;

        expect_parameter(params, at++, model->token_table,
                         config.vocab_size, config.d_model, "token table");
        expect_parameter(params, at++, model->position_table,
                         config.block_size, config.d_model, "position table");
        for (int layer = 0; layer < config.layer_count; layer++) {
            Block *block = &model->blocks[layer];

            expect_parameter(params, at++, block->norm1_gain,
                             1, config.d_model, "norm1 gain");
            expect_parameter(params, at++, block->norm1_bias,
                             1, config.d_model, "norm1 bias");
            expect_parameter(params, at++, block->qkv_weights,
                             QKV_STREAMS * config.d_model, config.d_model,
                             "QKV weights");
            expect_parameter(params, at++, block->proj_weights,
                             config.d_model, config.d_model,
                             "attention projection");
            expect_parameter(params, at++, block->norm2_gain,
                             1, config.d_model, "norm2 gain");
            expect_parameter(params, at++, block->norm2_bias,
                             1, config.d_model, "norm2 bias");
            expect_parameter(params, at++, block->up_weights,
                             MODEL_MLP_WIDENING * config.d_model,
                             config.d_model, "MLP up weights");
            expect_parameter(params, at++, block->down_weights,
                             config.d_model,
                             MODEL_MLP_WIDENING * config.d_model,
                             "MLP down weights");
        }
        expect_parameter(params, at++, model->final_gain,
                         1, config.d_model, "final gain");
        expect_parameter(params, at++, model->final_bias,
                         1, config.d_model, "final bias");
        expect(at == expected_count,
               "the ordered shape ledger consumes every parameter");
    }

    Rng *expected_rng = rng_new(SEED);
    float residual_stddev =
        INIT_STDDEV / sqrtf(2.0f * (float)config.layer_count);

    expect_gaussian(model->token_table, config.vocab_size, config.d_model,
                    INIT_STDDEV, expected_rng,
                    "token table replays the seeded Gaussian stream");
    expect_gaussian(model->position_table, config.block_size, config.d_model,
                    INIT_STDDEV, expected_rng,
                    "position table continues the seeded Gaussian stream");
    for (int layer = 0; layer < config.layer_count; layer++) {
        Block *block = &model->blocks[layer];

        expect_constant(block->norm1_gain, 1, config.d_model, 1.0f,
                        "norm1 gain begins at one");
        expect_constant(block->norm1_bias, 1, config.d_model, 0.0f,
                        "norm1 bias begins at zero");
        expect_gaussian(block->qkv_weights,
                        QKV_STREAMS * config.d_model, config.d_model,
                        INIT_STDDEV, expected_rng,
                        "QKV weights use the base initialization scale");
        expect_gaussian(block->proj_weights,
                        config.d_model, config.d_model,
                        residual_stddev, expected_rng,
                        "attention projection uses the residual scale");
        expect_constant(block->norm2_gain, 1, config.d_model, 1.0f,
                        "norm2 gain begins at one");
        expect_constant(block->norm2_bias, 1, config.d_model, 0.0f,
                        "norm2 bias begins at zero");
        expect_gaussian(block->up_weights,
                        MODEL_MLP_WIDENING * config.d_model, config.d_model,
                        INIT_STDDEV, expected_rng,
                        "MLP up weights use the base initialization scale");
        expect_gaussian(block->down_weights,
                        config.d_model,
                        MODEL_MLP_WIDENING * config.d_model,
                        residual_stddev, expected_rng,
                        "MLP down weights use the residual scale");
    }
    expect_constant(model->final_gain, 1, config.d_model, 1.0f,
                    "final gain begins at one");
    expect_constant(model->final_bias, 1, config.d_model, 0.0f,
                    "final bias begins at zero");
    rng_free(expected_rng);

    Model *replayed = blueprint_model(config, SEED);
    Model *changed  = blueprint_model(config, SEED + 1);

    expect(parameter_values_equal(model, replayed),
           "the same seed reproduces every initialized parameter bit");
    expect(!parameter_values_equal(model, changed),
           "a different seed changes the initialized Gaussian parameters");
    model_free(changed);
    model_free(replayed);
    model_free(model);
}

static void check_global_clipping(void)
{
    ModelConfig config = {
        .vocab_size  = 5,
        .block_size  = 2,
        .d_model     = 4,
        .head_count  = 1,
        .layer_count = 1,
        .batch_size  = 1,
    };
    Model *model = blueprint_model(config, 9);
    ModelParams params = model_params(model);

    for (int p = 0; p < params.count; p++) {
        Mat values = param_values(params.params[p]);
        Mat gradient = param_gradient(params.params[p]);

        memset(values.vals, 0, mat_size(values) * sizeof *values.vals);
        memset(gradient.vals, 0, mat_size(gradient) * sizeof *gradient.vals);
    }

    Mat gradient = param_gradient(params.params[0]);

    gradient.vals[0] = 3.0f;
    gradient.vals[1] = 4.0f;

    AdamW optimizer = {
        .learning_rate = 1.0f,
        .beta1         = 0.0f,
        .beta2         = 0.0f,
        .epsilon       = 1.0f,
        .weight_decay  = 0.0f,
    };

    expect(model_step(model, optimizer, 1) == 0,
           "model_step accepts a finite gradient and valid recipe");

    expect(fabsf(gradient.vals[0] - 0.6f) <= 1e-6f,
           "model_step globally clips the first component");
    expect(fabsf(gradient.vals[1] - 0.8f) <= 1e-6f,
           "model_step globally clips the second component");

    double norm_squared = 0.0;

    for (int p = 0; p < params.count; p++) {
        Mat each = param_gradient(params.params[p]);

        for (size_t i = 0; i < mat_size(each); i++)
            norm_squared += (double)each.vals[i] * each.vals[i];
    }
    expect(fabs(sqrt(norm_squared) - 1.0) <= 1e-6,
           "model_step caps the shared global norm at one");
    model_free(model);
}

static void check_update_rejection_and_overflow_fallback(void)
{
    ModelConfig config = {
        .vocab_size  = 5,
        .block_size  = 2,
        .d_model     = 4,
        .head_count  = 1,
        .layer_count = 1,
        .batch_size  = 1,
    };
    Model *model = blueprint_model(config, 19);
    ModelParams params = model_params(model);

    for (int p = 0; p < params.count; p++)
        param_zero_gradient(params.params[p]);

    Mat gradient = param_gradient(params.params[0]);

    gradient.vals[0] = FLT_MAX / 4.0f;
    gradient.vals[1] = FLT_MAX / 8.0f;

    AdamW no_move = {
        .learning_rate = 0.0f,
        .beta1         = 0.0f,
        .beta2         = 0.0f,
        .epsilon       = 1.0f,
        .weight_decay  = 0.0f,
    };

    expect(model_step(model, no_move, 1) == 0,
           "large finite gradients use the clipping overflow fallback");

    double norm_squared = 0.0;

    for (int p = 0; p < params.count; p++) {
        Mat each = param_gradient(params.params[p]);

        for (size_t i = 0; i < mat_size(each); i++)
            norm_squared += (double)each.vals[i] * each.vals[i];
    }
    expect(fabs(sqrt(norm_squared) - 1.0) <= 1e-6,
           "overflow fallback clips the exact shared norm to one");
    expect(fabsf(gradient.vals[0] / gradient.vals[1] - 2.0f) <= 1e-6f,
           "overflow fallback preserves gradient direction");

    for (int p = 0; p < params.count; p++)
        param_zero_gradient(params.params[p]);
    gradient.vals[0] = 3.0f;
    gradient.vals[1] = 4.0f;

    AdamW invalid = no_move;
    invalid.epsilon = 0.0f;

    expect(model_step(model, invalid, 1) != 0,
           "model_step rejects an invalid optimizer recipe");
    expect(gradient.vals[0] == 3.0f && gradient.vals[1] == 4.0f,
           "invalid recipe rejection happens before gradient clipping");

    Mat values = param_values(params.params[0]);
    uint32_t value_before = float_bits(values.vals[0]);
    uint32_t infinity_bits = 0x7F800000u;

    memcpy(&gradient.vals[0], &infinity_bits, sizeof infinity_bits);
    expect(model_step(model, no_move, 1) != 0,
           "model_step rejects a non-finite gradient");
    expect(float_bits(values.vals[0]) == value_before,
           "non-finite gradient rejection leaves parameter values unchanged");
    model_free(model);
}

int main(void)
{
    check_geometry();
    check_parameter_blueprint();
    check_global_clipping();
    check_update_rejection_and_overflow_fallback();

    if (failures != 0) {
        fprintf(stderr, "check-09: %d of %d checks failed\n", failures, checks);
        return EXIT_FAILURE;
    }
    printf("check-09: all %d model-contract checks passed\n", checks);
    return EXIT_SUCCESS;
}
