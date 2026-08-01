/*
 * Chapter 12 witness: every parameter family is connected to the loss.
 * Finite differences run separately against the same learner modules.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "param.h"

static int checks;
static int failures;
static const uint32_t FLOAT_EXPONENT_MASK = 0x7F800000u;
enum {
    BATCH_SIZE = 2,
    TIME_STEPS = 6,
    TOKEN_ROWS = BATCH_SIZE * TIME_STEPS,
    VOCABULARY_SIZE = 13,
    CONTEXT_SIZE = 8,
    MODEL_WIDTH = 16,
    HEAD_COUNT = 2,
    LAYER_COUNT = 2,
    INPUT_STRIDE = 3,
    INPUT_OFFSET = 1,
    TARGET_STRIDE = 5,
    TARGET_OFFSET = 2,
};
static const unsigned long long MODEL_SEED = 4242;

static void expect(int condition, const char *message);
static int finite_float(float value);
static ModelConfig witness_config(void);
static void fill_token_fixture(int *inputs, int *targets,
                               ModelConfig config);
static void check_parameter_connections(Model *model);
int main(void);

static void expect(int condition, const char *message)
{
    checks++;
    if (condition)
        return;
    fprintf(stderr, "check-12: %s\n", message);
    failures++;
}

static int finite_float(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof bits);
    return (bits & FLOAT_EXPONENT_MASK) != FLOAT_EXPONENT_MASK;
}

static ModelConfig witness_config(void)
{
    ModelConfig config = {
        .vocab_size  = VOCABULARY_SIZE,
        .block_size  = CONTEXT_SIZE,
        .d_model     = MODEL_WIDTH,
        .head_count  = HEAD_COUNT,
        .layer_count = LAYER_COUNT,
        .batch_size  = BATCH_SIZE,
    };

    return config;
}

static void fill_token_fixture(int *inputs, int *targets,
                               ModelConfig config)
{
    for (int i = 0; i < TOKEN_ROWS; i++) {
        inputs[i] = (INPUT_STRIDE * i + INPUT_OFFSET) % config.vocab_size;
        targets[i] =
            (TARGET_STRIDE * i + TARGET_OFFSET) % config.vocab_size;
    }
}

static void check_parameter_connections(Model *model)
{
    ModelParams params = model_params(model);

    for (int p = 0; p < params.count; p++) {
        Mat gradient = param_gradient(params.params[p]);
        int connected = 0;

        for (size_t i = 0; i < mat_size(gradient); i++) {
            expect(finite_float(gradient.vals[i]),
                   "every whole-model gradient is finite");
            connected |= gradient.vals[i] != 0.0f;
        }
        expect(connected,
               "every parameter tensor has a path to the training loss");
    }
}

int main(void)
{
    ModelConfig config = witness_config();
    int inputs[TOKEN_ROWS];
    int targets[TOKEN_ROWS];

    fill_token_fixture(inputs, targets, config);

    Model *model = model_new(config, MODEL_SEED);

    model_zero_gradients(model);
    float loss = model_forward(model, inputs, targets,
                               BATCH_SIZE, TIME_STEPS);
    model_backward(model);
    expect(finite_float(loss) && loss > 0.0f,
           "whole-model backward begins from a finite loss");
    check_parameter_connections(model);

    model_free(model);
    if (failures != 0) {
        fprintf(stderr, "check-12: %d of %d checks failed\n",
                failures, checks);
        return EXIT_FAILURE;
    }
    printf("check-12: all %d model-backward checks passed\n", checks);
    return EXIT_SUCCESS;
}
