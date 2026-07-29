/*
 * Chapter 10 witness: the learner's arena plan supports full-capacity
 * and shorter views when driven by known-good forward and backward code.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "param.h"

static int checks;
static int failures;

static void expect(int condition, const char *message)
{
    checks++;
    if (condition)
        return;
    fprintf(stderr, "check-10: %s\n", message);
    failures++;
}

static int finite_float(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof bits);
    return (bits & 0x7F800000u) != 0x7F800000u;
}

int main(void)
{
    enum { BATCH = 2, BLOCK = 4 };
    ModelConfig config = {
        .vocab_size  = 5,
        .block_size  = BLOCK,
        .d_model     = 8,
        .head_count  = 2,
        .layer_count = 1,
        .batch_size  = BATCH,
    };
    ModelMemory memory;

    expect(model_memory_requirements(config, &memory),
           "arena sizes are computable without allocating");
    expect(memory.activation_bytes > memory.token_bytes
           && memory.gradient_bytes > memory.token_bytes,
           "activation and gradient arenas dominate token caches");

    Model *model = model_new(config, 17);
    int full_inputs[BATCH * BLOCK];
    int full_targets[BATCH * BLOCK];

    for (int i = 0; i < BATCH * BLOCK; i++) {
        full_inputs[i] = i % config.vocab_size;
        full_targets[i] = (i + 1) % config.vocab_size;
    }

    model_zero_gradients(model);
    float full_loss =
        model_forward(model, full_inputs, full_targets, BATCH, BLOCK);
    model_backward(model);
    expect(finite_float(full_loss) && full_loss > 0.0f,
           "full-capacity arena views complete forward and backward");

    int short_inputs[] = { 0, 1, 2 };
    int short_targets[] = { 1, 2, 3 };

    model_zero_gradients(model);
    float short_loss = model_forward(model, short_inputs, short_targets, 1, 3);
    model_backward(model);
    expect(finite_float(short_loss) && short_loss > 0.0f,
           "short arena views reshape attention without stale capacity");

    ModelParams params = model_params(model);

    for (int p = 0; p < params.count; p++) {
        Mat gradient = param_gradient(params.params[p]);

        for (size_t i = 0; i < mat_size(gradient); i++)
            expect(finite_float(gradient.vals[i]),
                   "arena-backed model gradients stay finite");
    }

    model_free(model);
    if (failures != 0) {
        fprintf(stderr, "check-10: %d of %d checks failed\n",
                failures, checks);
        return EXIT_FAILURE;
    }
    printf("check-10: all %d arena checks passed\n", checks);
    return EXIT_SUCCESS;
}
