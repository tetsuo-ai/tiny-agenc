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
    return (bits & 0x7F800000u) != 0x7F800000u;
}

int main(void)
{
    enum { BATCH = 2, TIME = 6, ROWS = BATCH * TIME };
    ModelConfig config = {
        .vocab_size  = 13,
        .block_size  = 8,
        .d_model     = 16,
        .head_count  = 2,
        .layer_count = 2,
        .batch_size  = BATCH,
    };
    int inputs[ROWS];
    int targets[ROWS];

    for (int i = 0; i < ROWS; i++) {
        inputs[i] = (3 * i + 1) % config.vocab_size;
        targets[i] = (5 * i + 2) % config.vocab_size;
    }

    Model *model = model_new(config, 4242);

    model_zero_gradients(model);
    float loss = model_forward(model, inputs, targets, BATCH, TIME);
    model_backward(model);
    expect(finite_float(loss) && loss > 0.0f,
           "whole-model backward begins from a finite loss");

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

    model_free(model);
    if (failures != 0) {
        fprintf(stderr, "check-12: %d of %d checks failed\n",
                failures, checks);
        return EXIT_FAILURE;
    }
    printf("check-12: all %d model-backward checks passed\n", checks);
    return EXIT_SUCCESS;
}
