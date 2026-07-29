#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"

enum {
    VOCAB_SIZE = 5,
    BLOCK_SIZE = 8,
    D_MODEL = 16,
    HEAD_COUNT = 2,
    LAYER_COUNT = 2,
    BATCH_SIZE = 4,
    TRAINING_STEPS = 300,
};

static int finite_float(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof bits);
    return (bits & 0x7F800000u) != 0x7F800000u;
}

int main(void)
{
    ModelConfig config = {
        .vocab_size  = VOCAB_SIZE,
        .block_size  = BLOCK_SIZE,
        .d_model     = D_MODEL,
        .head_count  = HEAD_COUNT,
        .layer_count = LAYER_COUNT,
        .batch_size  = BATCH_SIZE,
    };
    AdamW optimizer = {
        .learning_rate = 1e-2f,
        .beta1         = 0.9f,
        .beta2         = 0.999f,
        .epsilon       = 1e-8f,
        .weight_decay  = 0.01f,
    };
    int inputs[BATCH_SIZE * BLOCK_SIZE];
    int targets[BATCH_SIZE * BLOCK_SIZE];

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int time = 0; time < BLOCK_SIZE; time++) {
            int at = batch * BLOCK_SIZE + time;

            inputs[at]  = (time + batch) % VOCAB_SIZE;
            targets[at] = (time + batch + 1) % VOCAB_SIZE;
        }
    }

    Model *model = model_new(config, 1337);
    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    for (int step = 1; step <= TRAINING_STEPS; step++) {
        model_zero_gradients(model);

        float loss = model_forward(model, inputs, targets,
                                   BATCH_SIZE, BLOCK_SIZE);

        if (step == 1)
            initial_loss = loss;
        model_backward(model);
        if (model_step(model, optimizer, step) != 0) {
            fputs("overfit: optimizer rejected a finite step\n", stderr);
            model_free(model);
            return EXIT_FAILURE;
        }
        final_loss = loss;
    }

    printf("overfit: initial loss %.6f\n", (double)initial_loss);
    printf("overfit: final loss   %.6f\n", (double)final_loss);

    int passed = finite_float(initial_loss)
              && finite_float(final_loss)
              && final_loss < 0.10f
              && final_loss < initial_loss * 0.10f;

    model_free(model);
    if (!passed) {
        fputs("overfit: FAILED to memorize the fixed batch\n", stderr);
        return EXIT_FAILURE;
    }
    puts("overfit: passed");
    return EXIT_SUCCESS;
}
