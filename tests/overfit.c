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
    FIRST_TRAINING_STEP = 1,
};

static const uint32_t FLOAT_EXPONENT_MASK = 0x7F800000u;
static const unsigned long long MODEL_SEED = 1337;
static const float MAXIMUM_FINAL_LOSS = 0.10f;
static const float MAXIMUM_REMAINING_FRACTION = 0.10f;
static const ModelConfig MODEL_CONFIG = {
    .vocab_size  = VOCAB_SIZE,
    .block_size  = BLOCK_SIZE,
    .d_model     = D_MODEL,
    .head_count  = HEAD_COUNT,
    .layer_count = LAYER_COUNT,
    .batch_size  = BATCH_SIZE,
};
static const AdamW OPTIMIZER = {
    .learning_rate = 1e-2f,
    .beta1         = 0.9f,
    .beta2         = 0.999f,
    .epsilon       = 1e-8f,
    .weight_decay  = 0.01f,
};

static int finite_float(float value);
static void build_fixed_batch(int *inputs, int *targets);
static int train_fixed_batch(Model *model, const int *inputs,
                             const int *targets, float *initial_loss,
                             float *final_loss);
static int losses_show_memorization(float initial_loss, float final_loss);
int main(void);

static int finite_float(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof bits);
    return (bits & FLOAT_EXPONENT_MASK) != FLOAT_EXPONENT_MASK;
}

static void build_fixed_batch(int *inputs, int *targets)
{
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int time = 0; time < BLOCK_SIZE; time++) {
            int at = batch * BLOCK_SIZE + time;

            inputs[at]  = (time + batch) % VOCAB_SIZE;
            targets[at] =
                (time + batch + FIRST_TRAINING_STEP) % VOCAB_SIZE;
        }
    }
}

static int train_fixed_batch(Model *model, const int *inputs,
                             const int *targets, float *initial_loss,
                             float *final_loss)
{
    for (int step = FIRST_TRAINING_STEP;
         step <= TRAINING_STEPS; step++) {
        model_zero_gradients(model);

        float loss = model_forward(model, inputs, targets,
                                   BATCH_SIZE, BLOCK_SIZE);

        if (step == FIRST_TRAINING_STEP)
            *initial_loss = loss;
        model_backward(model);
        if (model_step(model, OPTIMIZER, step) != 0)
            return -1;
        *final_loss = loss;
    }
    return 0;
}

static int losses_show_memorization(float initial_loss, float final_loss)
{
    return finite_float(initial_loss)
        && finite_float(final_loss)
        && final_loss < MAXIMUM_FINAL_LOSS
        && final_loss
               < initial_loss * MAXIMUM_REMAINING_FRACTION;
}

int main(void)
{
    int inputs[BATCH_SIZE * BLOCK_SIZE];
    int targets[BATCH_SIZE * BLOCK_SIZE];
    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    build_fixed_batch(inputs, targets);

    Model *model = model_new(MODEL_CONFIG, MODEL_SEED);

    if (train_fixed_batch(model, inputs, targets,
                          &initial_loss, &final_loss) != 0) {
        fputs("overfit: optimizer rejected a finite step\n", stderr);
        model_free(model);
        return EXIT_FAILURE;
    }

    printf("overfit: initial loss %.6f\n", (double)initial_loss);
    printf("overfit: final loss   %.6f\n", (double)final_loss);

    int passed = losses_show_memorization(initial_loss, final_loss);

    model_free(model);
    if (!passed) {
        fputs("overfit: FAILED to memorize the fixed batch\n", stderr);
        return EXIT_FAILURE;
    }
    puts("overfit: passed");
    return EXIT_SUCCESS;
}
