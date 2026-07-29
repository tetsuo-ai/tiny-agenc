/*
 * Chapter 11 witness: the learner's complete model forward pass answers
 * to fixed, nonuniform known results at full and shortened sequence sizes.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "model.h"

static int checks;
static int failures;

static void expect(int condition, const char *message)
{
    checks++;
    if (condition)
        return;
    fprintf(stderr, "check-11: %s\n", message);
    failures++;
}

int main(void)
{
    ModelConfig config = {
        .vocab_size  = 5,
        .block_size  = 4,
        .d_model     = 8,
        .head_count  = 2,
        .layer_count = 1,
        .batch_size  = 2,
    };
    int inputs[8];
    int targets[8];

    for (int i = 0; i < 8; i++) {
        inputs[i] = i % config.vocab_size;
        targets[i] = (i + 1) % config.vocab_size;
    }

    Model *model = model_new(config, 17);
    float full_loss = model_forward(model, inputs, targets, 2, 4);

    expect(fabsf(full_loss - 1.647271f) < 1e-5f,
           "seeded full-capacity forward matches its known loss");
    expect(model_forward(model, inputs, NULL, 2, 4) == 0.0f,
           "inference fills logits without inventing a training loss");

    int short_inputs[] = { 0, 1, 2 };
    int short_targets[] = { 1, 2, 3 };
    float short_loss =
        model_forward(model, short_inputs, short_targets, 1, 3);

    expect(fabsf(short_loss - 1.654009f) < 1e-5f,
           "short forward uses the current time shape, not arena capacity");

    short_targets[1] = 4;
    float changed_target_loss =
        model_forward(model, short_inputs, short_targets, 1, 3);

    expect(fabsf(changed_target_loss - short_loss) > 1e-4f,
           "whole-model cross-entropy responds to a changed target");

    model_free(model);
    if (failures != 0) {
        fprintf(stderr, "check-11: %d of %d checks failed\n",
                failures, checks);
        return EXIT_FAILURE;
    }
    printf("check-11: all %d model-forward checks passed\n", checks);
    return EXIT_SUCCESS;
}
