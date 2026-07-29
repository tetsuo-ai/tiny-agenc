/*
 * Chapter 15 model witness: learn a fixed batch, save it, and load it again.
 *
 * Sampling is intentionally absent. Chapter 15 must pass before the
 * Chapter 16 generation loop exists.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "param.h"
#include "tokenizer.h"

static int checks;
static int failures;

static void expect(int condition, const char *message)
{
    checks++;
    if (condition)
        return;
    fprintf(stderr, "check-15: %s\n", message);
    failures++;
}

static uint32_t float_bits(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof bits);
    return bits;
}

static int finite_float(float value)
{
    return (float_bits(value) & 0x7F800000u) != 0x7F800000u;
}

static int models_equal(const Model *first, const Model *second)
{
    ModelParams a = model_params(first);
    ModelParams b = model_params(second);

    if (a.count != b.count)
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

int main(int argc, char **argv)
{
    if (argc != 2) {
        fputs("usage: check-15 CHECKPOINT\n", stderr);
        return EXIT_FAILURE;
    }

    static const char TRAINING_TEXT[] = "\nabcabcabcabcabcabcabcabc\n";
    static const char INPUT_TEXT[] = "\nabc\nabc";
    static const char TARGET_TEXT[] = "abc\nabc\n";
    enum { BATCH = 2, TIME = 4, TOKENS = BATCH * TIME, STEPS = 60 };

    Tokenizer *tokenizer =
        tokenizer_new(TRAINING_TEXT, sizeof TRAINING_TEXT - 1);
    ModelConfig config = {
        .vocab_size  = 4,
        .block_size  = TIME,
        .d_model     = 8,
        .head_count  = 2,
        .layer_count = 1,
        .batch_size  = BATCH,
    };
    Model *model = model_new(config, 2024);
    int inputs[TOKENS];
    int targets[TOKENS];

    expect(tokenizer_encode(tokenizer, inputs, INPUT_TEXT,
                            sizeof INPUT_TEXT - 1) == TOKENS,
           "fixed inputs encode completely");
    expect(tokenizer_encode(tokenizer, targets, TARGET_TEXT,
                            sizeof TARGET_TEXT - 1) == TOKENS,
           "fixed targets encode completely");

    float initial_loss =
        model_forward(model, inputs, targets, BATCH, TIME);
    AdamW optimizer = {
        .learning_rate = 0.02f,
        .beta1         = 0.9f,
        .beta2         = 0.999f,
        .epsilon       = 1e-8f,
        .weight_decay  = 0.01f,
    };

    for (int step = 1; step <= STEPS; step++) {
        model_zero_gradients(model);
        (void)model_forward(model, inputs, targets, BATCH, TIME);
        model_backward(model);
        expect(model_step(model, optimizer, step) == 0,
               "the fixed-batch optimizer step stays finite");
    }

    float trained_loss =
        model_forward(model, inputs, targets, BATCH, TIME);

    expect(finite_float(initial_loss) && finite_float(trained_loss),
           "training losses remain finite");
    expect(trained_loss < initial_loss * 0.5f,
           "the fixed-batch loss falls by at least half");
    expect(model_save(model, tokenizer, argv[1]) == 0,
           "the trained model saves a checkpoint");

    Tokenizer *loaded_tokenizer = NULL;
    Model *loaded = model_load(&loaded_tokenizer, argv[1]);

    expect(loaded != NULL && loaded_tokenizer != NULL,
           "the checkpoint loads without sampling");
    if (loaded != NULL && loaded_tokenizer != NULL) {
        expect(models_equal(model, loaded),
               "checkpoint loading preserves every parameter bit");

        float loaded_loss =
            model_forward(loaded, inputs, targets, BATCH, TIME);

        expect(float_bits(loaded_loss) == float_bits(trained_loss),
               "checkpoint loading preserves the forward loss");
    }

    if (loaded != NULL)
        model_free(loaded);
    if (loaded_tokenizer != NULL)
        tokenizer_free(loaded_tokenizer);
    model_free(model);
    tokenizer_free(tokenizer);
    remove(argv[1]);

    if (failures != 0) {
        fprintf(stderr, "check-15: %d of %d checks failed\n", failures, checks);
        return EXIT_FAILURE;
    }
    printf("check-15: all %d training-loop checks passed\n", checks);
    return EXIT_SUCCESS;
}
