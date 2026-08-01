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
static const uint32_t FLOAT_EXPONENT_MASK = 0x7F800000u;
enum {
    FIXED_BATCH_SIZE = 2,
    FIXED_TIME = 4,
    FIXED_TOKEN_COUNT = FIXED_BATCH_SIZE * FIXED_TIME,
    TRAINING_STEPS = 60,
    FIRST_TRAINING_STEP = 1,
};

static const unsigned long long MODEL_SEED = 2024;
static const float LOSS_REMAINING_FRACTION = 0.5f;
static const float LEARNING_RATE = 0.02f;
static const float ADAM_BETA1 = 0.9f;
static const float ADAM_BETA2 = 0.999f;
static const float ADAM_EPSILON = 1e-8f;
static const float WEIGHT_DECAY = 0.01f;
static const char TRAINING_TEXT[] = "\nabcabcabcabcabcabcabcabc\n";
static const char INPUT_TEXT[] = "\nabc\nabc";
static const char TARGET_TEXT[] = "abc\nabc\n";

typedef struct {
    Tokenizer *tokenizer;
    Model *model;
    int inputs[FIXED_TOKEN_COUNT];
    int targets[FIXED_TOKEN_COUNT];
    float initial_loss;
    float trained_loss;
} TrainingFixture;

static void expect(int condition, const char *message);
static uint32_t float_bits(float value);
static int finite_float(float value);
static int models_equal(const Model *first, const Model *second);
static AdamW training_optimizer(void);
static TrainingFixture training_fixture(void);
static void train_fixed_batch(TrainingFixture *fixture);
static void check_checkpoint_round_trip(const TrainingFixture *fixture,
                                        const char *path);
static void training_fixture_free(TrainingFixture *fixture);
int main(int argc, char **argv);

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
    return (float_bits(value) & FLOAT_EXPONENT_MASK)
        != FLOAT_EXPONENT_MASK;
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

static AdamW training_optimizer(void)
{
    AdamW optimizer = {
        .learning_rate = LEARNING_RATE,
        .beta1         = ADAM_BETA1,
        .beta2         = ADAM_BETA2,
        .epsilon       = ADAM_EPSILON,
        .weight_decay  = WEIGHT_DECAY,
    };

    return optimizer;
}

static TrainingFixture training_fixture(void)
{
    ModelConfig config = {
        .vocab_size  = 4,
        .block_size  = FIXED_TIME,
        .d_model     = 8,
        .head_count  = 2,
        .layer_count = 1,
        .batch_size  = FIXED_BATCH_SIZE,
    };
    TrainingFixture fixture = {
        .tokenizer = tokenizer_new(
            TRAINING_TEXT, sizeof TRAINING_TEXT - 1),
    };

    fixture.model = model_new(config, MODEL_SEED);
    expect(tokenizer_encode(fixture.tokenizer, fixture.inputs,
                            INPUT_TEXT, sizeof INPUT_TEXT - 1)
               == FIXED_TOKEN_COUNT,
           "fixed inputs encode completely");
    expect(tokenizer_encode(fixture.tokenizer, fixture.targets,
                            TARGET_TEXT, sizeof TARGET_TEXT - 1)
               == FIXED_TOKEN_COUNT,
           "fixed targets encode completely");
    fixture.initial_loss = model_forward(
        fixture.model, fixture.inputs, fixture.targets,
        FIXED_BATCH_SIZE, FIXED_TIME);
    return fixture;
}

static void train_fixed_batch(TrainingFixture *fixture)
{
    AdamW optimizer = training_optimizer();

    for (int step = FIRST_TRAINING_STEP; step <= TRAINING_STEPS; step++) {
        model_zero_gradients(fixture->model);
        (void)model_forward(fixture->model, fixture->inputs,
                            fixture->targets,
                            FIXED_BATCH_SIZE, FIXED_TIME);
        model_backward(fixture->model);
        expect(model_step(fixture->model, optimizer, step) == 0,
               "the fixed-batch optimizer step stays finite");
    }

    fixture->trained_loss = model_forward(
        fixture->model, fixture->inputs, fixture->targets,
        FIXED_BATCH_SIZE, FIXED_TIME);
    expect(finite_float(fixture->initial_loss)
               && finite_float(fixture->trained_loss),
           "training losses remain finite");
    expect(fixture->trained_loss
               < fixture->initial_loss * LOSS_REMAINING_FRACTION,
           "the fixed-batch loss falls by at least half");
}

static void check_checkpoint_round_trip(const TrainingFixture *fixture,
                                        const char *path)
{
    expect(model_save(fixture->model, fixture->tokenizer, path) == 0,
           "the trained model saves a checkpoint");

    Tokenizer *loaded_tokenizer = NULL;
    Model *loaded = model_load(&loaded_tokenizer, path);

    expect(loaded != NULL && loaded_tokenizer != NULL,
           "the checkpoint loads without sampling");
    if (loaded != NULL && loaded_tokenizer != NULL) {
        expect(models_equal(fixture->model, loaded),
               "checkpoint loading preserves every parameter bit");
        float loaded_loss = model_forward(
            loaded, fixture->inputs, fixture->targets,
            FIXED_BATCH_SIZE, FIXED_TIME);

        expect(float_bits(loaded_loss) == float_bits(fixture->trained_loss),
               "checkpoint loading preserves the forward loss");
    }

    if (loaded != NULL)
        model_free(loaded);
    if (loaded_tokenizer != NULL)
        tokenizer_free(loaded_tokenizer);
}

static void training_fixture_free(TrainingFixture *fixture)
{
    model_free(fixture->model);
    tokenizer_free(fixture->tokenizer);
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fputs("usage: check-15 CHECKPOINT\n", stderr);
        return EXIT_FAILURE;
    }

    TrainingFixture fixture = training_fixture();

    train_fixed_batch(&fixture);
    check_checkpoint_round_trip(&fixture, argv[1]);
    training_fixture_free(&fixture);
    remove(argv[1]);

    if (failures != 0) {
        fprintf(stderr, "check-15: %d of %d checks failed\n", failures, checks);
        return EXIT_FAILURE;
    }
    printf("check-15: all %d training-loop checks passed\n", checks);
    return EXIT_SUCCESS;
}
