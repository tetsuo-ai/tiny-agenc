/*
 * Sampling witness: generation fills every requested position, a
 * seeded categorical draw follows a known uniform distribution,
 * temperature changes a nonuniform draw, and an evicted context uses
 * the freshest block.
 */
#include <stdio.h>
#include <stdlib.h>

#include "model.h"
#include "param.h"
#include "rng.h"

static int checks;
static int failures;

static void expect(int condition, const char *message)
{
    checks++;
    if (condition)
        return;
    fprintf(stderr, "check-16: %s\n", message);
    failures++;
}

static int uniform_expected(Rng *rng, int vocab_size)
{
    float draw = rng_uniform(rng);
    float probability = 1.0f / (float)vocab_size;
    float cumulative = 0.0f;

    for (int id = 0; id < vocab_size; id++) {
        cumulative += probability;
        if (draw < cumulative)
            return id;
    }
    return vocab_size - 1;
}

static void zero_parameters(Model *model)
{
    ModelParams params = model_params(model);

    for (int p = 0; p < params.count; p++) {
        Mat values = param_values(params.params[p]);

        for (size_t i = 0; i < mat_size(values); i++)
            values.vals[i] = 0.0f;
    }
}

static unsigned long long seed_with_first_draw_between(float low, float high)
{
    for (unsigned long long seed = 1; seed < 10000; seed++) {
        Rng  *rng = rng_new(seed);
        float draw = rng_uniform(rng);

        rng_free(rng);
        if (draw > low && draw < high)
            return seed;
    }
    fprintf(stderr, "check-16: could not find a bounded known-answer draw\n");
    exit(EXIT_FAILURE);
}

static void check_uniform_generation(void)
{
    enum { VOCAB = 5, BLOCK = 3, TOTAL = 12 };
    ModelConfig config = {
        .vocab_size  = VOCAB,
        .block_size  = BLOCK,
        .d_model     = 4,
        .head_count  = 1,
        .layer_count = 1,
        .batch_size  = 1,
    };
    Model *model = model_new(config, 17);

    /* Zero parameters make every logit exactly equal. Sampling then has
     * a simple independent answer: walk five equal probability bins. */
    zero_parameters(model);

    int actual[TOTAL];
    int expected[TOTAL];

    actual[0] = 0;
    expected[0] = 0;
    for (int i = 1; i < TOTAL; i++) {
        actual[i] = -1;
        expected[i] = -1;
    }

    Rng *actual_rng = rng_new(777);
    Rng *expected_rng = rng_new(777);

    for (int i = 1; i < TOTAL; i++)
        expected[i] = uniform_expected(expected_rng, VOCAB);
    model_sample(model, actual_rng, actual, 1, TOTAL, 0.8f);

    for (int i = 1; i < TOTAL; i++) {
        expect(actual[i] >= 0 && actual[i] < VOCAB,
               "sampling fills a valid vocabulary id");
        expect(actual[i] == expected[i],
               "seeded uniform sampling chooses the expected id");
    }

    rng_free(actual_rng);
    rng_free(expected_rng);
    model_free(model);
}

static Model *controlled_model(void)
{
    enum {
        TOKEN_TABLE = 0,
        NORM1_GAIN = 2,
        QKV_WEIGHTS = 4,
        PROJ_WEIGHTS = 5,
    };
    ModelConfig config = {
        .vocab_size  = 3,
        .block_size  = 3,
        .d_model     = 2,
        .head_count  = 1,
        .layer_count = 1,
        .batch_size  = 1,
    };
    Model *model = model_new(config, 17);

    zero_parameters(model);

    ModelParams params = model_params(model);
    Mat token_table = param_values(params.params[TOKEN_TABLE]);
    Mat norm1_gain = param_values(params.params[NORM1_GAIN]);
    Mat qkv_weights = param_values(params.params[QKV_WEIGHTS]);
    Mat proj_weights = param_values(params.params[PROJ_WEIGHTS]);
    Mat final_gain = param_values(params.params[params.count - 2]);

    /* Tokens 0 and 1 point in opposite directions. Zero queries and
     * keys make attention average the visible values. The projection
     * amplifies that average enough for two earlier token 1 values to
     * overturn a final token 0. Sampling therefore depends on the
     * retained prefix, not merely its newest token. */
    mat_row(token_table, 0)[0] = 1.0f;
    mat_row(token_table, 0)[1] = -1.0f;
    mat_row(token_table, 1)[0] = -1.0f;
    mat_row(token_table, 1)[1] = 1.0f;
    norm1_gain.vals[0] = 1.0f;
    norm1_gain.vals[1] = 1.0f;
    mat_row(qkv_weights, 4)[0] = 1.0f;
    mat_row(qkv_weights, 5)[1] = 1.0f;
    mat_row(proj_weights, 0)[0] = 4.0f;
    mat_row(proj_weights, 1)[1] = 4.0f;
    final_gain.vals[0] = 1.0f;
    final_gain.vals[1] = 1.0f;
    return model;
}

static void check_temperature(void)
{
    Model *model = controlled_model();
    unsigned long long seed = seed_with_first_draw_between(0.90f, 0.95f);
    int cold[2] = { 0, -1 };
    int hot[2] = { 0, -1 };
    Rng *cold_rng = rng_new(seed);
    Rng *hot_rng = rng_new(seed);

    model_sample(model, cold_rng, cold, 1, 2, 0.05f);
    model_sample(model, hot_rng, hot, 1, 2, 100.0f);

    expect(cold[1] == 0,
           "low temperature concentrates a nonuniform draw on the winner");
    expect(hot[1] != 0,
           "high temperature flattens the same seeded nonuniform draw");
    expect(cold[1] != hot[1],
           "temperature changes an observable categorical result");

    rng_free(cold_rng);
    rng_free(hot_rng);
    model_free(model);
}

static void check_fresh_context(void)
{
    Model *model = controlled_model();
    int evicted[7] = { 0, 0, 0, 1, 1, 0, -1 };
    int explicit_tail[4] = { 1, 1, 0, -1 };
    Rng *evicted_rng = rng_new(777);
    Rng *tail_rng = rng_new(777);

    model_sample(model, evicted_rng, evicted, 6, 7, 0.05f);
    model_sample(model, tail_rng, explicit_tail, 3, 4, 0.05f);

    expect(evicted[6] == 1,
           "a post-eviction draw uses information across the retained block");
    expect(evicted[6] == explicit_tail[3],
           "eviction agrees with an explicit last-block reference");

    rng_free(evicted_rng);
    rng_free(tail_rng);
    model_free(model);
}

int main(void)
{
    check_uniform_generation();
    check_temperature();
    check_fresh_context();

    if (failures != 0) {
        fprintf(stderr, "check-16: %d of %d checks failed\n", failures, checks);
        return EXIT_FAILURE;
    }
    printf("check-16: all %d sampling checks passed\n", checks);
    return EXIT_SUCCESS;
}
