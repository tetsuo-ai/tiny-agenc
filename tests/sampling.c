/*
 * Sampling witness: generation fills every requested position, a
 * seeded categorical draw follows a known uniform distribution, a
 * one-id vocabulary still consumes one draw, temperature changes a
 * nonuniform draw, and an evicted context uses the freshest block.
 */
#include <stdio.h>
#include <stdlib.h>

#include "model.h"
#include "param.h"
#include "rng.h"

static int checks;
static int failures;

enum {
    FIRST_SEED = 1,
    SEED_SEARCH_LIMIT = 10000,
    MODEL_SEED = 17,
    SAMPLE_SEED = 777,
    PROMPT_TOKEN_COUNT = 1,
};

static const float DEFAULT_TEST_TEMPERATURE = 0.8f;
static const float COLD_TEMPERATURE = 0.05f;
static const float HOT_TEMPERATURE = 100.0f;
static const float TEMPERATURE_DRAW_LOW = 0.90f;
static const float TEMPERATURE_DRAW_HIGH = 0.95f;

static void expect(int condition, const char *message);
static int uniform_expected(Rng *rng, int vocab_size);
static void zero_parameters(Model *model);
static unsigned long long seed_with_first_draw_between(float low, float high);
static void check_uniform_generation(void);
static void check_one_id_consumes_one_draw(void);
static Model *controlled_model(void);
static void check_temperature(void);
static void check_fresh_context(void);
int main(void);

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
    for (unsigned long long seed = FIRST_SEED;
         seed < SEED_SEARCH_LIMIT; seed++) {
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
    Model *model = model_new(config, MODEL_SEED);

    /* Zero parameters make every logit exactly equal. Sampling then has
     * a simple independent answer: walk five equal probability bins. */
    zero_parameters(model);

    int actual[TOTAL];
    int expected[TOTAL];

    actual[0] = 0;
    expected[0] = 0;
    for (int i = PROMPT_TOKEN_COUNT; i < TOTAL; i++) {
        actual[i] = -1;
        expected[i] = -1;
    }

    Rng *actual_rng = rng_new(SAMPLE_SEED);
    Rng *expected_rng = rng_new(SAMPLE_SEED);

    for (int i = PROMPT_TOKEN_COUNT; i < TOTAL; i++)
        expected[i] = uniform_expected(expected_rng, VOCAB);
    model_sample(model, actual_rng, actual, PROMPT_TOKEN_COUNT, TOTAL,
                 DEFAULT_TEST_TEMPERATURE);

    for (int i = PROMPT_TOKEN_COUNT; i < TOTAL; i++) {
        expect(actual[i] >= 0 && actual[i] < VOCAB,
               "sampling fills a valid vocabulary id");
        expect(actual[i] == expected[i],
               "seeded uniform sampling chooses the expected id");
    }

    rng_free(actual_rng);
    rng_free(expected_rng);
    model_free(model);
}

static void check_one_id_consumes_one_draw(void)
{
    ModelConfig config = {
        .vocab_size  = 1,
        .block_size  = 1,
        .d_model     = 1,
        .head_count  = 1,
        .layer_count = 1,
        .batch_size  = 1,
    };
    Model *model = model_new(config, MODEL_SEED);
    int ids[2] = { 0, -1 };
    Rng *actual_rng = rng_new(SAMPLE_SEED);
    Rng *expected_rng = rng_new(SAMPLE_SEED);

    (void)rng_uniform(expected_rng);
    model_sample(model, actual_rng, ids, PROMPT_TOKEN_COUNT,
                 (int)(sizeof ids / sizeof ids[0]),
                 DEFAULT_TEST_TEMPERATURE);

    expect(ids[1] == 0, "a one-id vocabulary can only select id zero");
    expect(rng_uniform(actual_rng) == rng_uniform(expected_rng),
           "a one-id vocabulary still consumes exactly one uniform draw");

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
        VALUE_FIRST_CHANNEL_ROW = 4,
        VALUE_SECOND_CHANNEL_ROW = 5,
    };
    ModelConfig config = {
        .vocab_size  = 3,
        .block_size  = 3,
        .d_model     = 2,
        .head_count  = 1,
        .layer_count = 1,
        .batch_size  = 1,
    };
    Model *model = model_new(config, MODEL_SEED);

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
    mat_row(qkv_weights, VALUE_FIRST_CHANNEL_ROW)[0] = 1.0f;
    mat_row(qkv_weights, VALUE_SECOND_CHANNEL_ROW)[1] = 1.0f;
    mat_row(proj_weights, 0)[0] = 4.0f;
    mat_row(proj_weights, 1)[1] = 4.0f;
    final_gain.vals[0] = 1.0f;
    final_gain.vals[1] = 1.0f;
    return model;
}

static void check_temperature(void)
{
    Model *model = controlled_model();
    unsigned long long seed =
        seed_with_first_draw_between(TEMPERATURE_DRAW_LOW,
                                     TEMPERATURE_DRAW_HIGH);
    int cold[2] = { 0, -1 };
    int hot[2] = { 0, -1 };
    Rng *cold_rng = rng_new(seed);
    Rng *hot_rng = rng_new(seed);

    model_sample(model, cold_rng, cold, PROMPT_TOKEN_COUNT,
                 (int)(sizeof cold / sizeof cold[0]), COLD_TEMPERATURE);
    model_sample(model, hot_rng, hot, PROMPT_TOKEN_COUNT,
                 (int)(sizeof hot / sizeof hot[0]), HOT_TEMPERATURE);

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
    enum {
        EVICTED_PROMPT_TOKENS = 6,
        EXPLICIT_TAIL_TOKENS = 3,
    };
    Model *model = controlled_model();
    int evicted[EVICTED_PROMPT_TOKENS + 1] =
        { 0, 0, 0, 1, 1, 0, -1 };
    int explicit_tail[EXPLICIT_TAIL_TOKENS + 1] = { 1, 1, 0, -1 };
    Rng *evicted_rng = rng_new(SAMPLE_SEED);
    Rng *tail_rng = rng_new(SAMPLE_SEED);

    model_sample(model, evicted_rng, evicted, EVICTED_PROMPT_TOKENS,
                 (int)(sizeof evicted / sizeof evicted[0]),
                 COLD_TEMPERATURE);
    model_sample(model, tail_rng, explicit_tail, EXPLICIT_TAIL_TOKENS,
                 (int)(sizeof explicit_tail / sizeof explicit_tail[0]),
                 COLD_TEMPERATURE);

    expect(evicted[EVICTED_PROMPT_TOKENS] == 1,
           "a post-eviction draw uses information across the retained block");
    expect(evicted[EVICTED_PROMPT_TOKENS]
               == explicit_tail[EXPLICIT_TAIL_TOKENS],
           "eviction agrees with an explicit last-block reference");

    rng_free(evicted_rng);
    rng_free(tail_rng);
    model_free(model);
}

int main(void)
{
    check_uniform_generation();
    check_one_id_consumes_one_draw();
    check_temperature();
    check_fresh_context();

    if (failures != 0) {
        fprintf(stderr, "check-16: %d of %d checks failed\n", failures, checks);
        return EXIT_FAILURE;
    }
    printf("check-16: all %d sampling checks passed\n", checks);
    return EXIT_SUCCESS;
}
