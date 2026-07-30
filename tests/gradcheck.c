/*
 * gradcheck.c -- the referee for every hand-written backward pass.
 *
 * A derivative is a promise about what happens when you nudge an
 * input, so nudge it: for every element x of every input we compare
 * the analytic gradient against the central difference
 *
 *   (loss(x + h) - loss(x - h)) / 2h
 *
 * To check every gradient of an op with ONE backward call, the loss is
 * a random projection: loss = sum(out . u) for a fixed random u, which
 * makes d_loss/d_out exactly u.  Where analytic and numeric agree, the
 * calculus is right; a wrong sign or a forgotten term has nowhere to
 * hide.  A final test aims the same nudge at every parameter of a
 * miniature Model, which checks the wiring between ops too.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mat.h"
#include "model.h"
#include "ops.h"
#include "param.h"
#include "rng.h"
#include "util.h"

static const float NUDGE              = 1e-2f;   /* h: clears float noise */
static const float RELATIVE_TOLERANCE = 2e-2f;
static const float ABSOLUTE_TOLERANCE = 1e-3f;

/* Arbitrary but fixed, so every run checks the same numbers. */
static const unsigned long long CHECK_SEED = 42;
static const unsigned long long MODEL_SEED = 7;

static int checks;
static int failures;

static void compare(const char *label, float analytic, float numeric)
{
    float allowed = ABSOLUTE_TOLERANCE
                  + RELATIVE_TOLERANCE * fmaxf(fabsf(analytic), fabsf(numeric));

    checks++;
    if (fabsf(analytic - numeric) <= allowed)
        return;
    printf("FAIL %-28s analytic % .6f  numeric % .6f\n",
           label, (double)analytic, (double)numeric);
    failures++;
}

static void expect(int condition, const char *label)
{
    checks++;
    if (condition)
        return;
    printf("FAIL %s\n", label);
    failures++;
}

/*
 * The driver: `measure` runs an op's forward pass and returns the
 * scalar loss.  nudge_all wiggles every element of `target` and holds
 * the matching element of `analytic` to the central-difference answer.
 */
static void nudge_all(const char *label, Mat target, Mat analytic,
                      float (*measure)(void *), void *context)
{
    for (size_t i = 0; i < mat_size(target); i++) {
        float saved = target.vals[i];

        target.vals[i] = saved + NUDGE;
        float above = measure(context);

        target.vals[i] = saved - NUDGE;
        float below = measure(context);

        target.vals[i] = saved;
        compare(label, analytic.vals[i], (above - below) / (2.0f * NUDGE));
    }
}

/* -------- small builders -------- */

static Mat mat_new_gaussian(Rng *rng, int rows, int cols)
{
    Mat m = mat_make(ecalloc((size_t)rows * cols, sizeof(float)), rows, cols);

    for (size_t i = 0; i < mat_size(m); i++)
        m.vals[i] = rng_gaussian(rng);
    return m;
}

static Mat mat_new_zeros(int rows, int cols)
{
    return mat_make(ecalloc((size_t)rows * cols, sizeof(float)), rows, cols);
}

static float projected(Mat out, Mat u)
{
    float sum = 0.0f;

    for (size_t i = 0; i < mat_size(out); i++)
        sum += out.vals[i] * u.vals[i];
    return sum;
}

static void fill(Mat matrix, float value)
{
    for (size_t i = 0; i < mat_size(matrix); i++)
        matrix.vals[i] = value;
}

static void check_accumulated(const char *label, Mat actual,
                              Mat contribution, float initial)
{
    for (size_t i = 0; i < mat_size(actual); i++)
        compare(label, actual.vals[i], initial + contribution.vals[i]);
}

/* -------- per-op checks -------- */

typedef struct {
    Mat out, u;
    Mat x, w;                       /* matmul */
    Mat gain, bias;                 /* layernorm */
    float *means, *rstds;
    Mat scores, qkv;                /* attention */
    int time, head_count;
    Mat a, b;                       /* residual */
    Mat token_table, position_table;/* embedding */
    const int *tokens;
    Mat probs, logits;              /* crossentropy */
    const int *targets;
} Case;

static float measure_matmul(void *context)
{
    Case *c = context;

    matmul_forward(c->out, c->x, c->w);
    return projected(c->out, c->u);
}

static void check_matmul(Rng *rng)
{
    enum { ROWS = 5, IN = 4, OUT = 3 };
    Case c = { .out = mat_new_zeros(ROWS, OUT),
               .u   = mat_new_gaussian(rng, ROWS, OUT),
               .x   = mat_new_gaussian(rng, ROWS, IN),
               .w   = mat_new_gaussian(rng, OUT, IN) };
    Mat d_x = mat_new_zeros(ROWS, IN);
    Mat d_w = mat_new_zeros(OUT, IN);

    matmul_backward(d_x, d_w, c.u, c.x, c.w);
    nudge_all("matmul d_x", c.x, d_x, measure_matmul, &c);
    nudge_all("matmul d_weights", c.w, d_w, measure_matmul, &c);

    free(c.out.vals); free(c.u.vals); free(c.x.vals); free(c.w.vals);
    free(d_x.vals); free(d_w.vals);
}

static void check_matmul_accumulation(void)
{
    float x_values[] = { 2.0f, -1.0f };
    float weight_values[] = { 3.0f, 4.0f };
    float d_output[] = { 5.0f };
    float d_input[] = { 10.0f, 20.0f };
    float d_weight[] = { 1.0f, 2.0f };

    matmul_backward(mat_make(d_input, 1, 2),
                    mat_make(d_weight, 1, 2),
                    mat_make(d_output, 1, 1),
                    mat_make(x_values, 1, 2),
                    mat_make(weight_values, 1, 2));

    expect(d_input[0] == 25.0f && d_input[1] == 40.0f,
           "matmul adds into a nonzero input gradient");
    expect(d_weight[0] == 11.0f && d_weight[1] == -3.0f,
           "matmul adds into a nonzero weight gradient");
}

static float measure_layernorm(void *context)
{
    Case *c = context;

    layernorm_forward(c->out, c->means, c->rstds, c->x, c->gain.vals, c->bias.vals);
    return projected(c->out, c->u);
}

static void check_layernorm(Rng *rng)
{
    enum { ROWS = 4, COLS = 6 };
    Case c = { .out  = mat_new_zeros(ROWS, COLS),
               .u    = mat_new_gaussian(rng, ROWS, COLS),
               .x    = mat_new_gaussian(rng, ROWS, COLS),
               .gain = mat_new_gaussian(rng, 1, COLS),
               .bias = mat_new_gaussian(rng, 1, COLS),
               .means = ecalloc(ROWS, sizeof(float)),
               .rstds = ecalloc(ROWS, sizeof(float)) };
    Mat d_x    = mat_new_zeros(ROWS, COLS);
    Mat d_gain = mat_new_zeros(1, COLS);
    Mat d_bias = mat_new_zeros(1, COLS);

    measure_layernorm(&c);   /* fill means/rstds for the backward pass */
    layernorm_backward(d_x, d_gain.vals, d_bias.vals, c.u, c.x, c.gain.vals,
                       c.means, c.rstds);

    Mat accumulated_x    = mat_new_zeros(ROWS, COLS);
    Mat accumulated_gain = mat_new_zeros(1, COLS);
    Mat accumulated_bias = mat_new_zeros(1, COLS);

    fill(accumulated_x, 0.25f);
    fill(accumulated_gain, -0.5f);
    fill(accumulated_bias, 0.75f);
    layernorm_backward(accumulated_x, accumulated_gain.vals,
                       accumulated_bias.vals, c.u, c.x, c.gain.vals,
                       c.means, c.rstds);
    check_accumulated("layernorm accumulates d_x",
                      accumulated_x, d_x, 0.25f);
    check_accumulated("layernorm accumulates d_gain",
                      accumulated_gain, d_gain, -0.5f);
    check_accumulated("layernorm accumulates d_bias",
                      accumulated_bias, d_bias, 0.75f);

    nudge_all("layernorm d_x", c.x, d_x, measure_layernorm, &c);
    nudge_all("layernorm d_gain", c.gain, d_gain, measure_layernorm, &c);
    nudge_all("layernorm d_bias", c.bias, d_bias, measure_layernorm, &c);

    free(c.out.vals); free(c.u.vals); free(c.x.vals);
    free(c.gain.vals); free(c.bias.vals); free(c.means); free(c.rstds);
    free(d_x.vals); free(d_gain.vals); free(d_bias.vals);
    free(accumulated_x.vals);
    free(accumulated_gain.vals);
    free(accumulated_bias.vals);
}

static float measure_attention(void *context)
{
    Case *c = context;

    attention_forward(c->out, c->scores, c->qkv, c->time, c->head_count);
    return projected(c->out, c->u);
}

static void check_attention(Rng *rng)
{
    enum { SEQS = 2, TIME = 4, CHANNELS = 8, HEADS = 2, ROWS = SEQS * TIME };
    static const float INVISIBLE_SENTINEL = 12345.0f;
    Case c = { .out    = mat_new_zeros(ROWS, CHANNELS),
               .u      = mat_new_gaussian(rng, ROWS, CHANNELS),
               .qkv    = mat_new_gaussian(rng, ROWS, QKV_STREAMS * CHANNELS),
               .scores = mat_new_zeros(SEQS * HEADS * TIME, TIME),
               .time   = TIME,
               .head_count = HEADS };
    Mat d_qkv    = mat_new_zeros(ROWS, QKV_STREAMS * CHANNELS);
    Mat d_scores = mat_new_zeros(SEQS * HEADS * TIME, TIME);

    fill(d_scores, INVISIBLE_SENTINEL);
    measure_attention(&c);   /* fill scores for the backward pass */
    attention_backward(d_qkv, d_scores, c.u, c.qkv, c.scores, TIME, HEADS);

    for (int seq = 0; seq < SEQS; seq++)
        for (int head = 0; head < HEADS; head++)
            for (int t = 0; t < TIME; t++) {
                const float *d_score =
                    mat_row(d_scores, (seq * HEADS + head) * TIME + t);

                for (int t2 = t + 1; t2 < TIME; t2++)
                    expect(d_score[t2] == INVISIBLE_SENTINEL,
                           "attention leaves invisible d_scores untouched");
            }

    Mat accumulated_qkv =
        mat_new_zeros(ROWS, QKV_STREAMS * CHANNELS);
    Mat accumulated_scores =
        mat_new_zeros(SEQS * HEADS * TIME, TIME);

    fill(accumulated_qkv, 0.25f);
    attention_backward(accumulated_qkv, accumulated_scores, c.u,
                       c.qkv, c.scores, TIME, HEADS);
    check_accumulated("attention accumulates d_qkv",
                      accumulated_qkv, d_qkv, 0.25f);

    nudge_all("attention d_qkv", c.qkv, d_qkv, measure_attention, &c);

    free(c.out.vals); free(c.u.vals); free(c.qkv.vals); free(c.scores.vals);
    free(d_qkv.vals); free(d_scores.vals);
    free(accumulated_qkv.vals); free(accumulated_scores.vals);
}

static float measure_gelu(void *context)
{
    Case *c = context;

    gelu_forward(c->out, c->x);
    return projected(c->out, c->u);
}

static void check_gelu(Rng *rng)
{
    enum { ROWS = 4, COLS = 5 };
    Case c = { .out = mat_new_zeros(ROWS, COLS),
               .u   = mat_new_gaussian(rng, ROWS, COLS),
               .x   = mat_new_gaussian(rng, ROWS, COLS) };
    Mat d_x = mat_new_zeros(ROWS, COLS);

    gelu_backward(d_x, c.u, c.x);
    nudge_all("gelu d_x", c.x, d_x, measure_gelu, &c);

    free(c.out.vals); free(c.u.vals); free(c.x.vals); free(d_x.vals);
}

static float measure_residual(void *context)
{
    Case *c = context;

    residual_forward(c->out, c->a, c->b);
    return projected(c->out, c->u);
}

static void check_residual(Rng *rng)
{
    enum { ROWS = 3, COLS = 4 };
    Case c = { .out = mat_new_zeros(ROWS, COLS),
               .u   = mat_new_gaussian(rng, ROWS, COLS),
               .a   = mat_new_gaussian(rng, ROWS, COLS),
               .b   = mat_new_gaussian(rng, ROWS, COLS) };
    Mat d_a = mat_new_zeros(ROWS, COLS);
    Mat d_b = mat_new_zeros(ROWS, COLS);

    residual_backward(d_a, d_b, c.u);
    nudge_all("residual d_a", c.a, d_a, measure_residual, &c);
    nudge_all("residual d_b", c.b, d_b, measure_residual, &c);

    free(c.out.vals); free(c.u.vals); free(c.a.vals); free(c.b.vals);
    free(d_a.vals); free(d_b.vals);
}

static float measure_embedding(void *context)
{
    Case *c = context;

    embedding_forward(c->out, c->tokens, c->token_table, c->position_table, c->time);
    return projected(c->out, c->u);
}

static void check_embedding(Rng *rng)
{
    enum { VOCAB = 3, TIME = 3, ROWS = 6, COLS = 4 };
    /* Repeated ids prove gradients accumulate instead of overwrite. */
    static const int tokens[ROWS] = { 0, 2, 1, 2, 0, 0 };
    Case c = { .out            = mat_new_zeros(ROWS, COLS),
               .u              = mat_new_gaussian(rng, ROWS, COLS),
               .token_table    = mat_new_gaussian(rng, VOCAB, COLS),
               .position_table = mat_new_gaussian(rng, TIME, COLS),
               .tokens         = tokens,
               .time           = TIME };
    Mat d_tokens    = mat_new_zeros(VOCAB, COLS);
    Mat d_positions = mat_new_zeros(TIME, COLS);

    embedding_backward(d_tokens, d_positions, c.u, tokens, TIME);
    nudge_all("embedding d_token_table", c.token_table, d_tokens,
              measure_embedding, &c);
    nudge_all("embedding d_position_table", c.position_table, d_positions,
              measure_embedding, &c);

    free(c.out.vals); free(c.u.vals);
    free(c.token_table.vals); free(c.position_table.vals);
    free(d_tokens.vals); free(d_positions.vals);
}

static float measure_crossentropy(void *context)
{
    Case *c = context;

    return crossentropy_forward(c->probs, c->logits, c->targets);
}

static void check_crossentropy(Rng *rng)
{
    enum { ROWS = 5, VOCAB = 7 };
    static const int targets[ROWS] = { 3, 0, 6, 2, 2 };
    Case c = { .probs   = mat_new_zeros(ROWS, VOCAB),
               .logits  = mat_new_gaussian(rng, ROWS, VOCAB),
               .targets = targets };
    Mat d_logits = mat_new_zeros(ROWS, VOCAB);

    measure_crossentropy(&c);   /* fill probs for the backward pass */
    crossentropy_backward(d_logits, c.probs, targets);
    nudge_all("crossentropy d_logits", c.logits, d_logits,
              measure_crossentropy, &c);

    free(c.probs.vals); free(c.logits.vals); free(d_logits.vals);
}

/* -------- the whole model -------- */

typedef struct {
    Model     *model;
    const int *tokens, *targets;
    int        batch, time;
} ModelCase;

static float measure_model(void *context)
{
    ModelCase *c = context;

    return model_forward(c->model, c->tokens, c->targets, c->batch, c->time);
}

static void check_model(Rng *rng)
{
    enum { VOCAB = 13, BLOCK = 8, D_MODEL = 16, HEADS = 2, LAYERS = 2,
           BATCH = 2, TIME = 6, ROWS = BATCH * TIME };
    ModelConfig cfg = { .vocab_size = VOCAB, .block_size = BLOCK,
                        .d_model = D_MODEL, .head_count = HEADS,
                        .layer_count = LAYERS, .batch_size = BATCH };
    int tokens[ROWS], targets[ROWS];

    for (int i = 0; i < ROWS; i++) {
        tokens[i]  = rng_below(rng, cfg.vocab_size);
        targets[i] = rng_below(rng, cfg.vocab_size);
    }

    ModelCase c = { .model = model_new(cfg, MODEL_SEED), .tokens = tokens,
                    .targets = targets, .batch = BATCH, .time = TIME };

    model_zero_gradients(c.model);
    measure_model(&c);
    model_backward(c.model);

    /* Every parameter tensor of the tied, two-block transformer. */
    ModelParams p = model_params(c.model);

    enum { LABEL_CAPACITY = 32 };

    for (int i = 0; i < p.count; i++) {
        char label[LABEL_CAPACITY];

        snprintf(label, sizeof label, "model param[%d]", i);
        nudge_all(label, param_values(p.params[i]),
                  param_gradient(p.params[i]), measure_model, &c);
    }

    model_free(c.model);
}

/* -------- the optimizer --------
 *
 * Not a gradient check: AdamW is walked against an independent
 * double-precision transcription of the published update rule
 * (Loshchilov & Hutter 2019), decay included, for a few steps.
 */
static void run_adamw_reference(const char *label, Param *p, AdamW opt,
                                int steps, float decay, Rng *rng)
{
    enum { MOMENTS = 2 };
    size_t  count = mat_size(param_values(p));
    double *shadow = ecalloc((MOMENTS + 1) * count, sizeof *shadow);
    double *first  = shadow + count;
    double *second = shadow + 2 * count;

    for (size_t i = 0; i < count; i++)
        shadow[i] = (double)param_values(p).vals[i];

    for (int step = 1; step <= steps; step++) {
        Mat gradient = param_gradient(p);

        for (size_t i = 0; i < count; i++)
            gradient.vals[i] = rng_gaussian(rng);
        for (size_t i = 0; i < count; i++) {
            double g = (double)gradient.vals[i];

            first[i]  = opt.beta1 * first[i] + (1.0 - opt.beta1) * g;
            second[i] = opt.beta2 * second[i] + (1.0 - opt.beta2) * g * g;

            double m_hat = first[i] / (1.0 - pow(opt.beta1, step));
            double v_hat = second[i] / (1.0 - pow(opt.beta2, step));

            shadow[i] -= opt.learning_rate
                       * (m_hat / (sqrt(v_hat) + opt.epsilon) + decay * shadow[i]);
        }
        checks++;
        if (param_adamw_step(p, opt, step) != 0) {
            printf("FAIL %-28s optimizer rejected a finite step\n", label);
            failures++;
            break;
        }
    }
    for (size_t i = 0; i < count; i++)
        compare(label, param_values(p).vals[i], (float)shadow[i]);
    free(shadow);
}

static void check_adamw(Rng *rng)
{
    enum { ROWS = 3, COLS = 2, STEPS = 4 };
    AdamW opt = { .learning_rate = 0.1f, .beta1 = 0.9f, .beta2 = 0.999f,
                  .epsilon = 1e-8f, .weight_decay = 0.01f };
    Param *matrix = param_new_gaussian(ROWS, COLS, 1.0f, rng);
    Param *vector = param_new_gaussian(1, COLS, 1.0f, rng);

    /* Matrices feel weight decay; vectors must not. */
    run_adamw_reference("adamw matrix", matrix, opt, STEPS, opt.weight_decay, rng);
    run_adamw_reference("adamw vector", vector, opt, STEPS, 0.0f, rng);

    param_free(matrix);
    param_free(vector);
}

static void usage(const char *program)
{
    fprintf(stderr, "usage: %s [backward|model|optimizer]\n", program);
}

int main(int argc, char **argv)
{
    const char *group = argc == 2 ? argv[1] : "all";
    int backward = strcmp(group, "all") == 0 || strcmp(group, "backward") == 0;
    int model = strcmp(group, "all") == 0 || strcmp(group, "model") == 0;
    int optimizer = strcmp(group, "all") == 0 || strcmp(group, "optimizer") == 0;

    if (argc > 2 || (!backward && !model && !optimizer)) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    Rng *rng = rng_new(CHECK_SEED);

    if (backward) {
        check_matmul(rng);
        check_matmul_accumulation();
        check_layernorm(rng);
        check_attention(rng);
        check_gelu(rng);
        check_residual(rng);
        check_embedding(rng);
        check_crossentropy(rng);
    }
    if (optimizer)
        check_adamw(rng);
    if (model)
        check_model(rng);

    rng_free(rng);
    if (failures > 0) {
        if (strcmp(group, "all") == 0)
            printf("gradcheck: %d of %d checks FAILED\n", failures, checks);
        else
            printf("gradcheck[%s]: %d of %d checks FAILED\n",
                   group, failures, checks);
        return EXIT_FAILURE;
    }
    if (strcmp(group, "all") == 0)
        printf("gradcheck: all %d checks passed\n", checks);
    else
        printf("gradcheck[%s]: all %d checks passed\n", group, checks);
    return EXIT_SUCCESS;
}
