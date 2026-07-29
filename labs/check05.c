/*
 * Chapter 5 witness: independent forward examples.
 *
 * No backward symbol is referenced here. A learner can complete the
 * forward half of ops.c before Chapter 6 exists.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ops.h"

static int checks;
static int failures;

static void expect(int condition, const char *message)
{
    checks++;
    if (condition)
        return;
    fprintf(stderr, "check-05: %s\n", message);
    failures++;
}

static int close_float(float actual, float expected, float tolerance)
{
    return fabsf(actual - expected) <= tolerance;
}

static void check_embedding(void)
{
    enum { TIME = 3, CHANNELS = 2 };
    float token_values[] = {
        0.0f, 0.0f,
        10.0f, 20.0f,
    };
    float position_values[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
    };
    int tokens[] = { 0, 1, 0, 0, 1, 0 };
    float output[2 * TIME * CHANNELS] = { 0 };

    embedding_forward(mat_make(output, 2 * TIME, CHANNELS), tokens,
                      mat_make(token_values, 2, CHANNELS),
                      mat_make(position_values, TIME, CHANNELS), TIME);

    expect(output[0] == 1.0f && output[1] == 2.0f,
           "embedding adds token zero and position zero");
    expect(output[2] == 13.0f && output[3] == 24.0f,
           "embedding adds a nonzero token row");
    expect(output[6] == 1.0f && output[7] == 2.0f,
           "position indices restart for the next sequence");
}

static void check_layernorm(void)
{
    float input[] = { 1.0f, 2.0f, 3.0f };
    float gain[] = { 2.0f, -1.0f, 0.5f };
    float bias[] = { 0.25f, 1.0f, -2.0f };
    float output[3] = { 0 };
    float mean = 0.0f;
    float rstd = 0.0f;

    layernorm_forward(mat_make(output, 1, 3), &mean, &rstd,
                      mat_make(input, 1, 3), gain, bias);

    expect(mean == 2.0f, "layernorm records the row mean");
    expect(rstd > 1.22f && rstd < 1.23f,
           "layernorm records the reciprocal standard deviation");
    expect(close_float(output[0], -2.19947f, 1e-5f)
           && close_float(output[1], 1.0f, 1e-6f)
           && close_float(output[2], -1.38763f, 1e-5f),
           "layernorm applies learned gain and bias after normalization");
}

static void check_matmul(void)
{
    float x_values[] = {
         1.0f, 2.0f, 3.0f,
        -1.0f, 0.0f, 2.0f,
    };
    float weight_values[] = {
        1.0f, 0.0f, -1.0f,
        2.0f, 1.0f,  0.5f,
    };
    float output[4] = { 0 };

    matmul_forward(mat_make(output, 2, 2),
                   mat_make(x_values, 2, 3),
                   mat_make(weight_values, 2, 3));

    expect(output[0] == -2.0f, "matmul row zero, output zero");
    expect(output[1] ==  5.5f, "matmul row zero, output one");
    expect(output[2] == -3.0f, "matmul row one, output zero");
    expect(output[3] == -1.0f, "matmul row one, output one");
}

static void check_attention(void)
{
    enum { TIME = 3, CHANNELS = 2, COLS = QKV_STREAMS * CHANNELS };
    float qkv_values[TIME * COLS] = {
        1.0f, 0.0f,  1.0f, 0.0f,  2.0f, 1.0f,
        0.0f, 1.0f,  0.0f, 1.0f,  4.0f, 3.0f,
        1.0f, 1.0f,  1.0f, 1.0f,  8.0f, 7.0f,
    };
    float output[TIME * CHANNELS] = { 0 };
    float scores[TIME * TIME] = { 0 };

    attention_forward(mat_make(output, TIME, CHANNELS),
                      mat_make(scores, TIME, TIME),
                      mat_make(qkv_values, TIME, COLS), TIME, 1);

    expect(close_float(scores[3], 0.330238f, 1e-5f)
           && close_float(scores[4], 0.669762f, 1e-5f),
           "attention uses query-key scores, scaling, and softmax");
    expect(close_float(output[2], 3.33952f, 1e-5f)
           && close_float(output[3], 2.33952f, 1e-5f),
           "attention mixes values with the exact nonuniform weights");

    float earlier[2 * CHANNELS];

    memcpy(earlier, output, sizeof earlier);
    for (int c = 0; c < COLS; c++)
        qkv_values[2 * COLS + c] = 1000.0f + (float)c;
    attention_forward(mat_make(output, TIME, CHANNELS),
                      mat_make(scores, TIME, TIME),
                      mat_make(qkv_values, TIME, COLS), TIME, 1);

    expect(memcmp(earlier, output, sizeof earlier) == 0,
           "future QKV cannot change an earlier output");

    enum {
        HEAD_TIME = 2,
        HEAD_CHANNELS = 4,
        HEAD_COLS = QKV_STREAMS * HEAD_CHANNELS,
    };
    float two_head_qkv[HEAD_TIME * HEAD_COLS] = {
        0, 0, 0, 0,  0, 0, 0, 0,   1, 2, 10, 20,
        0, 0, 0, 0,  0, 0, 0, 0,   3, 4, 30, 40,
    };
    float two_head_output[HEAD_TIME * HEAD_CHANNELS] = { 0 };
    float two_head_scores[2 * HEAD_TIME * HEAD_TIME] = { 0 };

    attention_forward(mat_make(two_head_output, HEAD_TIME, HEAD_CHANNELS),
                      mat_make(two_head_scores, 2 * HEAD_TIME, HEAD_TIME),
                      mat_make(two_head_qkv, HEAD_TIME, HEAD_COLS),
                      HEAD_TIME, 2);
    expect(two_head_output[4] == 2.0f && two_head_output[5] == 3.0f
           && two_head_output[6] == 20.0f
           && two_head_output[7] == 30.0f,
           "attention keeps head value slices isolated");
}

static void check_elementwise(void)
{
    float gelu_input[] = { -1.0f, 0.0f, 1.0f };
    float gelu_output[3] = { 0 };

    gelu_forward(mat_make(gelu_output, 1, 3),
                 mat_make(gelu_input, 1, 3));
    expect(close_float(gelu_output[0], -0.158808f, 1e-5f),
           "GELU maps negative one to its known value");
    expect(gelu_output[1] == 0.0f, "GELU maps zero to zero");
    expect(close_float(gelu_output[2], 0.841192f, 1e-5f),
           "GELU maps positive one to its known value");

    float left[] = { 1.0f, -2.0f, 4.5f };
    float right[] = { 3.0f, 5.0f, 0.5f };
    float sum[3] = { 0 };

    residual_forward(mat_make(sum, 1, 3),
                     mat_make(left, 1, 3),
                     mat_make(right, 1, 3));
    expect(sum[0] == 4.0f && sum[1] == 3.0f && sum[2] == 5.0f,
           "residual performs elementwise addition");
}

static void check_probabilities(void)
{
    float distribution[] = { 1000.0f, 1000.0f, 999.0f };

    softmax_in_place(distribution, 3);
    expect(close_float(distribution[0] + distribution[1] + distribution[2],
                       1.0f, 1e-6f),
           "stable softmax sums to one");
    expect(distribution[0] == distribution[1],
           "equal logits receive equal probabilities");
    expect(distribution[0] > distribution[2],
           "a larger logit receives more probability");

    float logits[] = {
         2.0f, 0.0f, -1.0f,
        -1.0f, 0.0f,  2.0f,
    };
    float probabilities[6] = { 0 };
    int targets[] = { 0, 1 };
    float loss = crossentropy_forward(mat_make(probabilities, 2, 3),
                                      mat_make(logits, 2, 3), targets);

    expect(close_float(loss, 1.169846f, 1e-6f),
           "cross-entropy selects the requested target in each row");
    expect(close_float(probabilities[0], 0.843795f, 1e-6f)
           && close_float(probabilities[1], 0.114195f, 1e-6f)
           && close_float(probabilities[2], 0.042010f, 1e-6f)
           && close_float(probabilities[3], 0.042010f, 1e-6f)
           && close_float(probabilities[4], 0.114195f, 1e-6f)
           && close_float(probabilities[5], 0.843795f, 1e-6f),
           "cross-entropy retains the exact nonuniform distributions");

    int changed_targets[] = { 2, 2 };
    float changed_loss =
        crossentropy_forward(mat_make(probabilities, 2, 3),
                             mat_make(logits, 2, 3), changed_targets);

    expect(close_float(changed_loss, 1.669846f, 1e-6f)
           && !close_float(changed_loss, loss, 1e-3f),
           "changing only target ids changes the loss");
}

int main(void)
{
    check_embedding();
    check_layernorm();
    check_matmul();
    check_attention();
    check_elementwise();
    check_probabilities();

    if (failures != 0) {
        fprintf(stderr, "check-05: %d of %d checks failed\n", failures, checks);
        return EXIT_FAILURE;
    }
    printf("check-05: all %d forward checks passed\n", checks);
    return EXIT_SUCCESS;
}
