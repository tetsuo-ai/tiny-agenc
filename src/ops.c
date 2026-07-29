#include <math.h>
#include <string.h>

#include "ops.h"

static const float LAYERNORM_EPSILON   = 1e-5f;       /* keeps 1/sqrt(var) finite */
static const float GELU_SQRT_2_OVER_PI = 0.7978845608f;
static const float GELU_CUBIC_COEFF    = 0.044715f;

/* Below this many independent pieces of work, forking a thread team
 * costs more than it saves; the `if` clauses keep small calls serial. */
enum { PARALLEL_THRESHOLD = 64 };

/* The two loops at the bottom of everything. */

static float dot(const float *a, const float *b, int count)
{
    float sum = 0.0f;

    for (int i = 0; i < count; i++)
        sum += a[i] * b[i];
    return sum;
}

static void add_scaled(float *out, float scale, const float *values, int count)
{
    for (int i = 0; i < count; i++)
        out[i] += scale * values[i];
}

/* -------- embeddings -------- */

void embedding_forward(Mat out, const int *tokens, Mat token_table,
                       Mat position_table, int time)
{
    assert(out.cols == token_table.cols && out.cols == position_table.cols);

    for (int row = 0; row < out.rows; row++) {
        const float *token_vector    = mat_row(token_table, tokens[row]);
        const float *position_vector = mat_row(position_table, row % time);
        float       *output          = mat_row(out, row);

        for (int c = 0; c < out.cols; c++)
            output[c] = token_vector[c] + position_vector[c];
    }
}

void embedding_backward(Mat d_token_table, Mat d_position_table, Mat d_out,
                        const int *tokens, int time)
{
    for (int row = 0; row < d_out.rows; row++) {
        const float *d_output = mat_row(d_out, row);

        add_scaled(mat_row(d_token_table, tokens[row]), 1.0f, d_output, d_out.cols);
        add_scaled(mat_row(d_position_table, row % time), 1.0f, d_output, d_out.cols);
    }
}

/* -------- layernorm -------- */

void layernorm_forward(Mat out, float *means, float *rstds, Mat x,
                       const float *gain, const float *bias)
{
    assert(out.rows == x.rows && out.cols == x.cols);

    for (int row = 0; row < x.rows; row++) {
        const float *input  = mat_row(x, row);
        float       *output = mat_row(out, row);

        float mean = 0.0f;
        for (int c = 0; c < x.cols; c++)
            mean += input[c];
        mean /= (float)x.cols;

        float variance = 0.0f;
        for (int c = 0; c < x.cols; c++) {
            float centered = input[c] - mean;

            variance += centered * centered;
        }
        variance /= (float)x.cols;

        float rstd = 1.0f / sqrtf(variance + LAYERNORM_EPSILON);

        for (int c = 0; c < x.cols; c++)
            output[c] = gain[c] * ((input[c] - mean) * rstd) + bias[c];

        means[row] = mean;
        rstds[row] = rstd;
    }
}

/*
 * For one row with normalized values n_c = (x_c - mean) * rstd and
 * d_n = d_out * gain, the chain rule gives
 *
 *   d_x = rstd * (d_n - mean(d_n) - n * mean(d_n . n))
 *
 * The two mean terms are how nudging one input moves the row's own
 * mean and variance, which every other output of the row flows through.
 */
void layernorm_backward(Mat d_x, float *d_gain, float *d_bias, Mat d_out,
                        Mat x, const float *gain,
                        const float *means, const float *rstds)
{
    for (int row = 0; row < x.rows; row++) {
        const float *input    = mat_row(x, row);
        const float *d_output = mat_row(d_out, row);
        float       *d_input  = mat_row(d_x, row);
        float        mean     = means[row];
        float        rstd     = rstds[row];

        float d_norm_mean      = 0.0f;
        float d_norm_norm_mean = 0.0f;
        for (int c = 0; c < x.cols; c++) {
            float norm   = (input[c] - mean) * rstd;
            float d_norm = d_output[c] * gain[c];

            d_norm_mean      += d_norm;
            d_norm_norm_mean += d_norm * norm;
        }
        d_norm_mean      /= (float)x.cols;
        d_norm_norm_mean /= (float)x.cols;

        for (int c = 0; c < x.cols; c++) {
            float norm   = (input[c] - mean) * rstd;
            float d_norm = d_output[c] * gain[c];

            d_input[c] += rstd * (d_norm - d_norm_mean - norm * d_norm_norm_mean);
            d_gain[c]  += d_output[c] * norm;
            d_bias[c]  += d_output[c];
        }
    }
}

/* -------- matmul -------- */

void matmul_forward(Mat out, Mat x, Mat weights)
{
    assert(x.cols == weights.cols && out.cols == weights.rows && out.rows == x.rows);

    #pragma omp parallel for if(out.rows >= PARALLEL_THRESHOLD)
    for (int row = 0; row < out.rows; row++) {
        const float *input  = mat_row(x, row);
        float       *output = mat_row(out, row);

        for (int o = 0; o < weights.rows; o++)
            output[o] = dot(input, mat_row(weights, o), weights.cols);
    }
}

void matmul_backward(Mat d_x, Mat d_weights, Mat d_out, Mat x, Mat weights)
{
    /* d_x[r] += sum_o d_out[r][o] * weights[o]: rows are independent. */
    #pragma omp parallel for if(x.rows >= PARALLEL_THRESHOLD)
    for (int row = 0; row < x.rows; row++) {
        const float *d_output = mat_row(d_out, row);
        float       *d_input  = mat_row(d_x, row);

        for (int o = 0; o < weights.rows; o++)
            add_scaled(d_input, d_output[o], mat_row(weights, o), weights.cols);
    }

    /* d_weights[o] += sum_r d_out[r][o] * x[r]: output channels are
     * independent, so this loop parallelizes without collisions. */
    #pragma omp parallel for if(weights.rows >= PARALLEL_THRESHOLD)
    for (int o = 0; o < weights.rows; o++) {
        float *d_neuron = mat_row(d_weights, o);

        for (int row = 0; row < x.rows; row++) {
            const float *d_output = mat_row(d_out, row);

            add_scaled(d_neuron, d_output[o], mat_row(x, row), weights.cols);
        }
    }
}

/* -------- attention -------- */

/*
 * qkv packs the query, key, and value projections side by side: row t
 * is [q | k | v], each `channels` wide, and each of those splits into
 * head_count slices of head_size.  scores holds one row of attention
 * weights per (sequence, head, query position).
 */

static const float *qkv_slice(Mat qkv, int row, int stream, int offset)
{
    return mat_row(qkv, row) + stream * (qkv.cols / QKV_STREAMS) + offset;
}

static float *d_qkv_slice(Mat d_qkv, int row, int stream, int offset)
{
    return mat_row(d_qkv, row) + stream * (d_qkv.cols / QKV_STREAMS) + offset;
}

/* One (sequence, head) pair of the forward pass: for each query
 * position, score the visible past, soften scores into weights, and
 * mix the values. */
static void attention_head_forward(Mat out, Mat scores, Mat qkv, int seq,
                                   int head, int time, int head_count)
{
    int   head_size = out.cols / head_count;
    int   offset    = head * head_size;
    float scale     = 1.0f / sqrtf((float)head_size);

    for (int t = 0; t < time; t++) {
        const float *query   = qkv_slice(qkv, seq * time + t, QUERIES, offset);
        float       *weights = mat_row(scores, (seq * head_count + head) * time + t);

        /* Causality is a loop bound: position t sees 0..t only. */
        for (int t2 = 0; t2 <= t; t2++)
            weights[t2] = scale * dot(query, qkv_slice(qkv, seq * time + t2, KEYS, offset), head_size);
        softmax_in_place(weights, t + 1);

        float *output = mat_row(out, seq * time + t) + offset;

        memset(output, 0, (size_t)head_size * sizeof *output);
        for (int t2 = 0; t2 <= t; t2++)
            add_scaled(output, weights[t2], qkv_slice(qkv, seq * time + t2, VALUES, offset), head_size);
    }
}

void attention_forward(Mat out, Mat scores, Mat qkv, int time, int head_count)
{
    int sequence_count = out.rows / time;

    assert(qkv.cols == QKV_STREAMS * out.cols && out.cols % head_count == 0);
    assert(scores.cols == time && out.rows % time == 0);

    #pragma omp parallel for collapse(2) if(sequence_count * head_count >= PARALLEL_THRESHOLD)
    for (int seq = 0; seq < sequence_count; seq++)
        for (int head = 0; head < head_count; head++)
            attention_head_forward(out, scores, qkv, seq, head, time, head_count);
}

/* d_raw = w . (d_w - dot(w, d_w)): softmax couples every weight in a
 * row through the shared normalizer. */
static void softmax_backward_in_place(float *d_weights, const float *weights, int count)
{
    float coupled = dot(weights, d_weights, count);

    for (int i = 0; i < count; i++)
        d_weights[i] = weights[i] * (d_weights[i] - coupled);
}

/* The same (sequence, head) pair, unwound in reverse. */
static void attention_head_backward(Mat d_qkv, Mat d_scores, Mat d_out,
                                    Mat qkv, Mat scores, int seq, int head,
                                    int time, int head_count)
{
    int   head_size = d_out.cols / head_count;
    int   offset    = head * head_size;
    float scale     = 1.0f / sqrtf((float)head_size);

    for (int t = 0; t < time; t++) {
        int          score_row = (seq * head_count + head) * time + t;
        const float *weights   = mat_row(scores, score_row);
        float       *d_weights = mat_row(d_scores, score_row);
        const float *d_output  = mat_row(d_out, seq * time + t) + offset;

        /* out = sum w[t2] v[t2], so each v earns w[t2] of the output
         * gradient and each w earns v . d_out. */
        for (int t2 = 0; t2 <= t; t2++) {
            d_weights[t2] = dot(d_output, qkv_slice(qkv, seq * time + t2, VALUES, offset), head_size);
            add_scaled(d_qkv_slice(d_qkv, seq * time + t2, VALUES, offset),
                       weights[t2], d_output, head_size);
        }

        softmax_backward_in_place(d_weights, weights, t + 1);

        /* raw[t2] = scale * (q . k[t2]) fans out to both sides. */
        const float *query   = qkv_slice(qkv, seq * time + t, QUERIES, offset);
        float       *d_query = d_qkv_slice(d_qkv, seq * time + t, QUERIES, offset);

        for (int t2 = 0; t2 <= t; t2++) {
            float d_raw = scale * d_weights[t2];

            add_scaled(d_query, d_raw, qkv_slice(qkv, seq * time + t2, KEYS, offset), head_size);
            add_scaled(d_qkv_slice(d_qkv, seq * time + t2, KEYS, offset),
                       d_raw, query, head_size);
        }
    }
}

void attention_backward(Mat d_qkv, Mat d_scores, Mat d_out, Mat qkv,
                        Mat scores, int time, int head_count)
{
    int sequence_count = d_out.rows / time;

    /* (seq, head) pairs touch disjoint slices of d_qkv, so the pair
     * loop parallelizes; the t loops inside share those slices. */
    #pragma omp parallel for collapse(2) if(sequence_count * head_count >= PARALLEL_THRESHOLD)
    for (int seq = 0; seq < sequence_count; seq++)
        for (int head = 0; head < head_count; head++)
            attention_head_backward(d_qkv, d_scores, d_out, qkv, scores,
                                    seq, head, time, head_count);
}

/* -------- gelu -------- */

void gelu_forward(Mat out, Mat x)
{
    size_t count = mat_size(x);

    for (size_t i = 0; i < count; i++) {
        float value = x.vals[i];
        float inner = GELU_SQRT_2_OVER_PI * (value + GELU_CUBIC_COEFF * value * value * value);

        out.vals[i] = 0.5f * value * (1.0f + tanhf(inner));
    }
}

void gelu_backward(Mat d_x, Mat d_out, Mat x)
{
    size_t count = mat_size(x);

    for (size_t i = 0; i < count; i++) {
        float value   = x.vals[i];
        float inner   = GELU_SQRT_2_OVER_PI * (value + GELU_CUBIC_COEFF * value * value * value);
        float tanh_of = tanhf(inner);
        float d_inner = GELU_SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_CUBIC_COEFF * value * value);
        float slope   = 0.5f * (1.0f + tanh_of)
                      + 0.5f * value * (1.0f - tanh_of * tanh_of) * d_inner;

        d_x.vals[i] += slope * d_out.vals[i];
    }
}

/* -------- residual -------- */

void residual_forward(Mat out, Mat a, Mat b)
{
    size_t count = mat_size(out);

    for (size_t i = 0; i < count; i++)
        out.vals[i] = a.vals[i] + b.vals[i];
}

void residual_backward(Mat d_a, Mat d_b, Mat d_out)
{
    size_t count = mat_size(d_out);

    for (size_t i = 0; i < count; i++) {
        d_a.vals[i] += d_out.vals[i];
        d_b.vals[i] += d_out.vals[i];
    }
}

/* -------- softmax + cross-entropy -------- */

static void softmax_with_details(float *values, int count,
                                 float *maximum_out, float *sum_out)
{
    assert(count >= 1);

    float max = values[0];

    for (int i = 1; i < count; i++)
        if (values[i] > max)
            max = values[i];

    /* For finite inputs, subtracting the max changes nothing
     * mathematically and keeps every exponential at most 1.  Terms
     * below float range become zero in the returned distribution. */
    float total = 0.0f;

    for (int i = 0; i < count; i++) {
        values[i] = expf(values[i] - max);
        total += values[i];
    }
    for (int i = 0; i < count; i++)
        values[i] /= total;
    if (maximum_out != NULL)
        *maximum_out = max;
    if (sum_out != NULL)
        *sum_out = total;
}

void softmax_in_place(float *values, int count)
{
    softmax_with_details(values, count, NULL, NULL);
}

float crossentropy_forward(Mat probs, Mat logits, const int *targets)
{
    double total_loss = 0.0;   /* accumulate in double: many small terms */

    assert(probs.rows == logits.rows && probs.cols == logits.cols);

    for (int row = 0; row < logits.rows; row++) {
        float *prob = mat_row(probs, row);
        int    target = targets[row];
        float  maximum;
        float  exponential_sum;

        assert(target >= 0 && target < logits.cols);
        memcpy(prob, mat_row(logits, row), (size_t)logits.cols * sizeof *prob);
        softmax_with_details(prob, logits.cols, &maximum, &exponential_sum);

        /* This log-sum-exp form stays finite when the target probability
         * itself underflows in the float distribution above. */
        total_loss += log((double)exponential_sum)
                    + (double)maximum
                    - (double)mat_row(logits, row)[target];
    }
    return (float)(total_loss / logits.rows);
}

void crossentropy_backward(Mat d_logits, Mat probs, const int *targets)
{
    float mean_scale = 1.0f / (float)probs.rows;

    for (int row = 0; row < probs.rows; row++) {
        const float *prob     = mat_row(probs, row);
        float       *d_logit  = mat_row(d_logits, row);

        for (int c = 0; c < probs.cols; c++) {
            float indicator = (c == targets[row]) ? 1.0f : 0.0f;

            d_logit[c] += (prob[c] - indicator) * mean_scale;
        }
    }
}
