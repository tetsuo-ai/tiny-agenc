#include <math.h>
#include <string.h>

#include "ops.h"

static const float LAYERNORM_EPSILON   = 1e-5f;       /* keeps 1/sqrt(var) finite */
static const float GELU_SQRT_2_OVER_PI = 0.7978845608f;
static const float GELU_CUBIC_COEFF    = 0.044715f;
static const float GELU_HALF           = 0.5f;
static const float CUBIC_DERIVATIVE_FACTOR = 3.0f;

/* Below this many independent pieces of work, forking a thread team
 * costs more than it saves; the `if` clauses keep small calls serial. */
enum { PARALLEL_THRESHOLD = 64 };

typedef struct {
    Mat          out;
    float       *means;
    float       *rstds;
    Mat          x;
    const float *gain;
    const float *bias;
} LayernormForward;

typedef struct {
    Mat          d_x;
    float       *d_gain;
    float       *d_bias;
    Mat          d_out;
    Mat          x;
    const float *gain;
    const float *means;
    const float *rstds;
} LayernormBackward;

typedef struct {
    const float *input;
    const float *d_output;
    float       *d_input;
    float       *d_gain;
    float       *d_bias;
    const float *gain;
    int          channels;
    float        mean;
    float        rstd;
} LayernormBackwardRow;

typedef struct {
    float d_norm_mean;
    float d_norm_norm_mean;
} LayernormGradientMeans;

typedef struct {
    Mat out;
    Mat scores;
    Mat qkv;
    int time;
    int head_count;
} AttentionForward;

typedef struct {
    Mat d_qkv;
    Mat d_scores;
    Mat d_out;
    Mat qkv;
    Mat scores;
    int time;
    int head_count;
} AttentionBackward;

typedef struct {
    Mat          d_qkv;
    Mat          qkv;
    const float *weights;
    float       *d_weights;
    const float *d_output;
    int          sequence;
    int          offset;
    int          time_index;
    int          time;
    int          head_size;
    float        scale;
} AttentionBackwardPosition;

typedef struct {
    Mat   d_qkv;
    Mat   d_scores;
    Mat   d_out;
    Mat   qkv;
    Mat   scores;
    int   sequence;
    int   head;
    int   time;
    int   head_count;
    int   head_size;
    int   offset;
    float scale;
} AttentionBackwardHead;

static float dot(const float *a, const float *b, int count);
static void add_scaled(float *out, float scale, const float *values,
                       int count);
static float layernorm_row_mean(const float *input, int channels);
static float layernorm_row_variance(const float *input, int channels,
                                    float mean);
static void layernorm_transform_row(float *output, const float *input,
                                    const float *gain, const float *bias,
                                    int channels, float mean, float rstd);
static void layernorm_forward_row(const LayernormForward *forward, int row);
static LayernormGradientMeans layernorm_gradient_means(
    const LayernormBackwardRow *row);
static void layernorm_accumulate_row_gradients(
    const LayernormBackwardRow *row, LayernormGradientMeans means);
static void layernorm_backward_row(const LayernormBackward *backward,
                                   int row);
static void matmul_accumulate_input_row(Mat d_x, Mat d_out, Mat weights,
                                        int row);
static void matmul_accumulate_weight_row(Mat d_weights, Mat d_out, Mat x,
                                         Mat weights, int output);
static const float *qkv_slice(Mat qkv, int row, int stream, int offset);
static float *d_qkv_slice(Mat d_qkv, int row, int stream, int offset);
static void attention_head_forward(const AttentionForward *forward,
                                   int sequence, int head);
static void softmax_backward_in_place(float *d_weights,
                                      const float *weights, int count);
static void attention_value_backward(
    const AttentionBackwardPosition *position);
static void attention_score_backward(
    const AttentionBackwardPosition *position);
static void attention_position_backward(const AttentionBackwardHead *head,
                                        int time_index);
static void attention_head_backward(const AttentionBackward *backward,
                                    int sequence, int head);
static void softmax_with_details(float *values, int count,
                                 float *maximum_out, float *sum_out);
static void crossentropy_backward_row(float *d_logits, const float *probs,
                                      int target, int count,
                                      float mean_scale);

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

static float layernorm_row_mean(const float *input, int channels)
{
    float mean = 0.0f;

    for (int c = 0; c < channels; c++)
        mean += input[c];
    mean /= (float)channels;
    return mean;
}

static float layernorm_row_variance(const float *input, int channels,
                                    float mean)
{
    float variance = 0.0f;

    for (int c = 0; c < channels; c++) {
        float centered = input[c] - mean;

        variance += centered * centered;
    }
    variance /= (float)channels;
    return variance;
}

static void layernorm_transform_row(float *output, const float *input,
                                    const float *gain, const float *bias,
                                    int channels, float mean, float rstd)
{
    for (int c = 0; c < channels; c++)
        output[c] = gain[c] * ((input[c] - mean) * rstd) + bias[c];
}

static void layernorm_forward_row(const LayernormForward *forward, int row)
{
    const float *input  = mat_row(forward->x, row);
    float       *output = mat_row(forward->out, row);
    float        mean   = layernorm_row_mean(input, forward->x.cols);
    float        variance =
        layernorm_row_variance(input, forward->x.cols, mean);
    float rstd = 1.0f / sqrtf(variance + LAYERNORM_EPSILON);

    layernorm_transform_row(output, input, forward->gain, forward->bias,
                            forward->x.cols, mean, rstd);
    forward->means[row] = mean;
    forward->rstds[row] = rstd;
}

void layernorm_forward(Mat out, float *means, float *rstds, Mat x,
                       const float *gain, const float *bias)
{
    assert(out.rows == x.rows && out.cols == x.cols);

    LayernormForward forward = {
        .out = out,
        .means = means,
        .rstds = rstds,
        .x = x,
        .gain = gain,
        .bias = bias,
    };

    for (int row = 0; row < x.rows; row++)
        layernorm_forward_row(&forward, row);
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
static LayernormGradientMeans layernorm_gradient_means(
    const LayernormBackwardRow *row)
{
    LayernormGradientMeans means = {
        .d_norm_mean = 0.0f,
        .d_norm_norm_mean = 0.0f,
    };

    for (int c = 0; c < row->channels; c++) {
        float norm   = (row->input[c] - row->mean) * row->rstd;
        float d_norm = row->d_output[c] * row->gain[c];

        means.d_norm_mean      += d_norm;
        means.d_norm_norm_mean += d_norm * norm;
    }
    means.d_norm_mean      /= (float)row->channels;
    means.d_norm_norm_mean /= (float)row->channels;
    return means;
}

static void layernorm_accumulate_row_gradients(
    const LayernormBackwardRow *row, LayernormGradientMeans means)
{
    for (int c = 0; c < row->channels; c++) {
        float norm   = (row->input[c] - row->mean) * row->rstd;
        float d_norm = row->d_output[c] * row->gain[c];

        row->d_input[c] +=
            row->rstd
            * (d_norm - means.d_norm_mean
               - norm * means.d_norm_norm_mean);
        row->d_gain[c] += row->d_output[c] * norm;
        row->d_bias[c] += row->d_output[c];
    }
}

static void layernorm_backward_row(const LayernormBackward *backward,
                                   int row)
{
    LayernormBackwardRow backward_row = {
        .input = mat_row(backward->x, row),
        .d_output = mat_row(backward->d_out, row),
        .d_input = mat_row(backward->d_x, row),
        .d_gain = backward->d_gain,
        .d_bias = backward->d_bias,
        .gain = backward->gain,
        .channels = backward->x.cols,
        .mean = backward->means[row],
        .rstd = backward->rstds[row],
    };
    LayernormGradientMeans means =
        layernorm_gradient_means(&backward_row);

    layernorm_accumulate_row_gradients(&backward_row, means);
}

void layernorm_backward(Mat d_x, float *d_gain, float *d_bias, Mat d_out,
                        Mat x, const float *gain,
                        const float *means, const float *rstds)
{
    LayernormBackward backward = {
        .d_x = d_x,
        .d_gain = d_gain,
        .d_bias = d_bias,
        .d_out = d_out,
        .x = x,
        .gain = gain,
        .means = means,
        .rstds = rstds,
    };

    for (int row = 0; row < x.rows; row++)
        layernorm_backward_row(&backward, row);
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

static void matmul_accumulate_input_row(Mat d_x, Mat d_out, Mat weights,
                                        int row)
{
    const float *d_output = mat_row(d_out, row);
    float       *d_input  = mat_row(d_x, row);

    for (int o = 0; o < weights.rows; o++)
        add_scaled(d_input, d_output[o], mat_row(weights, o), weights.cols);
}

static void matmul_accumulate_weight_row(Mat d_weights, Mat d_out, Mat x,
                                         Mat weights, int output)
{
    float *d_neuron = mat_row(d_weights, output);

    for (int row = 0; row < x.rows; row++) {
        const float *d_output = mat_row(d_out, row);

        add_scaled(d_neuron, d_output[output], mat_row(x, row), weights.cols);
    }
}

void matmul_backward(Mat d_x, Mat d_weights, Mat d_out, Mat x, Mat weights)
{
    /* d_x[r] += sum_o d_out[r][o] * weights[o]: rows are independent. */
    #pragma omp parallel for if(x.rows >= PARALLEL_THRESHOLD)
    for (int row = 0; row < x.rows; row++)
        matmul_accumulate_input_row(d_x, d_out, weights, row);

    /* d_weights[o] += sum_r d_out[r][o] * x[r]: output channels are
     * independent, so this loop parallelizes without collisions. */
    #pragma omp parallel for if(weights.rows >= PARALLEL_THRESHOLD)
    for (int o = 0; o < weights.rows; o++)
        matmul_accumulate_weight_row(d_weights, d_out, x, weights, o);
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
static void attention_head_forward(const AttentionForward *forward,
                                   int sequence, int head)
{
    int   head_size = forward->out.cols / forward->head_count;
    int   offset    = head * head_size;
    float scale     = 1.0f / sqrtf((float)head_size);

    for (int time_index = 0; time_index < forward->time; time_index++) {
        int row = sequence * forward->time + time_index;
        int score_row =
            (sequence * forward->head_count + head) * forward->time
            + time_index;
        const float *query =
            qkv_slice(forward->qkv, row, QUERIES, offset);
        float *weights = mat_row(forward->scores, score_row);

        /* Causality is a loop bound: this position sees its past only. */
        for (int source = 0; source <= time_index; source++)
            weights[source] =
                scale
                * dot(query,
                      qkv_slice(forward->qkv,
                                sequence * forward->time + source,
                                KEYS, offset),
                      head_size);
        softmax_in_place(weights, time_index + 1);

        float *output = mat_row(forward->out, row) + offset;

        memset(output, 0, (size_t)head_size * sizeof *output);
        for (int source = 0; source <= time_index; source++)
            add_scaled(output, weights[source],
                       qkv_slice(forward->qkv,
                                 sequence * forward->time + source,
                                 VALUES, offset),
                       head_size);
    }
}

void attention_forward(Mat out, Mat scores, Mat qkv, int time, int head_count)
{
    int sequence_count = out.rows / time;
    AttentionForward forward = {
        .out = out,
        .scores = scores,
        .qkv = qkv,
        .time = time,
        .head_count = head_count,
    };

    assert(qkv.cols == QKV_STREAMS * out.cols && out.cols % head_count == 0);
    assert(scores.cols == time && out.rows % time == 0);

    #pragma omp parallel for collapse(2) if(sequence_count * head_count >= PARALLEL_THRESHOLD)
    for (int sequence = 0; sequence < sequence_count; sequence++)
        for (int head = 0; head < head_count; head++)
            attention_head_forward(&forward, sequence, head);
}

/* d_raw = w . (d_w - dot(w, d_w)): softmax couples every weight in a
 * row through the shared normalizer. */
static void softmax_backward_in_place(float *d_weights, const float *weights, int count)
{
    float coupled = dot(weights, d_weights, count);

    for (int i = 0; i < count; i++)
        d_weights[i] = weights[i] * (d_weights[i] - coupled);
}

static void attention_value_backward(
    const AttentionBackwardPosition *position)
{
    /* out = sum w[t2] v[t2], so each v earns w[t2] of the output
     * gradient and each w earns v . d_out. */
    for (int source = 0; source <= position->time_index; source++) {
        position->d_weights[source] =
            dot(position->d_output,
                qkv_slice(position->qkv,
                          position->sequence * position->time + source,
                          VALUES, position->offset),
                position->head_size);
        add_scaled(
            d_qkv_slice(position->d_qkv,
                        position->sequence * position->time + source,
                        VALUES, position->offset),
            position->weights[source], position->d_output,
            position->head_size);
    }
}

static void attention_score_backward(
    const AttentionBackwardPosition *position)
{
    /* raw[t2] = scale * (q . k[t2]) fans out to both sides. */
    const float *query =
        qkv_slice(position->qkv,
                  position->sequence * position->time
                      + position->time_index,
                  QUERIES, position->offset);
    float *d_query =
        d_qkv_slice(position->d_qkv,
                    position->sequence * position->time
                        + position->time_index,
                    QUERIES, position->offset);

    for (int source = 0; source <= position->time_index; source++) {
        float d_raw = position->scale * position->d_weights[source];

        add_scaled(d_query, d_raw,
                   qkv_slice(position->qkv,
                             position->sequence * position->time + source,
                             KEYS, position->offset),
                   position->head_size);
        add_scaled(
            d_qkv_slice(position->d_qkv,
                        position->sequence * position->time + source,
                        KEYS, position->offset),
            d_raw, query, position->head_size);
    }
}

static void attention_position_backward(const AttentionBackwardHead *head,
                                        int time_index)
{
    int score_row =
        (head->sequence * head->head_count + head->head) * head->time
        + time_index;
    AttentionBackwardPosition position = {
        .d_qkv = head->d_qkv,
        .qkv = head->qkv,
        .weights = mat_row(head->scores, score_row),
        .d_weights = mat_row(head->d_scores, score_row),
        .d_output =
            mat_row(head->d_out,
                    head->sequence * head->time + time_index)
            + head->offset,
        .sequence = head->sequence,
        .offset = head->offset,
        .time_index = time_index,
        .time = head->time,
        .head_size = head->head_size,
        .scale = head->scale,
    };

    attention_value_backward(&position);
    softmax_backward_in_place(position.d_weights, position.weights,
                              time_index + 1);
    attention_score_backward(&position);
}

/* The same (sequence, head) pair, unwound in reverse. */
static void attention_head_backward(const AttentionBackward *backward,
                                    int sequence, int head)
{
    int   head_size = backward->d_out.cols / backward->head_count;
    int   offset    = head * head_size;
    float scale     = 1.0f / sqrtf((float)head_size);
    AttentionBackwardHead head_state = {
        .d_qkv = backward->d_qkv,
        .d_scores = backward->d_scores,
        .d_out = backward->d_out,
        .qkv = backward->qkv,
        .scores = backward->scores,
        .sequence = sequence,
        .head = head,
        .time = backward->time,
        .head_count = backward->head_count,
        .head_size = head_size,
        .offset = offset,
        .scale = scale,
    };

    for (int time_index = 0; time_index < backward->time; time_index++)
        attention_position_backward(&head_state, time_index);
}

void attention_backward(Mat d_qkv, Mat d_scores, Mat d_out, Mat qkv,
                        Mat scores, int time, int head_count)
{
    int sequence_count = d_out.rows / time;
    AttentionBackward backward = {
        .d_qkv = d_qkv,
        .d_scores = d_scores,
        .d_out = d_out,
        .qkv = qkv,
        .scores = scores,
        .time = time,
        .head_count = head_count,
    };

    /* (sequence, head) pairs touch disjoint slices of d_qkv, so the pair
     * loop parallelizes; the t loops inside share those slices. */
    #pragma omp parallel for collapse(2) if(sequence_count * head_count >= PARALLEL_THRESHOLD)
    for (int sequence = 0; sequence < sequence_count; sequence++)
        for (int head = 0; head < head_count; head++)
            attention_head_backward(&backward, sequence, head);
}

/* -------- gelu -------- */

void gelu_forward(Mat out, Mat x)
{
    size_t count = mat_size(x);

    for (size_t i = 0; i < count; i++) {
        float value = x.vals[i];
        float inner = GELU_SQRT_2_OVER_PI * (value + GELU_CUBIC_COEFF * value * value * value);

        out.vals[i] = GELU_HALF * value * (1.0f + tanhf(inner));
    }
}

void gelu_backward(Mat d_x, Mat d_out, Mat x)
{
    size_t count = mat_size(x);

    for (size_t i = 0; i < count; i++) {
        float value   = x.vals[i];
        float inner   = GELU_SQRT_2_OVER_PI * (value + GELU_CUBIC_COEFF * value * value * value);
        float tanh_of = tanhf(inner);
        float d_inner =
            GELU_SQRT_2_OVER_PI
            * (1.0f + CUBIC_DERIVATIVE_FACTOR * GELU_CUBIC_COEFF
                          * value * value);
        float slope = GELU_HALF * (1.0f + tanh_of)
                    + GELU_HALF * value * (1.0f - tanh_of * tanh_of)
                          * d_inner;

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

    for (int row = 0; row < probs.rows; row++)
        crossentropy_backward_row(mat_row(d_logits, row),
                                  mat_row(probs, row), targets[row],
                                  probs.cols, mean_scale);
}

static void crossentropy_backward_row(float *d_logits, const float *probs,
                                      int target, int count,
                                      float mean_scale)
{
    for (int channel = 0; channel < count; channel++) {
        float target_probability = channel == target ? 1.0f : 0.0f;

        d_logits[channel] +=
            (probs[channel] - target_probability) * mean_scale;
    }
}
