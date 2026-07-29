/*
 * ops.h -- the mathematics of a transformer, one pair of functions per
 * operation.
 *
 * Every operation appears twice: `x_forward` computes outputs from
 * inputs, and `x_backward` propagates loss gradients from outputs back
 * to inputs by the chain rule.  Gradients flowing into model inputs and
 * parameters ACCUMULATE (+=), so a tensor used in two places -- the
 * tied embedding table, a residual stream -- collects its whole
 * gradient.  Private scratch may be overwritten; callers zero all
 * externally accumulated buffers once per step.
 *
 * Activations flow through matrices of shape (batch*time, channels).
 * Only attention needs to know where one sequence ends and the next
 * begins, so only attention and the embeddings take time dimensions.
 *
 * Weight matrices store one output channel per row (n_out x n_in), the
 * same layout as a PyTorch Linear, so out = x . weights^T everywhere
 * and the tied language-model head can reuse matmul with the token
 * embedding table as its weights.
 */
#ifndef TINY_AGENC_OPS_H
#define TINY_AGENC_OPS_H

#include "mat.h"

/* qkv matrices pack three projections side by side per row: [q | k | v]. */
enum { QUERIES, KEYS, VALUES, QKV_STREAMS };

/* out[r] = token_table[tokens[r]] + position_table[r mod time] */
void embedding_forward(Mat out, const int *tokens, Mat token_table,
                       Mat position_table, int time);
void embedding_backward(Mat d_token_table, Mat d_position_table, Mat d_out,
                        const int *tokens, int time);

/* Each row is rescaled to mean 0 and variance approximately 1 (epsilon
 * keeps the denominator finite), then redressed with a learned gain and
 * bias; means and rstds save one float per row for the backward pass. */
void layernorm_forward(Mat out, float *means, float *rstds, Mat x,
                       const float *gain, const float *bias);
void layernorm_backward(Mat d_x, float *d_gain, float *d_bias, Mat d_out,
                        Mat x, const float *gain,
                        const float *means, const float *rstds);

/* out = x . weights^T */
void matmul_forward(Mat out, Mat x, Mat weights);
void matmul_backward(Mat d_x, Mat d_weights, Mat d_out, Mat x, Mat weights);

/* Causal multi-head self-attention over qkv = (queries|keys|values),
 * the three projections concatenated per row.  scores has one row of T
 * attention weights per (sequence, head, position); it is written by
 * the forward pass and consumed by the backward pass, whose d_scores
 * twin is pure scratch. */
void attention_forward(Mat out, Mat scores, Mat qkv, int time, int head_count);
void attention_backward(Mat d_qkv, Mat d_scores, Mat d_out, Mat qkv,
                        Mat scores, int time, int head_count);

/* gelu(x) = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))),
 * the transformer's smooth cousin of relu. */
void gelu_forward(Mat out, Mat x);
void gelu_backward(Mat d_x, Mat d_out, Mat x);

/* out = a + b: the residual stream that lets gradients skip layers. */
void residual_forward(Mat out, Mat a, Mat b);
void residual_backward(Mat d_a, Mat d_b, Mat d_out);

/* Softmax each row of logits into probs, then return the mean negative
 * log-likelihood of the targets.  The fused backward is one subtraction:
 * d_logits = (probs - onehot(targets)) / rows. */
float crossentropy_forward(Mat probs, Mat logits, const int *targets);
void  crossentropy_backward(Mat d_logits, Mat probs, const int *targets);

/* Exposed for sampling: rescale values into a probability distribution. */
void softmax_in_place(float *values, int count);

#endif
