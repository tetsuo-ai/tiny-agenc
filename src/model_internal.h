/*
 * model_internal.h -- the private transformer blueprint.
 *
 * The public interface keeps Model opaque.  The implementation files
 * share this definition so construction, forward propagation, backward
 * propagation, sampling, and checkpointing remain separately readable.
 */
#ifndef TINY_AGENC_MODEL_INTERNAL_H
#define TINY_AGENC_MODEL_INTERNAL_H

#include <stddef.h>

#include "model.h"
#include "ops.h"

enum {
    MODEL_MLP_WIDENING      = 4,
    MODEL_TENSORS_PER_BLOCK = 8,
    MODEL_TENSORS_ELSEWHERE = 4,
};

typedef struct {
    Mat normed1;
    Mat qkv;
    Mat scores;
    Mat attended;
    Mat projected;
    Mat after_attention;
    Mat normed2;
    Mat up;
    Mat activated;
    Mat down;
    Mat after_mlp;
} BlockTensors;

typedef struct {
    Param *norm1_gain;
    Param *norm1_bias;
    Param *qkv_weights;
    Param *proj_weights;
    Param *norm2_gain;
    Param *norm2_bias;
    Param *up_weights;
    Param *down_weights;

    BlockTensors acts;
    BlockTensors grads;
    float *means1;
    float *rstds1;
    float *means2;
    float *rstds2;
} Block;

struct Model {
    ModelConfig cfg;

    Param  *token_table;
    Param  *position_table;
    Block  *blocks;
    Param  *final_gain;
    Param  *final_bias;

    Param **params;
    int     param_count;

    float  *values_arena;
    float  *gradient_arena;
    size_t  gradient_floats;

    Mat     embedded;
    Mat     d_embedded;
    Mat     final_normed;
    Mat     d_final_normed;
    Mat     logits;
    Mat     d_logits;
    Mat     probs;
    float  *final_means;
    float  *final_rstds;

    int    *tokens;
    int    *targets;
    int     batch;
    int     time;
    int     has_targets;
};

Mat          model_stream_into(const Model *m, int layer, int rows);
Mat          model_d_stream_into(const Model *m, int layer, int rows);
BlockTensors model_block_views(const BlockTensors *bt, int rows, int time,
                               int head_count);
void         model_create_parameters(Model *m, Rng *rng);
size_t       model_parameter_float_count(ModelConfig cfg);

#endif
