#include <assert.h>
#include <math.h>

#include "model_internal.h"
#include "util.h"

static const float INIT_STDDEV = 0.02f;

typedef struct {
    Model *model;
    Rng   *rng;
    int    at;
    int    wide;
    float  residual_stddev;
} ParameterCursor;

static ParameterCursor parameter_registry_create(Model *m, Rng *rng)
{
    ModelConfig cfg  = m->cfg;
    int         wide = MODEL_MLP_WIDENING * cfg.d_model;
    float residual_stddev =
        INIT_STDDEV / sqrtf(2.0f * (float)cfg.layer_count);

    m->param_count =
        MODEL_TENSORS_ELSEWHERE + MODEL_TENSORS_PER_BLOCK * cfg.layer_count;
    m->params = emalloc((size_t)m->param_count * sizeof *m->params);

    ParameterCursor cursor = { m, rng, 0, wide, residual_stddev };

    return cursor;
}

static void parameter_registry_add(ParameterCursor *cursor, Param **named,
                                   Param *parameter)
{
    cursor->model->params[cursor->at++] = parameter;
    *named = parameter;
}

static void create_embedding_parameters(ParameterCursor *cursor)
{
    Model       *m   = cursor->model;
    ModelConfig  cfg = m->cfg;
    Param       *token_table;
    Param       *position_table;

    token_table =
        param_new_gaussian(cfg.vocab_size, cfg.d_model,
                           INIT_STDDEV, cursor->rng);
    parameter_registry_add(cursor, &m->token_table, token_table);

    position_table =
        param_new_gaussian(cfg.block_size, cfg.d_model,
                           INIT_STDDEV, cursor->rng);
    parameter_registry_add(cursor, &m->position_table, position_table);
}

static void create_norm1_parameters(ParameterCursor *cursor, Block *block)
{
    ModelConfig cfg = cursor->model->cfg;
    Param *norm1_gain;
    Param *norm1_bias;

    norm1_gain = param_new_constant(1, cfg.d_model, 1.0f);
    parameter_registry_add(cursor, &block->norm1_gain, norm1_gain);

    norm1_bias = param_new_constant(1, cfg.d_model, 0.0f);
    parameter_registry_add(cursor, &block->norm1_bias, norm1_bias);
}

static void create_attention_parameters(ParameterCursor *cursor, Block *block)
{
    ModelConfig cfg = cursor->model->cfg;
    Param *qkv_weights;
    Param *proj_weights;

    qkv_weights =
        param_new_gaussian(QKV_STREAMS * cfg.d_model, cfg.d_model,
                           INIT_STDDEV, cursor->rng);
    parameter_registry_add(cursor, &block->qkv_weights, qkv_weights);

    proj_weights =
        param_new_gaussian(cfg.d_model, cfg.d_model,
                           cursor->residual_stddev, cursor->rng);
    parameter_registry_add(cursor, &block->proj_weights, proj_weights);
}

static void create_norm2_parameters(ParameterCursor *cursor, Block *block)
{
    ModelConfig cfg = cursor->model->cfg;
    Param *norm2_gain;
    Param *norm2_bias;

    norm2_gain = param_new_constant(1, cfg.d_model, 1.0f);
    parameter_registry_add(cursor, &block->norm2_gain, norm2_gain);

    norm2_bias = param_new_constant(1, cfg.d_model, 0.0f);
    parameter_registry_add(cursor, &block->norm2_bias, norm2_bias);
}

static void create_mlp_parameters(ParameterCursor *cursor, Block *block)
{
    ModelConfig cfg = cursor->model->cfg;
    Param *up_weights;
    Param *down_weights;

    up_weights =
        param_new_gaussian(cursor->wide, cfg.d_model,
                           INIT_STDDEV, cursor->rng);
    parameter_registry_add(cursor, &block->up_weights, up_weights);

    down_weights =
        param_new_gaussian(cfg.d_model, cursor->wide,
                           cursor->residual_stddev, cursor->rng);
    parameter_registry_add(cursor, &block->down_weights, down_weights);
}

static void create_block_parameters(ParameterCursor *cursor, Block *block)
{
    create_norm1_parameters(cursor, block);
    create_attention_parameters(cursor, block);
    create_norm2_parameters(cursor, block);
    create_mlp_parameters(cursor, block);
}

static void create_all_block_parameters(ParameterCursor *cursor)
{
    Model *m = cursor->model;

    for (int layer = 0; layer < m->cfg.layer_count; layer++)
        create_block_parameters(cursor, &m->blocks[layer]);
}

static void create_final_parameters(ParameterCursor *cursor)
{
    Model       *m   = cursor->model;
    ModelConfig  cfg = m->cfg;
    Param       *final_gain;
    Param       *final_bias;

    final_gain = param_new_constant(1, cfg.d_model, 1.0f);
    parameter_registry_add(cursor, &m->final_gain, final_gain);

    final_bias = param_new_constant(1, cfg.d_model, 0.0f);
    parameter_registry_add(cursor, &m->final_bias, final_bias);
}

void model_create_parameters(Model *m, Rng *rng)
{
    ParameterCursor cursor = parameter_registry_create(m, rng);

    create_embedding_parameters(&cursor);
    create_all_block_parameters(&cursor);
    create_final_parameters(&cursor);
    assert(cursor.at == m->param_count);
}
