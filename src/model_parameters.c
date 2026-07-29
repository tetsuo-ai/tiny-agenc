#include <assert.h>
#include <math.h>

#include "model_internal.h"
#include "util.h"

static const float INIT_STDDEV = 0.02f;

void model_create_parameters(Model *m, Rng *rng)
{
    ModelConfig cfg  = m->cfg;
    int         wide = MODEL_MLP_WIDENING * cfg.d_model;
    float residual_stddev =
        INIT_STDDEV / sqrtf(2.0f * (float)cfg.layer_count);

    m->param_count =
        MODEL_TENSORS_ELSEWHERE + MODEL_TENSORS_PER_BLOCK * cfg.layer_count;
    m->params = emalloc((size_t)m->param_count * sizeof *m->params);

    int at = 0;

    m->token_table = m->params[at++] =
        param_new_gaussian(cfg.vocab_size, cfg.d_model, INIT_STDDEV, rng);
    m->position_table = m->params[at++] =
        param_new_gaussian(cfg.block_size, cfg.d_model, INIT_STDDEV, rng);

    for (int layer = 0; layer < cfg.layer_count; layer++) {
        Block *b = &m->blocks[layer];

        b->norm1_gain = m->params[at++] =
            param_new_constant(1, cfg.d_model, 1.0f);
        b->norm1_bias = m->params[at++] =
            param_new_constant(1, cfg.d_model, 0.0f);
        b->qkv_weights = m->params[at++] =
            param_new_gaussian(QKV_STREAMS * cfg.d_model, cfg.d_model,
                               INIT_STDDEV, rng);
        b->proj_weights = m->params[at++] =
            param_new_gaussian(cfg.d_model, cfg.d_model,
                               residual_stddev, rng);
        b->norm2_gain = m->params[at++] =
            param_new_constant(1, cfg.d_model, 1.0f);
        b->norm2_bias = m->params[at++] =
            param_new_constant(1, cfg.d_model, 0.0f);
        b->up_weights = m->params[at++] =
            param_new_gaussian(wide, cfg.d_model, INIT_STDDEV, rng);
        b->down_weights = m->params[at++] =
            param_new_gaussian(cfg.d_model, wide, residual_stddev, rng);
    }

    m->final_gain = m->params[at++] =
        param_new_constant(1, cfg.d_model, 1.0f);
    m->final_bias = m->params[at++] =
        param_new_constant(1, cfg.d_model, 0.0f);
    assert(at == m->param_count);
}
