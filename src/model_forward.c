#include <assert.h>
#include <string.h>

#include "model_internal.h"

static Mat block_forward(const Model *m, Block *b, Mat stream,
                         int batch, int time)
{
    BlockTensors a =
        model_block_views(&b->acts, batch * time, time, m->cfg.head_count);

    layernorm_forward(a.normed1, b->means1, b->rstds1, stream,
                      param_values(b->norm1_gain).vals,
                      param_values(b->norm1_bias).vals);
    matmul_forward(a.qkv, a.normed1, param_values(b->qkv_weights));
    attention_forward(a.attended, a.scores, a.qkv, time, m->cfg.head_count);
    matmul_forward(a.projected, a.attended, param_values(b->proj_weights));
    residual_forward(a.after_attention, stream, a.projected);

    layernorm_forward(a.normed2, b->means2, b->rstds2, a.after_attention,
                      param_values(b->norm2_gain).vals,
                      param_values(b->norm2_bias).vals);
    matmul_forward(a.up, a.normed2, param_values(b->up_weights));
    gelu_forward(a.activated, a.up);
    matmul_forward(a.down, a.activated, param_values(b->down_weights));
    residual_forward(a.after_mlp, a.after_attention, a.down);

    return a.after_mlp;
}

float model_forward(Model *m, const int *tokens, const int *targets,
                    int batch, int time)
{
    assert(batch >= 1 && batch <= m->cfg.batch_size);
    assert(time >= 1 && time <= m->cfg.block_size);

    int rows = batch * time;

    m->batch       = batch;
    m->time        = time;
    m->has_targets = targets != NULL;
    memcpy(m->tokens, tokens, (size_t)rows * sizeof *tokens);
    if (targets != NULL)
        memcpy(m->targets, targets, (size_t)rows * sizeof *targets);

    Mat stream = mat_first_rows(m->embedded, rows);

    embedding_forward(stream, m->tokens, param_values(m->token_table),
                      param_values(m->position_table), time);
    for (int layer = 0; layer < m->cfg.layer_count; layer++)
        stream = block_forward(m, &m->blocks[layer], stream, batch, time);

    Mat final_normed = mat_first_rows(m->final_normed, rows);
    Mat logits       = mat_first_rows(m->logits, rows);

    layernorm_forward(final_normed, m->final_means, m->final_rstds, stream,
                      param_values(m->final_gain).vals,
                      param_values(m->final_bias).vals);
    matmul_forward(logits, final_normed, param_values(m->token_table));

    if (!m->has_targets)
        return 0.0f;
    return crossentropy_forward(mat_first_rows(m->probs, rows), logits,
                                m->targets);
}
