#include <assert.h>

#include "model_internal.h"

static void block_backward(const Model *m, Block *b, Mat stream,
                           Mat d_stream, int batch, int time);

static void block_backward(const Model *m, Block *b, Mat stream, Mat d_stream,
                           int batch, int time)
{
    BlockTensors a =
        model_block_views(&b->acts, batch * time, time, m->cfg.head_count);
    BlockTensors g =
        model_block_views(&b->grads, batch * time, time, m->cfg.head_count);

    residual_backward(g.after_attention, g.down, g.after_mlp);
    matmul_backward(g.activated, param_gradient(b->down_weights), g.down,
                    a.activated, param_values(b->down_weights));
    gelu_backward(g.up, g.activated, a.up);
    matmul_backward(g.normed2, param_gradient(b->up_weights), g.up,
                    a.normed2, param_values(b->up_weights));
    layernorm_backward(g.after_attention,
                       param_gradient(b->norm2_gain).vals,
                       param_gradient(b->norm2_bias).vals,
                       g.normed2, a.after_attention,
                       param_values(b->norm2_gain).vals,
                       b->means2, b->rstds2);

    residual_backward(d_stream, g.projected, g.after_attention);
    matmul_backward(g.attended, param_gradient(b->proj_weights), g.projected,
                    a.attended, param_values(b->proj_weights));
    attention_backward(g.qkv, g.scores, g.attended, a.qkv, a.scores,
                       time, m->cfg.head_count);
    matmul_backward(g.normed1, param_gradient(b->qkv_weights), g.qkv,
                    a.normed1, param_values(b->qkv_weights));
    layernorm_backward(d_stream,
                       param_gradient(b->norm1_gain).vals,
                       param_gradient(b->norm1_bias).vals,
                       g.normed1, stream,
                       param_values(b->norm1_gain).vals,
                       b->means1, b->rstds1);
}

void model_backward(Model *m)
{
    assert(m->has_targets);

    int rows = m->batch * m->time;
    int last = m->cfg.layer_count - 1;

    Mat final_normed   = mat_first_rows(m->final_normed, rows);
    Mat d_final_normed = mat_first_rows(m->d_final_normed, rows);
    Mat d_logits       = mat_first_rows(m->d_logits, rows);
    Mat last_stream    = model_stream_into(m, last + 1, rows);
    Mat d_last_stream  = model_d_stream_into(m, last + 1, rows);

    crossentropy_backward(d_logits, mat_first_rows(m->probs, rows),
                          m->targets);
    matmul_backward(d_final_normed, param_gradient(m->token_table), d_logits,
                    final_normed, param_values(m->token_table));
    layernorm_backward(d_last_stream,
                       param_gradient(m->final_gain).vals,
                       param_gradient(m->final_bias).vals,
                       d_final_normed, last_stream,
                       param_values(m->final_gain).vals,
                       m->final_means, m->final_rstds);

    for (int layer = last; layer >= 0; layer--)
        block_backward(m, &m->blocks[layer],
                       model_stream_into(m, layer, rows),
                       model_d_stream_into(m, layer, rows),
                       m->batch, m->time);

    embedding_backward(param_gradient(m->token_table),
                       param_gradient(m->position_table),
                       mat_first_rows(m->d_embedded, rows),
                       m->tokens, m->time);
}
