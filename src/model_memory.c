#include <assert.h>

#include "model_internal.h"
#include "util.h"

static Mat place(float *base, size_t *offset, int rows, int cols)
{
    Mat m = mat_make(base == NULL ? NULL : base + *offset, rows, cols);

    *offset += mat_size(m);
    return m;
}

static float *place_floats(float *base, size_t *offset, size_t count)
{
    float *start = base == NULL ? NULL : base + *offset;

    *offset += count;
    return start;
}

static void place_block_tensors(BlockTensors *bt, float *base, size_t *offset,
                                ModelConfig cfg)
{
    int rows = cfg.batch_size * cfg.block_size;
    int wide = MODEL_MLP_WIDENING * cfg.d_model;

    bt->normed1         = place(base, offset, rows, cfg.d_model);
    bt->qkv             = place(base, offset, rows,
                                QKV_STREAMS * cfg.d_model);
    bt->scores          = place(base, offset, rows * cfg.head_count,
                                cfg.block_size);
    bt->attended        = place(base, offset, rows, cfg.d_model);
    bt->projected       = place(base, offset, rows, cfg.d_model);
    bt->after_attention = place(base, offset, rows, cfg.d_model);
    bt->normed2         = place(base, offset, rows, cfg.d_model);
    bt->up              = place(base, offset, rows, wide);
    bt->activated       = place(base, offset, rows, wide);
    bt->down            = place(base, offset, rows, cfg.d_model);
    bt->after_mlp       = place(base, offset, rows, cfg.d_model);
}

static size_t lay_out_values(Model *m, float *base)
{
    ModelConfig cfg    = m->cfg;
    int         rows   = cfg.batch_size * cfg.block_size;
    size_t      offset = 0;

    m->embedded = place(base, &offset, rows, cfg.d_model);
    for (int layer = 0; layer < cfg.layer_count; layer++) {
        Block *b = &m->blocks[layer];

        place_block_tensors(&b->acts, base, &offset, cfg);
        b->means1 = place_floats(base, &offset, (size_t)rows);
        b->rstds1 = place_floats(base, &offset, (size_t)rows);
        b->means2 = place_floats(base, &offset, (size_t)rows);
        b->rstds2 = place_floats(base, &offset, (size_t)rows);
    }
    m->final_normed = place(base, &offset, rows, cfg.d_model);
    m->final_means  = place_floats(base, &offset, (size_t)rows);
    m->final_rstds  = place_floats(base, &offset, (size_t)rows);
    m->logits       = place(base, &offset, rows, cfg.vocab_size);
    m->probs        = place(base, &offset, rows, cfg.vocab_size);
    return offset;
}

static size_t lay_out_gradients(Model *m, float *base)
{
    ModelConfig cfg    = m->cfg;
    int         rows   = cfg.batch_size * cfg.block_size;
    size_t      offset = 0;

    m->d_embedded = place(base, &offset, rows, cfg.d_model);
    for (int layer = 0; layer < cfg.layer_count; layer++)
        place_block_tensors(&m->blocks[layer].grads, base, &offset, cfg);
    m->d_final_normed = place(base, &offset, rows, cfg.d_model);
    m->d_logits       = place(base, &offset, rows, cfg.vocab_size);
    return offset;
}

Model *model_new(ModelConfig cfg, unsigned long long seed)
{
    ModelMemory memory;
    int memory_ok = model_memory_requirements(cfg, &memory);

    assert(model_config_valid(cfg));
    assert(memory_ok);
    if (!memory_ok)
        die("invalid or unrepresentable model configuration");

    Model *m   = ecalloc(1, sizeof *m);
    Rng   *rng = rng_new(seed);
    int max_tokens = cfg.batch_size * cfg.block_size;

    m->cfg    = cfg;
    m->blocks = ecalloc((size_t)cfg.layer_count, sizeof *m->blocks);
    model_create_parameters(m, rng);
    rng_free(rng);

    size_t value_floats = lay_out_values(m, NULL);

    m->values_arena = emalloc(value_floats * sizeof *m->values_arena);
    lay_out_values(m, m->values_arena);

    m->gradient_floats = lay_out_gradients(m, NULL);
    m->gradient_arena =
        emalloc(m->gradient_floats * sizeof *m->gradient_arena);
    lay_out_gradients(m, m->gradient_arena);

    m->tokens  = emalloc((size_t)max_tokens * sizeof *m->tokens);
    m->targets = emalloc((size_t)max_tokens * sizeof *m->targets);
    return m;
}

Mat model_stream_into(const Model *m, int layer, int rows)
{
    Mat full =
        layer == 0 ? m->embedded : m->blocks[layer - 1].acts.after_mlp;

    return mat_first_rows(full, rows);
}

Mat model_d_stream_into(const Model *m, int layer, int rows)
{
    Mat full =
        layer == 0 ? m->d_embedded : m->blocks[layer - 1].grads.after_mlp;

    return mat_first_rows(full, rows);
}

BlockTensors model_block_views(const BlockTensors *bt, int rows, int time,
                               int head_count)
{
    BlockTensors view;

    view.normed1         = mat_first_rows(bt->normed1, rows);
    view.qkv             = mat_first_rows(bt->qkv, rows);
    view.scores          =
        mat_make(bt->scores.vals, (rows / time) * head_count * time, time);
    view.attended        = mat_first_rows(bt->attended, rows);
    view.projected       = mat_first_rows(bt->projected, rows);
    view.after_attention = mat_first_rows(bt->after_attention, rows);
    view.normed2         = mat_first_rows(bt->normed2, rows);
    view.up              = mat_first_rows(bt->up, rows);
    view.activated       = mat_first_rows(bt->activated, rows);
    view.down            = mat_first_rows(bt->down, rows);
    view.after_mlp       = mat_first_rows(bt->after_mlp, rows);
    return view;
}
