#include <assert.h>
#include <stdint.h>

#include "model_internal.h"
#include "util.h"

typedef struct {
    float  *base;
    size_t  next;
    size_t  capacity;
} ArenaCursor;

typedef struct {
    size_t values;
    size_t gradients;
} ArenaLayout;

static ArenaCursor arena_cursor(float *base, size_t capacity)
{
    ArenaCursor cursor = { base, 0, capacity };

    return cursor;
}

static float *arena_take(ArenaCursor *cursor, size_t count)
{
    if (cursor->next > cursor->capacity
        || count > cursor->capacity - cursor->next)
        die("model arena layout exceeded its measured capacity");

    float *start = NULL;

    if (cursor->base != NULL)
        start = cursor->base + cursor->next;
    cursor->next += count;
    return start;
}

static Mat arena_place_mat(ArenaCursor *cursor, int rows, int cols)
{
    size_t count = (size_t)rows * (size_t)cols;

    return mat_make(arena_take(cursor, count), rows, cols);
}

static float *arena_place_floats(ArenaCursor *cursor, size_t count)
{
    return arena_take(cursor, count);
}

static void place_block_tensors(BlockTensors *bt, ArenaCursor *cursor,
                                ModelConfig cfg)
{
    int rows = cfg.batch_size * cfg.block_size;
    int wide = MODEL_MLP_WIDENING * cfg.d_model;

    bt->normed1 = arena_place_mat(cursor, rows, cfg.d_model);
    bt->qkv = arena_place_mat(cursor, rows, QKV_STREAMS * cfg.d_model);
    bt->scores =
        arena_place_mat(cursor, rows * cfg.head_count, cfg.block_size);
    bt->attended = arena_place_mat(cursor, rows, cfg.d_model);
    bt->projected = arena_place_mat(cursor, rows, cfg.d_model);
    bt->after_attention = arena_place_mat(cursor, rows, cfg.d_model);
    bt->normed2 = arena_place_mat(cursor, rows, cfg.d_model);
    bt->up = arena_place_mat(cursor, rows, wide);
    bt->activated = arena_place_mat(cursor, rows, wide);
    bt->down = arena_place_mat(cursor, rows, cfg.d_model);
    bt->after_mlp = arena_place_mat(cursor, rows, cfg.d_model);
}

static void place_block_statistics(Block *block, ArenaCursor *cursor,
                                   size_t rows)
{
    block->means1 = arena_place_floats(cursor, rows);
    block->rstds1 = arena_place_floats(cursor, rows);
    block->means2 = arena_place_floats(cursor, rows);
    block->rstds2 = arena_place_floats(cursor, rows);
}

static void place_block_values(Block *block, ArenaCursor *cursor,
                               ModelConfig cfg, size_t rows)
{
    place_block_tensors(&block->acts, cursor, cfg);
    place_block_statistics(block, cursor, rows);
}

static void place_value_views(Model *m, ArenaCursor *cursor)
{
    ModelConfig cfg  = m->cfg;
    int         rows = cfg.batch_size * cfg.block_size;

    m->embedded = arena_place_mat(cursor, rows, cfg.d_model);
    for (int layer = 0; layer < cfg.layer_count; layer++)
        place_block_values(&m->blocks[layer], cursor, cfg, (size_t)rows);
    m->final_normed = arena_place_mat(cursor, rows, cfg.d_model);
    m->final_means = arena_place_floats(cursor, (size_t)rows);
    m->final_rstds = arena_place_floats(cursor, (size_t)rows);
    m->logits = arena_place_mat(cursor, rows, cfg.vocab_size);
    m->probs = arena_place_mat(cursor, rows, cfg.vocab_size);
}

static void place_gradient_views(Model *m, ArenaCursor *cursor)
{
    ModelConfig cfg  = m->cfg;
    int         rows = cfg.batch_size * cfg.block_size;

    m->d_embedded = arena_place_mat(cursor, rows, cfg.d_model);
    for (int layer = 0; layer < cfg.layer_count; layer++)
        place_block_tensors(&m->blocks[layer].grads, cursor, cfg);
    m->d_final_normed = arena_place_mat(cursor, rows, cfg.d_model);
    m->d_logits = arena_place_mat(cursor, rows, cfg.vocab_size);
}

static ArenaLayout measure_arena_layout(Model *m)
{
    ArenaCursor values = arena_cursor(NULL, SIZE_MAX);
    ArenaCursor gradients = arena_cursor(NULL, SIZE_MAX);
    ArenaLayout layout;

    place_value_views(m, &values);
    place_gradient_views(m, &gradients);
    layout.values = values.next;
    layout.gradients = gradients.next;
    return layout;
}

static int memory_total_matches_components(ModelMemory memory)
{
    size_t remainder = memory.total_bytes;

    if (memory.parameter_bytes > remainder)
        return 0;
    remainder -= memory.parameter_bytes;
    if (memory.activation_bytes > remainder)
        return 0;
    remainder -= memory.activation_bytes;
    if (memory.gradient_bytes > remainder)
        return 0;
    remainder -= memory.gradient_bytes;
    return remainder == memory.token_bytes;
}

static int memory_report_matches_layout(const Model *m, ModelMemory memory,
                                        ArenaLayout layout)
{
    size_t parameter_unit = 4 * sizeof(float);
    size_t token_unit = 2 * sizeof(int);
    size_t max_tokens =
        (size_t)m->cfg.batch_size * (size_t)m->cfg.block_size;

    return memory.parameter_bytes % parameter_unit == 0
        && memory.parameter_bytes / parameter_unit
            == model_parameter_count(m)
        && memory.activation_bytes % sizeof(float) == 0
        && memory.gradient_bytes % sizeof(float) == 0
        && memory.activation_bytes / sizeof(float) == layout.values
        && memory.gradient_bytes / sizeof(float) == layout.gradients
        && memory.token_bytes % token_unit == 0
        && memory.token_bytes / token_unit == max_tokens
        && memory_total_matches_components(memory);
}

static void require_matching_memory_report(const Model *m,
                                           ModelMemory memory,
                                           ArenaLayout layout)
{
    if (!memory_report_matches_layout(m, memory, layout))
        die("model storage layout disagrees with its memory report");
}

static void require_full_arena(ArenaCursor cursor)
{
    if (cursor.next != cursor.capacity)
        die("model arena placement did not fill its measured capacity");
}

static Model *allocate_model_record(ModelConfig cfg)
{
    Model *m = ecalloc(1, sizeof *m);

    m->cfg = cfg;
    return m;
}

static void create_parameter_storage(Model *m, Rng *rng)
{
    m->blocks =
        ecalloc((size_t)m->cfg.layer_count, sizeof *m->blocks);
    model_create_parameters(m, rng);
}

static void create_value_arena(Model *m, size_t value_floats)
{
    m->values_arena =
        emalloc(value_floats * sizeof *m->values_arena);

    ArenaCursor cursor =
        arena_cursor(m->values_arena, value_floats);

    place_value_views(m, &cursor);
    require_full_arena(cursor);
}

static void create_gradient_arena(Model *m, size_t gradient_floats)
{
    m->gradient_floats = gradient_floats;
    m->gradient_arena =
        emalloc(gradient_floats * sizeof *m->gradient_arena);

    ArenaCursor cursor =
        arena_cursor(m->gradient_arena, gradient_floats);

    place_gradient_views(m, &cursor);
    require_full_arena(cursor);
}

static void create_token_caches(Model *m)
{
    size_t max_tokens =
        (size_t)m->cfg.batch_size * (size_t)m->cfg.block_size;

    m->tokens = emalloc(max_tokens * sizeof *m->tokens);
    m->targets = emalloc(max_tokens * sizeof *m->targets);
}

static void create_model_storage(Model *m, ModelMemory memory)
{
    ArenaLayout layout = measure_arena_layout(m);

    require_matching_memory_report(m, memory, layout);
    create_value_arena(m, layout.values);
    create_gradient_arena(m, layout.gradients);
    create_token_caches(m);
}

Model *model_new(ModelConfig cfg, unsigned long long seed)
{
    ModelMemory memory;
    int memory_ok = model_memory_requirements(cfg, &memory);

    assert(model_config_valid(cfg));
    assert(memory_ok);
    if (!memory_ok)
        die("invalid or unrepresentable model configuration");

    Model *m = allocate_model_record(cfg);
    Rng *rng = rng_new(seed);

    create_parameter_storage(m, rng);
    rng_free(rng);
    create_model_storage(m, memory);
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
