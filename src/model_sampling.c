#include <assert.h>

#include "model_internal.h"

static float maximum_logit(const float *logits, int vocab);
static void build_distribution(float *distribution, const float *logits,
                               int vocab, float temperature);
static int draw_from_distribution(const float *distribution, int vocab,
                                  Rng *rng);
static int sample_from_logits(Model *m, const float *logits, Rng *rng,
                              float temperature);

static float maximum_logit(const float *logits, int vocab)
{
    float maximum = logits[0];

    for (int id = 1; id < vocab; id++)
        if (logits[id] > maximum)
            maximum = logits[id];
    return maximum;
}

static void build_distribution(float *distribution, const float *logits,
                               int vocab, float temperature)
{
    float maximum = maximum_logit(logits, vocab);

    for (int id = 0; id < vocab; id++)
        distribution[id] = (logits[id] - maximum) / temperature;
    softmax_in_place(distribution, vocab);
}

static int draw_from_distribution(const float *distribution, int vocab,
                                  Rng *rng)
{
    float draw       = rng_uniform(rng);
    float cumulative = 0.0f;

    for (int id = 0; id < vocab; id++) {
        cumulative += distribution[id];
        if (draw < cumulative)
            return id;
    }
    return vocab - 1;
}

static int sample_from_logits(Model *m, const float *logits, Rng *rng,
                              float temperature)
{
    float *distribution = mat_row(m->probs, 0);
    int    vocab        = m->cfg.vocab_size;

    build_distribution(distribution, logits, vocab, temperature);
    return draw_from_distribution(distribution, vocab, rng);
}

void model_sample(Model *m, Rng *rng, int *ids, int prompt_count,
                  int total_count, float temperature)
{
    assert(prompt_count >= 1);

    for (int known = prompt_count; known < total_count; known++) {
        int        window  =
            known < m->cfg.block_size ? known : m->cfg.block_size;
        const int *context = ids + known - window;

        model_forward(m, context, NULL, 1, window);
        ids[known] = sample_from_logits(m, mat_row(m->logits, window - 1),
                                        rng, temperature);
    }
}
