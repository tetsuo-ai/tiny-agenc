/*
 * Answer key for the Chapter 1 executable architecture specification.
 * The production model repeats the predicate with named ModelConfig fields.
 */
#include "spec.h"

int lab_config_valid(int vocab, int block, int width, int heads, int layers,
                     int batch);
size_t lab_parameter_count(int vocab, int block, int width, int layers);

int lab_config_valid(int vocab, int block, int width,
                     int heads, int layers, int batch)
{
    long long tokens = (long long)batch * block;

    return vocab >= 1 && vocab <= MODEL_MAX_VOCAB_SIZE
        && block >= 1 && block <= MODEL_MAX_BLOCK_SIZE
        && width >= 1 && width <= MODEL_MAX_D_MODEL
        && heads >= 1 && heads <= MODEL_MAX_HEAD_COUNT
        && layers >= 1 && layers <= MODEL_MAX_LAYER_COUNT
        && batch >= 1 && tokens <= MODEL_MAX_TOKENS_PER_PASS
        && width % heads == 0;
}

size_t lab_parameter_count(int vocab, int block, int width, int layers)
{
    size_t c = (size_t)width;

    return ((size_t)vocab + (size_t)block) * c
         + (size_t)layers * (12 * c * c + 4 * c)
         + 2 * c;
}
