/*
 * integration.c -- the promises around the mathematics.
 *
 * gradcheck asks whether each derivative agrees with its forward pass.
 * This program checks the rest of the teaching story: reproducible
 * randomness, text-to-token behavior, shifted training pairs, the model
 * shape contract, durable checkpoints, and a complete tiny training run.
 */
#define _POSIX_C_SOURCE 200809L

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "dataset.h"
#include "mat.h"
#include "model.h"
#include "ops.h"
#include "param.h"
#include "rng.h"
#include "tokenizer.h"
#include "util.h"

static int checks;
static int failures;

static void expect(int condition, const char *label)
{
    checks++;
    if (condition)
        return;
    printf("FAIL %s\n", label);
    failures++;
}

static uint32_t float_bits(float value)
{
    uint32_t bits;

    memcpy(&bits, &value, sizeof bits);
    return bits;
}

static float float_from_bits(uint32_t bits)
{
    float value;

    memcpy(&value, &bits, sizeof value);
    return value;
}

static int finite_float(float value)
{
    return (float_bits(value) & 0x7F800000u) != 0x7F800000u;
}

static void check_file_slurp(void)
{
    char path[] = "/tmp/tiny-agenc-integration-slurp-XXXXXX";
    int descriptor = mkstemp(path);

    expect(descriptor >= 0, "bounded slurp fixture path is created");
    if (descriptor < 0)
        return;

    FILE *stream = fdopen(descriptor, "wb");
    static const unsigned char BYTES[] = { 'A', 0, 'Z' };

    expect(stream != NULL, "bounded slurp fixture stream opens");
    if (stream == NULL) {
        close(descriptor);
        remove(path);
        return;
    }
    int wrote_fixture =
        fwrite(BYTES, 1, sizeof BYTES, stream) == sizeof BYTES;
    int closed_fixture = fclose(stream) == 0;

    expect(wrote_fixture && closed_fixture,
           "bounded slurp fixture is written");

    char *text = (char *)1;
    size_t size = 99;

    expect(file_slurp_bounded(path, sizeof BYTES, &text, &size)
               == FILE_SLURP_OK
           && text != NULL && size == sizeof BYTES
           && memcmp(text, BYTES, sizeof BYTES) == 0
           && text[size] == '\0',
           "bounded slurp accepts an exact ceiling and appends a sentinel");
    free(text);

    text = (char *)1;
    size = 99;
    expect(file_slurp_bounded(path, sizeof BYTES - 1, &text, &size)
               == FILE_SLURP_TOO_LARGE
           && text == NULL && size == 0,
           "bounded slurp rejects an oversized file without output");

    stream = fopen(path, "wb");
    int closed_empty = stream != NULL && fclose(stream) == 0;

    expect(closed_empty,
           "empty bounded slurp fixture is written");
    text = (char *)1;
    size = 99;
    expect(file_slurp_bounded(path, 0, &text, &size) == FILE_SLURP_OK
           && text != NULL && size == 0 && text[0] == '\0',
           "bounded slurp accepts an empty exact-ceiling file");
    free(text);

    text = (char *)1;
    size = 99;
    expect(file_slurp_bounded(NULL, 0, &text, &size)
               == FILE_SLURP_IO_ERROR
           && text == NULL && size == 0,
           "bounded slurp rejects a missing path and clears outputs");
    remove(path);
}

static int configs_equal(ModelConfig a, ModelConfig b)
{
    return a.vocab_size  == b.vocab_size
        && a.block_size  == b.block_size
        && a.d_model     == b.d_model
        && a.head_count  == b.head_count
        && a.layer_count == b.layer_count
        && a.batch_size  == b.batch_size;
}

/* -------- deterministic randomness -------- */

static void check_rng(void)
{
    /*
     * These are the high 24 bits of the published PCG32 stream after
     * Tiny AgenC's seed procedure, represented as exact float bits.
     * A replay-only test could let two identical mistakes agree.  Known
     * answers make the implementation answer to a fixed external fact.
     */
    static const uint32_t UNIFORM_BITS[] = {
        0x3F42F57Bu, 0x3ED60F88u, 0x3EE56F64u, 0x3E8842A6u,
        0x3F75AF5Eu, 0x3ED17D6Cu, 0x3F4BC731u, 0x3F55EFC7u,
    };
    static const int BELOW_TEN[] = { 7, 4, 4, 2, 9, 4, 7, 8, 4, 9, 7, 4 };

    Rng *uniform = rng_new(42);

    for (size_t i = 0; i < sizeof UNIFORM_BITS / sizeof UNIFORM_BITS[0]; i++)
        expect(float_bits(rng_uniform(uniform)) == UNIFORM_BITS[i],
               "PCG32 uniform known answer");
    rng_free(uniform);

    Rng *bounded = rng_new(42);

    for (size_t i = 0; i < sizeof BELOW_TEN / sizeof BELOW_TEN[0]; i++)
        expect(rng_below(bounded, 10) == BELOW_TEN[i],
               "PCG32 bounded known answer");
    rng_free(bounded);

    Rng *first  = rng_new(991);
    Rng *second = rng_new(991);

    for (int i = 0; i < 16; i++)
        expect(float_bits(rng_gaussian(first)) == float_bits(rng_gaussian(second)),
               "Box-Muller stream replays exactly");
    rng_free(first);
    rng_free(second);

    static const float GAUSSIAN_ANSWERS[] = {
        -0.782266140f, 1.062945604f, 0.196520776f, 0.952153027f,
        -0.718591213f, -1.716835856f, 0.370696634f, 0.374907821f,
    };
    Rng *gaussian = rng_new(1337);

    for (size_t i = 0;
         i < sizeof GAUSSIAN_ANSWERS / sizeof GAUSSIAN_ANSWERS[0];
         i++)
        expect(fabsf(rng_gaussian(gaussian) - GAUSSIAN_ANSWERS[i]) <= 1e-6f,
               "Box-Muller gaussian known answer");
    rng_free(gaussian);

    Rng *seed_1337 = rng_new(1337);
    Rng *seed_1338 = rng_new(1338);

    expect(float_bits(rng_uniform(seed_1337)) != float_bits(rng_uniform(seed_1338)),
           "different RNG seeds change the stream");
    rng_free(seed_1337);
    rng_free(seed_1338);
}

/* -------- tokenizer and dataset -------- */

static Tokenizer *read_vocabulary_fixture(int32_t count,
                                          const unsigned char *bytes,
                                          size_t stored)
{
    FILE *stream = tmpfile();

    if (stream == NULL
        || fwrite(&count, sizeof count, 1, stream) != 1
        || fwrite(bytes, 1, stored, stream) != stored
        || fseek(stream, 0, SEEK_SET) != 0) {
        if (stream != NULL)
            fclose(stream);
        expect(0, "tokenizer malformed fixture is writable");
        return NULL;
    }

    Tokenizer *tk = tokenizer_read(stream);

    expect(fclose(stream) == 0, "tokenizer malformed fixture closes");
    return tk;
}

static void check_tokenizer(void)
{
    static const char VOCABULARY_TEXT[] = "zaba\n";
    static const char TO_ENCODE[]       = "az?b\n";
    static const int  EXPECTED_IDS[]    = { 1, 3, 2, 0 };
    static const char EXPECTED_BYTES[]  = { '\n', 'a', 'b', 'z' };

    Tokenizer *tk = tokenizer_new(VOCABULARY_TEXT, sizeof VOCABULARY_TEXT - 1);

    expect(tokenizer_vocab_size(tk) == 4, "tokenizer deduplicates its vocabulary");
    for (int id = 0; id < 4; id++)
        expect(tokenizer_decode(tk, id) == EXPECTED_BYTES[id],
               "tokenizer ids are byte-sorted");

    int    ids[sizeof TO_ENCODE - 1];
    size_t encoded = tokenizer_encode(tk, ids, TO_ENCODE, sizeof TO_ENCODE - 1);

    expect(encoded == sizeof EXPECTED_IDS / sizeof EXPECTED_IDS[0],
           "tokenizer skips bytes outside the vocabulary");
    for (size_t i = 0; i < encoded; i++)
        expect(ids[i] == EXPECTED_IDS[i], "tokenizer encoding has stable ids");

    char decoded[sizeof TO_ENCODE - 1];

    if (encoded <= sizeof decoded)
        for (size_t i = 0; i < encoded; i++)
            decoded[i] = tokenizer_decode(tk, ids[i]);
    expect(encoded == 4 && memcmp(decoded, "azb\n", 4) == 0,
           "tokenizer encode and decode round-trip known bytes");

    FILE *serialized = tmpfile();

    expect(serialized != NULL, "tokenizer test opens a temporary stream");
    if (serialized != NULL) {
        expect(tokenizer_write(tk, serialized) == 0,
               "tokenizer writes its vocabulary");
        rewind(serialized);

        Tokenizer *loaded = tokenizer_read(serialized);

        expect(loaded != NULL, "tokenizer reads its vocabulary");
        if (loaded != NULL) {
            expect(tokenizer_vocab_size(loaded) == tokenizer_vocab_size(tk),
                   "tokenizer round-trip keeps vocabulary size");
            for (int id = 0; id < tokenizer_vocab_size(tk); id++)
                expect(tokenizer_decode(loaded, id) == tokenizer_decode(tk, id),
                       "tokenizer round-trip keeps id order");
            tokenizer_free(loaded);
        }
        fclose(serialized);
    }

    static const unsigned char DESCENDING[] = { 'z', 'a' };
    static const unsigned char DUPLICATE[] = { 'a', 'a' };
    static const unsigned char SHORT_BODY[] = { 'a' };
    static const unsigned char HIGH_BYTES[] = {
        0x00u, 0x7Fu, 0x80u, 0xFFu,
    };

    expect(read_vocabulary_fixture(2, DESCENDING,
                                   sizeof DESCENDING) == NULL,
           "tokenizer rejects descending saved bytes");
    expect(read_vocabulary_fixture(2, DUPLICATE,
                                   sizeof DUPLICATE) == NULL,
           "tokenizer rejects duplicate saved bytes");
    expect(read_vocabulary_fixture(2, SHORT_BODY,
                                   sizeof SHORT_BODY) == NULL,
           "tokenizer rejects a short saved vocabulary");
    expect(read_vocabulary_fixture(0, HIGH_BYTES, 0) == NULL,
           "tokenizer rejects a zero saved vocabulary count");
    expect(read_vocabulary_fixture(257, HIGH_BYTES, 0) == NULL,
           "tokenizer rejects an oversized saved vocabulary count");

    Tokenizer *high = read_vocabulary_fixture(4, HIGH_BYTES,
                                              sizeof HIGH_BYTES);

    expect(high != NULL && tokenizer_vocab_size(high) == 4,
           "tokenizer accepts canonical unsigned high bytes");
    if (high != NULL) {
        for (int id = 0; id < 4; id++)
            expect((unsigned char)tokenizer_decode(high, id)
                       == HIGH_BYTES[id],
                   "tokenizer preserves unsigned high-byte ids");
        tokenizer_free(high);
    }
    tokenizer_free(tk);
}

static void check_dataset(void)
{
    static const char MINIMUM[] = "abcd";

    Tokenizer *minimum_tk = tokenizer_new(MINIMUM, sizeof MINIMUM - 1);
    Dataset   *minimum_ds = dataset_new(minimum_tk, MINIMUM, sizeof MINIMUM - 1);
    Rng       *minimum_rng = rng_new(19);
    int        inputs[3];
    int        targets[3];

    expect(dataset_token_count(minimum_ds) == sizeof MINIMUM - 1,
           "dataset reports encoded token count");
    dataset_batch(minimum_ds, minimum_rng, inputs, targets, 1, 3);
    for (int i = 0; i < 3; i++) {
        expect(inputs[i] == i, "minimum dataset uses its only legal window");
        expect(targets[i] == i + 1, "minimum dataset targets are shifted once");
    }
    rng_free(minimum_rng);
    dataset_free(minimum_ds);
    tokenizer_free(minimum_tk);

    static const char DIGITS[] = "0123456789";
    enum { ROWS = 256, BLOCK = 3 };

    Tokenizer *digits_tk = tokenizer_new(DIGITS, sizeof DIGITS - 1);
    Dataset   *digits_ds = dataset_new(digits_tk, DIGITS, sizeof DIGITS - 1);
    Rng       *digits_rng = rng_new(42);
    int        many_inputs[ROWS * BLOCK];
    int        many_targets[ROWS * BLOCK];
    int        saw_first = 0;
    int        saw_last = 0;

    dataset_batch(digits_ds, digits_rng, many_inputs, many_targets, ROWS, BLOCK);
    for (int row = 0; row < ROWS; row++) {
        int start = many_inputs[row * BLOCK];

        expect(start >= 0 && start <= 6, "dataset start stays inside legal bounds");
        saw_first |= start == 0;
        saw_last  |= start == 6;
        for (int t = 0; t < BLOCK; t++) {
            expect(many_inputs[row * BLOCK + t] == start + t,
                   "dataset input window is consecutive");
            expect(many_targets[row * BLOCK + t] == start + t + 1,
                   "dataset target window is shifted once");
        }
    }
    expect(saw_first, "dataset can draw its first legal window");
    expect(saw_last, "dataset can draw its last legal window");

    rng_free(digits_rng);
    dataset_free(digits_ds);
    tokenizer_free(digits_tk);

    Tokenizer *filtered_tk = tokenizer_new("ab", 2);
    Dataset   *filtered_ds = dataset_new(filtered_tk, "a?b", 3);

    expect(dataset_token_count(filtered_ds) == 2,
           "dataset count reflects tokenizer filtering");
    dataset_free(filtered_ds);

    size_t corpus_limit = dataset_max_text_bytes();

    expect(corpus_limit <= (size_t)256 * 1024 * 1024,
           "dataset source policy is capped at 256 MiB");
    expect(corpus_limit <= SIZE_MAX / sizeof(int),
           "dataset source policy fits its token allocation");
    expect(corpus_limit <= (size_t)INT_MAX,
           "dataset source policy fits integer run indexes");
    expect(dataset_new(filtered_tk, "a", corpus_limit + 1) == NULL,
           "dataset rejects an oversized source before reading it");
    tokenizer_free(filtered_tk);
}

/* -------- matrix views -------- */

static void check_mat(void)
{
    float storage[12];

    for (int i = 0; i < 12; i++)
        storage[i] = (float)i;

    Mat full = mat_make(storage, 3, 4);

    expect(full.vals == storage, "Mat keeps the supplied storage");
    expect(full.rows == 3 && full.cols == 4, "Mat keeps its shape");
    expect(mat_size(full) == 12, "mat_size multiplies rows and columns");
    expect(mat_row(full, 0) == storage, "first matrix row starts at storage");
    expect(mat_row(full, 2) == storage + 8, "matrix rows use row-major offsets");
    expect(mat_row(full, 1)[3] == 7.0f, "row-major indexing reaches known value");

    Mat prefix = mat_first_rows(full, 2);

    expect(prefix.vals == full.vals, "leading-row view shares storage");
    expect(prefix.rows == 2 && prefix.cols == 4,
           "leading-row view changes only row count");
    expect(mat_size(prefix) == 8, "leading-row view reports its visible size");

    mat_row(prefix, 1)[2] = 99.0f;
    expect(mat_row(full, 1)[2] == 99.0f, "writes through a Mat view reach its source");

    Mat flattened_batch = mat_make(storage, 2 * 3, 2);

    expect(mat_row(flattened_batch, 4) == storage + 8,
           "flattened batch rows use batch times position order");
}

/* -------- independent forward examples -------- */

static int close_float(float actual, float expected, float tolerance)
{
    return fabsf(actual - expected) <= tolerance;
}

static void check_forward(void)
{
    float x_values[] = {
         1.0f, 2.0f, 3.0f,
        -1.0f, 0.0f, 2.0f,
    };
    float weight_values[] = {
        1.0f, 0.0f, -1.0f,
        2.0f, 1.0f,  0.5f,
    };
    float product_values[4] = { 0 };
    Mat x       = mat_make(x_values, 2, 3);
    Mat weights = mat_make(weight_values, 2, 3);
    Mat product = mat_make(product_values, 2, 2);

    matmul_forward(product, x, weights);
    expect(product_values[0] == -2.0f, "known matmul row 0 column 0");
    expect(product_values[1] ==  5.5f, "known matmul row 0 column 1");
    expect(product_values[2] == -3.0f, "known matmul row 1 column 0");
    expect(product_values[3] == -1.0f, "known matmul row 1 column 1");

    float softmax[] = { 1000.0f, 1000.0f, 999.0f };

    softmax_in_place(softmax, 3);
    expect(close_float(softmax[0] + softmax[1] + softmax[2], 1.0f, 1e-6f),
           "stable softmax sums to one");
    expect(close_float(softmax[0], softmax[1], 1e-7f),
           "equal logits receive equal probability");
    expect(softmax[0] > softmax[2], "larger logit receives larger probability");

    float logits_values[8] = { 0 };
    float probs_values[8]  = { 0 };
    int   class_targets[]   = { 0, 3 };
    Mat   logits = mat_make(logits_values, 2, 4);
    Mat   probs  = mat_make(probs_values, 2, 4);
    float uniform_loss = crossentropy_forward(probs, logits, class_targets);

    expect(close_float(uniform_loss, logf(4.0f), 1e-6f),
           "uniform logits produce log vocabulary loss");
    for (int i = 0; i < 8; i++)
        expect(probs_values[i] == 0.25f,
               "uniform logits produce uniform probabilities");

    float left_values[]  = { 1.0f, -2.0f, 4.5f };
    float right_values[] = { 3.0f,  5.0f, 0.5f };
    float sum_values[3]  = { 0 };

    residual_forward(mat_make(sum_values, 1, 3),
                     mat_make(left_values, 1, 3),
                     mat_make(right_values, 1, 3));
    expect(sum_values[0] == 4.0f, "residual adds first element");
    expect(sum_values[1] == 3.0f, "residual adds second element");
    expect(sum_values[2] == 5.0f, "residual adds third element");

    enum { TIME = 3, CHANNELS = 2, QKV_COLS = QKV_STREAMS * CHANNELS };
    float qkv_values[TIME * QKV_COLS] = {
        1.0f, 0.0f,  1.0f, 0.0f,  2.0f, 1.0f,
        0.0f, 1.0f,  0.0f, 1.0f,  4.0f, 3.0f,
        1.0f, 1.0f,  1.0f, 1.0f,  8.0f, 7.0f,
    };
    float attended_values[TIME * CHANNELS] = { 0 };
    float score_values[TIME * TIME] = { 0 };
    Mat qkv      = mat_make(qkv_values, TIME, QKV_COLS);
    Mat attended = mat_make(attended_values, TIME, CHANNELS);
    Mat scores   = mat_make(score_values, TIME, TIME);

    attention_forward(attended, scores, qkv, TIME, 1);

    float earlier[2 * CHANNELS];

    memcpy(earlier, attended_values, sizeof earlier);
    for (int c = 0; c < QKV_COLS; c++)
        mat_row(qkv, 2)[c] = 1000.0f + (float)c;
    attention_forward(attended, scores, qkv, TIME, 1);
    expect(memcmp(earlier, attended_values, sizeof earlier) == 0,
           "future QKV cannot change earlier attention outputs");

    float token_values[2] = { 0 };
    float position_values[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
    };
    float embedded_values[2 * TIME * CHANNELS] = { 0 };
    int   tokens[2 * TIME] = { 0 };

    embedding_forward(mat_make(embedded_values, 2 * TIME, CHANNELS),
                      tokens, mat_make(token_values, 1, CHANNELS),
                      mat_make(position_values, TIME, CHANNELS), TIME);
    expect(memcmp(embedded_values, position_values, sizeof position_values) == 0,
           "first sequence receives positions zero through time");
    expect(memcmp(embedded_values + TIME * CHANNELS,
                  position_values, sizeof position_values) == 0,
           "position indices restart for the next sequence");
}

static void choose_test_threads(int count)
{
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(count);
#else
    (void)count;
#endif
}

static void check_parallel_contracts(void)
{
    enum {
        ROWS = 64,
        INPUTS = 7,
        OUTPUTS = 64,
        SEQUENCES = 32,
        TIME = 3,
        CHANNELS = 4,
        HEADS = 2,
        ATTENTION_ROWS = SEQUENCES * TIME,
        SCORE_ROWS = SEQUENCES * HEADS * TIME,
    };
    float x[ROWS * INPUTS];
    float weights[OUTPUTS * INPUTS];
    float d_out[ROWS * OUTPUTS];
    float serial_out[ROWS * OUTPUTS];
    float parallel_out[ROWS * OUTPUTS];
    float serial_d_x[ROWS * INPUTS] = { 0 };
    float parallel_d_x[ROWS * INPUTS] = { 0 };
    float serial_d_weights[OUTPUTS * INPUTS] = { 0 };
    float parallel_d_weights[OUTPUTS * INPUTS] = { 0 };

    for (size_t i = 0; i < sizeof x / sizeof x[0]; i++)
        x[i] = (float)((int)(i % 17) - 8) / 11.0f;
    for (size_t i = 0; i < sizeof weights / sizeof weights[0]; i++)
        weights[i] = (float)((int)(i % 13) - 6) / 9.0f;
    for (size_t i = 0; i < sizeof d_out / sizeof d_out[0]; i++)
        d_out[i] = (float)((int)(i % 19) - 9) / 7.0f;

    choose_test_threads(1);
    matmul_forward(mat_make(serial_out, ROWS, OUTPUTS),
                   mat_make(x, ROWS, INPUTS),
                   mat_make(weights, OUTPUTS, INPUTS));
    matmul_backward(mat_make(serial_d_x, ROWS, INPUTS),
                    mat_make(serial_d_weights, OUTPUTS, INPUTS),
                    mat_make(d_out, ROWS, OUTPUTS),
                    mat_make(x, ROWS, INPUTS),
                    mat_make(weights, OUTPUTS, INPUTS));

    choose_test_threads(4);
    matmul_forward(mat_make(parallel_out, ROWS, OUTPUTS),
                   mat_make(x, ROWS, INPUTS),
                   mat_make(weights, OUTPUTS, INPUTS));
    matmul_backward(mat_make(parallel_d_x, ROWS, INPUTS),
                    mat_make(parallel_d_weights, OUTPUTS, INPUTS),
                    mat_make(d_out, ROWS, OUTPUTS),
                    mat_make(x, ROWS, INPUTS),
                    mat_make(weights, OUTPUTS, INPUTS));

    expect(memcmp(serial_out, parallel_out, sizeof serial_out) == 0,
           "threshold-crossing matmul forward is thread-count invariant");
    expect(memcmp(serial_d_x, parallel_d_x, sizeof serial_d_x) == 0
           && memcmp(serial_d_weights, parallel_d_weights,
                     sizeof serial_d_weights) == 0,
           "threshold-crossing matmul backward is thread-count invariant");

    float qkv[ATTENTION_ROWS * QKV_STREAMS * CHANNELS];
    float attention_d_out[ATTENTION_ROWS * CHANNELS];
    float serial_attention[ATTENTION_ROWS * CHANNELS];
    float parallel_attention[ATTENTION_ROWS * CHANNELS];
    float serial_scores[SCORE_ROWS * TIME] = { 0 };
    float parallel_scores[SCORE_ROWS * TIME] = { 0 };
    float serial_d_qkv[ATTENTION_ROWS * QKV_STREAMS * CHANNELS] = { 0 };
    float parallel_d_qkv[ATTENTION_ROWS * QKV_STREAMS * CHANNELS] = { 0 };
    float serial_d_scores[SCORE_ROWS * TIME] = { 0 };
    float parallel_d_scores[SCORE_ROWS * TIME] = { 0 };

    for (size_t i = 0; i < sizeof qkv / sizeof qkv[0]; i++)
        qkv[i] = (float)((int)(i % 23) - 11) / 10.0f;
    for (size_t i = 0;
         i < sizeof attention_d_out / sizeof attention_d_out[0]; i++)
        attention_d_out[i] = (float)((int)(i % 11) - 5) / 8.0f;

    choose_test_threads(1);
    attention_forward(mat_make(serial_attention, ATTENTION_ROWS, CHANNELS),
                      mat_make(serial_scores, SCORE_ROWS, TIME),
                      mat_make(qkv, ATTENTION_ROWS,
                               QKV_STREAMS * CHANNELS),
                      TIME, HEADS);
    attention_backward(
        mat_make(serial_d_qkv, ATTENTION_ROWS, QKV_STREAMS * CHANNELS),
        mat_make(serial_d_scores, SCORE_ROWS, TIME),
        mat_make(attention_d_out, ATTENTION_ROWS, CHANNELS),
        mat_make(qkv, ATTENTION_ROWS, QKV_STREAMS * CHANNELS),
        mat_make(serial_scores, SCORE_ROWS, TIME), TIME, HEADS);

    choose_test_threads(4);
    attention_forward(mat_make(parallel_attention, ATTENTION_ROWS, CHANNELS),
                      mat_make(parallel_scores, SCORE_ROWS, TIME),
                      mat_make(qkv, ATTENTION_ROWS,
                               QKV_STREAMS * CHANNELS),
                      TIME, HEADS);
    attention_backward(
        mat_make(parallel_d_qkv, ATTENTION_ROWS, QKV_STREAMS * CHANNELS),
        mat_make(parallel_d_scores, SCORE_ROWS, TIME),
        mat_make(attention_d_out, ATTENTION_ROWS, CHANNELS),
        mat_make(qkv, ATTENTION_ROWS, QKV_STREAMS * CHANNELS),
        mat_make(parallel_scores, SCORE_ROWS, TIME), TIME, HEADS);

    expect(memcmp(serial_attention, parallel_attention,
                  sizeof serial_attention) == 0
           && memcmp(serial_scores, parallel_scores,
                     sizeof serial_scores) == 0,
           "threshold-crossing attention forward is thread-count invariant");
    expect(memcmp(serial_d_qkv, parallel_d_qkv, sizeof serial_d_qkv) == 0
           && memcmp(serial_d_scores, parallel_d_scores,
                     sizeof serial_d_scores) == 0,
           "threshold-crossing attention backward is thread-count invariant");
}

/* -------- model contract and complete smoke run -------- */

static ModelConfig tiny_config(void)
{
    ModelConfig cfg = {
        .vocab_size  = 4,
        .block_size  = 4,
        .d_model     = 8,
        .head_count  = 2,
        .layer_count = 1,
        .batch_size  = 2,
    };

    return cfg;
}

static ModelConfig maximum_config(void)
{
    ModelConfig cfg = {
        .vocab_size  = MODEL_MAX_VOCAB_SIZE,
        .block_size  = MODEL_MAX_BLOCK_SIZE,
        .d_model     = MODEL_MAX_D_MODEL,
        .head_count  = MODEL_MAX_HEAD_COUNT,
        .layer_count = MODEL_MAX_LAYER_COUNT,
        .batch_size  = MODEL_MAX_TOKENS_PER_PASS / MODEL_MAX_BLOCK_SIZE,
    };

    return cfg;
}

static size_t architecture_parameter_count(ModelConfig cfg)
{
    size_t width = (size_t)cfg.d_model;

    return ((size_t)cfg.vocab_size + (size_t)cfg.block_size) * width
         + (size_t)cfg.layer_count * (12u * width * width + 4u * width)
         + 2u * width;
}

static int models_equal(const Model *first, const Model *second);

static int memory_reports_equal(ModelMemory first, ModelMemory second)
{
    return first.parameter_bytes == second.parameter_bytes
        && first.activation_bytes == second.activation_bytes
        && first.gradient_bytes == second.gradient_bytes
        && first.token_bytes == second.token_bytes
        && first.total_bytes == second.total_bytes;
}

static void check_model_config_contract(ModelConfig cfg)
{
    expect(model_config_valid(cfg), "representative model config is valid");

    ModelConfig minimum = {
        .vocab_size  = 1,
        .block_size  = 1,
        .d_model     = 1,
        .head_count  = 1,
        .layer_count = 1,
        .batch_size  = 1,
    };
    expect(model_config_valid(minimum), "model dimension minima are inclusive");

    ModelConfig invalid = cfg;
    invalid.vocab_size = 0;
    expect(!model_config_valid(invalid), "model rejects an empty vocabulary");

    invalid = cfg;
    invalid.block_size = 0;
    expect(!model_config_valid(invalid), "model rejects an empty context");

    invalid = cfg;
    invalid.d_model = 7;
    expect(!model_config_valid(invalid), "model width must divide into heads");

    invalid = cfg;
    invalid.d_model = 0;
    expect(!model_config_valid(invalid), "model rejects an empty width");

    invalid = cfg;
    invalid.head_count = 0;
    expect(!model_config_valid(invalid), "model rejects an empty head count");

    invalid = cfg;
    invalid.layer_count = 0;
    expect(!model_config_valid(invalid), "model needs at least one layer");

    invalid = cfg;
    invalid.batch_size = 0;
    expect(!model_config_valid(invalid), "model rejects an empty batch");

    ModelConfig boundary = maximum_config();

    expect(model_config_valid(boundary), "documented model limits are inclusive");

    invalid = boundary;
    invalid.vocab_size = MODEL_MAX_VOCAB_SIZE + 1;
    expect(!model_config_valid(invalid), "model rejects vocabulary above its limit");

    invalid = boundary;
    invalid.block_size = MODEL_MAX_BLOCK_SIZE + 1;
    invalid.batch_size = 1;
    expect(!model_config_valid(invalid), "model rejects context above its limit");

    invalid = boundary;
    invalid.d_model = MODEL_MAX_D_MODEL + 1;
    invalid.head_count = 1;
    expect(!model_config_valid(invalid), "model rejects width above its limit");

    invalid = boundary;
    invalid.head_count = MODEL_MAX_HEAD_COUNT + 1;
    invalid.d_model = MODEL_MAX_HEAD_COUNT + 1;
    expect(!model_config_valid(invalid), "model rejects heads above their limit");

    invalid = boundary;
    invalid.layer_count = MODEL_MAX_LAYER_COUNT + 1;
    expect(!model_config_valid(invalid), "model rejects layers above their limit");

    invalid = boundary;
    invalid.batch_size++;
    expect(!model_config_valid(invalid), "model caps tokens in one pass");
}

static void check_model_memory_contract(void)
{
    ModelConfig showcase = {
        .vocab_size  = 80,
        .block_size  = 128,
        .d_model     = 128,
        .head_count  = 4,
        .layer_count = 4,
        .batch_size  = 32,
    };

    expect(model_config_valid(showcase), "showcase model config is valid");
    expect(architecture_parameter_count(showcase) == 815360,
           "showcase architecture has 815360 parameters");

    ModelMemory showcase_memory;

    expect(model_memory_requirements(showcase, &showcase_memory),
           "showcase memory requirements are computable");
    expect(showcase_memory.parameter_bytes == 13045760,
           "showcase parameter storage is exact");
    expect(showcase_memory.activation_bytes == 191660032,
           "showcase value arena storage is exact");
    expect(showcase_memory.gradient_bytes == 190054400,
           "showcase gradient arena storage is exact");
    expect(showcase_memory.token_bytes == 32768,
           "showcase token cache storage is exact");
    expect(showcase_memory.total_bytes == 394792960,
           "showcase total storage is exact");

    ModelMemory maximum_memory;

    expect(model_memory_requirements(maximum_config(), &maximum_memory)
           && maximum_memory.total_bytes
              > MODEL_MAX_CHECKPOINT_RESIDENT_BYTES,
           "maximum valid geometry has a representable memory report");

    ModelMemory sentinel = {
        .parameter_bytes  = 11,
        .activation_bytes = 22,
        .gradient_bytes   = 33,
        .token_bytes      = 44,
        .total_bytes      = 55,
    };
    ModelMemory unchanged = sentinel;
    ModelConfig invalid = showcase;

    invalid.vocab_size = 0;
    expect(!model_memory_requirements(invalid, &unchanged)
           && memory_reports_equal(unchanged, sentinel),
           "failed memory report leaves caller storage unchanged");
    expect(!model_memory_requirements(showcase, NULL),
           "memory report rejects a missing output record");

    ModelConfig resource_bomb = {
        .vocab_size  = 1,
        .block_size  = 1024,
        .d_model     = 1,
        .head_count  = 1,
        .layer_count = 1,
        .batch_size  = 1024,
    };
    ModelMemory bomb_memory;

    expect(model_config_valid(resource_bomb),
           "resource-bomb geometry is individually admissible");
    expect(model_memory_requirements(resource_bomb, &bomb_memory)
           && bomb_memory.total_bytes > MODEL_MAX_CHECKPOINT_RESIDENT_BYTES,
           "checkpoint memory preflight identifies an oversized arena");
}

static void check_constructed_model_contract(ModelConfig cfg)
{
    expect(param_new_constant(0, 1, 0.0f) == NULL,
           "Param rejects a non-positive shape");
    expect(param_new_constant(INT_MAX, INT_MAX, 0.0f) == NULL,
           "Param rejects an unrepresentable four-buffer allocation");

    Model *m = model_new(cfg, 17);

    expect(configs_equal(model_config(m), cfg), "model reports its construction config");

    expect(model_parameter_count(m) == architecture_parameter_count(cfg),
           "model parameter count matches the architecture formula");

    ModelMemory memory;

    expect(model_memory_requirements(cfg, &memory),
           "constructed model memory requirements are computable");
    expect(memory.parameter_bytes
           == model_parameter_count(m) * 4 * sizeof(float),
           "parameter storage formula matches the constructed model");

    static const int EXPECTED_ROWS[] = { 4, 4, 1, 1, 24, 8, 1, 1, 32, 8, 1, 1 };
    static const int EXPECTED_COLS[] = { 8, 8, 8, 8,  8, 8, 8, 8,  8, 32, 8, 8 };
    ModelParams params = model_params(m);

    expect(params.count == 12, "one-layer model exposes twelve parameter tensors");
    if (params.count == 12) {
        for (int i = 0; i < params.count; i++) {
            Mat values = param_values(params.params[i]);

            expect(values.rows == EXPECTED_ROWS[i], "parameter tensor has expected rows");
            expect(values.cols == EXPECTED_COLS[i], "parameter tensor has expected columns");
        }
    }
    model_free(m);
}

static void check_model_contract(void)
{
    ModelConfig cfg = tiny_config();

    check_model_config_contract(cfg);
    check_model_memory_contract();
    check_constructed_model_contract(cfg);
}

static void check_optimizer_integration(void)
{
    Model *m = model_new(tiny_config(), 5);
    ModelParams params = model_params(m);

    for (int p = 0; p < params.count; p++) {
        Mat values  = param_values(params.params[p]);
        Mat gradient = param_gradient(params.params[p]);

        memset(values.vals, 0, mat_size(values) * sizeof *values.vals);
        memset(gradient.vals, 0, mat_size(gradient) * sizeof *gradient.vals);
    }

    Mat first_values   = param_values(params.params[0]);
    Mat first_gradient = param_gradient(params.params[0]);

    first_gradient.vals[0] = 3.0f;
    first_gradient.vals[1] = 4.0f;

    AdamW opt = {
        .learning_rate = 1.0f,
        .beta1         = 0.0f,
        .beta2         = 0.0f,
        .epsilon       = 1.0f,
        .weight_decay  = 0.0f,
    };

    expect(model_step(m, opt, 1) == 0,
           "ordinary finite optimizer step succeeds");

    expect(close_float(first_gradient.vals[0], 0.6f, 1e-6f),
           "gradient clipping scales first component");
    expect(close_float(first_gradient.vals[1], 0.8f, 1e-6f),
           "gradient clipping scales second component");

    double norm_squared = 0.0;

    for (int p = 0; p < params.count; p++) {
        Mat gradient = param_gradient(params.params[p]);

        for (size_t i = 0; i < mat_size(gradient); i++)
            norm_squared += (double)gradient.vals[i] * gradient.vals[i];
    }
    expect(close_float((float)sqrt(norm_squared), 1.0f, 1e-6f),
           "global gradient norm is clipped to one");
    expect(close_float(first_gradient.vals[0] / first_gradient.vals[1],
                       3.0f / 4.0f, 1e-6f),
           "gradient clipping preserves direction");
    expect(close_float(first_values.vals[0], -0.375f, 1e-6f),
           "AdamW sees clipped first component");
    expect(close_float(first_values.vals[1], -4.0f / 9.0f, 1e-6f),
           "AdamW sees clipped second component");

    for (size_t i = 2; i < mat_size(first_values); i++)
        expect(first_values.vals[i] == 0.0f,
               "zero gradient leaves other values unchanged");

    model_free(m);

    Model *huge = model_new(tiny_config(), 6);
    ModelParams huge_params = model_params(huge);
    Mat huge_gradient = param_gradient(huge_params.params[0]);
    AdamW no_move = opt;

    no_move.learning_rate = 0.0f;
    huge_gradient.vals[0] = 3e20f;
    huge_gradient.vals[1] = 4e20f;
    expect(model_step(huge, no_move, 1) == 0,
           "huge finite gradients take the overflow-safe clipping path");
    expect(finite_float(huge_gradient.vals[0])
           && finite_float(huge_gradient.vals[1]),
           "huge finite gradients stay finite after clipping");
    expect(close_float(huge_gradient.vals[0], 0.6f, 1e-6f)
           && close_float(huge_gradient.vals[1], 0.8f, 1e-6f),
           "overflow-safe clipping preserves a huge gradient's direction");
    model_free(huge);

    static const uint32_t NONFINITE_BITS[] = {
        0x7FC00000u,
        0x7F800000u,
    };
    static const char *NONFINITE_LABELS[] = {
        "NaN gradient is rejected before any parameter update",
        "infinite gradient is rejected before any parameter update",
    };

    for (size_t kind = 0;
         kind < sizeof NONFINITE_BITS / sizeof NONFINITE_BITS[0];
         kind++) {
        Model *bad = model_new(tiny_config(), 7);
        Model *reference = model_new(tiny_config(), 7);
        ModelParams bad_params = model_params(bad);
        Mat bad_gradient = param_gradient(bad_params.params[0]);

        bad_gradient.vals[0] = float_from_bits(NONFINITE_BITS[kind]);
        bad_gradient.vals[1] = 4.0f;
        expect(model_step(bad, opt, 1) == -1
               && float_bits(bad_gradient.vals[0]) == NONFINITE_BITS[kind]
               && bad_gradient.vals[1] == 4.0f
               && models_equal(bad, reference),
               NONFINITE_LABELS[kind]);
        model_free(reference);
        model_free(bad);
    }

    Model *invalid_opt_model = model_new(tiny_config(), 8);
    Model *invalid_opt_reference = model_new(tiny_config(), 8);
    ModelParams invalid_params = model_params(invalid_opt_model);
    Mat invalid_gradient = param_gradient(invalid_params.params[0]);
    AdamW invalid_opt = opt;

    invalid_gradient.vals[0] = 3.0f;
    invalid_gradient.vals[1] = 4.0f;
    invalid_opt.learning_rate = float_from_bits(0x7FC00000u);
    expect(model_step(invalid_opt_model, invalid_opt, 1) == -1
           && invalid_gradient.vals[0] == 3.0f
           && invalid_gradient.vals[1] == 4.0f
           && models_equal(invalid_opt_model, invalid_opt_reference),
           "invalid optimizer settings are rejected before clipping");
    model_free(invalid_opt_reference);
    model_free(invalid_opt_model);
}

static int models_equal(const Model *first, const Model *second)
{
    ModelParams a = model_params(first);
    ModelParams b = model_params(second);

    if (!configs_equal(model_config(first), model_config(second))
        || a.count != b.count)
        return 0;
    for (int i = 0; i < a.count; i++) {
        Mat av = param_values(a.params[i]);
        Mat bv = param_values(b.params[i]);

        if (av.rows != bv.rows || av.cols != bv.cols
            || memcmp(av.vals, bv.vals, mat_size(av) * sizeof *av.vals) != 0)
            return 0;
    }
    return 1;
}

static int temporary_path(char *path)
{
    int descriptor = mkstemp(path);

    if (descriptor < 0)
        return -1;
    if (close(descriptor) != 0) {
        remove(path);
        return -1;
    }
    return 0;
}

static void expect_load_failure(const char *path, Tokenizer *sentinel,
                                const char *label)
{
    Tokenizer *loaded_tk = sentinel;
    Model     *loaded = model_load(&loaded_tk, path);

    expect(loaded == NULL && loaded_tk == NULL, label);
    if (loaded != NULL)
        model_free(loaded);
    if (loaded_tk != NULL && loaded_tk != sentinel)
        tokenizer_free(loaded_tk);
}

static int overwrite_file_bytes(const char *path, long offset,
                                const void *bytes, size_t count)
{
    FILE *stream = fopen(path, "r+b");

    if (stream == NULL)
        return -1;
    int failed = fseek(stream, offset, SEEK_SET) != 0
              || fwrite(bytes, 1, count, stream) != count;

    if (fclose(stream) != 0)
        failed = 1;
    return failed ? -1 : 0;
}

static uint32_t fixture_crc32(uint32_t crc, const unsigned char *bytes,
                              size_t count)
{
    for (size_t i = 0; i < count; i++) {
        crc ^= bytes[i];
        for (int bit = 0; bit < 8; bit++)
            crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
    return crc;
}

static int rewrite_checkpoint_checksum(const char *path)
{
    FILE *stream = fopen(path, "r+b");

    if (stream == NULL || fseek(stream, 0, SEEK_END) != 0) {
        if (stream != NULL)
            fclose(stream);
        return -1;
    }

    long end = ftell(stream);

    if (end < (long)sizeof(uint32_t)
        || fseek(stream, 0, SEEK_SET) != 0) {
        fclose(stream);
        return -1;
    }

    long remaining = end - (long)sizeof(uint32_t);
    uint32_t crc = 0xFFFFFFFFu;
    unsigned char buffer[4096];

    while (remaining > 0) {
        size_t wanted =
            remaining < (long)sizeof buffer ? (size_t)remaining : sizeof buffer;
        size_t received = fread(buffer, 1, wanted, stream);

        if (received != wanted) {
            fclose(stream);
            return -1;
        }
        crc = fixture_crc32(crc, buffer, received);
        remaining -= (long)received;
    }

    crc = ~crc;
    int failed = fseek(stream, end - (long)sizeof crc, SEEK_SET) != 0
              || fwrite(&crc, sizeof crc, 1, stream) != 1;

    if (fclose(stream) != 0)
        failed = 1;
    return failed ? -1 : 0;
}

static int write_resource_checkpoint(const char *path, ModelConfig cfg)
{
    int32_t fields[] = {
        (int32_t)0x43474154u,
        1,
        cfg.vocab_size,
        cfg.block_size,
        cfg.d_model,
        cfg.head_count,
        cfg.layer_count,
        cfg.batch_size,
        cfg.vocab_size,
    };
    FILE *stream = fopen(path, "wb");
    unsigned char token = '\n';
    float zero = 0.0f;
    uint32_t checksum = 0;

    if (stream == NULL)
        return -1;
    int failed =
        fwrite(fields, sizeof fields[0],
               sizeof fields / sizeof fields[0], stream)
            != sizeof fields / sizeof fields[0]
        || fwrite(&token, sizeof token, 1, stream) != 1;

    for (size_t i = 0; !failed && i < architecture_parameter_count(cfg); i++)
        failed = fwrite(&zero, sizeof zero, 1, stream) != 1;
    if (!failed)
        failed = fwrite(&checksum, sizeof checksum, 1, stream) != 1;
    if (fclose(stream) != 0)
        failed = 1;
    return failed || rewrite_checkpoint_checksum(path) != 0 ? -1 : 0;
}

static void check_training_checkpoint_and_sampling(void)
{
    static const char TRAINING_TEXT[] = "\nabcabcabcabcabcabcabcabc\n";
    static const char INPUT_TEXT[]    = "\nabc\nabc";
    static const char TARGET_TEXT[]   = "abc\nabc\n";
    enum { TOKENS = 8, TRAIN_STEPS = 60, SAMPLE_TOKENS = 32 };

    Tokenizer *tk = tokenizer_new(TRAINING_TEXT, sizeof TRAINING_TEXT - 1);
    ModelConfig cfg = tiny_config();
    Model *m = model_new(cfg, 2024);
    int inputs[TOKENS], targets[TOKENS];

    expect(tokenizer_encode(tk, inputs, INPUT_TEXT, sizeof INPUT_TEXT - 1) == TOKENS,
           "smoke inputs encode completely");
    expect(tokenizer_encode(tk, targets, TARGET_TEXT, sizeof TARGET_TEXT - 1) == TOKENS,
           "smoke targets encode completely");

    float initial_loss = model_forward(m, inputs, targets, cfg.batch_size, cfg.block_size);
    AdamW opt = {
        .learning_rate = 0.02f,
        .beta1         = 0.9f,
        .beta2         = 0.999f,
        .epsilon       = 1e-8f,
        .weight_decay  = 0.01f,
    };

    int update_failed = 0;

    for (int step = 1; step <= TRAIN_STEPS; step++) {
        model_zero_gradients(m);
        (void)model_forward(m, inputs, targets, cfg.batch_size, cfg.block_size);
        model_backward(m);
        if (model_step(m, opt, step) != 0) {
            update_failed = 1;
            break;
        }
    }
    expect(!update_failed, "tiny training accepts every finite optimizer step");

    float trained_loss = model_forward(m, inputs, targets, cfg.batch_size, cfg.block_size);

    expect(finite_float(initial_loss) && finite_float(trained_loss),
           "tiny training losses stay finite");
    expect(trained_loss < initial_loss * 0.5f,
           "tiny model overfits a repeating batch");

    char checkpoint[] = "/tmp/tiny-agenc-integration-checkpoint-XXXXXX";
    char bad_magic[]  = "/tmp/tiny-agenc-integration-magic-XXXXXX";
    char bad_version[] = "/tmp/tiny-agenc-integration-version-XXXXXX";
    char bad_config[] = "/tmp/tiny-agenc-integration-config-XXXXXX";
    char bad_tokens[] = "/tmp/tiny-agenc-integration-tokens-XXXXXX";
    char corrupt[] = "/tmp/tiny-agenc-integration-corrupt-XXXXXX";
    char nonfinite[] = "/tmp/tiny-agenc-integration-nonfinite-XXXXXX";
    char truncated[]  = "/tmp/tiny-agenc-integration-truncated-XXXXXX";
    char trailing[]   = "/tmp/tiny-agenc-integration-trailing-XXXXXX";
    char resource[]   = "/tmp/tiny-agenc-integration-resource-XXXXXX";
    char too_large[]  = "/tmp/tiny-agenc-integration-too-large-XXXXXX";

    int paths_ready = temporary_path(checkpoint) == 0
                   && temporary_path(bad_magic) == 0
                   && temporary_path(bad_version) == 0
                   && temporary_path(bad_config) == 0
                   && temporary_path(bad_tokens) == 0
                   && temporary_path(corrupt) == 0
                   && temporary_path(nonfinite) == 0
                   && temporary_path(truncated) == 0
                   && temporary_path(trailing) == 0
                   && temporary_path(resource) == 0
                   && temporary_path(too_large) == 0;

    expect(paths_ready, "checkpoint tests reserve temporary paths");
    if (!paths_ready) {
        remove(checkpoint);
        remove(bad_magic);
        remove(bad_version);
        remove(bad_config);
        remove(bad_tokens);
        remove(corrupt);
        remove(nonfinite);
        remove(truncated);
        remove(trailing);
        remove(resource);
        remove(too_large);
        model_free(m);
        tokenizer_free(tk);
        return;
    }

    expect(model_save(m, tk, checkpoint) == 0, "trained model saves a checkpoint");

    Tokenizer *loaded_tk = NULL;
    Model     *loaded = model_load(&loaded_tk, checkpoint);

    expect(loaded != NULL && loaded_tk != NULL, "saved checkpoint loads");
    if (loaded != NULL && loaded_tk != NULL) {
        expect(models_equal(m, loaded), "checkpoint preserves every parameter bit");
        expect(tokenizer_vocab_size(loaded_tk) == tokenizer_vocab_size(tk),
               "checkpoint preserves vocabulary size");
        for (int id = 0; id < tokenizer_vocab_size(tk); id++)
            expect(tokenizer_decode(loaded_tk, id) == tokenizer_decode(tk, id),
                   "checkpoint preserves vocabulary ids");

        float loaded_loss =
            model_forward(loaded, inputs, targets, cfg.batch_size, cfg.block_size);

        expect(float_bits(loaded_loss) == float_bits(trained_loss),
               "checkpoint preserves forward loss exactly");

        int original_ids[SAMPLE_TOKENS] = { 0 };
        int repeated_ids[SAMPLE_TOKENS] = { 0 };
        int loaded_ids[SAMPLE_TOKENS]   = { 0 };
        int prompt[2];

        expect(tokenizer_encode(tk, prompt, "\na", 2) == 2,
               "sampling prompt encodes completely");
        memcpy(original_ids, prompt, sizeof prompt);
        memcpy(repeated_ids, prompt, sizeof prompt);
        memcpy(loaded_ids, prompt, sizeof prompt);

        Rng *original_rng = rng_new(777);
        Rng *repeated_rng = rng_new(777);
        Rng *loaded_rng   = rng_new(777);

        model_sample(m, original_rng, original_ids, 2, SAMPLE_TOKENS, 0.8f);
        model_sample(m, repeated_rng, repeated_ids, 2, SAMPLE_TOKENS, 0.8f);
        model_sample(loaded, loaded_rng, loaded_ids, 2, SAMPLE_TOKENS, 0.8f);

        expect(memcmp(original_ids, repeated_ids, sizeof original_ids) == 0,
               "seeded sampling replays exactly");
        expect(memcmp(original_ids, loaded_ids, sizeof original_ids) == 0,
               "loaded model continues the same seeded sample");
        for (int i = 2; i < SAMPLE_TOKENS; i++)
            expect(original_ids[i] >= 0 && original_ids[i] < cfg.vocab_size,
                   "sampling emits valid vocabulary ids");

        rng_free(original_rng);
        rng_free(repeated_rng);
        rng_free(loaded_rng);
    }

    expect(model_save(m, tk, bad_magic) == 0,
           "malformed checkpoint fixture saves");
    uint32_t legacy_magic = 0x4B524754u;

    expect(overwrite_file_bytes(bad_magic, 0, &legacy_magic,
                                sizeof legacy_magic) == 0
           && rewrite_checkpoint_checksum(bad_magic) == 0,
           "legacy checkpoint fixture rewrites magic and checksum");
    expect_load_failure(bad_magic, tk,
                        "checkpoint cleanly rejects the legacy format");

    int32_t changed_version = 2;

    expect(model_save(m, tk, bad_version) == 0,
           "bad-version checkpoint fixture saves");
    expect(overwrite_file_bytes(bad_version, (long)sizeof(int32_t),
                                &changed_version, sizeof changed_version) == 0
           && rewrite_checkpoint_checksum(bad_version) == 0,
           "bad-version checkpoint fixture changes its version");
    expect_load_failure(bad_version, tk, "checkpoint rejects unknown version");

    int32_t invalid_vocab_size = 0;

    expect(model_save(m, tk, bad_config) == 0,
           "invalid-config checkpoint fixture saves");
    expect(overwrite_file_bytes(bad_config, 2L * (long)sizeof(int32_t),
                                &invalid_vocab_size,
                                sizeof invalid_vocab_size) == 0
           && rewrite_checkpoint_checksum(bad_config) == 0,
           "invalid-config checkpoint fixture changes a dimension");
    expect_load_failure(bad_config, tk, "checkpoint rejects invalid dimensions");

    ModelConfig resource_bomb = {
        .vocab_size  = 1,
        .block_size  = 1024,
        .d_model     = 1,
        .head_count  = 1,
        .layer_count = 1,
        .batch_size  = 1024,
    };

    expect(write_resource_checkpoint(resource, resource_bomb) == 0,
           "resource-bomb fixture is complete and has a valid checksum");
    expect_load_failure(resource, tk,
                        "checkpoint rejects oversized arenas before payload parsing");

    FILE *too_large_stream = fopen(too_large, "r+b");

    expect(too_large_stream != NULL, "oversized-file fixture opens");
    if (too_large_stream != NULL) {
        expect(ftruncate(fileno(too_large_stream),
                         (off_t)MODEL_MAX_CHECKPOINT_FILE_BYTES + 1) == 0,
               "oversized-file fixture creates a sparse file");
        expect(fclose(too_large_stream) == 0,
               "oversized-file fixture closes");
        expect_load_failure(too_large, tk,
                            "checkpoint rejects oversized files before checksum work");
    }

    unsigned char descending_tokens[2] = { 'z', 'a' };
    long vocabulary_offset = 9L * (long)sizeof(int32_t);

    expect(model_save(m, tk, bad_tokens) == 0,
           "bad-tokenizer checkpoint fixture saves");
    expect(overwrite_file_bytes(bad_tokens, vocabulary_offset,
                                descending_tokens,
                                sizeof descending_tokens) == 0
           && rewrite_checkpoint_checksum(bad_tokens) == 0,
           "bad-tokenizer checkpoint fixture changes vocabulary order");
    expect_load_failure(bad_tokens, tk,
                        "checkpoint rejects malformed tokenizer order");

    uint32_t nan_bits = 0x7FC00000u;
    long parameter_offset =
        vocabulary_offset + (long)tokenizer_vocab_size(tk);

    unsigned char finite_bit_flip = 1;

    expect(model_save(m, tk, corrupt) == 0,
           "checksum-corruption checkpoint fixture saves");
    expect(overwrite_file_bytes(corrupt, parameter_offset,
                                &finite_bit_flip,
                                sizeof finite_bit_flip) == 0,
           "checksum-corruption fixture flips one finite payload byte");
    expect_load_failure(corrupt, tk,
                        "checkpoint checksum rejects finite bit corruption");

    expect(model_save(m, tk, nonfinite) == 0,
           "non-finite checkpoint fixture saves");
    expect(overwrite_file_bytes(nonfinite, parameter_offset,
                                &nan_bits, sizeof nan_bits) == 0
           && rewrite_checkpoint_checksum(nonfinite) == 0,
           "non-finite checkpoint fixture changes a parameter");
    expect_load_failure(nonfinite, tk,
                        "checkpoint rejects non-finite parameter payload");

    expect(model_save(m, tk, truncated) == 0,
           "truncated-payload checkpoint fixture saves");
    FILE *truncated_stream = fopen(truncated, "r+b");

    expect(truncated_stream != NULL, "truncated-payload fixture opens");
    if (truncated_stream != NULL) {
        expect(fseek(truncated_stream, -1, SEEK_END) == 0,
               "truncated-payload fixture finds its last byte");

        long shortened_size = ftell(truncated_stream);

        expect(shortened_size > 0, "truncated-payload fixture has a payload");
        expect(shortened_size > 0
               && ftruncate(fileno(truncated_stream), shortened_size) == 0,
               "truncated-payload fixture removes its last byte");
        expect(fclose(truncated_stream) == 0,
               "truncated-payload fixture closes");
        expect_load_failure(truncated, tk, "checkpoint rejects truncated payload");
    }

    expect(model_save(m, tk, trailing) == 0,
           "trailing-payload checkpoint fixture saves");
    FILE *trailing_stream = fopen(trailing, "ab");

    expect(trailing_stream != NULL, "trailing-payload fixture opens");
    if (trailing_stream != NULL) {
        unsigned char extra = 0xA5u;

        expect(fwrite(&extra, 1, 1, trailing_stream) == 1,
               "trailing-payload fixture appends a byte");
        expect(fclose(trailing_stream) == 0,
               "trailing-payload fixture closes");
        expect_load_failure(trailing, tk, "checkpoint rejects trailing payload");
    }

    ModelParams params = model_params(m);
    Mat first_values = param_values(params.params[0]);
    uint32_t saved_parameter_bits = float_bits(first_values.vals[0]);

    memcpy(&first_values.vals[0], &nan_bits, sizeof nan_bits);
    expect(model_save(m, tk, checkpoint) != 0,
           "checkpoint save rejects non-finite parameters");
    memcpy(&first_values.vals[0], &saved_parameter_bits,
           sizeof saved_parameter_bits);

    Tokenizer *preserved_tk = NULL;
    Model *preserved = model_load(&preserved_tk, checkpoint);

    expect(preserved != NULL && preserved_tk != NULL
           && models_equal(m, preserved),
           "failed atomic save preserves the previous checkpoint");
    if (preserved != NULL)
        model_free(preserved);
    if (preserved_tk != NULL)
        tokenizer_free(preserved_tk);

    if (loaded != NULL)
        model_free(loaded);
    if (loaded_tk != NULL)
        tokenizer_free(loaded_tk);
    remove(checkpoint);
    remove(bad_magic);
    remove(bad_version);
    remove(bad_config);
    remove(bad_tokens);
    remove(corrupt);
    remove(nonfinite);
    remove(truncated);
    remove(trailing);
    remove(resource);
    remove(too_large);
    model_free(m);
    tokenizer_free(tk);
}

static void usage(const char *program)
{
    fprintf(stderr,
            "usage: %s [foundations|data|mat|forward|parallel|model|optimizer|smoke]\n",
            program);
}

int main(int argc, char **argv)
{
    const char *group = argc == 2 ? argv[1] : "all";

    if (argc > 2) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (strcmp(group, "all") == 0 || strcmp(group, "foundations") == 0) {
        check_file_slurp();
        check_rng();
    }
    if (strcmp(group, "all") == 0 || strcmp(group, "data") == 0) {
        check_tokenizer();
        check_dataset();
    }
    if (strcmp(group, "all") == 0 || strcmp(group, "mat") == 0)
        check_mat();
    if (strcmp(group, "all") == 0 || strcmp(group, "forward") == 0)
        check_forward();
    if (strcmp(group, "all") == 0 || strcmp(group, "parallel") == 0)
        check_parallel_contracts();
    if (strcmp(group, "all") == 0 || strcmp(group, "model") == 0)
        check_model_contract();
    if (strcmp(group, "all") == 0 || strcmp(group, "optimizer") == 0)
        check_optimizer_integration();
    if (strcmp(group, "all") == 0 || strcmp(group, "smoke") == 0)
        check_training_checkpoint_and_sampling();

    if (checks == 0) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (failures > 0) {
        printf("integration[%s]: %d of %d checks FAILED\n",
               group, failures, checks);
        return EXIT_FAILURE;
    }
    printf("integration[%s]: all %d checks passed\n", group, checks);
    return EXIT_SUCCESS;
}
